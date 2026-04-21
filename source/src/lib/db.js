/**
 * PostgreSQL client (pg). Used when DATABASE_URL is set, or when DB_HOST/DB_USER/DB_PASSWORD/DB_NAME are set.
 * Replaces Supabase for local / self-hosted Postgres + pgvector.
 */

import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { config } from '../../config/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { Pool } = pg;

let pool = null;
/** True if the documents table has an embedding (vector) column. Set by ensureSchema(). */
let _hasVector = null;
/** Expected embedding dimension for pgvector columns (must match embedding service output). */
const EXPECTED_VECTOR_DIM = 768;

function isDatabaseConfigured() {
  const url = config.database?.url || '';
  const useParts = config.database?.host != null && (config.database?.database || config.database?.user);
  return url.length > 0 || (useParts && (config.database.database || '').trim().length > 0);
}

function getPool() {
  if (!isDatabaseConfigured()) return null;
  if (pool) return pool;

  const url = (config.database?.url || '').trim();
  if (url) {
    // Enable SSL for cloud DBs (AWS RDS etc.): set DB_SSL=true or include ?sslmode=require in DATABASE_URL
    const needsSsl = config.database?.ssl || url.includes('sslmode=require') || url.includes('rds.amazonaws.com');
    pool = new Pool({
      connectionString: url,
      ...(needsSsl ? { ssl: { rejectUnauthorized: false } } : {}),
    });
    return pool;
  }

  const host = config.database?.host ?? 'localhost';
  const port = config.database?.port ?? 5432;
  const user = (config.database?.user || '').trim();
  const password = (config.database?.password || '').trim();
  const database = (config.database?.database || '').trim();
  if (!database) return null;

  const isLocalHost = ['localhost', '127.0.0.1'].includes((host || '').toLowerCase());
  const useSsl = config.database?.ssl && !isLocalHost;

  const ssl = useSsl ? { rejectUnauthorized: false } : false;

  pool = new Pool({ host, port, user, password, database, ssl });
  return pool;
}

/**
 * Run a query. Use when LOCAL_DATABASE_URL or DATABASE_URL is set.
 * @param {string} text - SQL query
 * @param {any[]} [params] - Query parameters
 * @returns {Promise<{ rows: any[], rowCount: number }>}
 */
export async function query(text, params = []) {
  const p = getPool();
  if (!p) throw new Error('Database not configured. Set LOCAL_DATABASE_URL or DATABASE_URL in .env');
  const result = await p.query(text, params);
  return { rows: result.rows || [], rowCount: result.rowCount ?? 0 };
}

/**
 * Check if app is using local/Postgres (URL or DB_HOST/DB_USER/DB_NAME).
 */
export function isUsingPostgres() {
  return isDatabaseConfigured();
}

/**
 * True if the DB has the embedding (vector) column on public.documents.
 * Call ensureSchema() first so this is accurate.
 */
export function hasVector() {
  return _hasVector === true;
}

/**
 * Create tables and match_documents function. Safe to call on every startup.
 * 1) Runs base schema (no pgvector): documents, scheduled_posts, audience_insights, text-only match_documents.
 * 2) Tries full schema (pgvector): extension, embedding column, vector match_documents. On failure, keeps base schema.
 */
export async function ensureSchema() {
  const p = getPool();
  if (!p) throw new Error('Database not configured. Set LOCAL_DATABASE_URL or DATABASE_URL in .env');

  const basePath = path.join(__dirname, '..', '..', 'supabase', 'schema-base-no-vector.sql');
  const fullPath = path.join(__dirname, '..', '..', 'supabase', 'schema-with-embeddings.sql');

  const baseSql = fs.readFileSync(basePath, 'utf8');
  const fullSql = fs.readFileSync(fullPath, 'utf8');

  try {
    await p.query(baseSql);
  } catch (e) {
    throw new Error(`Base schema failed: ${e.message}`);
  }

  _hasVector = false;
  try {
    await p.query(fullSql);
    _hasVector = true;
  } catch (e) {
    const msg = (e.message || '').toLowerCase();
    if (msg.includes('vector') || msg.includes('extension')) {
      console.warn('[db] pgvector not available; using text-only search. Install pgvector for semantic search.');
    } else {
      console.warn('[db] Full schema skipped:', e.message);
    }
  }

  try {
    const { rows } = await p.query(
      `
      SELECT format_type(a.atttypid, a.atttypmod) AS type
      FROM pg_attribute a
      JOIN pg_class c ON c.oid = a.attrelid
      JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'public'
        AND c.relname = 'documents'
        AND a.attname = 'embedding'
        AND a.attnum > 0
        AND NOT a.attisdropped
      LIMIT 1
      `
    );

    const colType = rows?.[0]?.type || '';
    const m = String(colType).match(/vector\((\d+)\)/i);
    const dim = m ? Number(m[1]) : null;

    if (!colType) {
      _hasVector = false;
    } else if (dim === EXPECTED_VECTOR_DIM) {
      _hasVector = true;
    } else {
      _hasVector = false;
      console.warn(
        `[db] Found embedding column type "${colType}", but expected vector(${EXPECTED_VECTOR_DIM}). ` +
        `Update your schema to vector(${EXPECTED_VECTOR_DIM}) (or re-create the column) to enable semantic search.`
      );
    }
  } catch (_) { }

  return { hasVector: _hasVector };
}

/**
 * Helper to extract integer from string (e.g. "user-123" -> 123).
 * Falls back to null if not parsable.
 */
function parseIntegerId(id) {
  if (id == null) return null;
  const num = parseInt(String(id).replace(/[^0-9]/g, ''), 10);
  return isNaN(num) ? null : num;
}

/**
 * Fetch active platforms for a given workspace using the external API instead of DB queries.
 * @param {string|number} userId      (kept for backwards compatibility, not used for filtering here)
 * @param {string|number} workspaceId The workspace whose connected platforms we want
 * @returns {Promise<string[]|null>} Array of active platform names, or null on hard failure
 */
export async function getActivePlatforms(userId, workspaceId) {
  if (!workspaceId) return null;

  const numWorkspaceId = parseIntegerId(workspaceId);
  if (numWorkspaceId === null) return null;

  try {
    const apiUrl = `https://devapi.zunosync.com/api/users/workspaces-ai/${numWorkspaceId}/social-accounts`;
    const response = await fetch(apiUrl);

    if (!response.ok) {
      console.warn(`[db] Error fetching active platforms from API: HTTP ${response.status}`);
      return null;
    }

    const data = await response.json();

    if (data.Status !== 1 || !Array.isArray(data.SocialAccounts)) {
      console.warn('[db] Unexpected API response shape or no social accounts.', data);
      return [];
    }

    // Filter only active ones, get the distinct platform names
    const activePlatforms = data.SocialAccounts
      .filter(account => account.isActive === true)
      .map(account => (account.platform || '').toLowerCase());

    const platforms = Array.from(new Set(activePlatforms));

    console.log(
      '[db] getActivePlatforms (API) | workspaceId=%s | Count=%s | platforms=%j',
      numWorkspaceId,
      data.Count,
      platforms,
    );

    return platforms;
  } catch (e) {
    console.warn('[db] Error fetching active platforms from API:', e?.message || e);
    return null;
  }
}

/**
 * Fetch complete social account details for a given workspace using the external API.
 * This can be used later if you need access to the full JSON including nested pages, followers, etc.
 * @param {string|number} workspaceId
 */
export async function getDetailedSocialAccounts(workspaceId) {
  if (!workspaceId) return null;

  const numWorkspaceId = parseIntegerId(workspaceId);
  if (numWorkspaceId === null) return null;

  try {
    const apiUrl = `https://devapi.zunosync.com/api/users/workspaces-ai/${numWorkspaceId}/social-accounts`;
    const response = await fetch(apiUrl);

    if (!response.ok) {
      console.warn(`[db] Error fetching detailed social accounts from API: HTTP ${response.status}`);
      return null;
    }

    const data = await response.json();
    return data;
  } catch (e) {
    console.warn('[db] Error fetching detailed social accounts from API:', e?.message || e);
    return null;
  }
}

/**
 * Return a flattened list of active accounts (including per-page entries) for a workspace.
 * This does not change orchestration behaviour; it is only for consumers that
 * explicitly want userId/accountId/accountName lists.
 */
export async function getActiveAccounts(userId, workspaceId) {
  const data = await getDetailedSocialAccounts(workspaceId);
  if (!data || !Array.isArray(data.SocialAccounts)) return [];

  function toBool(v) {
    if (v === true) return true;
    if (v === false) return false;
    if (v == null) return null;
    const s = String(v).trim().toLowerCase();
    if (['true', '1', 'yes', 'y', 'active', 'enabled'].includes(s)) return true;
    if (['false', '0', 'no', 'n', 'inactive', 'disabled'].includes(s)) return false;
    return null;
  }

  const accounts = [];

  for (const acc of data.SocialAccounts) {
    // Upstream API isn't always consistent about the "active" flag shape.
    // Prefer an explicit false to exclude; otherwise include.
    const activeFlag =
      toBool(acc?.isActive) ??
      toBool(acc?.is_active) ??
      toBool(acc?.active) ??
      toBool(acc?.enabled) ??
      null;
    if (activeFlag === false) continue;

    const platform = (acc?.platform || acc?.platformName || acc?.provider || '').toLowerCase();
    if (!platform) continue;

    const pages =
      (Array.isArray(acc.pages) && acc.pages) ||
      (Array.isArray(acc.Pages) && acc.Pages) ||
      (Array.isArray(acc.accounts) && acc.accounts) ||
      (Array.isArray(acc.Accounts) && acc.Accounts) ||
      [];

    if (Array.isArray(pages) && pages.length > 0) {
      for (const page of pages) {
        const pageName =
          page?.page_name ||
          page?.pageName ||
          page?.organization_name ||
          page?.organizationName ||
          page?.name ||
          acc.accountName ||
          acc.account_name ||
          acc.username ||
          acc.handle ||
          acc.socialAccountId;
        const pageId =
          page?.page_id ||
          page?.pageId ||
          page?.organization_id ||
          page?.organizationId ||
          page?.id ||
          acc.socialAccountId;

        accounts.push({
          platform,
          accountName: pageName,
          accountId: pageId,
          type: acc.accountType || 'page',
        });
      }
    } else {
      accounts.push({
        platform,
        accountName: acc.accountName || acc.account_name || acc.username || acc.handle || acc.socialAccountId,
        accountId: acc.socialAccountId,
        type: acc.accountType || 'personal',
      });
    }
  }

  const unique = [];
  const seen = new Set();
  for (const acc of accounts) {
    if (!seen.has(acc.accountId)) {
      seen.add(acc.accountId);
      unique.push(acc);
    }
  }

  console.log('[db] getActiveAccounts (API) parsed %s unique accounts from workspace %s', unique.length, workspaceId);
  return unique;
}

/**
 * Get all uploads for a workspace that have at least one ingested chunk.
 * @param {string} workspaceId
 * @returns {Promise<Array<{ id: string, original_filename: string, created_at: Date }>>}
 */
export async function getWorkspaceUploads(workspaceId) {
  if (!workspaceId) return [];
  try {
    const withChunksSql = `
      SELECT u.id, u.original_filename, u.created_at
      FROM public.knowledge_document_uploads u
      WHERE {WORKSPACE_FILTER}
        AND EXISTS (
          SELECT 1 FROM public.documents d
          WHERE d.source_upload_id = u.id
             OR d.metadata->>'filename' = u.original_filename
        )
      ORDER BY u.created_at DESC`;

    // 1st try: exact workspace match
    const { rows: exactRows } = await query(
      withChunksSql.replace('{WORKSPACE_FILTER}', 'u.workspace_id = $1'),
      [String(workspaceId)]
    );
    if (exactRows.length > 0) {
      console.log(`[db] getWorkspaceUploads workspace=${workspaceId} → ${exactRows.length} upload(s) (exact match)`);
      return exactRows;
    }

    // 2nd try: no workspace filter (uploads stored under a different workspace_id in dev/test env)
    const { rows: anyRows } = await query(
      withChunksSql.replace('{WORKSPACE_FILTER}', '1=1')
    );
    if (anyRows.length > 0) {
      console.log(`[db] getWorkspaceUploads workspace=${workspaceId} → 0 exact, falling back to all workspaces → ${anyRows.length} upload(s)`);
    }
    return anyRows || [];
  } catch (e) {
    console.warn('[db] getWorkspaceUploads failed:', e.message);
    return [];
  }
}

/**
 * Get all document chunks for a specific upload, ordered by chunk_index.
 * Matches by source_upload_id first; falls back to metadata->>'filename' match.
 * @param {string} uploadId - UUID from knowledge_document_uploads.id
 * @param {string} [filename] - Original filename for metadata fallback
 * @param {number} [limit=50] - Safety cap
 * @returns {Promise<Array<{ id: string, content: string, metadata: object }>>}
 */
export async function getChunksByUploadId(uploadId, filename = null, limit = 50) {
  if (!uploadId && !filename) return [];
  try {
    const { rows } = await query(
      `SELECT id, content, metadata
       FROM public.documents
       WHERE source_upload_id = $1
          OR (source_upload_id IS NULL AND $2::text IS NOT NULL AND metadata->>'filename' = $2)
       ORDER BY (metadata->>'chunk_index')::int ASC NULLS LAST
       LIMIT $3`,
      [uploadId ? String(uploadId) : null, filename ? String(filename) : null, limit]
    );
    return rows || [];
  } catch (e) {
    console.warn('[db] getChunksByUploadId failed:', e.message);
    return [];
  }
}

/**
 * Delete an upload and all its document chunks from both tables.
 * Deletes from `documents` (by source_upload_id) first, then `knowledge_document_uploads`.
 * Scoped by userId + workspaceId so users can only delete their own docs.
 * @param {string} uploadId - UUID from knowledge_document_uploads.id
 * @param {string} userId
 * @param {string} workspaceId
 * @returns {Promise<{ deleted: boolean, chunksRemoved: number }>}
 */
export async function deleteUploadAndChunks(uploadId, userId, workspaceId) {
  if (!uploadId || !userId || !workspaceId) {
    throw new Error('deleteUploadAndChunks requires uploadId, userId, and workspaceId');
  }

  // 1. Verify the upload belongs to this user + workspace before deleting anything
  const { rows: ownerCheck } = await query(
    'SELECT id FROM public.knowledge_document_uploads WHERE id = $1 AND user_id = $2 AND workspace_id = $3',
    [uploadId, userId, workspaceId]
  );
  if (ownerCheck.length === 0) {
    return { deleted: false, chunksRemoved: 0 };
  }

  // 2. Delete all chunks linked to this upload
  const { rowCount: chunksRemoved } = await query(
    'DELETE FROM public.documents WHERE source_upload_id = $1',
    [uploadId]
  );
  console.log(`[db] deleteUploadAndChunks: removed ${chunksRemoved} chunk(s) for upload ${uploadId}`);

  // 3. Delete the upload record itself
  await query(
    'DELETE FROM public.knowledge_document_uploads WHERE id = $1 AND user_id = $2 AND workspace_id = $3',
    [uploadId, userId, workspaceId]
  );
  console.log(`[db] deleteUploadAndChunks: removed upload record ${uploadId}`);

  return { deleted: true, chunksRemoved };
}

/**
 * Update metadata (e.g. filename) of an existing upload.
 * Scoped by userId + workspaceId.
 * @param {string} uploadId
 * @param {string} userId
 * @param {string} workspaceId
 * @param {{ original_filename?: string }} fields - Fields to update
 * @returns {Promise<boolean>} true if a row was updated
 */
export async function updateUpload(uploadId, userId, workspaceId, fields = {}) {
  if (!uploadId || !userId || !workspaceId) {
    throw new Error('updateUpload requires uploadId, userId, and workspaceId');
  }

  const sets = [];
  const params = [];
  let idx = 1;

  if (fields.original_filename) {
    sets.push(`original_filename = $${idx++}`);
    params.push(fields.original_filename);
  }

  if (sets.length === 0) {
    throw new Error('updateUpload: no fields to update');
  }

  params.push(uploadId, userId, workspaceId);
  const { rowCount } = await query(
    `UPDATE public.knowledge_document_uploads SET ${sets.join(', ')} WHERE id = $${idx++} AND user_id = $${idx++} AND workspace_id = $${idx}`,
    params
  );
  return rowCount > 0;
}

/**
 * Replace a document: delete only the old chunks, update the upload row (keep same ID),
 * and return the uploadId so the caller can re-ingest chunks under the same ID.
 * @param {string} uploadId
 * @param {string} userId
 * @param {string} workspaceId
 * @param {{ s3_url?: string, original_filename?: string, file_size_bytes?: number, mime_type?: string }} updateFields
 * @returns {Promise<{ found: boolean, chunksRemoved: number }>}
 */
export async function replaceUploadChunks(uploadId, userId, workspaceId, updateFields = {}) {
  if (!uploadId || !userId || !workspaceId) {
    throw new Error('replaceUploadChunks requires uploadId, userId, and workspaceId');
  }

  // 1. Verify ownership
  const { rows: ownerCheck } = await query(
    'SELECT id FROM public.knowledge_document_uploads WHERE id = $1 AND user_id = $2 AND workspace_id = $3',
    [uploadId, userId, workspaceId]
  );
  if (ownerCheck.length === 0) {
    return { found: false, chunksRemoved: 0 };
  }

  // 2. Delete old chunks only (keep the upload row)
  const { rowCount: chunksRemoved } = await query(
    'DELETE FROM public.documents WHERE source_upload_id = $1',
    [uploadId]
  );
  console.log(`[db] replaceUploadChunks: removed ${chunksRemoved} old chunk(s) for upload ${uploadId}`);

  // 3. Update the upload row with new file info
  const sets = [];
  const params = [];
  let idx = 1;
  if (updateFields.s3_url) { sets.push(`s3_url = $${idx++}`); params.push(updateFields.s3_url); }
  if (updateFields.original_filename) { sets.push(`original_filename = $${idx++}`); params.push(updateFields.original_filename); }
  if (updateFields.file_size_bytes != null) { sets.push(`file_size_bytes = $${idx++}`); params.push(updateFields.file_size_bytes); }
  if (updateFields.mime_type) { sets.push(`mime_type = $${idx++}`); params.push(updateFields.mime_type); }

  if (sets.length > 0) {
    params.push(uploadId);
    await query(
      `UPDATE public.knowledge_document_uploads SET ${sets.join(', ')} WHERE id = $${idx}`,
      params
    );
    console.log(`[db] replaceUploadChunks: updated upload row ${uploadId}`);
  }

  return { found: true, chunksRemoved };
}

export default {
  query,
  isUsingPostgres,
  getPool,
  ensureSchema,
  hasVector,
  getActivePlatforms,
  getDetailedSocialAccounts,
  getActiveAccounts,
  getWorkspaceUploads,
  getChunksByUploadId,
  deleteUploadAndChunks,
  updateUpload,
  replaceUploadChunks,
};


