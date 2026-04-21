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
    pool = new Pool({ connectionString: url });
    return pool;
  }

  const host = config.database?.host ?? 'localhost';
  const port = config.database?.port ?? 5432;
  const user = (config.database?.user || '').trim();
  const password = (config.database?.password || '').trim();
  const database = (config.database?.database || '').trim();
  if (!database) return null;

  pool = new Pool({ host, port, user, password, database });
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
 * Check if app is using local/Postgres (not Supabase).
 */
export function isUsingPostgres() {
  return (config.database?.url || '').trim().length > 0;
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
  } catch (_) {}

  return { hasVector: _hasVector };
}

export default { query, isUsingPostgres, getPool, ensureSchema, hasVector };
