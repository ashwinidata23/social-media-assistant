/**
 * PDF → text → chunk → embed → documents table (PostgreSQL or Supabase).
 * Uses LOCAL_DATABASE_URL (pg) when set; else Supabase client.
 */

// Supabase not required when using local Postgres (LOCAL_DATABASE_URL or DB_*).
// import { createClient } from '@supabase/supabase-js';
import { PDFParse } from 'pdf-parse';
import { config } from '../../config/index.js';
import { chunkText } from './chunk-text.js';
import { getEmbedding, formatVectorForSupabase } from './embedding-client.js';
import * as db from './db.js';
import { uploadDocumentToS3 } from './s3-upload.js';

// Only local Postgres is used for document ingest; no SUPABASE_URL or service key required.
// const supabase =
//   !db.isUsingPostgres() && config.supabase.url && config.supabase.serviceKey
//     ? createClient(config.supabase.url, config.supabase.serviceKey)
//     : null;
const supabase = null;

const CHUNK_SIZE = 600;
const CHUNK_OVERLAP = 100;

/**
 * Ingest a PDF buffer into Supabase documents table.
 * Extracts text, chunks it, gets embeddings from EMBEDDING_SERVICE_URL, stores each chunk with embedding.
 * If EMBEDDING_SERVICE_URL is not set, stores a single row without embedding (fallback).
 * @param {Buffer} buffer - Raw PDF file buffer
 * @param {{ filename?: string }} options - Optional filename for metadata
 * @returns {Promise<{ ok: boolean, id?: string, ids?: string[], chunks?: number, error?: string, pages?: number }>}
 */
export async function ingestPdf(buffer, options = {}) {
  const startTime = Date.now();
  const usePg = db.isUsingPostgres();
  // company_id can be string or int; normalize to string for DB (text column).
  const raw = options.companyId ?? config.defaults?.companyId ?? null;
  const companyId = raw != null && raw !== '' ? String(raw) : null;
  console.log(`\n${'='.repeat(60)}`);
  console.log(`[ingest-pdf] ▶ START  |  DB: ${usePg ? 'PostgreSQL (local)' : supabase ? 'Supabase' : 'NONE'}  |  company_id: ${companyId || '(none)'}`);

  if (!usePg) {
    const dbCfg = config.database || {};
    console.log('[ingest-pdf] ✗ No database configured. Aborting.');
    console.log('[ingest-pdf]   DB config from .env:', {
      hasUrl: !!(dbCfg.url && dbCfg.url.trim()),
      host: dbCfg.host || '(not set)',
      user: dbCfg.user || '(not set)',
      database: dbCfg.database || '(not set)',
      port: dbCfg.port,
    });
    return { ok: false, error: 'Database not configured. Set LOCAL_DATABASE_URL or DB_HOST, DB_USER, DB_PASSWORD, and DB_NAME in .env' };
  }

  const filename = options.filename ?? 'upload.pdf';
  const uid = options.userId || 'anon';
  const wid = options.workspaceId || 'default';

  // --- Step 0: Upload to S3 ---
  console.log('[ingest-pdf] Step 0 — Uploading to S3...');
  const s3Start = Date.now();
  let s3Details = {};
  // If caller provides an existing uploadId (e.g. PUT update), reuse it — skip creating a new upload row.
  let uploadId = options.existingUploadId || null;

  if (usePg) {
    try {
      const uploadResult = await uploadDocumentToS3(buffer, {
        filename, userId: uid, workspaceId: wid, contentType: options.mimeType || 'application/pdf'
      });
      s3Details = {
        s3_url: uploadResult.s3Url,
        s3_key: uploadResult.key,
        s3_bucket: uploadResult.bucket
      };
      console.log(`[ingest-pdf]   ✓ Uploaded to S3: ${uploadResult.s3Url} (${Date.now() - s3Start}ms)`);

      if (!uploadId) {
        // New upload — create a row in knowledge_document_uploads
        const { rows } = await db.query(
          `INSERT INTO public.knowledge_document_uploads
          (user_id, workspace_id, s3_url, original_filename, mime_type, file_size_bytes)
          VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`,
          [uid, wid, uploadResult.s3Url, filename, uploadResult.contentType, options.sizeBytes || buffer.length]
        );
        if (rows.length > 0) {
          uploadId = rows[0].id;
          console.log(`[ingest-pdf]   ✓ Upload tracked in database: ${uploadId}`);
        }
      } else {
        console.log(`[ingest-pdf]   ✓ Reusing existing uploadId: ${uploadId} (update mode)`);
      }
    } catch (e) {
      console.log(`[ingest-pdf]   ✗ S3 Upload/DB Insert failed: ${e.message}`);
      return { ok: false, error: `S3 Upload/Tracking failed: ${e.message}` };
    }
  }

  // --- Step 1: Parse PDF ---
  console.log('[ingest-pdf] Step 1/4 — Parsing PDF...');
  const parseStart = Date.now();
  let text;
  let numpages;
  let parser;
  try {
    parser = new PDFParse({ data: buffer });
    const result = await parser.getText();
    text = result?.text ?? '';
    numpages = result?.total ?? 0;
    console.log(`[ingest-pdf]   ✓ Parsed ${numpages} page(s), ${text.length} chars extracted  (${Date.now() - parseStart}ms)`);
  } catch (e) {
    console.log(`[ingest-pdf]   ✗ PDF parse failed: ${e.message}`);
    return { ok: false, error: `PDF parse failed: ${e.message}` };
  } finally {
    if (parser) await parser.destroy().catch(() => { });
  }

  const trimmed = (text || '').trim();
  if (!trimmed) {
    console.log('[ingest-pdf]   ✗ PDF produced no text (might be image-only or empty)');
    return { ok: false, error: 'PDF produced no text (might be image-only or empty)' };
  }

  const hasExternalEmbed = (config.embedding?.serviceUrl || '').trim().length > 0;
  // Embeddings are generated ONLY via external service (no OpenAI embeddings).
  const useEmbedding = hasExternalEmbed;
  const embedSource = hasExternalEmbed ? config.embedding.serviceUrl : 'none';
  console.log(`[ingest-pdf]   File: ${filename}  |  Embedding: ${embedSource}`);

  if (useEmbedding) {
    // --- Step 2: Chunk text ---
    console.log(`[ingest-pdf] Step 2/4 — Chunking text (size=${CHUNK_SIZE}, overlap=${CHUNK_OVERLAP})...`);
    const chunks = chunkText(trimmed, { chunkSize: CHUNK_SIZE, overlap: CHUNK_OVERLAP });
    if (chunks.length === 0) {
      console.log('[ingest-pdf]   ✗ Chunking produced 0 chunks. Aborting.');
      return { ok: false, error: 'Chunking produced no chunks' };
    }
    console.log(`[ingest-pdf]   ✓ ${chunks.length} chunk(s) created  [avg ${Math.round(trimmed.length / chunks.length)} chars/chunk]`);

    // --- Step 3: Embed + Store each chunk ---
    console.log(`[ingest-pdf] Step 3/4 — Embedding & storing ${chunks.length} chunks...`);
    const ids = [];
    let embeddingOk = 0;
    let embeddingFailed = false;
    // If pgvector/embedding column is misconfigured, trying to insert vectors will fail every time.
    // We'll probe once (or trust db.hasVector()) and then decide the store mode.
    let canStoreEmbeddingInDb = usePg ? db.hasVector() : true;
    let embeddingStoreProbed = !usePg || canStoreEmbeddingInDb;
    for (let i = 0; i < chunks.length; i++) {
      const content = chunks[i];
      let embeddingStr = null;
      const chunkLabel = `  chunk ${i + 1}/${chunks.length}`;

      // 3a: embed
      try {
        const embedStart = Date.now();
        const vec = await getEmbedding(content);
        embeddingStr = formatVectorForSupabase(vec);
        embeddingOk++;
        console.log(`[ingest-pdf] ${chunkLabel} → embedded (${vec.length}d vector, ${Date.now() - embedStart}ms)`);
      } catch (e) {
        embeddingFailed = true;
        console.warn(`[ingest-pdf] ${chunkLabel} → embedding FAILED: ${e.message}`);
      }

      // 3b: store
      const metadata = {
        source: 'pdf',
        filename,
        pages: numpages,
        chunk_index: i,
        total_chunks: chunks.length,
        ...s3Details
      };
      const storeStart = Date.now();
      if (usePg) {
        // Try to store embeddings whenever we have them. If the DB rejects it (no column, no pgvector,
        // wrong vector dims, etc.), fall back to text-only and stop trying for subsequent chunks.
        if (embeddingStr !== null && (canStoreEmbeddingInDb || !embeddingStoreProbed)) {
          try {
            const { rows } = await db.query(
              'INSERT INTO public.documents (company_id, content, metadata, embedding, source_upload_id) VALUES ($1, $2, $3, $4::vector, $5) RETURNING id',
              [companyId, content, JSON.stringify(metadata), embeddingStr, uploadId]
            );
            embeddingStoreProbed = true;
            canStoreEmbeddingInDb = true;
            if (rows[0]?.id) ids.push(rows[0].id);
            console.log(
              `[ingest-pdf] ${chunkLabel} → stored WITH vector  id=${rows[0]?.id}  (${Date.now() - storeStart}ms)`
            );
            continue;
          } catch (e) {
            embeddingStoreProbed = true;
            canStoreEmbeddingInDb = false;
            console.warn(
              `[ingest-pdf] ${chunkLabel} → embedding column insert failed; falling back to text-only: ${e.message}`
            );
          }
        }

        const { rows } = await db.query(
          'INSERT INTO public.documents (company_id, content, metadata, source_upload_id) VALUES ($1, $2, $3, $4) RETURNING id',
          [companyId, content, JSON.stringify(metadata), uploadId]
        );
        if (rows[0]?.id) ids.push(rows[0].id);
        console.log(`[ingest-pdf] ${chunkLabel} → stored text-only  id=${rows[0]?.id}  (${Date.now() - storeStart}ms)`);
      } else {
        const insertPayload = { content, metadata, ...(companyId && { company_id: companyId }), ...(embeddingStr && { embedding: embeddingStr }) };
        const { data: row, error } = await supabase
          .from('documents')
          .insert(insertPayload)
          .select('id')
          .single();
        if (error) {
          console.log(`[ingest-pdf] ${chunkLabel} → Supabase INSERT failed: ${error.message}`);
          return { ok: false, error: error.message };
        }
        if (row?.id) ids.push(row.id);
        console.log(`[ingest-pdf] ${chunkLabel} → stored in Supabase  id=${row?.id}  (${Date.now() - storeStart}ms)`);
      }
    }

    // --- Step 4: Summary ---
    const elapsed = Date.now() - startTime;
    console.log(`[ingest-pdf] Step 4/4 — Done!`);
    console.log(`[ingest-pdf]   ✓ ${ids.length}/${chunks.length} chunks stored  |  embeddings: ${embeddingOk}/${chunks.length} ok`);
    if (embeddingFailed) console.log(`[ingest-pdf]   ⚠ Some embeddings failed — stored for text-only search`);
    console.log(`[ingest-pdf] ▶ FINISH  |  ${elapsed}ms total  |  ${numpages} pages → ${chunks.length} chunks → ${ids.length} rows`);
    console.log('='.repeat(60) + '\n');

    return {
      ok: true,
      uploadId,
      ids,
      chunks: chunks.length,
      pages: numpages,
      ...(embeddingFailed && { warning: 'Embedding service unavailable; documents stored for text-only search.' }),
    };
  }

  // No embedding service — store entire text as one row
  console.log('[ingest-pdf] Step 2/4 — No embedding service; storing full text as single document...');
  const metadata = { source: 'pdf', filename, pages: numpages, ...s3Details };
  if (usePg) {
    const storeStart = Date.now();
    const { rows } = await db.query(
      'INSERT INTO public.documents (company_id, content, metadata, source_upload_id) VALUES ($1, $2, $3, $4) RETURNING id',
      [companyId, trimmed, JSON.stringify(metadata), uploadId]
    );
    const elapsed = Date.now() - startTime;
    console.log(`[ingest-pdf]   ✓ Stored as single row  id=${rows[0]?.id}  (${Date.now() - storeStart}ms)`);
    console.log(`[ingest-pdf] ▶ FINISH  |  ${elapsed}ms total  |  ${numpages} pages → 1 row (no chunking)`);
    console.log('='.repeat(60) + '\n');
    return { ok: true, uploadId, id: rows[0]?.id, pages: numpages };
  }
  const singlePayload = { content: trimmed, metadata, ...(companyId && { company_id: companyId }) };
  const { data, error } = await supabase
    .from('documents')
    .insert(singlePayload)
    .select('id')
    .single();
  if (error) {
    console.log(`[ingest-pdf]   ✗ Supabase insert failed: ${error.message}`);
    return { ok: false, error: error.message };
  }
  const elapsed = Date.now() - startTime;
  console.log(`[ingest-pdf]   ✓ Stored in Supabase  id=${data?.id}  (${elapsed}ms)`);
  console.log(`[ingest-pdf] ▶ FINISH  |  ${elapsed}ms total  |  ${numpages} pages → 1 row (no chunking)`);
  console.log('='.repeat(60) + '\n');
  return { ok: true, uploadId, id: data?.id, pages: numpages };
}

export default ingestPdf;
