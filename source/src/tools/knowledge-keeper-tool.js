import { createClient } from '@supabase/supabase-js';
import { config } from '../../config/index.js';
import { getEmbedding, formatVectorForSupabase } from '../lib/embedding-client.js';
import * as db from '../lib/db.js';
import { getWorkspaceUploads, getChunksByUploadId } from '../lib/db.js';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

function clampNumber(n, min, max, fallback) {
  const x = Number(n);
  if (!Number.isFinite(x)) return fallback;
  return Math.min(max, Math.max(min, x));
}

// ── "given document" intent detection ────────────────────────────────────────
// Matches explicit doc references ("the uploaded pdf") AND generic knowledge
// references ("the data I provided", "use my knowledge", "info I gave you").
const DOC_REFERENCE_RE = new RegExp([
  // Explicit doc references: "the uploaded document", "this pdf", "that file"
  /\b(given|uploaded|this|the|that|my)\s+(document|doc|pdf|file|upload)\b/.source,
  // Action + doc target: "from the document", "based on the file", "use the upload"
  /\b(from|based on|using|use|refer to|check)\s+(the\s+|my\s+)?(document|doc|pdf|file|upload|knowledge\s*base)\b/.source,
  // Generic knowledge references: "the data I provided", "my knowledge", "the info"
  /\b(the|my|this|that)\s+(data|knowledge|info|information|content|material|stuff|resource|reference)\s+(i\s+)?(provided|uploaded|gave|shared|added|submitted)\b/.source,
  // Shorter generic: "use my data", "from my knowledge", "based on my info"
  /\b(from|based on|using|use|with)\s+(the\s+|my\s+)(data|knowledge|info|information|content|material|stuff|resources?|references?)\b/.source,
  // Possessive patterns: "my uploaded data", "the provided information"
  /\b(my|the)\s+(uploaded|provided|given|shared|added)\s+(data|knowledge|info|information|content|material|files?|docs?|documents?)\b/.source,
  // Short references: "what I uploaded", "what I provided", "what I gave you"
  /\bwhat\s+i\s+(uploaded|provided|gave|shared|added|submitted)\b/.source,
].join('|'), 'i');

function isDocumentReferenceIntent(prompt) {
  return DOC_REFERENCE_RE.test(String(prompt || ''));
}

/**
 * Extract lowercase search tokens from a filename (strip extension, UUID suffixes, split on
 * non-alphanumeric boundaries). Returns tokens with length >= 3.
 */
function filenameTokens(filename) {
  return String(filename || '')
    .replace(/\.[^.]+$/, '')             // strip extension
    .replace(/[_-]/g, ' ')              // separators → space
    .replace(/[^a-zA-Z0-9 ]/g, ' ')    // strip non-alphanum
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length >= 3 && !/^[0-9a-f]{6,}$/.test(t)); // skip hex UUIDs
}

/**
 * Given a list of uploads and the user prompt, return the best matching upload.
 * Scoring: count how many filename tokens appear in the (lowercased) prompt.
 * Returns null if no upload scores > 0.
 */
function matchUploadByName(uploads, prompt) {
  const lp = String(prompt || '').toLowerCase();
  let best = null;
  let bestScore = 0;
  for (const u of uploads) {
    const tokens = filenameTokens(u.original_filename);
    const score = tokens.filter((t) => lp.includes(t)).length;
    if (score > bestScore) {
      bestScore = score;
      best = u;
    }
  }
  return bestScore > 0 ? best : null;
}

/**
 * Fetch all chunks for a specific upload (doc-scoped retrieval).
 * Returns { chunks } in the same shape as searchKnowledgeBase.
 */
async function fetchAllChunksForUpload(uploadId, filename) {
  const rows = await getChunksByUploadId(uploadId, filename, 50);
  console.log(`[KnowledgeKeeper] Doc-scoped: upload="${filename}" → ${rows.length} chunk(s) fetched`);
  return {
    chunks: rows.map((r) => ({ content: r.content, metadata: r.metadata || {} })),
    docScoped: true,
    uploadId,
    filename,
  };
}


const supabase =
  !db.isUsingPostgres() && config.supabase.url && config.supabase.serviceKey
    ? createClient(config.supabase.url, config.supabase.serviceKey)
    : null;

const KnowledgeKeeperSearchSchema = z.object({
  query: z.string().describe('Natural language search query over brand knowledge (PDFs, URLs, docs)'),
  topK: z.number().optional().default(5).describe('Max number of chunks to return'),
});

/**
 * Searches the Knowledge Base (vector store) and returns relevant facts/chunks.
 * When EMBEDDING_SERVICE_URL is set, uses query embedding for vector similarity search; else text search.
 * @param {string} query - Search query
 * @param {number} [topK=5] - Max chunks to return
 * @param {string|number|null} [companyId] - Optional company id to filter documents
 * @param {string|null} [workspaceId] - Optional workspace id; enables doc-scoped retrieval
 */
export async function searchKnowledgeBase(query, topK = 5, companyId = null, workspaceId = null) {
  const usePg = db.isUsingPostgres();

  if (!usePg && !supabase) {
    return { chunks: [], message: 'Database not configured. Set LOCAL_DATABASE_URL or SUPABASE_URL.' };
  }

  // ── Doc-scoped retrieval ──────────────────────────────────────────────────
  // If the prompt references "the uploaded document / given doc / this PDF" and
  // we have a workspaceId, fetch ALL chunks from the best matching upload.
  if (workspaceId && isDocumentReferenceIntent(query)) {
    try {
      const uploads = await getWorkspaceUploads(String(workspaceId));
      console.log(`[KnowledgeKeeper] Doc-scoped: workspace=${workspaceId} → ${uploads.length} upload(s) with chunks: [${uploads.map((u) => u.original_filename).join(', ')}]`);
      if (uploads.length > 0) {
        const matched = matchUploadByName(uploads, query);
        const target = matched || uploads[0]; // latest upload if no name matched
        console.log(`[KnowledgeKeeper] Doc-scoped: selected="${target.original_filename}" (${matched ? 'name match' : 'latest'})`);
        return await fetchAllChunksForUpload(target.id, target.original_filename);
      } else {
        console.log('[KnowledgeKeeper] Doc-scoped: no uploads with chunks found, falling back to similarity search');
      }
    } catch (e) {
      console.warn('[KnowledgeKeeper] Doc-scoped retrieval failed, falling back to similarity search:', e.message);
    }
  }

  // Embeddings are generated ONLY via external service (no OpenAI embeddings).
  const embeddingServiceConfigured = (config.embedding?.serviceUrl || '').trim().length > 0;
  // For local Postgres, only use embeddings if the DB can actually store/search vectors (and dims match).
  const useEmbedding = embeddingServiceConfigured && (!usePg || db.hasVector());
  const matchThreshold = clampNumber(config.knowledgeKeeper?.matchThreshold, 0, 1, 0.4);

  let queryEmbedding = null;
  if (useEmbedding && query && query.trim()) {
    try {
      const vec = await getEmbedding(query.trim());
      queryEmbedding = formatVectorForSupabase(vec);
    } catch (_) {
      queryEmbedding = null;
    }
  }
  try {
    if (usePg) {
      // Both schema-base-no-vector.sql and schema-with-embeddings.sql define
      // match_documents with 4 params: (query_embedding, match_threshold, match_count, query_text).
      // Only schema.sql (Supabase) has a 5th filter_company_id param — but ensureSchema()
      // runs schema-with-embeddings.sql, so always use 4 params for Postgres.
      const sql = db.hasVector()
        ? 'SELECT id, content, metadata FROM public.match_documents($1::vector, $2, $3, $4)'
        : 'SELECT id, content, metadata FROM public.match_documents($1, $2, $3, $4)';
      const initialQueryText = queryEmbedding ? null : (query || '');
      const { rows } = await db.query(sql, [queryEmbedding, matchThreshold, topK, initialQueryText]);
      const outRows = rows || [];
      console.log(`[KnowledgeKeeper] match_documents returned ${outRows.length} row(s) for query: "${(initialQueryText || '(embedding)').slice(0, 80)}"`);
      return { chunks: outRows.map((r) => ({ content: r.content, metadata: r.metadata || {} })) };
    }

    // Supabase path
    const rpcParams = {
      query_embedding: queryEmbedding,
      match_threshold: matchThreshold,
      match_count: topK,
      query_text: queryEmbedding ? null : query,
    };
    if (companyId != null && String(companyId).trim()) rpcParams.filter_company_id = String(companyId);
    const { data, error } = await supabase.rpc('match_documents', rpcParams);
    if (error) {
      const fallback = await supabase.from('documents').select('content, metadata').ilike('content', `%${query}%`).limit(topK);
      if (fallback.error) return { chunks: [], message: error.message };
      return { chunks: (fallback.data || []).map((r) => ({ content: r.content, metadata: r.metadata || {} })) };
    }
    return { chunks: (data || []).map((r) => ({ content: r.content, metadata: r.metadata || {} })) };
  } catch (e) {
    return { chunks: [], message: e.message };
  }
}

export const knowledgeKeeperTool = tool(
  async ({ query, topK }) => {
    const out = await searchKnowledgeBase(query, topK);
    return typeof out.chunks !== 'undefined'
      ? JSON.stringify(out.chunks.map((c) => c.content).join('\n\n'))
      : JSON.stringify(out);
  },
  {
    name: 'knowledge_keeper_search',
    description: 'Search the brand Knowledge Base (PDFs, URLs, docs) for facts to inject into copy. Call with a natural language query.',
    schema: KnowledgeKeeperSearchSchema,
  }
);

export default knowledgeKeeperTool;
