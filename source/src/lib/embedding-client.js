/**
 * Embedding: external service ONLY (EMBEDDING_SERVICE_URL).
 *
 * This project is configured to NEVER generate embeddings via OpenAI.
 * It will call your embedding service endpoint instead.
 *
 * External: POST { "texts": ["..."], "batch_size": 64 } to /embed → returns vector(s).
 */

import { config } from '../../config/index.js';

const DEBUG_EMBEDDING = process.env.DEBUG_EMBEDDING === '1';

/** Turn a fetch/network error into a clear message for logs */
function networkErrorDetail(e) {
  const msg = e?.message || 'unknown';
  const code = e?.cause?.code ?? e?.code;
  const cause = e?.cause?.message ?? e?.cause;
  if (code) return `${msg} (${code}${cause ? `: ${cause}` : ''})`;
  return msg;
}

/**
 * Get embedding vector for a single text.
 * Uses EMBEDDING_SERVICE_URL only.
 * @param {string} text
 * @returns {Promise<number[]>}
 */
export async function getEmbedding(text) {
  const url = (config.embedding?.serviceUrl || '').trim();
  if (!url) {
    throw new Error('No embedding source. Set EMBEDDING_SERVICE_URL in .env');
  }

  const embedUrl = /\/embed\/?$/.test(url) ? url.replace(/\/+$/, '') : url.replace(/\/+$/, '') + '/embed';
  const batchSize = Number(process.env.EMBEDDING_BATCH_SIZE) || 64;
  const bodyObj = { texts: [text || ''], batch_size: batchSize };

  try {
    const started = Date.now();
    const sample = String(text || '').replace(/\s+/g, ' ').slice(0, 80);
    if (DEBUG_EMBEDDING) {
      console.log(
        `[embedding] POST ${embedUrl}  texts=1  batch_size=${batchSize}  chars=${String(text || '').length}  sample="${sample}"`
      );
      console.log('[embedding] body:', JSON.stringify({ texts_count: 1, batch_size: batchSize }));
    }
    const res = await fetch(embedUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bodyObj),
    });

    if (!res.ok) {
      const errText = await res.text();
      if (DEBUG_EMBEDDING) {
        console.warn(`[embedding] Response ${res.status} in ${Date.now() - started}ms: ${errText.slice(0, 500)}`);
      }
      throw new Error(`Embedding API ${res.status}: ${errText}`);
    }

    const data = await res.json();
    if (DEBUG_EMBEDDING) {
      console.log(`[embedding] Response 200 in ${Date.now() - started}ms  keys=${Object.keys(data || {}).join(',')}`);
    }

    // Common response shapes
    if (Array.isArray(data)) {
      if (DEBUG_EMBEDDING) console.log('[embedding] Parsed shape: array');
      return data;
    }
    if (data?.embedding && Array.isArray(data.embedding)) {
      if (DEBUG_EMBEDDING) console.log('[embedding] Parsed shape: { embedding: number[] }');
      return data.embedding;
    }
    if (Array.isArray(data?.embeddings) && Array.isArray(data.embeddings?.[0])) {
      if (DEBUG_EMBEDDING) console.log('[embedding] Parsed shape: { embeddings: number[][] }');
      return data.embeddings[0];
    }
    if (Array.isArray(data?.data) && Array.isArray(data.data?.[0]?.embedding)) {
      if (DEBUG_EMBEDDING) console.log('[embedding] Parsed shape: { data: [{ embedding: number[] }] }');
      return data.data[0].embedding;
    }

    throw new Error('Embedding API response format not recognized');
  } catch (e) {
    const detail = networkErrorDetail(e);
    // Always log URL + request shape on failures for debugging timeouts.
    console.warn(
      `[embedding] External service failed: ${detail} | url=${embedUrl} | body=${JSON.stringify({ texts_count: 1, batch_size: batchSize, chars: String(text || '').length })}`
    );
    throw new Error(detail);
  }
}

/**
 * Format embedding array for Supabase pgvector: string like '[0.1,0.2,...]'
 * @param {number[]} vec
 * @returns {string}
 */
export function formatVectorForSupabase(vec) {
  if (!Array.isArray(vec)) return '[]';
  return '[' + vec.join(',') + ']';
}

export default getEmbedding;
