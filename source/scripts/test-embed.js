/**
 * Quick manual test for the external embedding service.
 *
 * Usage:
 *   node scripts/test-embed.js
 *   node scripts/test-embed.js "http://3.213.205.136:8007" "hi im vamsi"
 *   node scripts/test-embed.js "http://3.213.205.136:8007/embed" "hello" "world"
 *
 * Env:
 *   EMBEDDING_SERVICE_URL=http://host:8007 or http://host:8007/embed
 *   EMBEDDING_BATCH_SIZE=64
 */

import dotenv from 'dotenv';
dotenv.config();

function toEmbedUrl(base) {
  const url = String(base || '').trim().replace(/\/+$/, '');
  if (!url) return '';
  return /\/embed$/.test(url) ? url : `${url}/embed`;
}

function networkErrorDetail(e) {
  const msg = e?.message || 'unknown';
  const code = e?.cause?.code ?? e?.code;
  const cause = e?.cause?.message ?? e?.cause;
  if (code) return `${msg} (${code}${cause ? `: ${cause}` : ''})`;
  return msg;
}

const maybeUrl = process.argv[2];
const textsFromArgs = process.argv.slice(maybeUrl ? 3 : 2);

const serviceUrl =
  maybeUrl ||
  process.env.EMBEDDING_SERVICE_URL ||
  'http://3.213.205.136:8007/embed';

const embedUrl = toEmbedUrl(serviceUrl);
const batchSize = Number(process.env.EMBEDDING_BATCH_SIZE) || 64;
const texts = textsFromArgs.length ? textsFromArgs : ['hi im vamsi'];

if (!embedUrl) {
  console.error('Missing URL. Provide as arg or set EMBEDDING_SERVICE_URL in .env');
  process.exit(1);
}

const body = { texts, batch_size: batchSize };

try {
  const started = Date.now();
  console.log(`[test-embed] POST ${embedUrl}`);
  console.log(`[test-embed] texts=${texts.length} batch_size=${batchSize}`);

  const res = await fetch(embedUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const raw = await res.text();
  const ms = Date.now() - started;

  console.log(`[test-embed] status=${res.status} time_ms=${ms}`);

  // Pretty-print JSON when possible; otherwise show raw.
  try {
    const json = JSON.parse(raw);
    console.log(JSON.stringify(json, null, 2));
  } catch {
    console.log(raw);
  }

  if (!res.ok) process.exit(2);
} catch (e) {
  console.error(`[test-embed] request failed: ${networkErrorDetail(e)}`);
  process.exit(3);
}

