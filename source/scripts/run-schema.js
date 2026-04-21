/**
 * Run schema against local PostgreSQL.
 * Tries full schema (pgvector) first; if "vector" extension is not available, runs base schema (tables + text-only search).
 * Usage: node scripts/run-schema.js
 * Requires: LOCAL_DATABASE_URL (or DATABASE_URL) in .env.
 */

import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import pg from 'pg';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const connectionString = process.env.LOCAL_DATABASE_URL || process.env.DATABASE_URL;
if (!connectionString) {
  console.error('Set LOCAL_DATABASE_URL or DATABASE_URL in .env');
  process.exit(1);
}

const basePath = path.join(__dirname, '..', 'supabase', 'schema-base-no-vector.sql');
const fullPath = path.join(__dirname, '..', 'supabase', 'schema-with-embeddings.sql');
const baseSql = fs.readFileSync(basePath, 'utf8');
const fullSql = fs.readFileSync(fullPath, 'utf8');

async function run() {
  const client = new pg.Client({ connectionString });
  try {
    await client.connect();
    console.log('Connected to PostgreSQL.');
    await client.query(baseSql);
    console.log('Base schema applied (documents, scheduled_posts, audience_insights, match_documents).');
    try {
      await client.query(fullSql);
      console.log('Full schema applied (pgvector, embedding column, vector search).');
    } catch (e) {
      const msg = (e.message || '').toLowerCase();
      if (msg.includes('vector') || msg.includes('extension')) {
        console.log('pgvector not available; using text-only search. Install pgvector for semantic search.');
      } else if (e.message.includes('ivfflat') || e.message.includes('at least') || e.message.includes('empty')) {
        console.log('Tables and function created; vector index skipped (add rows then create index in pgAdmin if needed).');
      } else {
        console.warn('Full schema skipped:', e.message);
      }
    }
  } catch (e) {
    console.error('Schema error:', e.message);
    process.exit(1);
  } finally {
    await client.end();
  }
}

run();
