/**
 * CLI: Ingest a PDF into Supabase documents (Knowledge Base).
 * Run: node scripts/ingest-pdf.js <path-to-file.pdf>
 */

import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { ingestPdf } from '../src/lib/ingest-pdf.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const filePath = process.argv[2];
if (!filePath) {
  console.error('Usage: node scripts/ingest-pdf.js <path-to-file.pdf>');
  process.exit(1);
}

const absolutePath = path.isAbsolute(filePath) ? filePath : path.join(process.cwd(), filePath);
if (!fs.existsSync(absolutePath)) {
  console.error('File not found:', absolutePath);
  process.exit(1);
}

const buffer = fs.readFileSync(absolutePath);
const filename = path.basename(absolutePath);

const result = await ingestPdf(buffer, { filename });
if (result.ok) {
  console.log('OK: PDF ingested into Knowledge Base. id:', result.id, 'pages:', result.pages);
} else {
  console.error('Error:', result.error);
  process.exit(1);
}
