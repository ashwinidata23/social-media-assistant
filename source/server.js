/**
 * ZunoSync HTTP API
 * POST /chat with { "message": "..." } -> runs orchestration and returns final reply + buttons.
 * POST /chat (multipart/form-data with mediaFile) -> analyzes media, returns options (awaitingConfirmation: true).
 * POST /chat with { resumeWithSelection: true, selection: {...} } -> resumes from interrupt.
 * POST /api/documents/upload -> PDF upload to Supabase documents (Knowledge Base).
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import multer from 'multer';
import { processUserMessage } from './src/index.js';
import { config } from './config/index.js';
import { ingestPdf } from './src/lib/ingest-pdf.js';
import { createClient } from '@supabase/supabase-js';
import * as db from './src/lib/db.js';
import { pingRedis } from './src/lib/user-context-store.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

// ── Active request tracking for cancellation ────────────────────────────────
// Maps threadId → AbortController so a new message on the same thread
// automatically cancels any in-flight LLM generation.
const activeRequests = new Map();

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } }); // 10 MB for PDFs

const mediaUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const allowed = /^(image\/(jpeg|png|gif|webp)|video\/(mp4|quicktime|x-msvideo))$/;
    cb(null, allowed.test(file.mimetype));
  },
});

app.use(express.json());
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (_, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

/** Database connection check. GET /api/supabase/health (PostgreSQL or Supabase) */
app.get('/api/supabase/health', async (_, res) => {
  if (db.isUsingPostgres()) {
    try {
      await db.query('SELECT 1');
      return res.json({ status: 'ok', database: 'postgres', message: 'Connected (LOCAL_DATABASE_URL)' });
    } catch (e) {
      return res.json({ status: 'ok', database: 'error', message: e.message });
    }
  }
  const url = config.supabase?.url;
  const key = config.supabase?.serviceKey || config.supabase?.anonKey;
  if (!url || !key) {
    return res.json({ status: 'ok', database: 'not_configured', message: 'Set LOCAL_DATABASE_URL or SUPABASE_URL in .env' });
  }
  try {
    const client = createClient(url, key);
    const { error } = await client.from('documents').select('id').limit(1);
    if (error) return res.json({ status: 'ok', database: 'error', message: error.message });
    res.json({ status: 'ok', database: 'supabase', message: 'Connected' });
  } catch (e) {
    res.json({ status: 'ok', database: 'error', message: e.message });
  }
});

/** Redis connection check. GET /api/redis/health */
app.get('/api/redis/health', async (_, res) => {
  const out = await pingRedis();
  res.json(out);
});

/**
 * Workspace social accounts proxy.
 * GET /api/social-accounts?workspaceId=...
 * Returns the JSON from the upstream ZunoSync API so the frontend can render
 * platform → account selections without writing raw DB queries.
 */
app.get('/api/social-accounts', async (req, res) => {
  const workspaceId =
    req.query.workspaceId ||
    req.query.workspace_id ||
    req.headers['x-workspace-id'] ||
    null;

  if (!workspaceId) {
    return res.status(400).json({ error: 'Missing workspaceId' });
  }

  try {
    const data = await db.getDetailedSocialAccounts(workspaceId);
    if (!data) {
      return res.status(404).json({ error: 'No social accounts found for workspace' });
    }
    res.json(data);
  } catch (e) {
    console.error('[API] /api/social-accounts error:', e?.message || e);
    res.status(500).json({ error: e.message || 'Internal error' });
  }
});

/** PDF upload to Knowledge Base (Supabase documents). POST /api/documents/upload */
app.post('/api/documents/upload', upload.single('file'), async (req, res) => {
  const uploadStart = Date.now();
  console.log(`\n[upload] ▶ POST /api/documents/upload`);

  if (!req.file) {
    console.log('[upload]   ✗ No file in request');
    return res.status(400).json({ ok: false, error: 'No file uploaded. Use form field "file".' });
  }

  const fname = req.file.originalname || 'unknown';
  const sizeMB = (req.file.size / (1024 * 1024)).toFixed(2);
  console.log(`[upload]   File: ${fname}  |  Size: ${sizeMB} MB  |  MIME: ${req.file.mimetype}`);

  const isPdf = req.file.mimetype === 'application/pdf' || fname.toLowerCase().endsWith('.pdf');
  if (!isPdf) {
    console.log('[upload]   ✗ Rejected (not a PDF)');
    return res.status(400).json({ ok: false, error: 'Only PDF files are allowed.' });
  }

  try {
    const companyId = req.body?.companyId ?? req.headers['x-company-id'] ?? null;
    const userId = req.body?.userId ?? req.headers['x-user-id'] ?? 'default-user';
    const workspaceId = req.body?.workspaceId ?? companyId ?? 'default-workspace';

    const result = await ingestPdf(req.file.buffer, {
      filename: fname,
      companyId,
      userId,
      workspaceId,
      sizeBytes: req.file.size,
      mimeType: req.file.mimetype
    });
    const elapsed = Date.now() - uploadStart;

    if (!result.ok) {
      console.log(`[upload]   ✗ Ingest failed: ${result.error}  (${elapsed}ms)`);
      return res.status(400).json({ ok: false, error: result.error });
    }

    console.log(`[upload]   ✓ Response sent — ${result.chunks ?? 1} chunks, ${result.pages} pages  (${elapsed}ms total)`);
    if (result.warning) console.log(`[upload]   ⚠ ${result.warning}`);

    res.json({
      ok: true,
      uploadId: result.uploadId || null,
      id: result.id,
      ids: result.ids,
      chunks: result.chunks,
      pages: result.pages,
      ...(result.warning && { warning: result.warning }),
    });
  } catch (e) {
    console.log(`[upload]   ✗ Exception: ${e.message}`);
    res.status(500).json({ ok: false, error: e.message });
  }
});

/** List all uploaded documents for a workspace. GET /api/documents?workspaceId=...&userId=... */
app.get('/api/documents', async (req, res) => {
  const workspaceId = req.query.workspaceId || req.query.workspace_id || req.headers['x-workspace-id'];
  if (!workspaceId) {
    return res.status(400).json({ error: 'Missing workspaceId' });
  }
  try {
    const uploads = await db.getWorkspaceUploads(workspaceId);
    res.json({ ok: true, uploads });
  } catch (e) {
    console.error('[API] GET /api/documents error:', e.message);
    res.status(500).json({ error: e.message });
  }
});

/**
 * Delete a document upload + all its chunks.
 * DELETE /api/documents/:uploadId?userId=...&workspaceId=...
 *
 * Flow:
 *   1. Verifies ownership (userId + workspaceId must match the upload row)
 *   2. Deletes from `documents` where source_upload_id = uploadId
 *   3. Deletes from `knowledge_document_uploads` where id = uploadId
 */
app.delete('/api/documents/:uploadId', async (req, res) => {
  const { uploadId } = req.params;
  const userId = req.query.userId || req.query.user_id || req.headers['x-user-id'];
  const workspaceId = req.query.workspaceId || req.query.workspace_id || req.headers['x-workspace-id'];

  if (!uploadId || !userId || !workspaceId) {
    return res.status(400).json({ error: 'Missing uploadId, userId, or workspaceId' });
  }

  try {
    const result = await db.deleteUploadAndChunks(uploadId, userId, workspaceId);
    if (!result.deleted) {
      return res.status(404).json({ ok: false, error: 'Upload not found or does not belong to this user/workspace' });
    }
    console.log(`[API] DELETE /api/documents/${uploadId} — removed upload + ${result.chunksRemoved} chunk(s)`);
    res.json({ ok: true, uploadId, chunksRemoved: result.chunksRemoved });
  } catch (e) {
    console.error('[API] DELETE /api/documents error:', e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

/**
 * Replace (update) a document: keeps the same uploadId, deletes old chunks, re-ingests new PDF.
 * PUT /api/documents/:uploadId
 * Body (multipart): file, userId, workspaceId
 *
 * Flow:
 *   1. Verifies ownership, deletes old chunks (keeps upload row with same ID)
 *   2. Updates upload row (new S3 URL, filename, size)
 *   3. Re-ingests new PDF chunks under the same uploadId
 */
app.put('/api/documents/:uploadId', upload.single('file'), async (req, res) => {
  const { uploadId } = req.params;
  const userId = req.body?.userId || req.headers['x-user-id'];
  const workspaceId = req.body?.workspaceId || req.headers['x-workspace-id'];

  if (!uploadId || !userId || !workspaceId) {
    return res.status(400).json({ error: 'Missing uploadId, userId, or workspaceId' });
  }
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded. Use form field "file".' });
  }

  const isPdf = req.file.mimetype === 'application/pdf' || (req.file.originalname || '').toLowerCase().endsWith('.pdf');
  if (!isPdf) {
    return res.status(400).json({ error: 'Only PDF files are allowed.' });
  }

  try {
    // 1. Delete old chunks only + update upload row (keeps same uploadId)
    const replaceResult = await db.replaceUploadChunks(uploadId, userId, workspaceId, {
      original_filename: req.file.originalname || 'upload.pdf',
      file_size_bytes: req.file.size,
      mime_type: req.file.mimetype,
    });
    if (!replaceResult.found) {
      return res.status(404).json({ ok: false, error: 'Upload not found or does not belong to this user/workspace' });
    }
    console.log(`[API] PUT /api/documents/${uploadId} — old chunks removed (${replaceResult.chunksRemoved}). Re-ingesting under same ID...`);

    // 2. Re-ingest new PDF chunks under the SAME uploadId
    const result = await ingestPdf(req.file.buffer, {
      filename: req.file.originalname || 'upload.pdf',
      companyId: workspaceId,
      userId,
      workspaceId,
      sizeBytes: req.file.size,
      mimeType: req.file.mimetype,
      existingUploadId: uploadId,
    });

    if (!result.ok) {
      return res.status(400).json({ ok: false, error: result.error });
    }

    console.log(`[API] PUT /api/documents/${uploadId} — re-ingested: ${result.chunks ?? 1} chunk(s), ${result.pages} page(s)`);
    res.json({
      ok: true,
      uploadId,
      chunks: result.chunks,
      pages: result.pages,
      ...(result.warning && { warning: result.warning }),
    });
  } catch (e) {
    console.error('[API] PUT /api/documents error:', e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

/** Chat history retrieval. GET /api/chat/history?userId=...&threadId=... */
app.get('/api/chat/history', async (req, res) => {
  const userId = req.query.userId || req.query.user_id || req.headers['x-user-id'];
  const threadId = req.query.threadId || req.query.thread_id;

  if (!userId) {
    return res.status(400).json({ error: 'Missing userId' });
  }

  try {
    const { getHistoricalTurns, getUserThreads } = await import('./src/lib/conversation-summary.js');

    if (!threadId || threadId === 'null') {
      const threads = await getUserThreads(userId);
      return res.json({ threads });
    }

    const turns = await getHistoricalTurns(userId, threadId);

    // The DB returns { role, content }, but the frontend expects { finalReply, contentResult, etc. }
    // for assistant messages. To keep the frontend renderer simple and aligned with the API response shape, 
    // we map them so the frontend `appendMessage('assistant', message)` works seamlessly.
    const mappedTurns = turns.map(t => {
      if (t.role === 'assistant' && t.conversation_history) {
        return {
          role: t.role,
          content: t.conversation_history,
        };
      }
      return {
        role: t.role,
        content: t.content,
        ...(t.media != null && { media: t.media })
      };
    });

    res.json({ history: mappedTurns });
  } catch (e) {
    console.error('[Chat History] API error:', e.message);
    res.status(500).json({ error: e.message });
  }
});

/** Thread summary retrieval. GET /api/chat/summary?userId=...&threadId=...&workspaceId=... */
app.get('/api/chat/summary', async (req, res) => {
  const userId = req.query.userId || req.query.user_id || req.headers['x-user-id'];
  const threadId = req.query.threadId || req.query.thread_id || 'default';
  const workspaceId =
    req.query.workspaceId ||
    req.query.workspace_id ||
    req.headers['x-workspace-id'] ||
    null;

  if (!userId) {
    return res.status(400).json({ error: 'Missing userId' });
  }

  try {
    const { getLatestSummary } = await import('./src/lib/conversation-summary.js');
    const summary = await getLatestSummary(userId, threadId, workspaceId);
    res.json({ threadId, summary });
  } catch (e) {
    console.error('[Thread Summary] API error:', e.message);
    res.status(500).json({ error: e.message });
  }
});


app.post('/chat', mediaUpload.single('mediaFile'), async (req, res) => {
  console.log('\n[API] === NEW INCOMING /chat REQUEST ===');
  console.log('[Chat] req.body keys:', Object.keys(req.body || {}));
  console.log('[Chat] req.body.mediaUrl:', req.body?.mediaUrl);
  console.log('[Chat] req.body.mediaUrls:', req.body?.mediaUrls);
  console.log('[Chat] req.file:', req.file ? { fieldname: req.file.fieldname, originalname: req.file.originalname, mimetype: req.file.mimetype, size: req.file.size, hasBuffer: !!req.file.buffer } : null);

  let message = req.body?.message ?? req.body?.prompt ?? '';
  const resumeWithSelection =
    req.body?.resumeWithSelection === true ||
    req.body?.resumeWithSelection === 'true';

  // Detect when frontend sends image S3/CDN URL as the message text (no other text)
  // Move it to mediaUrl so the media flow kicks in correctly.
  // Matches URLs ending in image/video ext OR known S3/CDN hostnames (no extension needed)
  const hasImageExt = /^https?:\/\/.+\.(jpg|jpeg|png|gif|webp|mp4|mov|avi)(\?.*)?$/i;
  const isS3OrCdnHost = /^https?:\/\/[^/]*(s3[.-]|cloudfront\.net|supabase\.co|digitaloceanspaces\.com|blob\.core\.windows\.net)/i;
  const looksLikeMediaUrl = (url) => /^https?:\/\//i.test(url) && (hasImageExt.test(url) || isS3OrCdnHost.test(url));

  // 0. Prefer explicit mediaUrls array from caller (e.g. platform backend) over any blob: URLs in the message
  if (!req.file && !req.body?.mediaUrl && req.body?.mediaUrls) {
    let urls = [];
    if (Array.isArray(req.body.mediaUrls)) {
      urls = req.body.mediaUrls;
    } else if (typeof req.body.mediaUrls === 'string') {
      try {
        const parsed = JSON.parse(req.body.mediaUrls);
        urls = Array.isArray(parsed) ? parsed : [req.body.mediaUrls];
      } catch (_) {
        urls = [req.body.mediaUrls];
      }
    }
    const firstUrl = urls.find((u) => typeof u === 'string' && u.trim().length > 0);
    if (firstUrl && looksLikeMediaUrl(firstUrl)) {
      req.body.mediaUrl = firstUrl.trim();
      req.body.mediaMimeType = /\.(mp4|mov|avi)/i.test(firstUrl) ? 'video/mp4' : 'image/jpeg';
      console.log('[Chat] mediaUrls provided — using first URL for mediaInput:', req.body.mediaUrl.slice(0, 80));
    } else if (firstUrl) {
      console.log('[Chat] mediaUrls provided but first entry does not look like media URL:', firstUrl.slice(0, 80));
    }
  }

  // 1. Detect pure media URL message
  if (message && looksLikeMediaUrl(message.trim()) && !req.body?.mediaUrl && !req.file) {
    req.body.mediaUrl = message.trim();
    req.body.mediaMimeType = /\.(mp4|mov|avi)/i.test(message) ? 'video/mp4' : 'image/jpeg';
    message = ''; // clear so it's treated as pure media upload
    console.log('[Chat] Media URL detected in message — converting to mediaInput:', req.body.mediaUrl.slice(0, 80));
  }

  // 2. Detect frontend-injected media blob/url and boilerplate
  const frontendMediaRegex = /User uploaded media:\s*\[(image|video):\s*(.*?)\][\s\S]*?(?:IMPORTANT: The user has uploaded their own media[\s\S]*)?$/i;
  const frontendMatch = message.match(frontendMediaRegex);
  if (frontendMatch) {
    const isVideo = frontendMatch[1].toLowerCase() === 'video';
    const url = frontendMatch[2];
    if (!req.body?.mediaUrl && !req.file) {
      req.body.mediaUrl = url;
      req.body.mediaMimeType = isVideo ? 'video/mp4' : 'image/jpeg';
      console.log('[Chat] Extracted injected media blob/url:', url.slice(0, 80));
    }
    message = message.replace(frontendMediaRegex, '').trim();
  }

  if (!message && !resumeWithSelection && !req.body?.mediaUrl && !req.file) {
    return res.status(400).json({ error: 'Missing "message" or "prompt" in body' });
  }

  console.log(`[API] Message: "${String(message).slice(0, 80)}" | resume: ${resumeWithSelection}`);

  const userId = req.body?.userId ?? req.body?.sessionId ?? req.headers['x-user-id'];
  const companyId = req.body?.companyId ?? req.headers['x-company-id'];
  const workspaceId = req.body?.workspaceId ?? req.body?.companyId ?? null;

  const rawThreadId = req.body?.threadId;
  let threadId = rawThreadId;
  const resetContext =
    req.body?.resetContext === true ||
    req.body?.resetContext === 'true';
  const newChat =
    req.body?.newChat === true ||
    req.body?.newChat === 'true';

  const isMissingThreadId =
    !threadId || threadId === 'default' || threadId === 'null';

  // If frontend signals "new chat", force a fresh threadId so DB history can't leak
  // even if the client accidentally reuses an old threadId.
  // NEVER force a new thread on resume -- the resume MUST use the same thread that
  // was interrupted, otherwise the graph checkpoint won't be found.
  const shouldForceNewThread = !resumeWithSelection && (newChat || resetContext);

  if (!resumeWithSelection && (isMissingThreadId || shouldForceNewThread)) {
    try {
      const { randomUUID } = await import('crypto');
      threadId = 'thread_' + randomUUID().replace(/-/g, '').slice(0, 16);
    } catch (_) {
      threadId = 'thread_' + Math.random().toString(36).slice(2, 10) + '_' + Date.now().toString(36);
    }
  }

  const endChat = req.body?.endChat === true || req.body?.endChat === 'true';
  console.log(`[Chat] User: ${userId} | Workspace: ${workspaceId} | Thread: ${threadId}`);
  if (resumeWithSelection && isMissingThreadId) {
    console.warn(`[Chat] WARNING: resume request has no valid threadId (raw=${String(rawThreadId)}). The graph checkpoint may not be found.`);
  }
  if (isMissingThreadId || shouldForceNewThread) {
    console.log(`[Chat] New chat thread created (threadId missing/invalid). rawThreadId=${String(rawThreadId)} → generated=${threadId}`);
  }
  if (resetContext) {
    console.log(`[Chat] New chat reset requested (resetContext=true) for userId=${userId} workspaceId=${workspaceId} threadId=${threadId}`);
  }
  if (newChat) {
    console.log(`[Chat] New chat flag received (newChat=true) for userId=${userId} workspaceId=${workspaceId} threadId=${threadId}`);
  }

  let mediaInput = null;
  if (req.file) {
    mediaInput = {
      buffer: req.file.buffer,
      mimeType: req.file.mimetype,
      originalName: req.file.originalname,
      sizeBytes: req.file.size,
    };
    console.log(`[Chat] Media file received: ${req.file.originalname} (${req.file.mimetype}, ${(req.file.size / 1024).toFixed(1)} KB)`);
  } else if (req.body?.mediaUrl) {
    mediaInput = {
      url: req.body.mediaUrl,
      mimeType: req.body.mediaMimeType || 'image/jpeg',
    };
    console.log(`[Chat] Media URL received: ${req.body.mediaUrl}`);
  }
  console.log('[Chat] mediaInput built:', mediaInput ? { hasBuffer: !!mediaInput.buffer, url: mediaInput.url ? mediaInput.url.slice(0, 100) : undefined, mimeType: mediaInput.mimeType } : null);

  let resumeSelection = null;
  if (resumeWithSelection && req.body?.selection) {
    try {
      resumeSelection = typeof req.body.selection === 'string'
        ? JSON.parse(req.body.selection)
        : req.body.selection;
    } catch (e) {
      return res.status(400).json({ error: 'Invalid selection JSON in body.' });
    }
  }

  let platforms = req.body?.platforms;
  if (platforms && typeof platforms === 'string') {
    try { platforms = JSON.parse(platforms); } catch (_) { platforms = undefined; }
  }

  let accounts = req.body?.accounts;
  if (accounts && typeof accounts === 'string') {
    try { accounts = JSON.parse(accounts); } catch (_) { accounts = undefined; }
  }

  const options = {
    userId: userId || undefined,
    sessionId: req.body?.sessionId ?? undefined,
    threadId,
    companyId: companyId ?? undefined,
    workspaceId: workspaceId ?? undefined,
    endChat,
    scheduleAt: req.body?.scheduleAt,
    publishAllAtSameTime: req.body?.publishAllAtSameTime === true || req.body?.publishAllAtSameTime === 'true',
    platforms,
    accounts,
    mediaInput,
    resumeWithSelection,
    resumeSelection,
    // If client set newChat=true, treat as resetContext for backend state.
    resetContext: shouldForceNewThread,
  };

  // ── Cancel any in-flight request on the same thread ──────────────────────
  if (activeRequests.has(threadId)) {
    console.log(`[Chat] Cancelling previous in-flight request for thread: ${threadId}`);
    activeRequests.get(threadId).abort();
    activeRequests.delete(threadId);
  }
  const abortController = new AbortController();
  activeRequests.set(threadId, abortController);

  try {
    options.signal = abortController.signal;
    const result = await processUserMessage(message, options);

    // If the request was cancelled mid-flight, return a lightweight cancelled response
    if (result.cancelled) {
      console.log(`[Chat] Request cancelled for thread: ${threadId}`);
      return res.json({
        threadId,
        finalReply: '',
        cancelled: true,
        buttons: [],
        contentResult: null,
        mediaResult: null,
        previewResult: null,
        schedulerResult: null,
        schedulingOnlyReply: false,
        awaitingConfirmation: false,
      });
    }

    // Terminal logs (same as CLI in src/index.js)
    console.log('[Chat] User:', message?.slice(0, 80) + (message?.length > 80 ? '…' : ''));
    console.log('[Chat] Call agents:', result.callAgents ?? []);
    console.log('[Chat] Final reply:', result.finalReply?.slice(0, 120) + (result.finalReply?.length > 120 ? '…' : ''));
    if (result.buttons?.length) console.log('[Chat] Buttons:', result.buttons.map((b) => b.label || b.action).join(', '));
    if (result.contentResult) {
      if (result.contentResult.platforms?.length) {
        result.contentResult.platforms.forEach((p) => {
          const preview = p.content?.length > 80 ? p.content.slice(0, 80) + '…' : (p.content || '');
          console.log(`[Chat] [${p.type}]`, preview);
        });
      } else if (result.contentResult.copy) {
        console.log('[Chat] Content:', result.contentResult.copy.slice(0, 150) + (result.contentResult.copy.length > 150 ? '…' : ''));
      }
    }
    if (result.mediaResult) {
      if (result.mediaResult.outputs?.length) {
        const urls = (result.mediaResult.outputs[0]?.urls || []).flat();
        console.log('[Chat] Media:', result.mediaResult.outputs.length, 'output(s), URLs:', urls.length ? urls : '(none)');
        if (urls.length) urls.forEach((u) => console.log('[Chat]   ', u));
      } else if (result.mediaResult.error) {
        console.log('[Chat] Media error:', result.mediaResult.error);
      } else {
        console.log('[Chat] Media: (no outputs)');
      }
    }
    if (result.schedulerResult) {
      console.log('[Scheduler]', result.schedulerResult.message);
      if (result.schedulerResult.bestTime) console.log('[Scheduler] bestTime:', result.schedulerResult.bestTime);
      if (result.schedulerResult.scheduleResult) console.log('[Scheduler] scheduleResult:', result.schedulerResult.scheduleResult);
    }
    if (result.chatEnd && result.endSummary) {
      console.log('[Chat] End summary saved for user', userId || '(anon)');
    }
    if (result.awaitingConfirmation) {
      console.log('[Chat] Options presented:', (result.confirmationOptions || []).filter(o => o.action).map(o => o.action).join(', '));
    }

    res.json({
      threadId,
      finalReply: result.finalReply,
      buttons: result.buttons,
      contentResult: result.contentResult,
      mediaResult: result.mediaResult,
      previewResult: result.previewResult,
      schedulerResult: result.schedulerResult,
      schedulingOnlyReply: result.schedulingOnlyReply ?? false,
      endSummary: result.endSummary ?? undefined,
      chatEnd: result.chatEnd ?? undefined,
      awaitingConfirmation: result.awaitingConfirmation ?? false,
      confirmationOptions: result.confirmationOptions ?? undefined,
      mediaAnalysis: result.mediaAnalysis ?? undefined,
      // Billing / credits metadata
      generationType: result.generationType ?? 1,        // 1 = text, 2 = image, 3 = video
      contentCount: result.contentCount ?? 0,            // how many posts/captions
      imageCount: result.imageCount ?? 0,                // how many image URLs
      videoCount: result.videoCount ?? 0,                // how many video URLs
    });
  } catch (e) {
    // If aborted, return cancelled response instead of error
    if (e?.name === 'AbortError' || abortController.signal.aborted) {
      console.log(`[Chat] Request aborted for thread: ${threadId}`);
      return res.json({
        threadId,
        finalReply: '',
        cancelled: true,
        buttons: [],
        contentResult: null,
        mediaResult: null,
        previewResult: null,
        schedulerResult: null,
        schedulingOnlyReply: false,
        awaitingConfirmation: false,
      });
    }
    console.error('[Chat] Error:', e.message);
    const isDev = process.env.NODE_ENV !== 'production';
    res.status(500).json({ error: e.message, ...(isDev && { stack: e.stack }) });
  } finally {
    // Clean up active request tracker
    if (activeRequests.get(threadId) === abortController) {
      activeRequests.delete(threadId);
    }
  }
});

/** Explicitly cancel an in-flight request. POST /chat/cancel { threadId } */
app.post('/chat/cancel', (req, res) => {
  const threadId = req.body?.threadId;
  if (!threadId) {
    return res.status(400).json({ error: 'Missing threadId' });
  }
  if (activeRequests.has(threadId)) {
    activeRequests.get(threadId).abort();
    activeRequests.delete(threadId);
    console.log(`[Chat] Explicit cancel for thread: ${threadId}`);
    return res.json({ ok: true, cancelled: true, threadId });
  }
  res.json({ ok: true, cancelled: false, message: 'No active request for this thread' });
});

app.get('/health', (_, res) => res.json({ status: 'ok', service: 'zunosync' }));

const port = config.server.port;
async function start() {
  if (db.isUsingPostgres()) {
    try {
      const { hasVector } = await db.ensureSchema();
      console.log('PostgreSQL schema ready.', hasVector ? '(pgvector enabled)' : '(text search only)');
    } catch (e) {
      console.error('Schema setup failed:', e.message);
      process.exit(1);
    }
  }
  app.listen(port, () => {
    console.log(`ZunoSync API on http://localhost:${port}`);
  });
}
start();