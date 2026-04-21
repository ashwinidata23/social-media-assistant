import { StateGraph, Annotation, interrupt, MemorySaver } from '@langchain/langgraph';
import {
  runOrchestrator,
  analyzeMediaWithVision,
  detectIntentClarity,
  buildConfirmationOptions,
  buildPlatformConnectionOptions,
  buildSchedulingOptions,
  generateContextualReply,
} from '../agents/orchestrator.js';
import { runKnowledgeKeeper } from '../agents/knowledge-keeper.js';
import { runContentGenius } from '../agents/content-genius.js';
import { runMediaWizard } from '../agents/media-wizard.js';
import { runPreviewPro } from '../agents/preview-pro.js';
import { runSchedulerBoss } from '../agents/scheduler-boss.js';
import { getActivePlatforms, getActiveAccounts } from '../lib/db.js';

const checkpointer = new MemorySaver();

const ALL_PLATFORM_IDS = ['instagram', 'x', 'twitter', 'linkedin', 'facebook', 'tiktok', 'youtube', 'youtube_short'];

const PLATFORM_LABELS = {
  instagram: 'Instagram', x: 'X (Twitter)', twitter: 'X (Twitter)',
  linkedin: 'LinkedIn', facebook: 'Facebook', tiktok: 'TikTok', youtube: 'YouTube',
  youtube_short: 'YouTube Short',
};

const ZunoState = Annotation.Root({
  userPrompt: Annotation({ default: () => '' }),
  rawUserPrompt: Annotation({ default: () => '' }),
  canonicalUserId: Annotation({ default: () => null, reducer: (prev, right) => (right != null && right !== '' ? right : prev) }),
  canonicalCompanyId: Annotation({ default: () => null, reducer: (prev, right) => (right != null && right !== '' ? right : prev) }),
  directReply: Annotation({ default: () => null }),
  callAgents: Annotation({ default: () => [], reducer: (_, right) => (Array.isArray(right) ? right : _) }),
  subTasks: Annotation({ default: () => [], reducer: (_, right) => (Array.isArray(right) ? right : _) }),
  context: Annotation({
    default: () => ({}),
    reducer: (prev, right) => ({
      ...(typeof prev === 'object' && prev !== null ? prev : {}),
      ...(typeof right === 'object' && right !== null ? right : {}),
    }),
  }),
  knowledgeChunks: Annotation({ default: () => [], reducer: (_, right) => (Array.isArray(right) ? right : _) }),
  contentResult: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  mediaResult: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  previewResult: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  schedulerResult: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  finalReply: Annotation({ default: () => '' }),
  buttons: Annotation({ default: () => [], reducer: (_, right) => (Array.isArray(right) ? right : _) }),
  schedulingOnlyReply: Annotation({ default: () => false }),
  mediaInput: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  mediaAnalysis: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  intentClarity: Annotation({ default: () => 'unknown', reducer: (_, right) => (right !== undefined ? right : _) }),
  awaitingConfirmation: Annotation({ default: () => false, reducer: (_, right) => (right !== undefined ? right : _) }),
  confirmationOptions: Annotation({ default: () => [], reducer: (_, right) => (Array.isArray(right) ? right : _) }),
  confirmationReason: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  userSelection: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  guidedFlowStep: Annotation({ default: () => null, reducer: (_, right) => (right !== undefined ? right : _) }),
  pendingCallAgents: Annotation({ default: () => [], reducer: (_, right) => (Array.isArray(right) ? right : _) }),
});

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function resolveAgentsForAction(action) {
  switch (action) {
    case 'create_post': return ['content_genius', 'media_wizard', 'preview_pro'];
    case 'use_as_media': return ['content_genius', 'preview_pro'];
    case 'regenerate_media': return ['media_wizard'];
    case 'extract_content': return ['content_genius'];
    case 'analyze_brand': return [];
    case 'generate_media': return ['media_wizard'];
    case 'schedule': return ['scheduler_boss'];
    default: return ['content_genius', 'media_wizard', 'preview_pro'];
  }
}

function normalize(text) {
  return (text || '').toLowerCase().replace(/\s+/g, ' ').trim();
}

function normalizePlatformId(p) {
  const lower = String(p || '').toLowerCase().trim();
  if (!lower) return '';
  if (lower === 'twitter') return 'x';
  return lower;
}

function normalizePlatformList(platforms) {
  if (!Array.isArray(platforms)) return [];
  return platforms.map(normalizePlatformId).filter(Boolean);
}

function parsePlatformsFromText(text) {
  const n = normalize(text)
    .replace(/you\s*tube\s+shorts?\b/g, 'youtube_short')
    .replace(/yt\s+shorts?\b/g, 'youtube_short')
    .replace(/you\s*tube/g, 'youtube')
    .replace(/tik\s*tok/g, 'tiktok')
    .replace(/linked\s*in/g, 'linkedin')
    .replace(/face\s*book/g, 'facebook')
    // Common misspellings/typos from natural chat
    .replace(/\binsta+g?r?a?m+\b/g, 'instagram')
    .replace(/\blinkdn\b|\blinkdln\b|\blinkdin\b/g, 'linkedin')
    .replace(/\bfacebok\b|\bfaceboook\b|\bfb\b/g, 'facebook')
    .replace(/\byotube\b|\byoutu\b/g, 'youtube')
    .replace(/\btikok\b|\btiktokk\b/g, 'tiktok')
    .replace(/\btwiter\b|\btwiterr\b/g, 'twitter');
  return ALL_PLATFORM_IDS.filter(p => {
    if (p === 'x') return /\bx\b|twitter/.test(n);
    if (p === 'twitter') return false;
    if (p === 'youtube_short') return n.includes('youtube_short');
    // Don't match 'youtube' if 'youtube_short' was already matched (avoid double-match)
    if (p === 'youtube') return n.includes('youtube') && !n.includes('youtube_short');
    return n.includes(p);
  });
}

function isSmallTalkOnly(text) {
  const n = normalize(text);
  if (!n) return false;
  return /^(hi+|hii+|hiii+|hello+|hey+|thanks?|thank\s*you|thx|ty|ok(ay)?|great|cool|awesome|perfect|noted|got it|alright|cheers)[\s.!?]*$/i.test(n);
}

// Shared negation pattern: matches "no image", "without image", "text only", "only content",
// "not required image" (incl. typos), "don't need images", etc.
const NO_MEDIA_PATTERN = /text\s*only|no\s*image|without\s*image|no\s*media|copy\s*only|caption\s*only|only\s*(content|text|copy|caption)|not\s*req\w*\s*(image|media|picture|photo|visual)|don'?t\s*(need|want|require)\s*(any\s*)?(image|media|picture|photo|visual)/i;

function parseContentTypeFromText(text) {
  const n = normalize(text);
  // Explicit video-style requests
  if (/video|reel|tiktok|shorts?/i.test(n)) return { contentType: 'video', includeMedia: true };
  // Explicit text-only / no-media requests (MUST be checked BEFORE positive image match
  // to avoid false positives like "not required image" matching the word "image")
  if (NO_MEDIA_PATTERN.test(n)) {
    return { contentType: 'image', includeMedia: false };
  }
  // Explicit image / visual requests
  if (/\b(image|images|photo|photos|visual|visuals|picture|pictures|graphic|graphics)\b/i.test(n)) {
    return { contentType: 'image', includeMedia: true };
  }
  // No explicit format mentioned → let the workflow ask the user
  return { contentType: null, includeMedia: null };
}

function isMediaOnlyInstruction(text) {
  const n = normalize(text);
  if (!n) return false;
  // "generate images too", "image also", "add image", etc.
  if (/\b(generate|create|make)\s*(images?|pics?|pictures?|visuals?|graphics?|photos?|photo)\b/.test(n)) return true;
  if (/\bimages?\s*too\b/.test(n) || /\bimage\s*also\b/.test(n)) return true;
  if (/\b(add|include)\s*(image|images|visuals?|pictures?|graphics?)\b/.test(n)) return true;

  // Image edit/variation phrasing
  if (/\b(modify|edit|change|more\s+colou?r|more\s+color|regenerate)\b.*\b(image|images)\b/.test(n)) return true;
  if (/\bmore\s+colou?r\b/.test(n) || /\bmore\s+color\b/.test(n)) return true;

  // Removal instructions
  if (/\b(remove|delete|no)\b.*\b(image|images)\b/.test(n)) return true;
  if (/\bno\s*image\b|\bwithout\s*image\b|\bremove\s*image\b/.test(n)) return true;

  return false;
}

function parseVideoPurposeFromText(text) {
  const n = normalize(text);
  if (/reel|short|vertical|portrait|9.?16/i.test(n)) return 'reel';
  if (/landscape|horizontal|regular|normal|16.?9/i.test(n)) return 'regular';
  return 'both';
}

function parseScheduleFromText(text) {
  const n = normalize(text);
  if (/^(thank(s| you)?|ty|thx|ok(ay)?|great|cool|awesome|nice|perfect|got it|noted|sure|alright|cheers)[\.!]?$/i.test(n)) return { action: 'gratitude' };
  if (/\bnow\b|post now|immediately|right now/i.test(n)) return { action: 'schedule_now', scheduleAt: new Date().toISOString() };
  if (/skip|later|not now|maybe later|no thanks|don'?t|dont/i.test(n)) return { action: 'skip' };

  // Only treat as a scheduling request if we see an actual time/date signal.
  // This prevents unrelated messages (e.g. "images too") from being routed to the scheduler.
  const hasTimeSignal = /(\btoday\b|\btomorrow\b|\btonight\b|\bnext\b|\bthis\s+(week|month|weekend|morning|afternoon|evening)\b|\bweek\b|\bmonth\b|\bweekend\b|\bmorning\b|\bafternoon\b|\bevening\b|\bnoon\b|\bmidnight\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b|\bmon(day)?\b|\btue(s(day)?)?\b|\bwed(nesday)?\b|\bthu(r(sday)?)?\b|\bfri(day)?\b|\bsat(urday)?\b|\bsun(day)?\b|\bjan(uary)?\b|\bfeb(ruary)?\b|\bmar(ch)?\b|\bapr(il)?\b|\bmay\b|\bjun(e)?\b|\bjul(y)?\b|\baug(ust)?\b|\bsep(tember)?\b|\boct(ober)?\b|\bnov(ember)?\b|\bdec(ember)?\b)/i.test(n);
  if (!hasTimeSignal) return { action: 'unknown' };

  return { action: 'schedule_natural', rawTime: text.trim() };
}

function parseMediaActionFromText(text) {
  const n = normalize(text);
  if (/create|generate|write|make|post/i.test(n)) return 'create_post';
  if (/regenerat|new media|similar/i.test(n)) return 'regenerate_media';
  if (/extract|read|get text|ideas from/i.test(n)) return 'extract_content';
  if (/analyze|brand|check/i.test(n)) return 'analyze_brand';
  if (/use\s*(this|it|directly|image|video)/i.test(n)) return 'use_as_media';
  return null;
}

function parseContentActionFromText(text) {
  const n = normalize(text);
  if (/schedule|calendar|post now|tomorrow|\bnext\b|\btonight\b|\bmorning\b|\bevening\b|\bpm\b|\bam\b/i.test(n)) return 'schedule';

  // Keep this tolerant to typos: users often type "genrate"/"generte".
  const mediaVerb = /(generate|genrate|generte|gnerate|create|make|need|want|also|add|too)/i;
  const wantsImages = /(image|images|photo|photos|visual|visuals|graphic|graphics|picture|pictures|pics)/i.test(n);
  const wantsVideo = /(video|reel|shorts|short|tiktok|youtube short|yt short)/i.test(n);

  // FIX: Check if image/media words appear in a negation context BEFORE concluding
  // the user wants to generate media. "not required image" / "don't need images" / "only content"
  // should NOT be classified as generate_media.
  const isNegatingMedia = NO_MEDIA_PATTERN.test(n);

  // Allow short follow-ups like "images too" while mid-flow, and also long sentences
  // that mention the desired media anywhere (e.g. "... generate me a video for this").
  if (wantsImages && !isNegatingMedia && (mediaVerb.test(n) || /^(images?|pics?|visuals?|graphics?)\b/i.test(n))) return 'generate_media';
  if (wantsVideo && (mediaVerb.test(n) || /^(video|reel|shorts?)\b/i.test(n))) return 'generate_media';

  if (/(content|caption|post|thread|copy|write|rewrite|reword|paraphrase|edit|modify|update|revise|improve|enhance|polish|refine)/i.test(n)) return 'create_post';
  return null;
}

function isTextOnlyRequest(text) {
  const n = normalize(text);
  return NO_MEDIA_PATTERN.test(n);
}

/**
 * Returns true when the user's message contains a substantive topic beyond
 * generic action words (e.g. "create a post" has no topic; "create a post about
 * our product launch" does).
 */
function hasTopic(text) {
  const n = normalize(text);
  const stripped = n
    .replace(/\b(can|could|please|help|create|generate|make|write|produce|build|get|give|show|need|want|i|me|you|a|an|the|some|us|we|my|our)\b/g, ' ')
    .replace(/\b(post|content|caption|copy|image|images|video|reel|social|media)\b/g, ' ')
    .replace(/\b(for|to|about|on|with|using|in|at|from)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return stripped.length >= 5;
}

function resolvePlatformsWithFallback(selectionPlatforms, statePlatforms, stateActivePlatforms) {
  const fromSelection = Array.isArray(selectionPlatforms) && selectionPlatforms.length > 0 ? selectionPlatforms : null;
  if (fromSelection) return fromSelection;
  const fromActive = Array.isArray(stateActivePlatforms) && stateActivePlatforms.length > 0 ? stateActivePlatforms : null;
  if (fromActive) return fromActive;
  const fromState = Array.isArray(statePlatforms) && statePlatforms.length > 0 ? statePlatforms : null;
  if (fromState) return fromState;
  return null;
}

/** Build a descriptive contentTopic from mediaAnalysis description */
function buildContentTopicFromMedia(analysis, fallback = 'Create post using the uploaded media') {
  return analysis?.description
    ? `Create a post about this image: ${analysis.description}`
    : fallback;
}

/**
 * Copy contentType / includeMedia / rawUserPrompt into pendingInterruptContext so graph
 * state lost at interrupt() can be merged back on resume.
 */
function formatFieldsForPendingContext(context = {}) {
  const job = context.job && typeof context.job === 'object' ? context.job : {};
  const contentType = context.contentType ?? job.contentType;
  let includeMedia = context.includeMedia !== undefined ? context.includeMedia : job.includeMedia;
  if (includeMedia === undefined && contentType != null && String(contentType).trim() !== '') {
    const ctNorm = String(contentType).toLowerCase().replace(/[\s_\-]+/g, '');
    if (ctNorm === 'textonly') includeMedia = false;
  }
  const out = {};
  if (contentType != null && String(contentType).trim() !== '') out.contentType = contentType;
  if (includeMedia !== undefined) out.includeMedia = !!includeMedia;
  return out;
}

function inferIncludeMediaFromContentType(contentType) {
  if (contentType == null || String(contentType).trim() === '') return undefined;
  const ctNorm = String(contentType).toLowerCase().replace(/[\s_\-]+/g, '');
  return ctNorm === 'textonly' ? false : undefined;
}

function sessionWantsTextOnly(ctx = {}, job = {}) {
  const j = job && typeof job === 'object' ? job : {};
  const im = ctx.includeMedia !== undefined ? ctx.includeMedia : j.includeMedia;
  if (im === false) return true;
  const ct = ctx.contentType ?? j.contentType;
  if (inferIncludeMediaFromContentType(ct) === false) return true;
  const ctNorm = String(ct || '').toLowerCase().replace(/[\s_\-]+/g, '');
  return ctNorm === 'textonly';
}

/** Many clients send image+true as canned defaults when the user only picks platforms. */
function isLikelyDefaultImageWithMediaPayload(sel) {
  return sel?.contentType === 'image' && sel?.includeMedia === true;
}

/**
 * After a content_request interrupt, merged context has the real format (e.g. text_only).
 * Prefer that over spurious client defaults on platform-only resume.
 */
function shouldIgnoreSelectionFormatForContentRequestResume(state, sel) {
  if (String(state.confirmationReason || '') !== 'content_request') return false;
  if (sel?.action !== 'create_post') return false;
  const ctx = state.context || {};
  const j = (ctx.job && typeof ctx.job === 'object') ? ctx.job : {};
  if (!sessionWantsTextOnly(ctx, j)) return false;
  if (!isLikelyDefaultImageWithMediaPayload(sel)) return false;
  return true;
}

function clampInt(n, min, max) {
  const x = Number(n);
  if (!Number.isFinite(x)) return min;
  return Math.max(min, Math.min(max, Math.trunc(x)));
}

function parseExplicitMediaIntent(text) {
  const n = normalize(text);
  // Explicit requests to change/regenerate visuals (keep conservative; avoid false positives)
  return (
    /\b(regenerat|regen|new\s+(image|images|visual|visuals|graphic|graphics|picture|pictures|video|reel|short))\b/i.test(n) ||
    /\b(change|modify|edit|update)\b.*\b(image|visual|graphic|picture|video)\b/i.test(n) ||
    /\b(make|generate|create)\b.*\b(image|images|visual|visuals|graphic|graphics|picture|pictures|video|reel|shorts?)\b/i.test(n) ||
    /\bmore\s+(images?|visuals?|graphics?|pictures?)\b/i.test(n)
  );
}

function buildPreferredOptions(missing = [], context = {}) {
  const opts = [];

  // Topic input (only when missing topic)
  if (missing.includes('topic')) {
    opts.push({
      type: 'topic_input',
      label: 'What is this post about?',
      placeholder: 'e.g., Holi celebration at our office, product launch, weekend sale...',
      defaultValue: context?.contentTopic || context?.job?.topic || '',
    });
  }

  // Platforms selector (only when missing platforms)
  if (missing.includes('platforms')) {
    const connected = Array.isArray(context.activePlatforms) && context.activePlatforms.length > 0
      ? context.activePlatforms
      : (Array.isArray(context.platforms) ? context.platforms : []);

    // Keep schema compatible with existing UI: platform_selector expects string IDs
    if (connected.length > 0) {
      opts.push({
        type: 'platform_selector',
        label: 'Select platforms',
        options: connected,
        defaultSelected: connected,
      });
    }
  }

  // Linked accounts selector (account holder / page / organization names).
  // We include this when the user is selecting platforms, so they can also pick
  // which connected accounts to use for those platforms.
  if (missing.includes('platforms')) {
    const activeAccounts = Array.isArray(context.activeAccounts) ? context.activeAccounts : [];
    if (activeAccounts.length > 0) {
      opts.push({
        type: 'linked_account_selector',
        label: 'Linked accounts',
        options: activeAccounts
          .filter(a => a && a.accountId != null && a.accountName)
          .map(a => ({
            id: String(a.accountId),
            label: String(a.accountName),
            platform: String(a.platform || '').toLowerCase(),
            accountType: a.type ? String(a.type) : undefined,
          })),
        defaultSelected: activeAccounts
          .filter(a => a && a.accountId != null)
          .map(a => String(a.accountId)),
      });
    }
  }

  // Format selector/buttons (only when missing format)
  if (missing.includes('format')) {
    const platformsForDefault =
      Array.isArray(context?.platforms)
        ? context.platforms
        : (Array.isArray(context?.activePlatforms) ? context.activePlatforms : []);
    const isYouTubeOnly = platformsForDefault.length === 1 && platformsForDefault[0] === 'youtube';

    opts.push({
      type: 'content_type_buttons',
      label: 'What to create',
      options: buildConfirmationOptions(false, { call_agents: [] }, null, context)
        .find(o => o?.type === 'content_type_buttons')?.options
        || [
          { action: 'content_type_selected', label: 'Images + Content', icon: 'image', contentType: 'image', includeMedia: true },
          { action: 'content_type_selected', label: 'Text only', icon: 'text', contentType: 'text_only', includeMedia: false },
          { action: 'content_type_selected', label: 'Video + Content', icon: 'video', contentType: 'video', includeMedia: true },
        ],
      defaultSelected: isYouTubeOnly ? 'text_only' : (context?.contentType || 'image'),
    });
  }

  // Schedule time (only when missing schedule time)
  if (missing.includes('schedule_time')) {
    // Reuse the same schedule button schema you already return elsewhere
    opts.push(...buildSchedulingOptions());
  }

  return opts;
}

/**
 * Build a concise session-state summary that is injected into every LLM routing
 * prompt so the LLM can make context-aware decisions without needing to parse
 * the full conversation history.
 */
function buildStateSummary(state) {
  const ctx = state.context || {};
  const lines = [];
  const topic = ctx.contentTopic || ctx.job?.topic;
  if (topic) lines.push(`Content topic: "${topic}"`);
  const platforms = ctx.platforms?.length ? ctx.platforms : (ctx.activePlatforms?.length ? ctx.activePlatforms : null);
  if (platforms?.length) lines.push(`Selected platforms: ${platforms.join(', ')}`);
  const fmtType = ctx.contentType || ctx.job?.contentType;
  const fmtMedia = ctx.includeMedia !== undefined ? ctx.includeMedia : ctx.job?.includeMedia;
  if (fmtType != null && String(fmtType).trim() !== '') {
    lines.push(`User-selected format: ${fmtType}${fmtMedia === false ? ' (no image/video generation)' : fmtMedia === true ? ' (with media)' : ''}`);
  } else if (fmtMedia === false) {
    lines.push('User-selected format: text only (no image/video generation)');
  }
  if (state.contentResult) lines.push(`Status: content has been generated`);
  if (state.mediaResult && !state.mediaResult.error) lines.push(`Status: media has been generated`);
  if (ctx.scheduledPostId) lines.push(`Scheduled post ID: ${ctx.scheduledPostId} (reschedule is available)`);
  if (ctx.brandKit) lines.push(`Brand kit: available`);
  if (lines.length === 0) return '';
  return `<session_state>\n${lines.join('\n')}\n</session_state>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// ORCHESTRATOR NODE
// ─────────────────────────────────────────────────────────────────────────────

async function orchestratorNode(state, config) {
  const rawPrompt = state.rawUserPrompt || state.userPrompt || '';
  const incomingCtx = state.context || {};
  const job = (incomingCtx.job && typeof incomingCtx.job === 'object') ? incomingCtx.job : {};
  const baseContext = { ...incomingCtx, job };

  // ── RESUME: explicit userSelection from button click ──────────────────────
  if (state.userSelection) {
    const sel = state.userSelection;
    console.log('[Orchestrator] RESUME PATH -- userSelection:', JSON.stringify(sel));

    // ── Preview refresh: always allow preview without regenerating ────────────
    // Button payload: { action: 'preview' }
    if (sel.action === 'preview') {
      return {
        callAgents: ['preview_pro'],
        context: {
          ...(state.context || {}),
          // Ensure platforms are present for preview rendering; fall back to last active/platforms.
          platforms: Array.isArray(state.context?.platforms) && state.context.platforms.length
            ? state.context.platforms
            : (state.context?.activePlatforms || []),
        },
        subTasks: ['Refresh previews'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    // Handle topic input from preferred options UI
    // Expected payload: { action: 'topic_selected', contentTopic, rawContent? }
    if (sel.action === 'topic_selected') {
      const contentTopic = String(sel.contentTopic || '').trim();
      const rawContent = typeof sel.rawContent === 'string' ? sel.rawContent.trim() : '';
      const nextCtx = {
        ...(state.context || {}),
        ...(contentTopic ? { contentTopic } : {}),
        ...(rawContent ? { rawContent } : {}),
        job: {
          ...((state.context?.job && typeof state.context.job === 'object') ? state.context.job : {}),
          ...(contentTopic ? { topic: contentTopic } : {}),
        },
      };
      const pending = Array.isArray(state.pendingCallAgents) && state.pendingCallAgents.length > 0
        ? state.pendingCallAgents
        : resolveAgentsForAction(nextCtx?.selectedAction || 'create_post');
      return {
        callAgents: pending,
        context: nextCtx,
        subTasks: ['Continue after topic input'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    // ── Handle YouTube video confirmation ──────────────────────────────────────
    // User clicked "Yes, generate video" after YouTube confirmation prompt → ask Shorts vs Regular
    if (sel.action === 'confirm_youtube_video') {
      console.log('[Orchestrator] RESUME: User confirmed YouTube video — asking Shorts vs Regular');
      const youtubeFormatOptions = [
        { action: 'youtube_format_regular', label: 'Regular Video (16:9 landscape)', icon: 'video' },
        { action: 'youtube_format_short', label: 'YouTube Short (9:16 vertical)', icon: 'short' },
        { action: 'youtube_format_both', label: 'Both formats', icon: 'both' },
      ];
      const ytCtx = { ...(state.context || {}), contentType: 'video' };
      interrupt({
        reason: 'youtube_format',
        options: youtubeFormatOptions,
        message: 'What format would you like for YouTube?\n\n**Regular Video** (landscape 16:9) — for longer content with a full title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.',
        buttons: [],
        pendingContext: {
          platforms: ytCtx.platforms,
          contentTopic: ytCtx.contentTopic || ytCtx.job?.topic,
          confirmationReason: 'youtube_format',
          ...formatFieldsForPendingContext(ytCtx),
        },
      });
      return {
        callAgents: [],
        awaitingConfirmation: true,
        confirmationReason: 'youtube_format',
        confirmationOptions: youtubeFormatOptions,
        finalReply: 'What format would you like for YouTube?\n\n**Regular Video** (landscape 16:9) — for longer content with a full title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.',
        context: ytCtx,
        pendingCallAgents: state.pendingCallAgents || [],
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    // ── Handle YouTube format selection (Regular / Short / Both) ─────────────
    if (sel.action === 'youtube_format_regular' || sel.action === 'youtube_format_short' || sel.action === 'youtube_format_both') {
      const formatLabel = sel.action === 'youtube_format_regular' ? 'regular' : sel.action === 'youtube_format_short' ? 'short' : 'both';
      console.log('[Orchestrator] RESUME: YouTube format selected:', formatLabel);
      const pending = Array.isArray(state.pendingCallAgents) && state.pendingCallAgents.length > 0
        ? state.pendingCallAgents
        : resolveAgentsForAction('create_post');

      // Keep 'youtube' in the platforms list (not 'youtube_short') so platform validation
      // works — the DB only stores 'youtube'. The youtubeConfig.videoType controls the
      // actual format (short=9:16, full=16:9) and media-wizard/content-genius use it.
      const currentPlatforms = [...(state.context?.platforms || [])];
      if (!currentPlatforms.includes('youtube')) currentPlatforms.push('youtube');
      // Remove youtube_short from platforms if present — it's not a real DB platform
      const cleanPlatforms = currentPlatforms.filter(p => p !== 'youtube_short');

      const nonYouTubePlatforms = cleanPlatforms.filter(p => p !== 'youtube');
      const isYouTubeOnly = nonYouTubePlatforms.length === 0;
      const nextContentType = isYouTubeOnly ? 'video' : (state.context?.contentType || 'image');
      const youtubeVideoType = sel.action === 'youtube_format_short' ? 'short'
        : sel.action === 'youtube_format_both' ? 'both' : 'full';

      return {
        callAgents: pending,
        context: {
          ...(state.context || {}),
          contentType: nextContentType,
          skipThumbnails: true,
          platforms: cleanPlatforms,
          activePlatforms: cleanPlatforms,
          youtubeFormat: formatLabel,
          // Used by Media Wizard to generate correct YouTube aspect ratio.
          youtubeConfig: { videoType: youtubeVideoType },
        },
        subTasks: [`Generate ${formatLabel} video for YouTube`],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
        pendingCallAgents: [],
      };
    }

    // User clicked "No, different platform" after YouTube confirmation prompt
    if (sel.action === 'select_platform') {
      console.log('[Orchestrator] RESUME: User declined YouTube, showing platform selection');
      const connectedPlatforms = state.context?.userId
        ? (await getActivePlatforms(state.context.userId, state.context.workspaceId) || [])
        : [];
      // Filter out YouTube since user declined it
      const imagePlatforms = connectedPlatforms.filter(p => p !== 'youtube');
      const names = imagePlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');

      const dynamicButtons = [];
      if (imagePlatforms.length > 1) {
        dynamicButtons.push({
          label: 'Quick: All platforms (images + content)',
          action: 'quick_create_post_all',
          platforms: imagePlatforms,
        });
      }
      imagePlatforms.forEach((p) => {
        dynamicButtons.push({
          label: `Quick: ${PLATFORM_LABELS[p] || p}`,
          action: 'quick_create_post_single',
          platforms: [p],
        });
      });

      const replyMsg = imagePlatforms.length > 0
        ? `No problem! Which platform would you like to use for images instead?\n\nAvailable: **${names}**`
        : 'No other platforms are connected for image generation. Please connect a platform in Settings.';

      return {
        directReply: replyMsg,
        finalReply: replyMsg,
        callAgents: [],
        awaitingConfirmation: true,
        confirmationReason: 'content_request',
        buttons: dynamicButtons,
        context: { ...(state.context || {}), platforms: [] },
        pendingCallAgents: state.pendingCallAgents || resolveAgentsForAction('create_post'),
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    // Handle format selection from preferred options (content_type_selected)
    // Expected payload from UI: { action: 'content_type_selected', contentType, includeMedia }
    if (sel.action === 'content_type_selected') {
      const ctxJob = (state.context?.job && typeof state.context.job === 'object') ? state.context.job : {};
      const contentType = sel.contentType || state.context?.contentType || ctxJob.contentType || 'image';
      const includeMedia = sel.includeMedia !== undefined
        ? !!sel.includeMedia
        : (state.context?.includeMedia !== undefined
          ? !!state.context.includeMedia
          : (ctxJob.includeMedia !== undefined
            ? !!ctxJob.includeMedia
            : (inferIncludeMediaFromContentType(contentType) ?? true)));

      // If we were waiting on format only, continue with whatever pendingCallAgents were set,
      // but respect "text only" by dropping media_wizard.
      const selectedPlatforms = Array.isArray(state.context?.platforms) ? state.context.platforms : [];
      const hasYouTubeInFormat = selectedPlatforms.some(p => p === 'youtube' || p === 'youtube_short');
      const nonYouTubeInFormat = selectedPlatforms.filter(p => p !== 'youtube' && p !== 'youtube_short');
      const isYouTubeOnly = hasYouTubeInFormat && nonYouTubeInFormat.length === 0;
      const forceYouTubeVideoOnly = !includeMedia && isYouTubeOnly;

      const effectiveContentType = forceYouTubeVideoOnly ? 'video' : contentType;

      let callAgents = Array.isArray(state.pendingCallAgents) && state.pendingCallAgents.length > 0
        ? [...state.pendingCallAgents]
        : resolveAgentsForAction(state.context?.selectedAction || 'create_post');

      if (!includeMedia) {
        // Normal "Text only" => drop media_wizard.
        // Special case: YouTube-only should be video-only, even when user chooses "Text only".
        if (!forceYouTubeVideoOnly) callAgents = callAgents.filter(a => a !== 'media_wizard');
      } else if (!callAgents.includes('media_wizard') && (state.context?.selectedAction !== 'extract_content')) {
        callAgents.push('media_wizard');
      }

      // Ensure media_wizard runs for YouTube-only text-only mode.
      if (forceYouTubeVideoOnly && !callAgents.includes('media_wizard')) callAgents.push('media_wizard');

      const formatCtx = {
        ...(state.context || {}),
        contentType: effectiveContentType,
        includeMedia,
        // Text-only for YouTube should avoid image-generation calls (including thumbnail stills).
        skipThumbnails: forceYouTubeVideoOnly,
        job: {
          ...((state.context?.job && typeof state.context.job === 'object') ? state.context.job : {}),
          contentType: effectiveContentType,
          includeMedia,
        },
      };

      // ── YouTube format check: ask Short vs Regular before proceeding ──
      // When YouTube is selected and media will be generated, ask for video format
      // if we don't already have a youtubeConfig.
      if (hasYouTubeInFormat && includeMedia && callAgents.includes('media_wizard') && !state.context?.youtubeConfig) {
        if (isYouTubeOnly) {
          // YouTube-only + image content type: warn no images, ask video confirmation
          console.log('[Orchestrator] YOUTUBE CONFIRMATION (content_type_selected): YouTube-only, asking video confirmation');
          const youtubeConfirmOptions = [
            { action: 'confirm_youtube_video', label: 'Yes, generate video for YouTube', icon: 'video' },
            { action: 'select_platform', label: 'No, choose a different platform', icon: 'switch' },
          ];
          interrupt({
            reason: 'youtube_image_request',
            options: youtubeConfirmOptions,
            message: "⚠️ YouTube doesn't support image generation, but I can generate a video for you instead. Would you like a video for YouTube, or prefer a different platform?",
            buttons: [],
            pendingContext: {
              platforms: selectedPlatforms,
              contentTopic: formatCtx.contentTopic || formatCtx.job?.topic,
              confirmationReason: 'youtube_image_request',
              ...formatFieldsForPendingContext(formatCtx),
            },
          });
          return {
            callAgents: [], awaitingConfirmation: true,
            confirmationReason: 'youtube_image_request',
            confirmationOptions: youtubeConfirmOptions,
            finalReply: "⚠️ YouTube doesn't support image generation, but I can generate a video for you instead. Would you like a video for YouTube, or prefer a different platform?",
            context: formatCtx, pendingCallAgents: callAgents,
            guidedFlowStep: null, intentClarity: 'clear', userSelection: null,
          };
        } else {
          // Multi-platform with YouTube: ask Short vs Regular
          console.log('[Orchestrator] YOUTUBE FORMAT (content_type_selected): multi-platform, asking Short vs Regular');
          const youtubeFormatOptions = [
            { action: 'youtube_format_regular', label: 'Regular Video (16:9 landscape)', icon: 'video' },
            { action: 'youtube_format_short', label: 'YouTube Short (9:16 vertical)', icon: 'short' },
            { action: 'youtube_format_both', label: 'Both formats', icon: 'both' },
          ];
          interrupt({
            reason: 'youtube_format',
            options: youtubeFormatOptions,
            message: "For YouTube, what video format would you like?\n\n**Regular Video** (landscape 16:9) — for longer content with title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.\n\nImages will be generated for your other platforms automatically.",
            buttons: [],
            pendingContext: {
              platforms: selectedPlatforms,
              contentTopic: formatCtx.contentTopic || formatCtx.job?.topic,
              confirmationReason: 'youtube_format',
              ...formatFieldsForPendingContext(formatCtx),
            },
          });
          return {
            callAgents: [], awaitingConfirmation: true,
            confirmationReason: 'youtube_format',
            confirmationOptions: youtubeFormatOptions,
            finalReply: "I see you've selected YouTube along with other platforms. For YouTube, I need to know the video format:\n\n**Regular Video** (landscape 16:9) — for longer content with title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.\n\nImages will be generated for your other platforms automatically.",
            context: formatCtx, pendingCallAgents: callAgents,
            guidedFlowStep: null, intentClarity: 'clear', userSelection: null,
          };
        }
      }

      return {
        callAgents,
        context: formatCtx,
        subTasks: ['Continue after format selection'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
        pendingCallAgents: [],
      };
    }

    if (['schedule_now', 'schedule_later', 'schedule_custom'].includes(sel.action)) {
      return {
        callAgents: ['scheduler_boss'],
        context: { ...(state.context || {}), scheduleAt: sel.scheduleAt || null, publishAllAtSameTime: true },
        subTasks: ['Schedule generated content'],
        directReply: null, awaitingConfirmation: false, confirmationReason: null,
        guidedFlowStep: null, intentClarity: 'clear', userSelection: null,
      };
    }
    if (sel.action === 'skip_schedule') {
      const reply = await generateContextualReply('user skipped scheduling, content is ready to post later whenever they want');
      return {
        callAgents: [], directReply: reply || "Content is ready whenever you want to post. Just say 'schedule it' to pick a time.",
        awaitingConfirmation: false, confirmationReason: null, guidedFlowStep: null,
        intentClarity: 'clear', userSelection: null,
      };
    }
    if (sel.action === 'connect_platform') {
      const platform = sel.platform || 'the selected platform';
      const reply = await generateContextualReply('user needs to connect a social platform to their workspace to start posting', { platform });
      return {
        callAgents: [], awaitingConfirmation: false, confirmationReason: null, guidedFlowStep: null,
        directReply: reply || `To connect ${platform}, go to Settings → Social Accounts and click Connect. Once connected, come back and I'll generate your content!`,
        intentClarity: 'clear', userSelection: null,
      };
    }

    // Media / ambiguous intent button selection
    const stripClientFormatDefaults = shouldIgnoreSelectionFormatForContentRequestResume(state, sel);
    if (stripClientFormatDefaults) {
      console.log('[Orchestrator] Resume: ignoring client default image+media; preserving session text-only format');
    }
    const selContentTypeEff = stripClientFormatDefaults ? undefined : sel.contentType;
    const selIncludeMediaEff = stripClientFormatDefaults ? undefined : sel.includeMedia;

    let callAgents = resolveAgentsForAction(sel.action);
    if (selIncludeMediaEff === false) {
      callAgents = callAgents.filter(a => a !== 'media_wizard');
    }
    const selAnalysis = state.mediaAnalysis || state.context?.mediaAnalysis;

    // When the interrupt was only asking for format (missing_format/missing_topic),
    // the user had already specified platforms. The frontend may send all platforms
    // as default in the selection — ignore those defaults and use the saved platforms.
    const wasFormatOrTopicOnly = state.confirmationReason === 'missing_format' || state.confirmationReason === 'missing_topic';
    // Check all possible sources for saved platforms (graph state, baseContext, job)
    const savedPlatforms =
      (Array.isArray(state.context?.platforms) && state.context.platforms.length > 0 && state.context.platforms) ||
      (Array.isArray(baseContext.platforms) && baseContext.platforms.length > 0 && baseContext.platforms) ||
      (Array.isArray(job.platforms) && job.platforms.length > 0 && job.platforms) ||
      null;
    console.log('[Orchestrator] Resume: confirmationReason:', state.confirmationReason, '| savedPlatforms:', savedPlatforms, '| sel.platforms:', sel.platforms?.length);

    let resolvedPlatforms;
    if (wasFormatOrTopicOnly && savedPlatforms) {
      // Interrupt was only for format/topic — platforms were already chosen, preserve them
      resolvedPlatforms = savedPlatforms;
      console.log('[Orchestrator] Resume: preserving saved platforms (interrupt was format/topic only):', resolvedPlatforms);
    } else {
      resolvedPlatforms = resolvePlatformsWithFallback(
        sel.platforms,
        state.context?.platforms,
        state.context?.activePlatforms
      );
    }
    const context = {
      ...baseContext,
      platforms: resolvedPlatforms,
      ...(Array.isArray(sel.accounts) ? { accounts: sel.accounts } : {}),
      contentType: selContentTypeEff || state.context?.contentType || job.contentType || 'image',
      includeMedia: selIncludeMediaEff !== undefined
        ? selIncludeMediaEff
        : (state.context?.includeMedia !== undefined
          ? state.context.includeMedia
          : (job.includeMedia !== undefined
            ? job.includeMedia
            : (inferIncludeMediaFromContentType(
              selContentTypeEff || state.context?.contentType || job.contentType
            ) ?? true))),
      selectedAction: sel.action,
      mediaAnalysis: selAnalysis,
      ...(sel.contentTopic ? { contentTopic: String(sel.contentTopic).trim() } : {}),
      ...(sel.rawContent ? { rawContent: String(sel.rawContent).trim() } : {}),
      ...(sel.action === 'extract_content' && selAnalysis?.description
        ? { rawContent: selAnalysis.description } : {}),
    };

    // Update job spec from selection
    context.job = {
      ...job,
      platforms: context.platforms || job.platforms,
      contentType: context.contentType || job.contentType,
      includeMedia: context.includeMedia !== undefined ? context.includeMedia : job.includeMedia,
      selectedAction: sel.action,
      topic: context.contentTopic || job.topic,
    };

    if (context.includeMedia === false) {
      callAgents = callAgents.filter(a => a !== 'media_wizard');
    }

    const needsPlatforms = callAgents.some(a => ['content_genius', 'media_wizard'].includes(a));
    const hasPlatforms = context.platforms && context.platforms.length > 0;

    if (needsPlatforms && !hasPlatforms) {
      const connectedPlatforms = context?.userId
        ? (await getActivePlatforms(context.userId, context.workspaceId) || [])
        : [];

      if (connectedPlatforms.length === 0) {
        const noPlatformsReply = await generateContextualReply('user has no social media accounts connected, prompt them to connect a platform in settings');
        return {
          callAgents: [], directReply: null, guidedFlowStep: null,
          awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(),
          confirmationReason: 'no_platforms',
          finalReply: noPlatformsReply || 'No social media accounts are connected yet. Connect a platform in Settings to get started.',
          buttons: [],
        };
      }

      const names = connectedPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');

      const confirmationOptions = buildConfirmationOptions(true, { call_agents: callAgents }, selAnalysis, {
        ...context,
        activePlatforms: connectedPlatforms,
        contentType: context.contentType,
      });

      const dynamicButtons = [];
      if (connectedPlatforms.length > 1) {
        dynamicButtons.push({
          label: 'Quick: All platforms (images + content)',
          action: 'quick_create_post_all',
          platforms: connectedPlatforms,
        });
      }
      connectedPlatforms.forEach((p) => {
        dynamicButtons.push({
          label: `Quick: ${PLATFORM_LABELS[p] || p}`,
          action: 'quick_create_post_single',
          platforms: [p],
        });
      });

      const confirmationMessage = `Please choose what you'd like to do next:`;

      interrupt({
        reason: 'content_request',
        options: confirmationOptions,
        message: confirmationMessage,
        buttons: dynamicButtons,
        mediaAnalysis: selAnalysis || null,
        pendingContext: {
          contentTopic: context.contentTopic || context.job?.topic,
          confirmationReason: 'content_request',
          rawUserPrompt: rawPrompt || '',
          ...formatFieldsForPendingContext(context),
        },
      });

      return {
        context: {
          ...context,
          // FIX 1 -- use actual image description as contentTopic
          contentTopic: buildContentTopicFromMedia(selAnalysis),
          activePlatforms: connectedPlatforms,
        },
        pendingCallAgents: callAgents,
        callAgents: [],
        awaitingConfirmation: true,
        confirmationOptions,
        confirmationReason: 'content_request',
        finalReply: confirmationMessage,
        buttons: dynamicButtons,
        intentClarity: 'clear',
        mediaAnalysis: selAnalysis,
      };
    }

    // ── YouTube Confirmation Check (Resume Path - Before returning callAgents) ────────
    const selectedPlatformsForYT = Array.isArray(context.platforms) ? context.platforms : [];
    const hasYouTubeInSelection = selectedPlatformsForYT.includes('youtube');
    const isImageContentInSelection = context.contentType === 'image' || !context.contentType;
    const isCallingMediaWizardInSelection = callAgents.includes('media_wizard');
    const nonYouTubeSelected = selectedPlatformsForYT.filter(p => p !== 'youtube' && p !== 'youtube_short');
    const isYouTubeOnlySelection = hasYouTubeInSelection && nonYouTubeSelected.length === 0;
    const hasYouTubeConfigAlready = !!(context.youtubeConfig);

    // YouTube-only + image request: warn about no image support, ask video confirmation
    if (isYouTubeOnlySelection && isImageContentInSelection && isCallingMediaWizardInSelection) {
      console.log('[Orchestrator] YOUTUBE CONFIRMATION REQUIRED (RESUME): YouTube-only for image content');

      const youtubeConfirmOptions = [
        { action: 'confirm_youtube_video', label: 'Yes, generate video for YouTube', icon: 'video' },
        { action: 'select_platform', label: 'No, choose a different platform', icon: 'switch' },
      ];
      interrupt({
        reason: 'youtube_image_request',
        options: youtubeConfirmOptions,
        message: "⚠️ I notice you selected YouTube. YouTube doesn't support image generation, but I can generate a video for you instead. Would you like me to generate a video for YouTube, or would you prefer to choose a different platform for images?",
        buttons: [],
        pendingContext: {
          platforms: context.platforms,
          contentTopic: context.contentTopic || context.job?.topic,
          confirmationReason: 'youtube_image_request',
          ...formatFieldsForPendingContext(context),
        },
      });
      return {
        callAgents: [],
        awaitingConfirmation: true, confirmationReason: 'youtube_image_request',
        confirmationOptions: youtubeConfirmOptions,
        finalReply: "⚠️ I notice you selected YouTube. YouTube doesn't support image generation, but I can generate a video for you instead. Would you like me to generate a video for YouTube, or would you prefer to choose a different platform for images?",
        context: { ...context, activePlatforms: context.platforms },
        pendingCallAgents: callAgents, userSelection: null,
      };
    }

    // Multi-platform with YouTube + no format selected yet: ask Short vs Regular
    if (hasYouTubeInSelection && !isYouTubeOnlySelection && isCallingMediaWizardInSelection && !hasYouTubeConfigAlready) {
      console.log('[Orchestrator] YOUTUBE FORMAT REQUIRED (RESUME): YouTube in multi-platform — asking Short vs Regular');

      const youtubeFormatOptions = [
        { action: 'youtube_format_regular', label: 'Regular Video (16:9 landscape)', icon: 'video' },
        { action: 'youtube_format_short', label: 'YouTube Short (9:16 vertical)', icon: 'short' },
        { action: 'youtube_format_both', label: 'Both formats', icon: 'both' },
      ];
      interrupt({
        reason: 'youtube_format',
        options: youtubeFormatOptions,
        message: "I see you've selected YouTube along with other platforms. For YouTube, I need to know the video format:\n\n**Regular Video** (landscape 16:9) — for longer content with a full title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.\n\nImages will be generated for your other platforms automatically.",
        buttons: [],
        pendingContext: {
          platforms: context.platforms,
          contentTopic: context.contentTopic || context.job?.topic,
          confirmationReason: 'youtube_format',
          ...formatFieldsForPendingContext(context),
        },
      });
      return {
        callAgents: [],
        awaitingConfirmation: true, confirmationReason: 'youtube_format',
        confirmationOptions: youtubeFormatOptions,
        finalReply: "I see you've selected YouTube along with other platforms. For YouTube, I need to know the video format:\n\n**Regular Video** (landscape 16:9) — for longer content with a full title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.\n\nImages will be generated for your other platforms automatically.",
        context: { ...context, activePlatforms: context.platforms },
        pendingCallAgents: callAgents, userSelection: null,
      };
    }

    return {
      callAgents,
      context: {
        ...context,
        activePlatforms: context.platforms,
      },
      subTasks: [`Execute action: ${sel.action}`],
      directReply: null, awaitingConfirmation: false, confirmationReason: null,
      guidedFlowStep: null, intentClarity: 'clear', userSelection: null,
    };
  }

  // ── TEXT-BASED RESUME: handled by LLM via session_state + conversation history ──
  if (false && state.confirmationReason === 'media_present' && rawPrompt) {
    const mediaAction = parseMediaActionFromText(rawPrompt);
    console.log('[Orchestrator] TEXT-BASED RESUME -- media_present:', rawPrompt, '→ action:', mediaAction);

    if (mediaAction) {
      const analysis = state.mediaAnalysis || state.context?.mediaAnalysis;
      const callAgents = resolveAgentsForAction(mediaAction);
      const context = {
        ...baseContext,
        selectedAction: mediaAction,
        contentType: state.context?.contentType || 'image',
        mediaAnalysis: analysis,
        confirmationReason: null,
        ...(mediaAction === 'extract_content' && analysis?.description
          ? { rawContent: analysis.description } : {}),
      };
      context.job = { ...job, selectedAction: mediaAction, contentType: context.contentType || job.contentType };

      if (mediaAction === 'analyze_brand') {
        return {
          callAgents: [], context, directReply: null,
          awaitingConfirmation: false, confirmationReason: null,
          guidedFlowStep: null, intentClarity: 'clear', userSelection: null,
          mediaAnalysis: analysis,
        };
      }

      if (['create_post', 'use_as_media', 'regenerate_media'].includes(mediaAction)) {
        const connectedPlatforms = context?.userId
          ? (await getActivePlatforms(context.userId, context.workspaceId) || [])
          : [];

        if (connectedPlatforms.length === 0) {
          return {
            callAgents: [], directReply: null, guidedFlowStep: null,
            awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(),
            confirmationReason: 'no_platforms',
            finalReply: 'No social media accounts are connected yet. Which platform would you like to connect?',
            buttons: [],
          };
        }

        const names = connectedPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
        return {
          context: {
            ...context,
            // FIX 2 -- use actual image description as contentTopic
            contentTopic: buildContentTopicFromMedia(analysis),
          },
          pendingCallAgents: callAgents,
          callAgents: [],
          directReply: `Sure! Which platforms would you like to post on?\n\nYour connected platforms: **${names}**`,
          guidedFlowStep: 'awaiting_platforms',
          awaitingConfirmation: false, confirmationReason: null,
          intentClarity: 'clear', userSelection: null, mediaAnalysis: analysis,
        };
      }

      // extract_content -- run directly
      return {
        callAgents, context,
        subTasks: [`Execute: ${mediaAction}`],
        directReply: null, awaitingConfirmation: false, confirmationReason: null,
        guidedFlowStep: null, intentClarity: 'clear', userSelection: null, mediaAnalysis: analysis,
      };
    }

    // Didn't recognise -- re-show options
    return {
      callAgents: [],
      directReply: `I didn't quite get that. Would you like to:\n• **Create a post** using this image\n• **Use this image directly** in your post\n• **Extract ideas** from it\n• **Check brand consistency**`,
      awaitingConfirmation: true,
      confirmationReason: 'media_present',
      confirmationOptions: state.confirmationOptions || [],
      guidedFlowStep: null, intentClarity: 'ambiguous',
    };
  }

  // ── TEXT-BASED RESUME: handled by LLM via session_state + conversation history ──
  if (false && state.confirmationReason === 'content_request' && rawPrompt) {
    const action = parseContentActionFromText(rawPrompt);
    if (action) {
      const { contentType, includeMedia } = parseContentTypeFromText(rawPrompt);
      let callAgents = resolveAgentsForAction(action);
      if (action === 'create_post') {
        // If user said text-only, drop media; otherwise keep media default.
        if (isTextOnlyRequest(rawPrompt) || includeMedia === false) callAgents = callAgents.filter(a => a !== 'media_wizard');
        else if (!callAgents.includes('media_wizard')) callAgents.push('media_wizard');
      }
      if (action === 'generate_media') {
        callAgents = ['media_wizard'];
      }

      const resolvedPlatforms = resolvePlatformsWithFallback(
        parsePlatformsFromText(rawPrompt),
        state.context?.platforms,
        state.context?.activePlatforms
      );

      const nextContext = {
        ...baseContext,
        platforms: resolvedPlatforms,
        contentType: contentType || state.context?.contentType || 'image',
        includeMedia: (action === 'generate_media') ? true : (includeMedia !== undefined ? includeMedia : true),
        selectedAction: action,
      };
      nextContext.job = {
        ...job,
        platforms: nextContext.platforms || job.platforms,
        contentType: nextContext.contentType || job.contentType,
        includeMedia: nextContext.includeMedia !== undefined ? nextContext.includeMedia : job.includeMedia,
        selectedAction: action,
        topic: nextContext.contentTopic || job.topic,
      };

      // If we need platforms and don't have them yet, ask (guided flow).
      const needsPlatforms = callAgents.some(a => ['content_genius', 'media_wizard'].includes(a));
      const hasPlatforms = Array.isArray(nextContext.platforms) && nextContext.platforms.length > 0;
      if (needsPlatforms && !hasPlatforms) {
        const connectedPlatforms = nextContext?.userId
          ? (await getActivePlatforms(nextContext.userId, nextContext.workspaceId) || [])
          : [];
        if (connectedPlatforms.length === 0) {
          return {
            callAgents: [], directReply: null, guidedFlowStep: null,
            awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(),
            confirmationReason: 'no_platforms',
            finalReply: 'No social media accounts are connected yet. Which platform would you like to connect?',
            buttons: [],
          };
        }
        const names = connectedPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
        return {
          context: { ...nextContext, platforms: connectedPlatforms },
          pendingCallAgents: callAgents,
          callAgents: [],
          directReply: `Sure! Which platforms would you like to post on?\n\nYour connected platforms: **${names}**`,
          guidedFlowStep: 'awaiting_platforms',
          awaitingConfirmation: false,
          confirmationReason: null,
          intentClarity: 'clear',
          userSelection: null,
        };
      }

      return {
        callAgents,
        context: nextContext,
        subTasks: [`Execute action: ${action}`],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }
  }

  // ── GUIDED FLOW STATE MACHINE ─────────────────────────────────────────────

  // Follow-up intent router (runs even mid-flow).
  // This prevents the system from assuming "done" after first generation.
  if (false /* Follow-up intent router retired -- LLM decides routing from full context */ && rawPrompt) {
    const n = normalize(rawPrompt);
    const scheduleParsed = state.guidedFlowStep ? parseScheduleFromText(rawPrompt) : null;
    const mentionedPlatforms = parsePlatformsFromText(rawPrompt);
    const hasPlatformChange = mentionedPlatforms.length > 0 && /\bonly\b|\bjust\b|\bremove\b|\bexclude\b|\badd\b/i.test(n);
    const variantCount = parseVariantCountFromText(rawPrompt);
    const editInstruction = parseCopyEditInstruction(rawPrompt);
    const wantsMediaAction = parseContentActionFromText(rawPrompt); // schedule/create_post/generate_media/null
    const topicSwitch = looksLikeTopicSwitch(rawPrompt);

    // Platform-only update (e.g. "only instagram") should not trigger scheduling.
    if (hasPlatformChange) {
      const resolvedPlatforms = mentionedPlatforms;
      const nextCtx = {
        ...baseContext,
        platforms: resolvedPlatforms,
        job: { ...job, platforms: resolvedPlatforms },
      };
      // If we are waiting to schedule, keep waiting but with updated platforms.
      if (state.guidedFlowStep === 'awaiting_schedule') {
        return {
          callAgents: [],
          context: nextCtx,
          guidedFlowStep: 'awaiting_schedule',
          directReply: `Got it -- updating platforms to **${resolvedPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ')}**.\n\n⏰ When should I schedule this? (e.g. *"tomorrow 9am"* or *"skip"*)`,
          awaitingConfirmation: false,
          intentClarity: 'clear',
          userSelection: null,
        };
      }
    }

    // Copy edits (rewrite/shorter/tone/etc.) -- route back to content_genius.
    if (editInstruction) {
      const mentionedPlatforms = parsePlatformsFromText(rawPrompt);
      const scopedPlatforms = mentionedPlatforms.length > 0 ? mentionedPlatforms : (baseContext.platforms || job.platforms);
      const nextCtx = {
        ...baseContext,
        selectedAction: 'create_post',
        // Provide instructions via rawContent so content_genius can adapt while keeping the topic.
        rawContent: rawPrompt.trim(),
        platforms: scopedPlatforms,
        job: { ...job, selectedAction: 'create_post', editInstruction, platforms: scopedPlatforms },
      };
      return {
        callAgents: ['content_genius', 'preview_pro'],
        context: nextCtx,
        subTasks: ['Apply copy edits'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    // More variants / "another" media -- route to media_wizard with a higher count.
    if (variantCount && (wantsMediaAction === 'generate_media' || /\bimage|images|video|reel|shorts?\b/i.test(n) || state.mediaResult)) {
      const { contentType } = parseContentTypeFromText(rawPrompt);
      const resolvedPlatforms = resolvePlatformsWithFallback(
        parsePlatformsFromText(rawPrompt),
        state.context?.platforms,
        state.context?.activePlatforms
      );
      const nextCtx = {
        ...baseContext,
        selectedAction: 'regenerate_media',
        contentType: contentType || baseContext.contentType || job.contentType || 'image',
        mediaCount: variantCount,
        // Keep old topic unless the user supplied a new topic prompt.
        ...(topicSwitch ? { rawContent: rawPrompt.trim(), contentTopic: rawPrompt.trim() } : {}),
        platforms: resolvedPlatforms || baseContext.platforms || job.platforms,
        job: { ...job, selectedAction: 'regenerate_media', mediaCount: variantCount },
      };
      return {
        callAgents: ['media_wizard'],
        context: nextCtx,
        subTasks: [`Generate ${variantCount} variant(s)`],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    // Topic switch with media request: regenerate content + media so they match.
    if (topicSwitch && wantsMediaAction === 'generate_media') {
      const { contentType } = parseContentTypeFromText(rawPrompt);
      const resolvedPlatforms = resolvePlatformsWithFallback(
        parsePlatformsFromText(rawPrompt),
        state.context?.platforms,
        state.context?.activePlatforms
      );
      const nextCtx = {
        ...baseContext,
        selectedAction: 'generate_media',
        includeMedia: true,
        contentType: contentType || job.contentType || 'image',
        rawContent: rawPrompt.trim(),
        contentTopic: rawPrompt.trim(),
        platforms: resolvedPlatforms || baseContext.platforms || job.platforms,
        job: { ...job, topic: rawPrompt.trim(), contentType: contentType || job.contentType, selectedAction: 'generate_media' },
      };
      return {
        callAgents: ['content_genius', 'media_wizard', 'preview_pro'],
        context: nextCtx,
        subTasks: ['Generate content + media for new topic'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    // If user gives a scheduling time while awaiting_schedule, keep routing to scheduler.
    if (state.guidedFlowStep === 'awaiting_schedule' && scheduleParsed && scheduleParsed.action !== 'unknown') {
      // Let the existing awaiting_schedule block handle it below.
    }

    // New topic content request while mid-flow (e.g. "I want content for my software company"
    // after content was already generated). Reset the guided flow and regenerate from scratch.
    if (topicSwitch && wantsMediaAction === 'create_post' && state.guidedFlowStep) {
      const { contentType, includeMedia } = parseContentTypeFromText(rawPrompt);
      const textOnly = isTextOnlyRequest(rawPrompt);
      const effectiveIncludeMedia = !textOnly && (includeMedia !== false);
      const resolvedPlatforms = resolvePlatformsWithFallback(
        parsePlatformsFromText(rawPrompt),
        state.context?.platforms,
        state.context?.activePlatforms
      );
      const agents = effectiveIncludeMedia
        ? ['content_genius', 'media_wizard', 'preview_pro']
        : ['content_genius', 'preview_pro'];
      const nextCtx = {
        ...baseContext,
        selectedAction: 'create_post',
        includeMedia: effectiveIncludeMedia,
        contentType: contentType || baseContext.contentType || job.contentType || 'image',
        rawContent: rawPrompt.trim(),
        contentTopic: rawPrompt.trim(),
        platforms: resolvedPlatforms || baseContext.platforms || job.platforms,
        job: { ...job, topic: rawPrompt.trim(), contentType: contentType || job.contentType, selectedAction: 'create_post', platforms: resolvedPlatforms || job.platforms },
      };
      console.log('[Orchestrator] Topic switch (create_post) mid-flow → resetting guided flow | agents:', agents);
      return {
        callAgents: agents,
        context: nextCtx,
        subTasks: ['Generate fresh content for new topic'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }
  }

  // Global override: retired -- LLM handles mid-flow media requests from conversation context.
  if (false && rawPrompt && state.guidedFlowStep && state.guidedFlowStep !== 'awaiting_platforms') {
    const overrideAction = parseContentActionFromText(rawPrompt);
    if (overrideAction === 'generate_media') {
      const { contentType } = parseContentTypeFromText(rawPrompt);

      // If the user provides a substantial prompt (topic switch) while mid-flow,
      // prefer regenerating BOTH content + media using the new prompt.
      const n = normalize(rawPrompt);
      const deCommanded = n
        .replace(/\b(generate|genrate|generte|gnerate|create|make|need|want|also|add|too)\b/g, ' ')
        .replace(/\b(video|reel|shorts?|tiktok|youtube|yt)\b/g, ' ')
        .replace(/\b(images?|photos?|pics?|visuals?|graphics?|picture|pictures)\b/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      const isTopicPrompt = deCommanded.length >= 25;
      const callAgents = isTopicPrompt ? ['content_genius', 'media_wizard', 'preview_pro'] : ['media_wizard'];

      const resolvedPlatforms = resolvePlatformsWithFallback(
        parsePlatformsFromText(rawPrompt),
        state.context?.platforms,
        state.context?.activePlatforms
      );
      return {
        callAgents,
        context: {
          ...baseContext,
          selectedAction: 'generate_media',
          includeMedia: true,
          contentType: contentType || state.context?.contentType || 'image',
          // Force media_wizard to use the new prompt rather than stale contentResult.
          rawContent: rawPrompt.trim(),
          contentTopic: rawPrompt.trim(),
          platforms: resolvedPlatforms || state.context?.platforms || state.context?.activePlatforms,
          videoPurpose: contentType === 'video' ? parseVideoPurposeFromText(rawPrompt) : state.context?.videoPurpose,
          job: { ...job, topic: rawPrompt.trim(), contentType: contentType || job.contentType, selectedAction: 'generate_media' },
        },
        subTasks: [isTopicPrompt ? 'Generate content + media for new topic' : 'Generate media from user request'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }
  }

  // Step 0 -- retired: LLM now asks for the topic directly in direct_reply.
  if (false && state.guidedFlowStep === 'awaiting_topic') {
    const topic = rawPrompt.trim();
    if (!topic || topic.length < 2) {
      return {
        callAgents: [],
        guidedFlowStep: 'awaiting_topic',
        directReply: "Please share what you'd like to post about -- a topic, product, idea or your own text.",
        awaitingConfirmation: false, intentClarity: 'clear',
      };
    }

    const connectedPlatforms = state.context?.userId
      ? (await getActivePlatforms(state.context.userId, state.context.workspaceId) || [])
      : [];

    if (connectedPlatforms.length === 0) {
      return {
        callAgents: [], directReply: null, guidedFlowStep: null,
        awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(),
        confirmationReason: 'no_platforms',
        finalReply: 'No social media accounts are connected yet. Which platform would you like to connect?',
        buttons: [],
      };
    }

    const contextWithTopic = {
      ...baseContext,
      contentTopic: topic,
      rawContent: topic,
      activePlatforms: connectedPlatforms,
      job: { ...job, topic, platforms: connectedPlatforms },
    };

    const confirmationOptions = buildConfirmationOptions(false, { call_agents: ['content_genius', 'media_wizard'] }, null, {
      ...contextWithTopic,
      contentType: contextWithTopic.contentType || 'image',
    });

    const dynamicButtons = [];
    if (connectedPlatforms.length > 1) {
      dynamicButtons.push({ label: 'Quick: All platforms (images + content)', action: 'quick_create_post_all', platforms: connectedPlatforms });
    }
    connectedPlatforms.forEach(p => {
      dynamicButtons.push({ label: `Quick: ${PLATFORM_LABELS[p] || p}`, action: 'quick_create_post_single', platforms: [p] });
    });

    console.log('[Orchestrator] Guided Step 0 → topic received:', topic.slice(0, 60), '| triggering platform selection');

    interrupt({ reason: 'content_request', options: confirmationOptions, message: '', buttons: dynamicButtons, mediaAnalysis: null });

    return {
      context: contextWithTopic, callAgents: [], awaitingConfirmation: true,
      confirmationOptions, confirmationReason: 'content_request', finalReply: '',
      buttons: dynamicButtons, guidedFlowStep: null, intentClarity: 'clear',
    };
  }

  // Step 1 -- retired: LLM decides platform handling from conversation context.
  if (false && state.guidedFlowStep === 'awaiting_platforms') {
    const platforms = parsePlatformsFromText(rawPrompt);

    if (platforms.length === 0) {
      const connectedPlatforms = state.context?.userId
        ? (await getActivePlatforms(state.context.userId, state.context.workspaceId) || [])
        : [];

      const isConfirmation = /^(ok(ay)?|fine|yes|sure|all( of them| platforms?)?|proceed|go ahead|sounds good|perfect|great|cool|yep|yeah|yup)[\s\.!]*$/i.test(rawPrompt.trim());
      if (isConfirmation && connectedPlatforms.length > 0) {
        const names = connectedPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
        console.log('[Orchestrator] Guided Step 1 → confirmation, using all connected:', connectedPlatforms);
        return {
          context: { ...(state.context || {}), platforms: connectedPlatforms },
          callAgents: [],
          guidedFlowStep: 'awaiting_content_type',
          directReply: `Got it -- posting to **${names}**! 🎯\n\nWhat should I create?\n• **Images + content** -- captions with matching visuals\n• **Text only** -- captions/copy without images\n• **Video** -- video content`,
          awaitingConfirmation: false, intentClarity: 'clear',
        };
      }

      const names = connectedPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
      return {
        callAgents: [], guidedFlowStep: 'awaiting_platforms',
        directReply: `I didn't catch that. Your connected platforms are: **${names || 'none'}**. Which one(s) would you like to post on?`,
        awaitingConfirmation: false, intentClarity: 'clear',
      };
    }

    const names = platforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
    console.log('[Orchestrator] Guided Step 1 → platforms resolved:', platforms);
    return {
      context: { ...(state.context || {}), platforms },
      callAgents: [],
      guidedFlowStep: 'awaiting_content_type',
      directReply: `Got it -- posting to **${names}**! 🎯\n\nWhat should I create?\n• **Images + content** -- captions with matching visuals\n• **Text only** -- captions/copy without images\n• **Video** -- video content`,
      awaitingConfirmation: false, intentClarity: 'clear',
    };
  }

  // Step 2 -- retired: LLM infers content type from user message.
  if (false && state.guidedFlowStep === 'awaiting_content_type') {
    const { contentType, includeMedia } = parseContentTypeFromText(rawPrompt);

    if (contentType === 'video') {
      console.log('[Orchestrator] Guided Step 2 → video, asking purpose');
      return {
        context: { ...(state.context || {}), contentType: 'video' },
        callAgents: [],
        guidedFlowStep: 'awaiting_video_purpose',
        directReply: `Great choice! 🎬 Is this for a **Reel or Short** (vertical 9:16) or a **regular video** (landscape 16:9)?\n\nOr say **both** and I'll generate all formats.`,
        awaitingConfirmation: false, intentClarity: 'clear',
      };
    }

    let callAgents = state.pendingCallAgents?.length ? [...state.pendingCallAgents] : ['content_genius', 'media_wizard'];
    if (!includeMedia) {
      callAgents = callAgents.filter(a => a !== 'media_wizard');
    } else if (!callAgents.includes('media_wizard')) {
      callAgents.push('media_wizard');
    }

    console.log('[Orchestrator] Guided Step 2 → contentType:', contentType, '| includeMedia:', includeMedia, '| callAgents:', callAgents);
    return {
      callAgents,
      context: { ...(state.context || {}), contentType },
      subTasks: [`Generate ${contentType} content`],
      directReply: null, awaitingConfirmation: false, confirmationReason: null,
      guidedFlowStep: null, intentClarity: 'clear', userSelection: null, pendingCallAgents: [],
    };
  }

  // Step 2b -- retired: LLM infers video purpose from user message.
  if (false && state.guidedFlowStep === 'awaiting_video_purpose') {
    const videoPurpose = parseVideoPurposeFromText(rawPrompt);
    let callAgents = state.pendingCallAgents?.length ? [...state.pendingCallAgents] : ['content_genius', 'media_wizard'];
    if (!callAgents.includes('media_wizard')) callAgents.push('media_wizard');
    console.log('[Orchestrator] Guided Step 2b → videoPurpose:', videoPurpose);
    return {
      callAgents,
      context: { ...(state.context || {}), contentType: 'video', videoPurpose },
      subTasks: ['Generate video content'],
      directReply: null, awaitingConfirmation: false, confirmationReason: null,
      guidedFlowStep: null, intentClarity: 'clear', userSelection: null, pendingCallAgents: [],
    };
  }

  // Step 3 -- retired: LLM handles scheduling intent from conversation context.
  if (false && state.guidedFlowStep === 'awaiting_schedule') {
    // While awaiting a schedule time, allow the user to change direction (e.g. "images too").
    const overrideAction = parseContentActionFromText(rawPrompt);
    if (overrideAction === 'generate_media') {
      const { contentType } = parseContentTypeFromText(rawPrompt);

      // If this looks like a fresh prompt/topic (not just "video please"),
      // regenerate content + media from the new prompt to avoid using stale contentResult.
      const n = normalize(rawPrompt);
      const deCommanded = n
        .replace(/\b(generate|genrate|generte|gnerate|create|make|need|want|also|add|too)\b/g, ' ')
        .replace(/\b(video|reel|shorts?|tiktok|youtube|yt)\b/g, ' ')
        .replace(/\b(images?|photos?|pics?|visuals?|graphics?|picture|pictures)\b/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      const isTopicPrompt = deCommanded.length >= 25;
      const callAgents = isTopicPrompt ? ['content_genius', 'media_wizard', 'preview_pro'] : ['media_wizard'];

      const resolvedPlatforms = resolvePlatformsWithFallback(
        parsePlatformsFromText(rawPrompt),
        state.context?.platforms,
        state.context?.activePlatforms
      );
      return {
        callAgents,
        context: {
          ...baseContext,
          selectedAction: 'generate_media',
          includeMedia: true,
          contentType: contentType || state.context?.contentType || 'image',
          rawContent: rawPrompt.trim(),
          contentTopic: rawPrompt.trim(),
          platforms: resolvedPlatforms || state.context?.platforms || state.context?.activePlatforms,
          videoPurpose: contentType === 'video' ? parseVideoPurposeFromText(rawPrompt) : state.context?.videoPurpose,
          job: { ...job, topic: rawPrompt.trim(), contentType: contentType || job.contentType, selectedAction: 'generate_media' },
        },
        subTasks: [isTopicPrompt ? 'Generate content + media for new topic' : 'Generate media from user request'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }
    if (overrideAction === 'create_post') {
      // Distinguish between a fresh topic request and a copy-edit while still in the scheduling step.
      const isNewTopic = looksLikeTopicSwitch(rawPrompt);
      const textOnly = isTextOnlyRequest(rawPrompt);
      const effectiveIncludeMedia = !textOnly;
      const currentPlatforms = state.context?.platforms || state.context?.activePlatforms;
      const agents = isNewTopic && effectiveIncludeMedia
        ? ['content_genius', 'media_wizard', 'preview_pro']
        : textOnly
          ? ['content_genius', 'preview_pro']
          : ['content_genius', 'preview_pro'];

      const newTopicCtx = isNewTopic
        ? { contentTopic: rawPrompt.trim(), rawContent: rawPrompt.trim() }
        : { rawContent: rawPrompt.trim() };

      console.log('[Orchestrator] awaiting_schedule → create_post override | isNewTopic:', isNewTopic, '| textOnly:', textOnly, '| agents:', agents);
      return {
        callAgents: agents,
        context: {
          ...baseContext,
          ...newTopicCtx,
          selectedAction: 'create_post',
          includeMedia: effectiveIncludeMedia,
          platforms: currentPlatforms,
          job: {
            ...job,
            selectedAction: 'create_post',
            ...(isNewTopic ? { topic: rawPrompt.trim(), platforms: currentPlatforms } : {}),
          },
        },
        subTasks: [isNewTopic ? 'Generate fresh content for new topic' : 'Apply copy edits to existing content'],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        userSelection: null,
      };
    }

    const parsed = parseScheduleFromText(rawPrompt);
    console.log('[Orchestrator] Guided Step 3 → schedule parsed:', parsed);

    if (parsed.action === 'gratitude') {
      return {
        callAgents: [], guidedFlowStep: 'awaiting_schedule',
        directReply: `You're welcome! 😊\n\n⏰ **Would you like to schedule this?**\nSay *"post now"*, *"tomorrow 9am"*, a specific date/time, or *"skip"*.`,
        awaitingConfirmation: false, intentClarity: 'clear',
      };
    }
    if (parsed.action === 'skip') {
      return {
        callAgents: [], directReply: "No problem! Your content is saved and ready. Just say 'schedule it' whenever you want to post.",
        guidedFlowStep: null, awaitingConfirmation: false, confirmationReason: null, intentClarity: 'clear',
      };
    }
    if (parsed.action === 'schedule_now') {
      return {
        callAgents: ['scheduler_boss'],
        context: { ...(state.context || {}), scheduleAt: parsed.scheduleAt, publishAllAtSameTime: true },
        subTasks: ['Schedule now'],
        directReply: null, awaitingConfirmation: false, confirmationReason: null,
        guidedFlowStep: null, intentClarity: 'clear',
      };
    }
    if (parsed.action === 'unknown') {
      return {
        callAgents: [],
        guidedFlowStep: 'awaiting_schedule',
        directReply: `⏰ **When should I schedule this?**\nSay *"post now"*, *"tomorrow 9am"*, a specific date/time, or *"skip"*.\n\nIf you want changes instead, say things like *"images too"* or *"rewrite the caption"*.`,
        awaitingConfirmation: false,
        intentClarity: 'clear',
      };
    }
    return {
      callAgents: ['scheduler_boss'],
      context: { ...(state.context || {}), rawScheduleRequest: parsed.rawTime, publishAllAtSameTime: true },
      subTasks: ['Schedule content'],
      directReply: null, awaitingConfirmation: false, confirmationReason: null,
      guidedFlowStep: 'awaiting_schedule', intentClarity: 'clear',
    };
  }

  // ── Media-only upload (no text) → vision + auto create_post flow ──────────
  if (state.mediaInput && !rawPrompt.trim()) {
    console.log('[Orchestrator] Media-only upload -- running vision, going to platform question');
    const ctx = state.context || {};
    const connectedPlatforms = ctx?.userId
      ? (await getActivePlatforms(ctx.userId, ctx.workspaceId) || [])
      : [];
    const analysis = await analyzeMediaWithVision(state.mediaInput, '');

    if (connectedPlatforms.length === 0) {
      const noPlatformsReply = await generateContextualReply('user has no social media accounts connected, prompt them to connect a platform in settings');
      return {
        callAgents: [], directReply: null, guidedFlowStep: null,
        awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(),
        confirmationReason: 'no_platforms',
        finalReply: noPlatformsReply || 'No social media accounts are connected yet. Connect a platform in Settings to get started.',
        buttons: [], mediaAnalysis: analysis,
      };
    }

    const isVideo = state.mediaInput.mimeType?.startsWith('video/');
    console.log(`[Orchestrator] Auto-selecting create_post → triggering confirmation for platforms (${isVideo ? 'video' : 'image'}):`, connectedPlatforms);

    const confirmationOptions = buildConfirmationOptions(true, { call_agents: ['content_genius', 'media_wizard'] }, analysis, {
      ...ctx,
      activePlatforms: connectedPlatforms,
      contentType: ctx?.contentType || (isVideo ? 'video' : 'image'),
    });

    const dynamicButtons = [];
    if (connectedPlatforms.length > 1) {
      dynamicButtons.push({
        label: 'Quick: All platforms (images + content)',
        action: 'quick_create_post_all',
        platforms: connectedPlatforms,
      });
    }
    connectedPlatforms.forEach((p) => {
      dynamicButtons.push({
        label: `Quick: ${PLATFORM_LABELS[p] || p}`,
        action: 'quick_create_post_single',
        platforms: [p],
      });
    });

    const mediaLabel = isVideo ? 'video' : 'media';
    const descriptionText = analysis?.description || (isVideo ? 'Video file uploaded.' : 'Media received.');
    const confirmationMessage = await generateContextualReply(
      'user uploaded media with no text, describe what you see and ask what they want to do with it',
      { mediaType: mediaLabel, description: descriptionText, capabilities: ['create social posts', 'use directly', 'regenerate similar visuals', 'extract text/ideas', 'analyze brand consistency'] }
    ) || `Here's what I see in your ${mediaLabel}: *${descriptionText}*\n\nWhat would you like to do with it?`;

    interrupt({
      reason: 'content_request',
      options: confirmationOptions,
      message: confirmationMessage,
      buttons: dynamicButtons,
      mediaAnalysis: analysis || null,
    });

    return {
      context: {
        ...ctx,
        selectedAction: 'create_post',
        mediaAnalysis: analysis,
        contentTopic: buildContentTopicFromMedia(analysis),
        activePlatforms: connectedPlatforms,
      },
      callAgents: [],
      awaitingConfirmation: true,
      confirmationOptions,
      confirmationReason: 'content_request',
      finalReply: confirmationMessage,
      buttons: dynamicButtons,
      intentClarity: 'clear',
      mediaAnalysis: analysis,
    };
  }

  // ── LLM routing -- always runs for text messages ────────────────────────────
  // Pass full conversation history (state.userPrompt) + compact session state summary
  // so the LLM can make context-aware decisions for every message.
  const stateSummary = buildStateSummary(state);
  const r = await runOrchestrator(state.userPrompt || rawPrompt, stateSummary, { signal: config?.signal });

  // Guard: LLM sometimes returns empty call_agents for topic-switch requests,
  // even when it also sets a direct_reply. Force content regeneration and clear
  // the stale direct_reply so the workflow doesn't exit early without generating.
  if (
    r.context?.topicSwitched === true &&
    r.context?.newTopic &&
    (!Array.isArray(r.call_agents) || r.call_agents.length === 0)
  ) {
    const textOnly = r.context?.contentType === 'text_only' ||
      /text\s*only|no\s*(image|video|media)/i.test(rawPrompt || '');
    r.call_agents = textOnly ? ['content_genius'] : ['content_genius', 'media_wizard'];
    r.direct_reply = null; // don't exit early — agents must run to generate new content
    console.log('[Workflow] topic-switch guard: forced call_agents =', r.call_agents);
  }

  const fromLlm = (r.context && typeof r.context === 'object') ? r.context : {};

  // Only apply non-empty LLM context values so that a direct_reply (like "Got it, adding Instagram")
  // returning empty arrays / nulls does NOT wipe out valid saved state (platforms, contentTopic, etc.).
  const cleanFromLlm = {};
  const isPlatformChange = String(fromLlm.followUpIntent || '') === 'platform_change';
  for (const [k, v] of Object.entries(fromLlm)) {
    // For platform_change, empty platforms [] is intentional ("user wants to change but didn't say which").
    // Don't skip it — let it override old platforms so the workflow asks which platform.
    if (Array.isArray(v) && v.length === 0 && !(k === 'platforms' && isPlatformChange)) continue;
    if (v === null || v === undefined || v === '') continue;
    cleanFromLlm[k] = v;
  }

  // If the router LLM set contentType to its default "image" but the user never
  // mentioned image/video/text-only, treat that as "unknown" so we can ask for format.
  const inferredFromUser = parseContentTypeFromText(rawPrompt || '');
  if (
    cleanFromLlm.contentType === 'image' &&
    inferredFromUser.contentType === null &&
    !state.mediaInput
  ) {
    delete cleanFromLlm.contentType;
  }

  const context = { ...baseContext, ...cleanFromLlm, ...(incomingCtx.brandKit ? { brandKit: incomingCtx.brandKit } : {}) };
  context.job = { ...job, ...((context.job && typeof context.job === 'object') ? context.job : {}) };

  // Guard: when adding platforms, ensure LLM's platforms list includes existing ones.
  // LLM sometimes only returns the new platform and forgets the existing ones.
  const existingPlatforms = normalizePlatformList(baseContext.platforms || []);
  const llmPlatforms = normalizePlatformList(context.platforms || []);
  if (
    existingPlatforms.length > 0 &&
    llmPlatforms.length > 0 &&
    /\b(add|also|too|as well|include)\b/i.test(String(rawPrompt || ''))
  ) {
    const merged = [...new Set([...existingPlatforms, ...llmPlatforms])];
    if (merged.length > llmPlatforms.length) {
      console.log('[Orchestrator] Platform add guard: merged', llmPlatforms, '+', existingPlatforms, '→', merged);
      context.platforms = merged;
    }
  }

  // ── Follow-up robustness: apply router's followUpIntent/topic switching ─────
  // Router LLM is the single classifier for copy_edit vs new_topic. The workflow
  // must apply that decision by clearing stale state so we don't merge old-topic
  // outputs into new-topic outputs.
  const followUpIntent = String(context?.followUpIntent || 'unknown');
  const regenerate = String(context?.regenerate || 'none');
  const mentionedPlatforms = parsePlatformsFromText(rawPrompt || '');
  const looksLikePlatformChange =
    mentionedPlatforms.length > 0 && /\b(add|remove|only|exclude|switch|change)\b/i.test(String(rawPrompt || ''));
  const looksLikeMediaOnly = isMediaOnlyInstruction(rawPrompt || '');
  const topicSwitchedFromRouter =
    context?.topicSwitched === true || followUpIntent === 'new_topic';
  const topicSwitched = topicSwitchedFromRouter && !looksLikePlatformChange && !looksLikeMediaOnly;

  // Ensure newTopic is reflected in contentTopic for downstream agents.
  if (topicSwitched) {
    const newTopic = (context?.newTopic != null ? String(context.newTopic) : '').trim();
    if (newTopic) context.contentTopic = newTopic;
    // Make sure content_genius/media_wizard get the actual user prompt for the new topic.
    // (Do not rely on old contentResult as a prompt source.)
    if (rawPrompt && rawPrompt.trim()) context.rawContent = rawPrompt.trim();
  }

  // When switching topics, clear prior results so executeAgentsNode does not merge.
  // - Always clear content + preview
  // - Clear media only when router requested it (content_and_media / media_only)
  const shouldClearForTopicSwitch = topicSwitched;
  const shouldClearMediaForTopicSwitch =
    shouldClearForTopicSwitch &&
    // For platform add/remove, keep existing media so previews can render for the
    // newly selected platforms.
    !looksLikePlatformChange && !looksLikeMediaOnly &&
    (regenerate === 'content_and_media' || regenerate === 'media_only');

  const topicSwitchResetPayload = shouldClearForTopicSwitch
    ? {
        contentResult: null,
        previewResult: null,
        schedulerResult: null,
        ...(shouldClearMediaForTopicSwitch ? { mediaResult: null } : {}),
      }
    : {};

  let mediaAnalysis = state.mediaAnalysis;
  const hasMedia = !!(state.mediaInput);
  if (hasMedia && !mediaAnalysis) {
    console.log('[Orchestrator] Media detected -- running vision analysis...');
    mediaAnalysis = await analyzeMediaWithVision(state.mediaInput, rawPrompt);
  }

  let hasAgentsToRun = Array.isArray(r.call_agents) && r.call_agents.length > 0;

  // FIX: Platform change — when LLM returns no agents but intent is platform_change
  // and content already exists, auto-route to content_genius for new platforms
  // or preview_pro for platform removal.
  if (followUpIntent === 'platform_change' && !hasAgentsToRun && state.contentResult) {
    const updatedPlatforms = normalizePlatformList(context.platforms || []);
    if (updatedPlatforms.length > 0) {
      const existingTypes = new Set(
        (state.contentResult?.platforms || []).map(p => normalizePlatformId(p?.type))
      );
      const hasNewPlatforms = updatedPlatforms.some(p => !existingTypes.has(p));

      if (hasNewPlatforms) {
        // Adding new platform(s) — need content_genius to generate content for them
        console.log('[Orchestrator] platform_change with new platforms detected, auto-routing to content_genius');
        r.call_agents = ['content_genius'];
      } else {
        // Platform removal only — just refresh previews with the filtered list
        const existingPlatforms = state.contentResult?.platforms || [];
        const filteredContentPlatforms = existingPlatforms.filter(p =>
          updatedPlatforms.includes(normalizePlatformId(p?.type))
        );
        const filteredContentResult = filteredContentPlatforms.length > 0
          ? { ...state.contentResult, platforms: filteredContentPlatforms }
          : state.contentResult;

        return {
          callAgents: ['preview_pro'],
          context: {
            ...context,
            prefaceReply: r.direct_reply,
            activePlatforms: updatedPlatforms,
          },
          contentResult: filteredContentResult,
          subTasks: ['Refresh previews after platform removal'],
          directReply: null,
          awaitingConfirmation: false,
          confirmationReason: null,
          guidedFlowStep: null,
          intentClarity: 'clear',
          mediaAnalysis,
        };
      }
    }
  }

  // Refresh after potential auto-routing above
  hasAgentsToRun = Array.isArray(r.call_agents) && r.call_agents.length > 0;

  if (r.direct_reply && !hasMedia && !hasAgentsToRun) {
    // ── YouTube confirmation: use interrupt() so the graph properly suspends ──
    if (context.confirmationNeeded === 'youtube_video_only') {
      const youtubeConfirmOptions = [
        { action: 'confirm_youtube_video', label: 'Yes, generate video for YouTube', icon: 'video' },
        { action: 'select_platform', label: 'No, choose a different platform', icon: 'switch' },
      ];
      interrupt({
        reason: 'youtube_image_request',
        options: youtubeConfirmOptions,
        message: r.direct_reply,
        buttons: [],
        pendingContext: {
          platforms: context.platforms,
          contentTopic: context.contentTopic || context.job?.topic,
          confirmationReason: 'youtube_image_request',
          ...formatFieldsForPendingContext(context),
        },
      });
      return {
        callAgents: [], subTasks: r.sub_tasks,
        context, mediaAnalysis, intentClarity: 'clear',
        awaitingConfirmation: true, confirmationReason: 'youtube_image_request',
        confirmationOptions: youtubeConfirmOptions,
        finalReply: r.direct_reply || "⚠️ YouTube doesn't support image generation, but I can generate a video for you instead. Would you like a video for YouTube, or prefer a different platform?",
        pendingCallAgents: ['content_genius', 'media_wizard', 'preview_pro'],
        guidedFlowStep: null, userSelection: null,
      };
    }
    if (context.confirmationNeeded === 'youtube_format_multiplatform') {
      const youtubeFormatOptions = [
        { action: 'youtube_format_regular', label: 'Regular Video (16:9 landscape)', icon: 'video' },
        { action: 'youtube_format_short', label: 'YouTube Short (9:16 vertical)', icon: 'short' },
        { action: 'youtube_format_both', label: 'Both formats', icon: 'both' },
      ];
      interrupt({
        reason: 'youtube_format',
        options: youtubeFormatOptions,
        message: r.direct_reply,
        buttons: [],
        pendingContext: {
          platforms: context.platforms,
          contentTopic: context.contentTopic || context.job?.topic,
          confirmationReason: 'youtube_format',
          ...formatFieldsForPendingContext(context),
        },
      });
      return {
        callAgents: [], subTasks: r.sub_tasks,
        context, mediaAnalysis, intentClarity: 'clear',
        awaitingConfirmation: true, confirmationReason: 'youtube_format',
        confirmationOptions: youtubeFormatOptions,
        finalReply: r.direct_reply || "I see you've selected YouTube along with other platforms. For YouTube, I need to know the video format:\n\n**Regular Video** (landscape 16:9) — for longer content with title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.\n\nImages will be generated for your other platforms automatically.",
        pendingCallAgents: ['content_genius', 'media_wizard', 'preview_pro'],
        guidedFlowStep: null, userSelection: null,
      };
    }

    // LLM decided this is a direct reply with no agents to run.
    // Return only the reply — never pass through old content/media/preview.
    // Old results stay in graph state for when agents actually need them later.
    return {
      directReply: r.direct_reply, callAgents: [], subTasks: r.sub_tasks,
      context, mediaAnalysis, intentClarity: 'clear', awaitingConfirmation: false,
      guidedFlowStep: null, confirmationReason: null,
    };
  }
  // If the router returned both a direct_reply and agents to run, keep the message
  // but DO NOT end the graph early (directReply triggers __end__ routing).
  if (r.direct_reply && !hasMedia && hasAgentsToRun) {
    context.prefaceReply = String(r.direct_reply);
  } else {
    // Prevent stale prefaceReply (e.g. "I'll remove the image...") from leaking
    // into later turns like "add image" or "modify the content".
    delete context.prefaceReply;
  }

  // FORCE video context if the uploaded media is a video
  const isVideoInput = hasMedia && state.mediaInput?.mimeType?.startsWith('video/');
  if (isVideoInput) {
    context.contentType = 'video';
  }

  const clarity = detectIntentClarity(rawPrompt, r);
  console.log('[Orchestrator] Intent clarity:', clarity, '| hasMedia:', hasMedia, '| isVideoInput:', isVideoInput);

  const connectedPlatforms = context?.userId
    ? (await getActivePlatforms(context.userId, context.workspaceId) || [])
    : [];

  // Fetch linked account holder/page names for confirmation UI (optional).
  // This is separate from platform IDs and lets the frontend show "Linked accounts".
  const activeAccounts = context?.userId
    ? (await getActiveAccounts(context.userId, context.workspaceId) || [])
    : [];

  // Media uploaded with text → auto create_post, ask for platforms
  if (hasMedia) {
    console.log('[Orchestrator] Media with text -- skipping options, going to platform question');

    if (connectedPlatforms.length === 0) {
      const noPlatformsReply = await generateContextualReply('user has no social media accounts connected, prompt them to connect a platform in settings');
      return {
        callAgents: [], directReply: null, guidedFlowStep: null,
        awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(),
        confirmationReason: 'no_platforms',
        finalReply: noPlatformsReply || 'No social media accounts are connected yet. Connect a platform in Settings to get started.',
        buttons: [], mediaAnalysis,
      };
    }

    const isVideo = state.mediaInput.mimeType?.startsWith('video/');
    console.log(`[Orchestrator] Auto-selecting create_post -> triggering confirmation for platforms (${isVideo ? 'video' : 'image'}):`, connectedPlatforms);

    const confirmationOptions = buildConfirmationOptions(true, r, mediaAnalysis, {
      ...context,
      activePlatforms: connectedPlatforms,
      activeAccounts,
      contentType: context?.contentType || (isVideo ? 'video' : 'image'),
    });

    const dynamicButtons = [];
    if (connectedPlatforms.length > 1) {
      dynamicButtons.push({
        label: 'Quick: All platforms (images + content)',
        action: 'quick_create_post_all',
        platforms: connectedPlatforms,
      });
    }
    connectedPlatforms.forEach((p) => {
      dynamicButtons.push({
        label: `Quick: ${PLATFORM_LABELS[p] || p}`,
        action: 'quick_create_post_single',
        platforms: [p],
      });
    });

    const isVideo2 = state.mediaInput?.mimeType?.startsWith('video/');
    const mediaLabel2 = isVideo2 ? 'video' : 'media';
    const descriptionText2 = mediaAnalysis?.description || (isVideo2 ? 'Video file uploaded.' : 'Media received.');
    const confirmationMessage = await generateContextualReply(
      'user uploaded media with text, describe what you see and ask what they want to do with it',
      { mediaType: mediaLabel2, description: descriptionText2, userText: rawPrompt }
    ) || `Here's what I see in your ${mediaLabel2}: *${descriptionText2}*\n\nWhat would you like to do with it?`;

    interrupt({
      reason: 'content_request',
      options: confirmationOptions,
      message: confirmationMessage,
      buttons: dynamicButtons,
      mediaAnalysis: mediaAnalysis || null,
    });

    return {
      context: {
        ...context,
        selectedAction: 'create_post',
        mediaAnalysis,
        contentTopic: rawPrompt
          ? `${rawPrompt}. Image context: ${mediaAnalysis?.description || ''}`
          : buildContentTopicFromMedia(mediaAnalysis),
        activePlatforms: connectedPlatforms,
        activeAccounts,
      },
      callAgents: [],
      awaitingConfirmation: true,
      confirmationOptions,
      confirmationReason: 'content_request',
      finalReply: confirmationMessage,
      buttons: dynamicButtons,
      intentClarity: 'clear',
      mediaAnalysis,
    };
  }

  // Clear intent for content/media
  const wantsContentOrMedia = r.call_agents?.some(a => ['content_genius', 'media_wizard'].includes(a));
  if (clarity === 'clear' && wantsContentOrMedia) {
    // If no social accounts are connected yet, ask the user to connect one first.
    if (connectedPlatforms.length === 0) {
      const noPlatformsReply = await generateContextualReply('user has no social media accounts connected, prompt them to connect a platform in settings');
      return {
        callAgents: [], directReply: null, guidedFlowStep: null,
        awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(),
        confirmationReason: 'no_platforms',
        finalReply: noPlatformsReply || 'No social media accounts are connected yet. Connect a platform in Settings to get started.',
        buttons: [],
      };
    }

    const hasUserPlatforms = Array.isArray(context.platforms) && context.platforms.length > 0;
    const hasTopic = !!(context.contentTopic || context.job?.topic);
    const { contentType: inferredContentType } = parseContentTypeFromText(rawPrompt);
    // Format is "known" if: user said it explicitly, LLM set it, OR content already exists (previous generation used a format)
    // Format is "known" if: user said it explicitly, OR this is a follow-up (platform change/copy edit)
    // where content already exists with a format. For new requests, always ask.
    const isFollowUp = followUpIntent === 'platform_change' || followUpIntent === 'copy_edit';
    const hasExplicitFormat = !!context.contentType || !!inferredContentType || (isFollowUp && !!state.contentResult);

    // ── Case A: user already specified platforms (e.g. "for facebook") ─────────
    if (hasUserPlatforms) {
      const normalizedConnected = normalizePlatformList(connectedPlatforms || []);
      const requestedPlatforms = Array.isArray(context.platforms) ? context.platforms : [];
      // youtube_short maps to 'youtube' for connection checks — it's the same platform
      const isConnected = (p) => normalizedConnected.includes(p) || (p === 'youtube_short' && normalizedConnected.includes('youtube'));
      const notLinked = requestedPlatforms.filter(p => !isConnected(p));
      // Filter out unconnected platforms — only proceed with connected ones
      const effectivePlatforms = requestedPlatforms.filter(p => isConnected(p));
      const notLinkedNames = notLinked.map(p => PLATFORM_LABELS[p] || p).join(', ');
      const notLinkedNote = notLinked.length > 0
        ? `\n\nNote: ${notLinkedNames} ${notLinked.length === 1 ? "isn't" : "aren't"} linked to this workspace yet. Please connect ${notLinked.length === 1 ? 'it' : 'them'} in your settings before posting.`
        : '';

      // If ANY requested platform is not connected, inform the user and re-show platform selection
      if (notLinked.length > 0) {
        const connectedNames = normalizedConnected.map(p => PLATFORM_LABELS[p] || p).join(', ');
        const replyMsg = `${notLinkedNames} ${notLinked.length === 1 ? "isn't" : "aren't"} connected to your account. Your connected platforms are: ${connectedNames}. Please select from your connected platforms.`;
        console.log('[Orchestrator] Requested platforms not connected:', notLinked, '| connected:', normalizedConnected);
        const confirmationOptions = buildConfirmationOptions(false, { call_agents: r.call_agents || [] }, null, {
          ...context,
          activePlatforms: normalizedConnected,
        });
        return {
          callAgents: [],
          context: { ...context, platforms: [], activePlatforms: normalizedConnected, activeAccounts },
          pendingCallAgents: r.call_agents || [],
          awaitingConfirmation: true,
          confirmationOptions,
          confirmationReason: 'content_request',
          finalReply: replyMsg,
          buttons: [],
          intentClarity: 'clear',
        };
      }

      // If platforms are set but topic is missing, ask only for the missing topic/format (preferred options UI).
      if (!hasTopic) {
        const names = effectivePlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
        const nextCtx = {
          ...context,
          platforms: effectivePlatforms,
          activePlatforms: effectivePlatforms,
          activeAccounts,
        };
        const missing = ['topic'];
        if (!hasExplicitFormat) missing.push('format');
        const confirmationMessage = (await generateContextualReply(
          'platforms are set but topic is missing, ask only for the topic to proceed',
          { platforms: names, notLinkedNote }
        ) || `Posting to ${names}. What should the post be about?`) + notLinkedNote;
        const confirmationOptions = buildPreferredOptions(missing, nextCtx);
        interrupt({
          reason: 'missing_topic',
          options: confirmationOptions,
          message: confirmationMessage,
          buttons: [],
          mediaAnalysis: mediaAnalysis || null,
          pendingContext: {
            platforms: effectivePlatforms,
            confirmationReason: 'missing_topic',
            rawUserPrompt: rawPrompt || '',
            ...formatFieldsForPendingContext(context),
          },
        });
        return {
          ...topicSwitchResetPayload,
          context: nextCtx,
          pendingCallAgents: r.call_agents || [],
          callAgents: [],
          awaitingConfirmation: true,
          confirmationOptions,
          confirmationReason: 'missing_topic',
          finalReply: confirmationMessage,
          buttons: [],
          intentClarity: 'clear',
          mediaAnalysis,
        };
      }

      // If we have topic + platforms but NO explicit format (image/video/text-only),
      // interrupt with ONLY the missing format options (so API returns confirmationOptions).
      if (hasTopic && effectivePlatforms.length > 0 && !hasExplicitFormat) {
        const names = effectivePlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
        const nextCtx = {
          ...context,
          platforms: effectivePlatforms,
          activePlatforms: effectivePlatforms,
          activeAccounts,
        };
        const confirmationMessage = (await generateContextualReply(
          'topic and platforms are set but content format is missing, ask only for format: text only, images + content, or video',
          { platforms: names, topic: context.contentTopic }
        ) || `Posting to ${names} about "${context.contentTopic || 'this topic'}". What format do you want?`) + notLinkedNote;

        const confirmationOptions = buildPreferredOptions(['format'], nextCtx);

        interrupt({
          reason: 'missing_format',
          options: confirmationOptions,
          message: confirmationMessage,
          buttons: [],
          mediaAnalysis: mediaAnalysis || null,
          pendingContext: {
            platforms: effectivePlatforms,
            contentTopic: context.contentTopic || context.job?.topic,
            confirmationReason: 'missing_format',
            rawUserPrompt: rawPrompt || '',
            ...formatFieldsForPendingContext(context),
          },
        });

        return {
          ...topicSwitchResetPayload,
          context: nextCtx,
          pendingCallAgents: r.call_agents || [],
          callAgents: [],
          awaitingConfirmation: true,
          confirmationOptions,
          confirmationReason: 'missing_format',
          finalReply: confirmationMessage,
          buttons: [],
          intentClarity: 'clear',
          mediaAnalysis,
        };
      }

      const nextContext = {
        ...context,
        platforms: effectivePlatforms,
        activePlatforms: effectivePlatforms,
        activeAccounts,
        job: {
          ...context.job,
          platforms: effectivePlatforms,
          selectedAction: 'create_post',
        },
      };

      // If we already have media, do NOT regenerate it unless user explicitly asked.
      // This keeps existing images stable across copy edits / platform adds.
      const hasExistingMedia = !!(state.mediaResult && !state.mediaResult.error);
      const explicitMediaIntent = parseExplicitMediaIntent(rawPrompt) || parseContentActionFromText(rawPrompt) === 'generate_media';
      // If user explicitly asked to change/generate media and the instruction is media-only,
      // do not re-run content_genius (preserve existing copy).
      const wantsMediaOnlyEdit = !!explicitMediaIntent && isMediaOnlyInstruction(rawPrompt || '');
      // If the user explicitly asked to "add image/video" during a copy edit, ensure we don't
      // remain stuck in text-only mode from a previous selection.
      if (explicitMediaIntent) {
        const ct = String(nextContext.contentType || context?.contentType || '').toLowerCase();
        if (ct === 'text_only' || ct === 'text only' || ct === 'text') nextContext.contentType = 'image';
        nextContext.includeMedia = true;
        if (nextContext.job && typeof nextContext.job === 'object') nextContext.job.includeMedia = true;
      }
      // FIX: For platform_change, never re-run media_wizard unless user explicitly
      // requested new images. The LLM sometimes includes media_wizard by mistake
      // (misclassifying platform switch as a new content request via Rule 5).
      let filteredAgents = !explicitMediaIntent && (hasExistingMedia || followUpIntent === 'platform_change')
        ? (r.call_agents || []).filter(a => a !== 'media_wizard')
        : (r.call_agents || []);

      if (wantsMediaOnlyEdit) {
        filteredAgents = filteredAgents.filter(a => a !== 'content_genius');
        if (!filteredAgents.includes('media_wizard')) filteredAgents.push('media_wizard');
      }
      // FIX: If user explicitly requested images/media (e.g. "with image", "generate images"),
      // ensure media_wizard is included even if the LLM omitted it (common for platform_change
      // where Rule 6 only returns content_genius but user also wants images).
      if (explicitMediaIntent && !filteredAgents.includes('media_wizard')) {
        filteredAgents.push('media_wizard');
        console.log('[Orchestrator] explicitMediaIntent detected -- adding media_wizard to agents');
      }
      // FIX: When regenerate is "content_and_media" and content type includes media,
      // ensure media_wizard is included so images/videos get regenerated along with content.
      // Without this, media gets cleared by topicSwitchResetPayload but never regenerated.
      const ctxContentTypeForMedia = String(context?.contentType || '').toLowerCase();
      if (
        regenerate === 'content_and_media' &&
        !filteredAgents.includes('media_wizard') &&
        ctxContentTypeForMedia !== 'text' &&
        ctxContentTypeForMedia !== 'text only' &&
        ctxContentTypeForMedia !== 'text_only'
      ) {
        filteredAgents.push('media_wizard');
        console.log('[Orchestrator] content_and_media regenerate with media contentType -- adding media_wizard to agents');
      }
      // Always run preview_pro when generating content/media so the UI can show previews (incl. images) consistently.
      if (filteredAgents.some(a => a === 'content_genius' || a === 'media_wizard') && !filteredAgents.includes('preview_pro')) {
        filteredAgents = [...filteredAgents, 'preview_pro'];
      }

      console.log('[Orchestrator] Clear content intent detected -- routing directly to agents:', r.call_agents, '| platforms:', effectivePlatforms);

      return {
        ...topicSwitchResetPayload,
        callAgents: filteredAgents,
        context: nextContext,
        subTasks: r.sub_tasks || [],
        directReply: null,
        awaitingConfirmation: false,
        confirmationReason: null,
        guidedFlowStep: null,
        intentClarity: 'clear',
        mediaAnalysis,
      };
    }

    // ── Case B: platforms not chosen yet → interrupt with ONLY missing fields ────
    // Use connected platforms as AVAILABLE options, but do not assume they are selected.
    const preferredPlatforms = connectedPlatforms;
    const names = preferredPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');

    const nextCtx = {
      ...context,
      platforms: [], // not yet selected
      activePlatforms: preferredPlatforms,
      activeAccounts,
      contentType: context?.contentType || context?.job?.contentType || 'image',
    };

    const missing = [];
    if (!hasUserPlatforms) missing.push('platforms');
    if (!hasTopic) missing.push('topic');
    if (hasTopic && !hasExplicitFormat) missing.push('format');

    // If the router asked for scheduling but did not provide a time, ask schedule options.
    const wantsSchedule = Array.isArray(r.call_agents) && r.call_agents.includes('scheduler_boss');
    const hasScheduleTime = !!(nextCtx.scheduleAt || nextCtx.rawScheduleRequest);
    if (wantsSchedule && !hasScheduleTime) missing.push('schedule_time');

    const confirmationOptions = buildPreferredOptions(missing, nextCtx);

    const dynamicButtons = [];
    if (preferredPlatforms.length > 1) {
      dynamicButtons.push({
        label: 'Quick: All platforms (images + content)',
        action: 'quick_create_post_all',
        platforms: preferredPlatforms,
      });
    }
    preferredPlatforms.forEach((p) => {
      dynamicButtons.push({
        label: `Quick: ${PLATFORM_LABELS[p] || p}`,
        action: 'quick_create_post_single',
        platforms: [p],
      });
    });

    const confirmationMessage = await generateContextualReply(
      hasTopic
        ? 'platforms missing but topic is known, ask user to select platforms to post on'
        : 'both platforms and topic are missing, ask user for the missing details',
      { topic: context.contentTopic, availablePlatforms: names }
    ) || (hasTopic
      ? `Got it! Which platforms should I post on? Available: ${names}`
      : 'Please select the platforms and topic to continue.');

    interrupt({
      reason: 'content_request',
      options: confirmationOptions,
      message: confirmationMessage,
      buttons: dynamicButtons,
      mediaAnalysis: mediaAnalysis || null,
      pendingContext: {
        contentTopic: context.contentTopic || context.job?.topic,
        confirmationReason: 'content_request',
        rawUserPrompt: rawPrompt || '',
        ...formatFieldsForPendingContext(context),
      },
    });

    return {
      ...topicSwitchResetPayload,
      context: {
        ...nextCtx,
        activePlatforms: preferredPlatforms,
        job: {
          ...context.job,
          platforms: preferredPlatforms,
          selectedAction: 'create_post',
          ...formatFieldsForPendingContext({ ...context, ...nextCtx }),
        },
      },
      callAgents: [],
      awaitingConfirmation: true,
      confirmationOptions,
      confirmationReason: 'content_request',
      finalReply: confirmationMessage,
      buttons: dynamicButtons,
      intentClarity: 'clear',
      mediaAnalysis,
    };
  }

  // Non-content agents (scheduler_boss, __account_info__, etc.) -- route directly.
  // These have no platform-selection UI step so they go straight to executeAgentsNode.
  const hasNonContentAgents = Array.isArray(r.call_agents) && r.call_agents.length > 0;
  if (hasNonContentAgents) {
    console.log('[Orchestrator] Non-content agents requested:', r.call_agents, '-- routing directly');
    return {
      ...topicSwitchResetPayload,
      callAgents: r.call_agents,
      context,
      subTasks: r.sub_tasks || [],
      directReply: null,
      awaitingConfirmation: false,
      confirmationReason: null,
      guidedFlowStep: null,
      intentClarity: 'clear',
      mediaAnalysis,
    };
  }

  // Truly ambiguous / off-topic -- LLM generates a contextual reply
  console.log('[Orchestrator] Ambiguous -- replying directly');
  const ambiguousReply = await generateContextualReply(
    'user sent an off-topic or unclear message, redirect them to social media content creation',
    { userMessage: rawPrompt }
  ) || "I can help you create posts, generate images or videos, and schedule content to your connected platforms. What would you like to create?";
  return {
    directReply: ambiguousReply,
    callAgents: [], awaitingConfirmation: false, confirmationOptions: [], confirmationReason: null,
    mediaAnalysis, intentClarity: 'ambiguous', guidedFlowStep: null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// EXECUTE AGENTS NODE
// ─────────────────────────────────────────────────────────────────────────────

const HAS_SCHEDULING_ONLY_HINT = /\[We already have generated content and media from the previous turn\. User is only specifying when to schedule\.\]/i;

async function executeAgentsNode(state, config) {
  const signal = config?.signal;
  const { userPrompt, callAgents, context } = state;

  // analyze_brand
  if (context?.selectedAction === 'analyze_brand') {
    const analysis = state.mediaAnalysis || context?.mediaAnalysis;
    const replyText = analysis
      ? await generateContextualReply('present brand analysis results to user in a clear, actionable way', {
          description: analysis.description,
          brandMatch: analysis.brandMatch,
          detectedObjects: analysis.detectedObjects,
          extractedText: analysis.extractedText,
        }) || `**Brand Analysis**\n\n${analysis.description}\n\nBrand consistency: ${analysis.brandMatch ? 'Looks on-brand.' : 'May need review.'}\nDetected elements: ${(analysis.detectedObjects || []).join(', ') || 'None'}`
      : await generateContextualReply('brand analysis failed because no media was available') || 'Unable to analyze brand consistency from the provided media.';
    return {
      directReply: replyText, finalReply: replyText, buttons: [],
      contentResult: null, mediaResult: null, previewResult: null, schedulerResult: null,
      knowledgeChunks: [], schedulingOnlyReply: false, mediaInput: null,
      awaitingConfirmation: false, userSelection: null, guidedFlowStep: null,
    };
  }

  // __account_info__
  if ((callAgents || []).includes('__account_info__')) {
    const dbPlatforms = await getActivePlatforms(context?.userId, context?.workspaceId);
    const platforms = normalizePlatformList(dbPlatforms || []);
    const replyText = await generateContextualReply(
      platforms.length > 0
        ? 'inform user about their connected social accounts and offer to create content'
        : 'user has no connected social accounts, prompt them to connect one',
      { connectedPlatforms: platforms.map(p => PLATFORM_LABELS[p] || p), count: platforms.length }
    ) || (platforms.length > 0
      ? `You have ${platforms.length} account(s) connected: ${platforms.map(p => PLATFORM_LABELS[p] || p).join(', ')}. Would you like to create content?`
      : 'No social media accounts are connected yet. Connect one in Settings to get started.');
    return {
      finalReply: replyText, directReply: replyText, buttons: [], callAgents: [],
      contentResult: null, mediaResult: null, previewResult: null, schedulerResult: null,
      knowledgeChunks: [], schedulingOnlyReply: false,
    };
  }

  const wantsContent = (callAgents || []).includes('content_genius');
  // Use the clean user message for knowledge search — NOT the full conversation
  // history string that state.userPrompt may contain.
  // Fallback chain: rawUserPrompt → restored rawUserPrompt from pendingContext → contentTopic → job.topic
  const job0 = (context?.job && typeof context.job === 'object') ? context.job : {};
  const rawMsg = (state.rawUserPrompt || '').trim();
  const knowledgeQuery = rawMsg
    || (context?.rawUserPrompt || '').trim()
    || context?.contentTopic
    || job0?.topic
    || '';
  // When user explicitly references their knowledge base / uploaded data, fetch more chunks
  const userReferencesKB = context?.useKnowledgeBase === true;
  const kbTopK = userReferencesKB ? 20 : (wantsContent ? 15 : 5);
  console.log('[Workflow] KnowledgeKeeper query:', knowledgeQuery?.slice(0, 120) || '(empty)', '| useKnowledgeBase:', userReferencesKB, '| topK:', kbTopK);
  if (signal?.aborted) return { finalReply: '', cancelled: true };
  const knowledge = await runKnowledgeKeeper(knowledgeQuery, {
    topK: kbTopK,
    companyId: context?.companyId,
    workspaceId: context?.workspaceId,
  });
  const knowledgeChunks = knowledge.chunks || [];
  console.log('[Workflow] KnowledgeKeeper chunks:', {
    count: knowledgeChunks.length,
    docScoped: knowledge.raw?.docScoped || false,
    sample: knowledgeChunks[0] ? String(knowledgeChunks[0]).slice(0, 160) : null,
  });

  // Platform resolution
  let activePlatforms = state.context?.platforms;
  console.log('[Workflow] executeAgentsNode | context IDs:', { userId: context?.userId, workspaceId: context?.workspaceId });

  if (context?.userId) {
    const dbPlatforms = await getActivePlatforms(context.userId, context.workspaceId);
    console.log('[Workflow] getActivePlatforms result:', dbPlatforms);
    const actualPlatforms = normalizePlatformList(dbPlatforms || []);
    activePlatforms = normalizePlatformList(activePlatforms || []);
    if (!activePlatforms || activePlatforms.length === 0 || activePlatforms.length === 6) {
      activePlatforms = actualPlatforms;
    } else {
      const requestedPlatforms = [...activePlatforms]; // preserve what user asked for
      // youtube_short is not a separate DB platform — it maps to 'youtube' for connection checks
      const matchesConnected = (p) => actualPlatforms.includes(p) || (p === 'youtube_short' && actualPlatforms.includes('youtube'));
      const intersection = activePlatforms.filter(matchesConnected);
      const notConnected = requestedPlatforms.filter(p => !matchesConnected(p));

      if (notConnected.length > 0) {
        // Some or all requested platforms are not connected — inform the user and re-show platform selection
        const notConnectedNames = notConnected.map(p => PLATFORM_LABELS[p] || p).join(', ');
        const connectedNames = actualPlatforms.map(p => PLATFORM_LABELS[p] || p).join(', ');
        const notConnectedMsg = intersection.length > 0
          ? `${notConnectedNames} is not connected to your account. Your connected platforms are: ${connectedNames}. Would you like to proceed with the connected platforms, or choose different ones?`
          : `The platform(s) you requested (${notConnectedNames}) are not connected to your account. Your connected platforms are: ${connectedNames}. Please select from your connected platforms.`;
        console.log('[Workflow] Requested platforms not connected:', notConnected, '| connected:', actualPlatforms, '| available intersection:', intersection);

        const confirmationOptions = buildConfirmationOptions(false, { call_agents: callAgents || [] }, null, {
          ...context,
          activePlatforms: actualPlatforms,
        });

        return {
          knowledgeChunks,
          contentResult: state.contentResult || null, mediaResult: state.mediaResult || null,
          previewResult: null, schedulerResult: null,
          finalReply: notConnectedMsg,
          buttons: [],
          schedulingOnlyReply: false,
          awaitingConfirmation: true,
          confirmationOptions,
          confirmationReason: 'content_request',
          pendingCallAgents: callAgents || [],
          context: { ...context, platforms: [], activePlatforms: actualPlatforms },
        };
      }

      activePlatforms = intersection;
    }
    context.platforms = activePlatforms;
  }
  console.log('[Workflow] Final activePlatforms:', activePlatforms);

  if (activePlatforms && activePlatforms.length === 0) {
    const noPlatsMsg = await generateContextualReply(
      'user has no social media accounts connected, prompt them to connect one to start creating and posting content',
      {}
    ) || 'No social media accounts are connected yet. Which platform would you like to connect?';
    return {
      knowledgeChunks,
      contentResult: null, mediaResult: null, previewResult: null, schedulerResult: null,
      finalReply: noPlatsMsg,
      buttons: [], schedulingOnlyReply: false,
      awaitingConfirmation: true, confirmationOptions: buildPlatformConnectionOptions(), confirmationReason: 'no_platforms',
    };
  }

  const ctx = {
    knowledgeChunks,
    platforms: activePlatforms || [],
    contentType: context?.contentType || 'image',
  };

  // When the user switches platforms (e.g. "only Instagram"), we should not keep
  // rendering or previewing stale platforms from previous turns. Filter outputs
  // to the currently active platform set.
  const activeSet = new Set((ctx.platforms || []).map(p => normalizePlatformId(p)));
  const filterContentResultToActivePlatforms = (contentResult) => {
    if (!contentResult || !Array.isArray(contentResult.platforms) || contentResult.platforms.length === 0) return contentResult;
    const filtered = contentResult.platforms.filter(p => activeSet.has(normalizePlatformId(p?.type)));
    return { ...contentResult, platforms: filtered };
  };
  const filterMediaResultToActivePlatforms = (mediaResult) => {
    if (!mediaResult || !Array.isArray(mediaResult.outputs) || mediaResult.outputs.length === 0) return mediaResult;
    // Keep only platform-tagged outputs that match active platforms. If an output
    // is not platform-tagged, keep it (backwards compatibility with older media_wizard).
    const filteredOutputs = mediaResult.outputs.filter(o => {
      const op = normalizePlatformId(o?.platform);
      if (!op) return true;
      return activeSet.has(op);
    });
    const filteredWarnings = Array.isArray(mediaResult.warnings)
      ? mediaResult.warnings.filter(w => {
          const wp = normalizePlatformId(w?.platform);
          if (!wp) return true;
          return activeSet.has(wp);
        })
      : mediaResult.warnings;
    return { ...mediaResult, outputs: filteredOutputs, warnings: filteredWarnings };
  };

  let toRun = callAgents || [];

  // ── YouTube handling (executeAgentsNode) ───────────────────────────────────
  // IMPORTANT: Do NOT force a global contentType='video' just because YouTube is selected.
  // Media Wizard can generate images for non-YouTube platforms AND a YouTube video in the same run
  // when contentType is 'image' (it skips YouTube images by design).
  const hasContentAndMedia = !!(state.contentResult && state.mediaResult);
  const hasSchedulingOnlyHint = userPrompt && HAS_SCHEDULING_ONLY_HINT.test(String(userPrompt));
  const wantsPreviewNow =
    /\b(preview|previews|view previews|show preview|show previews|open preview)\b/i.test(String(userPrompt || '')) ||
    parseContentActionFromText?.(String(userPrompt || '')) === 'preview';
  // Optimization: if user is only giving a time, run scheduler only.
  // BUT: do not skip previews if the user explicitly asks to view them.
  if (hasContentAndMedia && hasSchedulingOnlyHint && toRun.includes('scheduler_boss') && !wantsPreviewNow) {
    toRun = ['scheduler_boss'];
  }

  // Ensure previews are generated whenever content/media is involved so the UI can
  // consistently show image/video previews.
  const schedulingOnly = toRun.length === 1 && toRun[0] === 'scheduler_boss';
  const wantsPreview =
    toRun.includes('content_genius') ||
    toRun.includes('media_wizard') ||
    // Copy edits / content-only reruns should still show existing media previews when available.
    (toRun.includes('content_genius') && !!(state.mediaResult && !state.mediaResult.error) && context?.includeMedia !== false);
  if (!schedulingOnly && wantsPreview && !toRun.includes('preview_pro')) {
    toRun = [...toRun, 'preview_pro'];
  }

  const job = (context?.job && typeof context.job === 'object') ? context.job : {};
  //const agentPrompt = context?.contentTopic || job?.topic || userPrompt;
  let agentPrompt = context?.contentTopic || job?.topic || userPrompt;

  // If we have a mediaAnalysis description (e.g., from a sampled video frame or image),
  // always prefer a prompt built directly from that description over any stale contentTopic
  // or generic fallback. This keeps generated copy aligned with what was actually analyzed.
  if (state.mediaAnalysis?.description || context?.mediaAnalysis?.description) {
    const analysis = state.mediaAnalysis || context.mediaAnalysis;
    agentPrompt = buildContentTopicFromMedia(analysis, agentPrompt || 'Create a post using the uploaded media');
  }

  let thisContentResult = state.contentResult;
  // Preserve existing media when we are not rerunning `media_wizard`.
  // This is critical for "add platform" and copy-edit follow-ups where the UI
  // expects preview images to update for the newly added platforms.
  let thisMediaResult = null;
  if (schedulingOnly) {
    thisMediaResult = state.mediaResult ?? null;
  } else if (!toRun.includes('media_wizard')) {
    // If user explicitly asked for text-only / no images, drop media.
    // Otherwise, reuse previously generated media so previews can still render
    // for newly added platforms.
    // FIX: Also respect the LLM's contentType classification (e.g. "text only" / "text_only")
    // so that even if the regex doesn't match the user's phrasing, the LLM's intent is honored.
    const ctxContentType = String(context?.contentType || '').replace(/[_\s]+/g, ' ').trim().toLowerCase();
    const textOnly = isTextOnlyRequest(userPrompt) || ctxContentType === 'text only';
    thisMediaResult = textOnly ? null : (state.mediaResult ?? null);
  }

  if (context?.selectedAction === 'use_as_media' && state.mediaInput) {
    const uploadedUrl = state.mediaInput.url || (state.mediaInput.buffer ? '[uploaded-buffer]' : null);
    thisMediaResult = { outputs: [{ type: state.mediaInput.mimeType?.startsWith('video') ? 'video' : 'image', urls: uploadedUrl ? [uploadedUrl] : [], source: 'user_upload' }] };
    console.log('[Workflow] use_as_media: using uploaded media directly');
  }

  if (toRun.includes('content_genius')) {
    console.log('[Workflow] content_genius agentPrompt:', agentPrompt?.slice(0, 80));
    const mediaAnalysisCtx = state.mediaAnalysis || context?.mediaAnalysis || null;

    const followUpIntent = String(context?.followUpIntent || 'unknown');
    const regenerate = String(context?.regenerate || 'none');
    const topicSwitched = context?.topicSwitched === true || followUpIntent === 'new_topic';

    // For copy_edit: pass existing content to content_genius so it edits rather than regenerates
    let contentGeniusCtx = {
      ...ctx,
      rawContent: context?.rawContent,
      brandKit: context?.brandKit,
      mediaAnalysis: mediaAnalysisCtx,
      originalUserPrompt: userPrompt,
    };
    // YouTube should ALWAYS generate title + description when selected as a platform.
    // Never disable YouTube metadata — title and description are essential for YouTube.
    // Map youtube → youtube_short in content-genius platforms based on youtubeConfig.videoType
    // so the LLM generates the right style (short-form vs long-form).
    // The workflow platforms keep 'youtube' for DB validation, but content-genius needs the
    // specific type to generate appropriate title/description/tags.
    if (context?.youtubeConfig?.videoType && Array.isArray(contentGeniusCtx.platforms)) {
      const ytVideoType = context.youtubeConfig.videoType;
      if (ytVideoType === 'short') {
        contentGeniusCtx.platforms = contentGeniusCtx.platforms.map(p => p === 'youtube' ? 'youtube_short' : p);
      } else if (ytVideoType === 'both') {
        // Generate content for both youtube and youtube_short
        const hasYT = contentGeniusCtx.platforms.includes('youtube');
        const hasYTS = contentGeniusCtx.platforms.includes('youtube_short');
        if (hasYT && !hasYTS) contentGeniusCtx.platforms.push('youtube_short');
        if (!hasYT) contentGeniusCtx.platforms.push('youtube');
      }
      // 'full' → keep 'youtube' as-is
    }
    if (followUpIntent === 'copy_edit' && state.contentResult?.platforms?.length) {
      // Only pass existing content for platforms the user wants to keep (context.platforms)
      const requestedPlatforms = normalizePlatformList(context?.platforms || []);
      const existingContentMap = {};
      for (const p of state.contentResult.platforms) {
        const pType = normalizePlatformId(p?.type);
        if (p?.type && p?.content && (requestedPlatforms.length === 0 || requestedPlatforms.includes(pType))) {
          existingContentMap[p.type] = p.content;
        }
      }
      contentGeniusCtx.existingContent = existingContentMap;
      // Include the edit instruction in the prompt so the LLM knows what to modify
      if (context?.editInstruction) {
        agentPrompt = `${agentPrompt}\n\nEdit instruction: ${context.editInstruction}`;
      }
    }

    // For platform_change: only generate content for NEW platforms that don't already
    // have content. This preserves consistency and avoids unnecessary LLM calls.
    if (followUpIntent === 'platform_change' && state.contentResult?.platforms?.length) {
      const existingTypes = new Set(
        (state.contentResult.platforms || []).map(p => normalizePlatformId(p?.type))
      );
      const allRequestedPlatforms = normalizePlatformList(contentGeniusCtx.platforms || []);
      const newPlatformsOnly = allRequestedPlatforms.filter(p => !existingTypes.has(p));
      if (newPlatformsOnly.length > 0) {
        contentGeniusCtx.platforms = newPlatformsOnly;
        console.log('[Workflow] content_genius -- platform_change: generating only for new platforms:', newPlatformsOnly, '| preserving existing:', [...existingTypes]);
      } else {
        console.log('[Workflow] content_genius -- platform_change: all platforms already have content, regenerating all:', allRequestedPlatforms);
      }
    }

    if (signal?.aborted) return { finalReply: '', cancelled: true };
    const freshResult = await runContentGenius(agentPrompt, contentGeniusCtx);

    // Order matters:
    // - New topic: always replace
    // - Platform change: preserve existing content and merge (even if media is also regenerating)
    // - Otherwise: respect regenerate mode
    if (topicSwitched) {
      // New topic → always replace (never merge old-topic platforms).
      thisContentResult = freshResult;
      console.log('[Workflow] content_genius -- topic switch → full replacement');
    } else if (followUpIntent === 'platform_change' && Array.isArray(context?.platforms) && context.platforms.length > 0) {
      // Platform-change follow-up: preserve existing content for platforms that already
      // had content, and only use fresh LLM output for truly NEW platforms.
      // Merge ALL existing platforms + requested platforms so nothing gets dropped.
      const requested = normalizePlatformList(context.platforms);
      const existingByType = new Map((state.contentResult?.platforms || []).map(p => [normalizePlatformId(p?.type), p]));
      const freshByType = new Map((freshResult?.platforms || []).map(p => [normalizePlatformId(p?.type), p]));
      // Union of existing + requested platforms (preserves order: existing first, then new)
      const allPlatforms = [...new Set([...existingByType.keys(), ...requested])];
      const mergedPlatforms = allPlatforms
        .map(type => existingByType.get(type) || freshByType.get(type))
        .filter(Boolean);
      thisContentResult = mergedPlatforms.length > 0
        ? { ...freshResult, platforms: mergedPlatforms }
        : freshResult;
      console.log('[Workflow] content_genius -- platform_change merged:', allPlatforms, '| existing preserved:', [...existingByType.keys()], '| fresh:', [...freshByType.keys()]);
    } else if (regenerate === 'content_and_media') {
      // When regenerating both content and media (but not a platform-change merge), replace content.
      thisContentResult = freshResult;
      console.log('[Workflow] content_genius -- content_and_media → full replacement');
    } else if (followUpIntent === 'copy_edit' && state.contentResult?.platforms?.length) {
      // Copy edit: replace only the freshly edited platforms, keep others intact,
      // but also respect platform removal (only keep platforms in context.platforms).
      const existingByType = new Map(
        (state.contentResult.platforms || []).map(p => [normalizePlatformId(p?.type), p])
      );
      const freshByType = new Map(
        (freshResult?.platforms || []).map(p => [normalizePlatformId(p?.type), p])
      );
      // Overwrite existing entries with fresh ones
      for (const [type, entry] of freshByType) {
        existingByType.set(type, entry);
      }
      // If context.platforms is set, filter to only those platforms (respects removal)
      const requestedPlatforms = normalizePlatformList(context?.platforms || []);
      let mergedPlatforms;
      if (requestedPlatforms.length > 0) {
        mergedPlatforms = requestedPlatforms
          .map(type => existingByType.get(type) || freshByType.get(type))
          .filter(Boolean);
        const removed = [...existingByType.keys()].filter(k => !requestedPlatforms.includes(k));
        if (removed.length > 0) {
          console.log('[Workflow] content_genius -- copy_edit removed platforms:', removed);
        }
      } else {
        mergedPlatforms = [...existingByType.values()];
      }
      thisContentResult = mergedPlatforms.length > 0
        ? { ...freshResult, platforms: mergedPlatforms }
        : freshResult;
      console.log('[Workflow] content_genius -- copy_edit selective replacement:', [...freshByType.keys()], '| final platforms:', mergedPlatforms.map(p => p.type));
    } else {
      // Merge new platform entries with existing ones -- don't overwrite previously generated platforms.
      // This ensures "add instagram" generates Instagram content without losing existing Facebook content.
      const existingPlatforms = state.contentResult?.platforms || [];
      const newPlatforms = freshResult?.platforms || [];
      if (existingPlatforms.length > 0 && newPlatforms.length > 0) {
        const existingByType = new Map(
          existingPlatforms.map(p => [normalizePlatformId(p?.type), p])
        );
        // Overwrite existing with fresh, keep untouched platforms
        for (const p of newPlatforms) {
          existingByType.set(normalizePlatformId(p?.type), p);
        }
        const mergedPlatforms = [...existingByType.values()];
        thisContentResult = { ...freshResult, platforms: mergedPlatforms };
        console.log('[Workflow] content_genius -- smart merge:', mergedPlatforms.map(p => p.type));
      } else {
        thisContentResult = freshResult;
      }
    }
  }

  if (toRun.includes('media_wizard')) {
    const hasBrandKit = !!context?.brandKit && typeof context.brandKit === 'object';
    const raw = context?.rawContent?.trim();
    const rawLooksLikeMediaOnly = isMediaOnlyInstruction(raw || '');
    const mediaPromptSource =
      (!rawLooksLikeMediaOnly ? raw : null) ||
      thisContentResult?.platforms?.[0]?.content ||
      thisContentResult?.copy ||
      agentPrompt;
    console.log(
      '[Workflow] media_wizard agentPrompt source:',
      (!rawLooksLikeMediaOnly && raw) ? 'rawContent' : (thisContentResult ? 'contentResult' : 'contentTopic')
    );
    const count = clampInt(context?.mediaCount ?? job?.mediaCount ?? 1, 1, 6);
    const previousReferenceUrl =
      state.mediaResult?.outputs?.[0]?.urls?.[0] ||
      state.mediaResult?.outputs?.[0]?.thumbnailUrls?.[0] ||
      null;

    if (signal?.aborted) return { finalReply: '', cancelled: true };
    thisMediaResult = await runMediaWizard(mediaPromptSource, {
      contentType: ctx.contentType,
      videoPurpose: context?.videoPurpose || 'regular',
      count,
      platforms: ctx.platforms,
      brandKit: context?.brandKit,
      injectBrandKit: hasBrandKit,
      skipThumbnails: true,
      userId: context?.userId,
      workspaceId: context?.workspaceId,
      youtubeConfig: context?.youtubeConfig || null,
      referenceImageUrl: context?.selectedAction === 'regenerate_media'
        ? (state.mediaInput?.url || previousReferenceUrl || undefined)
        : undefined,
    });
  }

  let thisSchedulerResult = state.schedulerResult;
  // Track the scheduled post ID across turns so reschedule can UPDATE instead of INSERT
  let scheduledPostId = context?.scheduledPostId || null;

  if (toRun.includes('scheduler_boss')) {
    console.log('[Workflow] scheduler payload media:', {
      mediaUrl: thisMediaResult?.outputs?.[0]?.urls?.[0],
      mediaUrls: thisMediaResult?.outputs?.[0]?.urls,
      mediaResultOutputs: thisMediaResult?.outputs?.length,
    });
    const uid = state.canonicalUserId ?? context?.userId;
    const cid = state.canonicalCompanyId ?? context?.companyId;
    thisSchedulerResult = await runSchedulerBoss(agentPrompt, {
      ...context,
      content: thisContentResult?.copy || thisContentResult?.platforms?.[0]?.content || null,
      copy: thisContentResult?.copy || thisContentResult?.platforms?.[0]?.content || null,
      platformsContent: thisContentResult?.platforms,
      mediaUrl: thisMediaResult?.outputs?.[0]?.urls?.[0],
      mediaUrls: thisMediaResult?.outputs?.[0]?.urls,
      platforms: context?.platforms ?? ctx.platforms,
      scheduleAt: context?.scheduleAt,
      scheduleDate: context?.scheduleDate,
      rawScheduleRequest: context?.rawScheduleRequest,
      platformSchedule: context?.platformSchedule,
      publishAllAtSameTime: context?.publishAllAtSameTime,
      scheduledPostId,
      rawUserMessage: state.rawUserPrompt || '',
      userId: uid, companyId: cid,
    });
    // Persist the scheduled post ID so the next turn can reschedule it
    if (thisSchedulerResult?.scheduleResult?.ok && thisSchedulerResult.scheduleResult.id) {
      scheduledPostId = thisSchedulerResult.scheduleResult.id;
      console.log('[Workflow] Persisting scheduledPostId:', scheduledPostId);
    }
  }

  let thisPreviewResult = null;
  if (toRun.includes('preview_pro') && (thisContentResult || thisMediaResult)) {
    // Always preview only the currently active platforms (prevents stale platforms
    // from previous turns from leaking into previews after platform_change).
    const previewPlatforms = (ctx.platforms && ctx.platforms.length > 0)
      ? ctx.platforms
      : (thisContentResult?.platforms?.map(p => p.type) || context?.platforms);
    const filteredContentForPreview = filterContentResultToActivePlatforms(thisContentResult);
    const filteredMediaForPreview = filterMediaResultToActivePlatforms(thisMediaResult);
    thisPreviewResult = await runPreviewPro(filteredContentForPreview, filteredMediaForPreview, {
      platforms: previewPlatforms,
      brandKit: context?.brandKit,
    });
  }

  console.log('[Workflow] toRun:', toRun, '| platforms:', ctx.platforms, '| schedulingOnly:', schedulingOnly);

  const justGeneratedContent = toRun.includes('content_genius') || toRun.includes('media_wizard');
  const schedulerAlreadyRan = toRun.includes('scheduler_boss');
  const hasResult = !!(thisContentResult || thisMediaResult);

  // Filter what we SHOW (and persist) to the current active platform set.
  // This prevents replies like **[youtube short]** after the user says "only instagram".
  thisContentResult = filterContentResultToActivePlatforms(thisContentResult);
  thisMediaResult = filterMediaResultToActivePlatforms(thisMediaResult);

  if (justGeneratedContent && hasResult && !schedulerAlreadyRan) {
    const parts = [];
    if (context?.prefaceReply) parts.push(String(context.prefaceReply));
    if (thisContentResult?.platforms?.length) {
      parts.push(thisContentResult.platforms.map(p => {
        // YouTube platforms get a structured display with title/description/tags
        if ((p.type === 'youtube' || p.type === 'youtube_short') && p.title) {
          const label = p.type === 'youtube_short' ? 'youtube short' : 'youtube';
          let display = `**[${label}]**\n**Title:** ${p.title}\n**Description:** ${p.description || ''}`;
          if (Array.isArray(p.tags) && p.tags.length > 0) {
            display += `\n**Tags:** ${p.tags.join(', ')}`;
          }
          return display;
        }
        return `**[${p.type}]**\n${p.content}`;
      }).join('\n\n'));
    } else if (thisContentResult?.copy) {
      parts.push(thisContentResult.copy);
    }
    if (thisMediaResult?.outputs?.length) {
      const output = thisMediaResult.outputs[0];
      const urls = output?.urls || [];
      // Use the actual output type (media_wizard returns { type: 'video' | 'image' })
      // instead of ctx.contentType which may still be 'image' when YouTube is selected.
      const isVideo = output?.type === 'video' || urls[0]?.endsWith?.('.mp4') || ctx.contentType === 'video';
      if (isVideo) {
        const variants = output?.variants || urls.map((url, i) => ({ url, aspectRatio: ['9:16', '16:9', '1:1'][i] || `variant ${i + 1}` }));
        parts.push(variants.map(v => `📹 **${v.aspectRatio}**: ${v.url}`).join('\n'));
      } else if (urls.length > 0) {
        const imgMsg = await generateContextualReply(
          'images have been generated and are ready for the user to review',
          { count: urls.length }
        ) || `${urls.length} image(s) ready -- see attachments`;
        parts.push(imgMsg);
      }
    }

    // Surface YouTube warnings/errors to the user in the final reply
    if (Array.isArray(thisMediaResult?.warnings)) {
      for (const w of thisMediaResult.warnings) {
        if (w?.message) parts.push(w.message);
      }
    }

    console.log('[Workflow] Content generated -- asking scheduling question with buttons');
    const scheduleButtons = buildSchedulingOptions();
    return {
      knowledgeChunks,
      contentResult: thisContentResult, mediaResult: thisMediaResult, previewResult: thisPreviewResult,
      schedulerResult: null,
      finalReply: `${parts.join('\n\n')}\n\n---\n`,
      buttons: [
        { label: 'Copy', action: 'copy' },
        ...(thisMediaResult?.outputs?.length ? [{ label: 'Download', action: 'download' }] : []),
        ...scheduleButtons,
      ],
      schedulingOnlyReply: false,
      awaitingConfirmation: false, confirmationOptions: [], confirmationReason: null,
      mediaInput: null, userSelection: null,
      guidedFlowStep: 'awaiting_schedule',
    };
  }

  // Normal final reply
  const parts = [];
  if (!schedulingOnly) {
    if (context?.prefaceReply) parts.push(String(context.prefaceReply));
    if (thisContentResult?.platforms?.length) {
      parts.push(thisContentResult.platforms.map(p => {
        if ((p.type === 'youtube' || p.type === 'youtube_short') && p.title) {
          const label = p.type === 'youtube_short' ? 'youtube short' : 'youtube';
          let display = `**[${label}]**\n**Title:** ${p.title}\n**Description:** ${p.description || ''}`;
          if (Array.isArray(p.tags) && p.tags.length > 0) {
            display += `\n**Tags:** ${p.tags.join(', ')}`;
          }
          return display;
        }
        return `**[${p.type}]**\n${p.content}`;
      }).join('\n\n'));
    } else if (thisContentResult?.copy) {
      parts.push(thisContentResult.copy);
    }
    if (thisMediaResult?.outputs?.length) {
      const output = thisMediaResult.outputs[0];
      const urls = output?.urls || [];
      // Use the actual output type (media_wizard returns { type: 'video' | 'image' })
      // instead of ctx.contentType which may still be 'image' when YouTube is selected.
      const isVideo = output?.type === 'video' || urls[0]?.endsWith?.('.mp4') || ctx.contentType === 'video';
      if (isVideo) {
        const variants = output?.variants || urls.map((url, i) => ({ url, aspectRatio: ['9:16', '16:9', '1:1'][i] || `variant ${i + 1}` }));
        parts.push(variants.map(v => `📹 **${v.aspectRatio}**: ${v.url}`).join('\n'));
      } else {
        const imgMsg = await generateContextualReply(
          'images have been generated and are ready for the user to review',
          { count: urls.length }
        ) || (urls.length > 0 ? `${urls.length} image(s) ready -- see attachments` : 'See attachments');
        parts.push(imgMsg);
      }
    } else if (thisMediaResult?.error && toRun.includes('media_wizard')) {
      const err = typeof thisMediaResult.error === 'object' ? JSON.stringify(thisMediaResult.error) : String(thisMediaResult.error);
      const errMsg = await generateContextualReply(
        'media generation failed, inform user of the error and suggest alternatives',
        { error: err }
      ) || `Media generation failed: ${err}`;
      parts.push(errMsg);
    }
    if (thisPreviewResult?.previews?.length) {
      const previewMsg = await generateContextualReply(
        'content previews are ready, invite the user to review before posting',
        { platforms: thisPreviewResult.platforms }
      ) || `Previews ready for: ${thisPreviewResult.platforms.join(', ')}`;
      parts.push(previewMsg);
    }
  }
  // Surface YouTube warnings/errors to the user in the normal reply path too
  if (Array.isArray(thisMediaResult?.warnings)) {
    for (const w of thisMediaResult.warnings) {
      if (w?.message) parts.push(w.message);
    }
  }

  if (thisSchedulerResult?.message) parts.push(thisSchedulerResult.message);

  const doneMsg = await generateContextualReply(
    'content generation is complete, invite user to take next action',
    { hasContent: !!thisContentResult, hasMedia: !!thisMediaResult, hasSchedule: !!thisSchedulerResult }
  ) || "Done. Tell me what you'd like next.";
  const finalReply = parts.length ? parts.join('\n\n') : doneMsg;
  const buttons = [];
  if (!schedulingOnly) {
    if (thisContentResult) buttons.push({ label: 'Copy', action: 'copy' });
    if (thisMediaResult?.outputs?.length) buttons.push({ label: 'Download', action: 'download' });
    if (thisPreviewResult) buttons.push({ label: 'View previews', action: 'preview' });
  }
  if (thisSchedulerResult) buttons.push({ label: schedulingOnly ? 'Reschedule' : 'Schedule', action: 'schedule' });

  return {
    knowledgeChunks, contentResult: thisContentResult, mediaResult: thisMediaResult,
    previewResult: thisPreviewResult, schedulerResult: thisSchedulerResult,
    finalReply, buttons, schedulingOnlyReply: schedulingOnly,
    mediaInput: null, awaitingConfirmation: false, userSelection: null,
    confirmationReason: null,
    guidedFlowStep: schedulingOnly ? 'awaiting_schedule' : null,
    // Persist scheduledPostId so the next turn can reschedule by updating the existing DB row
    ...(scheduledPostId ? { context: { scheduledPostId } } : {}),
  };
}

// ─────────────────────────────────────────────────────────────────────────────

function routeAfterOrchestrator(state) {
  if (state.directReply) return '__end__';
  if (state.awaitingConfirmation) return '__end__';
  if (state.callAgents?.length > 0) return 'execute_agents';
  return '__end__';
}

export function buildGraph() {
  const graph = new StateGraph(ZunoState)
    .addNode('orchestrator', orchestratorNode)
    .addNode('execute_agents', executeAgentsNode)
    .addEdge('__start__', 'orchestrator')
    .addConditionalEdges('orchestrator', routeAfterOrchestrator)
    .addEdge('execute_agents', '__end__');
  return graph.compile({ checkpointer });
}

export default buildGraph;