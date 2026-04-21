/**
 * Scheduler Boss Agent
 * Role: Calendar + Posting
 * Finds best posting time (audience data), schedules/posts via APIs, handles approval flow.
 * Trigger: When user says "schedule", "post now", "calendar".
 * Tools: Platform APIs + Supabase
 */

import * as chrono from 'chrono-node';
import { createClient } from '@supabase/supabase-js';
import { config } from '../../config/index.js';
import * as db from '../lib/db.js';

const DEBUG_SCHEDULER_DB = process.env.DEBUG_SCHEDULER_DB === '1';

const supabase =
  !db.isUsingPostgres() && config.supabase.url && config.supabase.serviceKey
    ? createClient(config.supabase.url, config.supabase.serviceKey)
    : null;

const PLATFORM_TOKENS = {
  facebook: config.platform.facebook,
  instagram: config.platform.instagram,
  linkedin: config.platform.linkedin,
  twitter: config.platform.twitter,
  x: config.platform.twitter,
  tiktok: config.platform.tiktok,
  youtube: config.platform.youtube,
  youtube_short: config.platform.youtube,
};

/**
 * Get best posting time from stored audience/analytics (placeholder: default window).
 */
export async function getBestPostingTime(platforms = []) {
  if (db.isUsingPostgres() && platforms?.length) {
    try {
      const { rows } = await db.query(
        'SELECT best_time_utc, platform FROM public.audience_insights WHERE platform = ANY($1) LIMIT 5',
        [platforms]
      );
      if (rows?.length) return { times: rows, source: 'audience_insights' };
    } catch (_) { }
  }
  if (supabase && platforms?.length) {
    const { data } = await supabase.from('audience_insights').select('best_time_utc, platform').in('platform', platforms).limit(5);
    if (data?.length) return { times: data, source: 'audience_insights' };
  }
  return {
    times: (platforms || []).map((p) => ({ platform: p, best_time_utc: '14:00', timezone: 'UTC' })),
    source: 'default',
  };
}

/**
 * Schedule a post in Supabase (and optionally push to platform APIs).
 */
export async function schedulePost(payload) {
  const { content, mediaUrl, mediaUrls, platformContent, platforms, scheduleAt, postNow, userId, companyId } = payload;
  const plats = platforms || ['instagram'];
  const scheduledAt = postNow ? new Date().toISOString() : scheduleAt;
  const uid = userId ?? null;
  const rawCid = companyId ?? config.defaults?.companyId ?? null;
  const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(rawCid);
  const cid = rawCid != null && rawCid !== '' && !isUuid ? String(rawCid) : null;
  if (db.isUsingPostgres()) {
    try {
      if (DEBUG_SCHEDULER_DB) {
        console.log('[scheduler-db] insert scheduled_posts', {
          plats, scheduledAt, userId: uid, companyId: cid,
          hasContent: !!content && String(content).trim().length > 0,
          contentChars: String(content || '').length,
          mediaUrl: mediaUrl || null,
          mediaUrlsCount: Array.isArray(mediaUrls) ? mediaUrls.length : mediaUrl ? 1 : 0,
          platformContentKeys: platformContent && typeof platformContent === 'object' ? Object.keys(platformContent).length : 0,
          status: postNow ? 'pending' : 'scheduled',
        });
      }
      const { rows } = await db.query(
        `INSERT INTO public.scheduled_posts (company_id, user_id, content, media_url, media_urls, platform_content, platforms, scheduled_at, status, approved)
         VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8::timestamptz, $9, false)
         RETURNING id, scheduled_at`,
        [
          cid, uid, content, mediaUrl || null,
          Array.isArray(mediaUrls) ? mediaUrls : (mediaUrl ? [mediaUrl] : []),
          JSON.stringify(platformContent || {}), plats, scheduledAt,
          postNow ? 'pending' : 'scheduled',
        ]
      );
      if (rows[0]) return { ok: true, id: rows[0].id, scheduled_at: rows[0].scheduled_at };
    } catch (e) {
      if (DEBUG_SCHEDULER_DB) console.warn('[scheduler-db] insert failed:', e.message);
      return { ok: false, message: e.message };
    }
  }
  if (!supabase) return { ok: false, message: 'Database not configured' };
  const record = {
    content,
    media_url: mediaUrl,
    media_urls: Array.isArray(mediaUrls) ? mediaUrls : (mediaUrl ? [mediaUrl] : []),
    platform_content: platformContent || {},
    platforms: plats,
    scheduled_at: scheduledAt,
    status: postNow ? 'pending' : 'scheduled',
    approved: false,
    ...(cid && { company_id: cid }),
    ...(uid && { user_id: uid }),
  };
  const { data, error } = await supabase.from('scheduled_posts').insert(record).select().single();
  if (error) return { ok: false, message: error.message };
  return { ok: true, id: data.id, scheduled_at: data.scheduled_at };
}

// ─── Platform-specific post implementations ───────────────────────────────────

async function postToFacebook(content, mediaUrl, accessToken) {
  const pageId = process.env.FACEBOOK_PAGE_ID;
  if (!pageId) return { ok: false, reason: 'FACEBOOK_PAGE_ID env var not configured.' };

  const body = { message: content, access_token: accessToken };
  if (mediaUrl) body.link = mediaUrl;

  try {
    const resp = await fetch(`https://graph.facebook.com/v19.0/${pageId}/feed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.id) return { ok: true, postId: data.id };
    return { ok: false, reason: data.error?.message || `Facebook API error (${resp.status}).` };
  } catch (e) {
    return { ok: false, reason: `Facebook request failed: ${e.message}` };
  }
}

async function postToInstagram(content, mediaUrl, accessToken) {
  const igUserId = process.env.INSTAGRAM_BUSINESS_ACCOUNT_ID;
  if (!igUserId) return { ok: false, reason: 'INSTAGRAM_BUSINESS_ACCOUNT_ID env var not configured.' };
  if (!mediaUrl) return { ok: false, reason: 'Instagram requires an image or video URL.' };

  try {
    // Step 1: Create media container
    const containerResp = await fetch(`https://graph.facebook.com/v19.0/${igUserId}/media`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_url: mediaUrl, caption: content, access_token: accessToken }),
    });
    const container = await containerResp.json();
    if (!container.id) {
      return { ok: false, reason: container.error?.message || 'Failed to create Instagram media container.' };
    }

    // Step 2: Publish the container
    const publishResp = await fetch(`https://graph.facebook.com/v19.0/${igUserId}/media_publish`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ creation_id: container.id, access_token: accessToken }),
    });
    const published = await publishResp.json();
    if (published.id) return { ok: true, postId: published.id };
    return { ok: false, reason: published.error?.message || 'Instagram publish failed.' };
  } catch (e) {
    return { ok: false, reason: `Instagram request failed: ${e.message}` };
  }
}

async function postToLinkedIn(content, mediaUrl, accessToken) {
  const authorUrn = process.env.LINKEDIN_ORGANIZATION_URN || process.env.LINKEDIN_PERSON_URN;
  if (!authorUrn) {
    return { ok: false, reason: 'LINKEDIN_ORGANIZATION_URN or LINKEDIN_PERSON_URN env var not configured.' };
  }

  const body = {
    author: authorUrn,
    lifecycleState: 'PUBLISHED',
    specificContent: {
      'com.linkedin.ugc.ShareContent': {
        shareCommentary: { text: content },
        shareMediaCategory: mediaUrl ? 'ARTICLE' : 'NONE',
        ...(mediaUrl ? { media: [{ status: 'READY', originalUrl: mediaUrl }] } : {}),
      },
    },
    visibility: { 'com.linkedin.ugc.MemberNetworkVisibility': 'PUBLIC' },
  };

  try {
    const resp = await fetch('https://api.linkedin.com/v2/ugcPosts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
        'X-Restli-Protocol-Version': '2.0.0',
      },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (resp.ok && data.id) return { ok: true, postId: data.id };
    return { ok: false, reason: data.message || `LinkedIn API error (${resp.status}).` };
  } catch (e) {
    return { ok: false, reason: `LinkedIn request failed: ${e.message}` };
  }
}

async function postToX(content, _mediaUrl, _accessToken) {
  // X API v2 POST /2/tweets requires user-level OAuth (not a Bearer token).
  // Set TWITTER_ACCESS_TOKEN to a user OAuth 2.0 token with tweet.write scope.
  const userToken = process.env.TWITTER_ACCESS_TOKEN;
  if (!userToken) {
    return { ok: false, reason: 'TWITTER_ACCESS_TOKEN (user OAuth token with tweet.write scope) not configured. The bearer token cannot be used for posting.' };
  }

  const body = { text: String(content).slice(0, 280) };

  try {
    const resp = await fetch('https://api.twitter.com/2/tweets', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${userToken}`,
      },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.data?.id) return { ok: true, postId: data.data.id };
    return { ok: false, reason: data.detail || data.errors?.[0]?.message || `X API error (${resp.status}).` };
  } catch (e) {
    return { ok: false, reason: `X request failed: ${e.message}` };
  }
}

async function postToTikTok(content, mediaUrl, accessToken) {
  if (!mediaUrl) {
    return { ok: false, reason: 'TikTok requires a video URL. Text-only posts are not supported on TikTok.' };
  }

  try {
    const resp = await fetch('https://open.tiktokapis.com/v2/post/publish/video/init/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': `Bearer ${accessToken}`,
      },
      body: JSON.stringify({
        post_info: {
          title: String(content).slice(0, 150),
          privacy_level: 'PUBLIC_TO_EVERYONE',
          disable_comment: false,
        },
        source_info: {
          source: 'PULL_FROM_URL',
          video_url: mediaUrl,
        },
      }),
    });
    const data = await resp.json();
    if (resp.ok && data.data?.publish_id) return { ok: true, publishId: data.data.publish_id };
    return { ok: false, reason: data.error?.message || `TikTok API error (${resp.status}).` };
  } catch (e) {
    return { ok: false, reason: `TikTok request failed: ${e.message}` };
  }
}

async function postToYouTube(content, mediaUrl, accessToken, platformData) {
  if (!mediaUrl) {
    return { ok: false, reason: 'YouTube requires a video URL.' };
  }

  // Extract structured title/description/tags from platform data if available,
  // otherwise fall back to slicing the content string.
  const title = platformData?.title || String(content).slice(0, 100);
  const description = platformData?.description || content || '';
  const tags = Array.isArray(platformData?.tags) ? platformData.tags : [];

  // YouTube requires a resumable binary upload -- direct URL ingestion is not supported by the API.
  // This call creates the video metadata resource; the actual video bytes must be uploaded separately.
  try {
    const resp = await fetch('https://www.googleapis.com/youtube/v3/videos?part=snippet,status', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
      },
      body: JSON.stringify({
        snippet: {
          title: String(title).slice(0, 100),
          description,
          tags,
          categoryId: '22',
        },
        status: { privacyStatus: 'public', selfDeclaredMadeForKids: false },
      }),
    });
    const data = await resp.json();
    if (resp.ok && data.id) return { ok: true, videoId: data.id };
    return {
      ok: false,
      reason: data.error?.message || `YouTube API error (${resp.status}). Note: YouTube requires binary video upload via resumable upload API -- direct URL posting is not supported.`,
    };
  } catch (e) {
    return { ok: false, reason: `YouTube request failed: ${e.message}` };
  }
}

/**
 * Post now to configured platforms using real platform APIs.
 */
export async function postNowToPlatforms(payload) {
  const { content, mediaUrl, platforms, platformsContent } = payload;
  const results = {};

  // Build a map of structured platform data (title/description/tags for YouTube)
  const platformDataMap = {};
  if (Array.isArray(platformsContent)) {
    for (const pd of platformsContent) {
      if (pd?.type) platformDataMap[pd.type] = pd;
    }
  }

  for (const p of platforms || []) {
    const token = PLATFORM_TOKENS[p];
    if (!token) {
      results[p] = { ok: false, reason: `No access token configured for ${p}. Set the corresponding env var.` };
      continue;
    }

    try {
      switch (p) {
        case 'facebook':
          results[p] = await postToFacebook(content, mediaUrl, token);
          break;
        case 'instagram':
          results[p] = await postToInstagram(content, mediaUrl, token);
          break;
        case 'linkedin':
          results[p] = await postToLinkedIn(content, mediaUrl, token);
          break;
        case 'x':
        case 'twitter':
          results[p] = await postToX(content, mediaUrl, token);
          break;
        case 'tiktok':
          results[p] = await postToTikTok(content, mediaUrl, token);
          break;
        case 'youtube':
        case 'youtube_short':
          results[p] = await postToYouTube(content, mediaUrl, token, platformDataMap[p]);
          break;
        default:
          results[p] = { ok: false, reason: `Platform "${p}" is not yet supported for direct publishing.` };
      }
    } catch (e) {
      console.error(`[SchedulerBoss] Unexpected error posting to ${p}:`, e.message);
      results[p] = { ok: false, reason: e.message };
    }

    console.log(`[SchedulerBoss] Post to ${p}:`, results[p]);
  }

  return results;
}

/**
 * Update an existing scheduled post's time (reschedule).
 */
export async function reschedulePost(postId, newScheduleAt) {
  if (!postId || !newScheduleAt) return { ok: false, message: 'Missing post ID or new time.' };
  if (db.isUsingPostgres()) {
    try {
      const { rows } = await db.query(
        `UPDATE public.scheduled_posts
         SET scheduled_at = $1::timestamptz, status = 'scheduled'
         WHERE id = $2
         RETURNING id, scheduled_at`,
        [newScheduleAt, postId]
      );
      if (rows[0]) return { ok: true, id: rows[0].id, scheduled_at: rows[0].scheduled_at };
      return { ok: false, message: 'Scheduled post not found -- it may have already been published or deleted.' };
    } catch (e) {
      console.warn('[scheduler-boss] reschedulePost failed:', e.message);
      return { ok: false, message: e.message };
    }
  }
  if (supabase) {
    const { data, error } = await supabase
      .from('scheduled_posts')
      .update({ scheduled_at: newScheduleAt, status: 'scheduled' })
      .eq('id', postId)
      .select()
      .single();
    if (error) return { ok: false, message: error.message };
    return { ok: true, id: data.id, scheduled_at: data.scheduled_at };
  }
  return { ok: false, message: 'Database not configured.' };
}

/**
 * Parse natural language date/time from user prompt (e.g. "25th february morning 10", "tomorrow 9am").
 * Returns ISO string or null.
 */
function parseScheduleDate(userPrompt) {
  if (!userPrompt || typeof userPrompt !== 'string') return null;
  const normalized = userPrompt.replace(/\b(\d{1,2})\s+(st|nd|rd|th)\b/gi, '$1$2');
  const ref = new Date();
  const parsed = chrono.parseDate(normalized, ref, { forwardDate: true });
  if (!parsed || parsed < ref) return null;
  return parsed.toISOString();
}

/**
 * Normalize and validate a schedule time (from UI or context). Returns ISO string or null.
 * If in the past, shifts to same time next day so user never gets a confusing error.
 */
function normalizeScheduleAt(isoOrDate) {
  if (isoOrDate == null || !String(isoOrDate).trim()) return null;
  const d = new Date(isoOrDate);
  if (Number.isNaN(d.getTime())) return null;
  const now = new Date();
  if (d <= now) {
    const next = new Date(d);
    next.setUTCDate(next.getUTCDate() + 1);
    return next.toISOString();
  }
  return d.toISOString();
}

/** Format ISO date for display; use UTC so shown time matches user-requested time. */
function formatScheduledMessage(atIso, userRequested = false) {
  const str = new Date(atIso).toLocaleString(undefined, {
    timeZone: 'UTC',
    dateStyle: 'medium',
    timeStyle: 'short',
  });
  return userRequested ? `Scheduled for ${str} UTC as requested.` : `Scheduled for ${str} UTC.`;
}

const PLATFORM_IDS = ['x', 'instagram', 'linkedin', 'facebook', 'tiktok', 'youtube', 'youtube_short'];

/**
 * Build full ISO datetime for a platform from platformSchedule (e.g. "09:00") and base date.
 */
function toScheduledAtISO(baseDateStr, timeStr) {
  const d = baseDateStr ? new Date(baseDateStr + 'T00:00:00.000Z') : new Date();
  if (Number.isNaN(d.getTime())) return null;
  const s = String(timeStr || '14:00').trim().toLowerCase();
  let hours = 14;
  let minutes = 0;
  const match = s.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/i) || s.match(/^(\d{1,2}):(\d{2})$/);
  if (match) {
    hours = parseInt(match[1], 10);
    minutes = match[2] ? parseInt(match[2], 10) : 0;
    if (match[3]) {
      if (match[3].toLowerCase() === 'pm' && hours < 12) hours += 12;
      if (match[3].toLowerCase() === 'am' && hours === 12) hours = 0;
    }
  }
  d.setUTCHours(hours, minutes, 0, 0);
  if (d <= new Date()) d.setUTCDate(d.getUTCDate() + 1);
  return d.toISOString();
}

/**
 * Get base date (YYYY-MM-DD) from context.scheduleDate or user prompt or tomorrow.
 */
function getBaseScheduleDate(scheduleDate, userPrompt) {
  if (scheduleDate && /^\d{4}-\d{2}-\d{2}$/.test(String(scheduleDate).trim())) {
    return String(scheduleDate).trim();
  }
  const parsed = parseScheduleDate(userPrompt || '');
  if (parsed) {
    const d = new Date(parsed);
    return d.toISOString().slice(0, 10);
  }
  const tomorrow = new Date();
  tomorrow.setUTCDate(tomorrow.getUTCDate() + 1);
  return tomorrow.toISOString().slice(0, 10);
}

/**
 * Run Scheduler Boss: best time + schedule or post now.
 */
export async function runSchedulerBoss(userPrompt, context = {}) {
  const {
    content, mediaUrl, mediaUrls, platforms,
    scheduleAt: contextScheduleAt, scheduleDate: contextScheduleDate,
    platformSchedule: contextPlatformSchedule, rawScheduleRequest,
    postNow, userId, companyId, scheduledPostId,
    rawUserMessage,
  } = context;

  const timeExpression = (rawScheduleRequest && rawScheduleRequest.trim()) || userPrompt || '';
  // rawUserMessage is the actual user text (not just the extracted time portion).
  // Use it for intent detection so "reschedule" is not lost when the LLM strips it.
  const fullUserText = (rawUserMessage && rawUserMessage.trim()) || timeExpression;

  const schedulingIntent = postNow || /post now|publish now/i.test(fullUserText);
  const calendarIntent = /schedule|calendar|when to post/i.test(fullUserText);
  const scheduleForIntent = /schedule\s+for|schedule\s+it\s+for|schedule\s+it|on\s+\d|morning|afternoon|evening|night|february|march|april|may|june|july|august|september|october|november|december|tomorrow|next\s+week|day after|tonight/i.test(timeExpression);
  const rescheduleIntent = /reschedule|change\s+(the\s+)?(time|schedule|date)|update\s+(the\s+)?(time|schedule)/i.test(fullUserText);

  const uiScheduleAt = normalizeScheduleAt(contextScheduleAt);
  const parsedFromText = (scheduleForIntent || rescheduleIntent || !!scheduledPostId) ? parseScheduleDate(timeExpression) : null;
  const effectiveScheduleAt = uiScheduleAt || parsedFromText;

  const bestTime = await getBestPostingTime(platforms || ['instagram', 'linkedin', 'x']);

  // If a scheduledPostId already exists and the user provides a new time, always UPDATE the
  // existing row -- regardless of whether they said "reschedule" (handles typos and shorthand).
  // This prevents duplicate scheduled_posts rows for the same piece of content.
  const shouldReschedule = !!scheduledPostId && !!effectiveScheduleAt && !schedulingIntent;
  if (shouldReschedule) {
    console.log('[Scheduler] Rescheduling (UPDATE) post', scheduledPostId, '->', effectiveScheduleAt);
    const reschedResult = await reschedulePost(scheduledPostId, effectiveScheduleAt);
    return {
      bestTime,
      scheduleResult: reschedResult,
      message: reschedResult.ok
        ? `Rescheduled! ${formatScheduledMessage(effectiveScheduleAt, true)}`
        : `Could not reschedule: ${reschedResult.message}`,
    };
  }

  const platformContentMap = {};
  const platformsContentArray = context.platformsContent && Array.isArray(context.platformsContent) ? context.platformsContent : [];
  if (platformsContentArray.length) {
    for (const p of platformsContentArray) {
      if (p?.type && p?.content != null) platformContentMap[p.type] = p.content;
    }
  }

  if (schedulingIntent && (content || mediaUrl)) {
    const scheduleResult = await schedulePost({
      content: content || context.copy,
      mediaUrl, mediaUrls,
      platformContent: Object.keys(platformContentMap).length ? platformContentMap : undefined,
      platforms, scheduleAt: contextScheduleAt, postNow: true, userId, companyId,
    });
    const postResults = await postNowToPlatforms({ content: content || context.copy, mediaUrl, platforms, platformsContent: platformsContentArray });
    const failedPlatforms = Object.entries(postResults).filter(([, r]) => !r.ok).map(([p]) => p);
    const successPlatforms = Object.entries(postResults).filter(([, r]) => r.ok).map(([p]) => p);
    let message = scheduleResult.ok ? 'Post published.' : scheduleResult.message;
    if (successPlatforms.length) message = `Published to ${successPlatforms.join(', ')}.`;
    if (failedPlatforms.length) message += ` Could not post to: ${failedPlatforms.map(p => `${p} (${postResults[p].reason})`).join('; ')}.`;
    return { bestTime, scheduleResult, postResults, message };
  }

  const platformSchedule =
    contextPlatformSchedule && typeof contextPlatformSchedule === 'object' && Object.keys(contextPlatformSchedule).length > 0
      ? contextPlatformSchedule
      : null;
  if (platformSchedule && (content || mediaUrl)) {
    const baseDate = getBaseScheduleDate(contextScheduleDate, userPrompt);
    const plats = Object.keys(platformSchedule).filter((p) => PLATFORM_IDS.includes(p));
    if (plats.length > 0) {
      const results = [];
      const summaries = [];
      for (const platform of plats) {
        const timeStr = platformSchedule[platform];
        const atIso = toScheduledAtISO(baseDate, timeStr);
        if (!atIso) continue;
        const platformCopy = platformContentMap[platform] || content || context.copy || '';
        const scheduleResult = await schedulePost({
          content: platformCopy, mediaUrl: mediaUrl || null, mediaUrls,
          platformContent: { [platform]: platformCopy }, platforms: [platform],
          scheduleAt: atIso, postNow: false, userId, companyId,
        });
        if (scheduleResult.ok) {
          results.push(scheduleResult);
          const str = new Date(atIso).toLocaleString(undefined, { timeZone: 'UTC', dateStyle: 'medium', timeStyle: 'short' });
          summaries.push(`${platform}: ${str} UTC`);
        }
      }
      const message = summaries.length > 0
        ? `Scheduled per platform:\n${summaries.join('\n')}`
        : results[0]?.message || 'Scheduling completed.';
      return { bestTime, scheduleResult: results[0] ?? { ok: summaries.length > 0 }, message };
    }
  }

  if (effectiveScheduleAt && (content || mediaUrl)) {
    const platformContent = Object.keys(platformContentMap).length > 0 ? platformContentMap : undefined;
    const scheduleResult = await schedulePost({
      content: content || context.copy, mediaUrl, mediaUrls, platformContent,
      platforms, scheduleAt: effectiveScheduleAt, postNow: false, userId, companyId,
    });
    const userRequested = !!uiScheduleAt;
    return {
      bestTime, scheduleResult,
      message: scheduleResult.ok ? formatScheduledMessage(effectiveScheduleAt, userRequested) : scheduleResult.message,
    };
  }

  const scheduleItIntent = /schedule\s+it|schedule\s*$/i.test(timeExpression);
  if ((calendarIntent || scheduleItIntent) && (content || mediaUrl)) {
    if (effectiveScheduleAt) {
      const platformContent = Object.keys(platformContentMap).length > 0 ? platformContentMap : undefined;
      const scheduleResult = await schedulePost({
        content: content || context.copy || '', mediaUrl: mediaUrl || null, mediaUrls, platformContent,
        platforms: platforms || ['instagram', 'linkedin', 'x'], scheduleAt: effectiveScheduleAt,
        postNow: false, userId, companyId,
      });
      return {
        bestTime, scheduleResult,
        message: scheduleResult.ok ? formatScheduledMessage(effectiveScheduleAt, true) : scheduleResult.message,
      };
    }
    const bestTimeStr = bestTime?.times?.[0]?.best_time_utc
      ? ` Our recommended best time for your audience is **${bestTime.times[0].best_time_utc} UTC**.`
      : '';
    return {
      bestTime,
      message: `When would you like to schedule this post? 📅${bestTimeStr}\n\nSay something like *"tomorrow at 9am"*, *"day after tomorrow evening 6pm"*, or *"post now"* to publish immediately.`,
    };
  }

  if (calendarIntent || scheduleForIntent) {
    const hint = effectiveScheduleAt
      ? `Time set: ${new Date(effectiveScheduleAt).toLocaleString(undefined, { timeZone: 'UTC', dateStyle: 'medium', timeStyle: 'short' })} UTC. Say "schedule it" to confirm, or pick a different time.`
      : 'Pick a date and time above and tap Schedule, or say when you\'d like to post (e.g. "tomorrow at 9am"). You can also say "post now" to publish immediately.';
    return { bestTime, message: hint, ...(effectiveScheduleAt ? { parsedScheduleAt: effectiveScheduleAt } : {}) };
  }

  return {
    bestTime,
    message: 'Ready to schedule. Create or select your content, then pick a time and tap Schedule, or say "schedule it" / "post now".',
  };
}

export default runSchedulerBoss;
