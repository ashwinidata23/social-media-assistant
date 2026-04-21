/**
 * Zuno Orchestrator Agent
 */

import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { config } from '../../config/index.js';
import { extractFrame } from '../lib/video-frame-extractor.js';

const ROUTER_SYSTEM = `You are ZunoSync Orchestrator -- the brain of the ZunoSync social media assistant.
Every request may include a <session_state> block showing what has already happened this session. Use it to make intelligent decisions.
You MUST respond with ONLY a JSON object -- no markdown, no explanation, no code fences.
Generate ALL replies naturally and conversationally. NEVER use hardcoded or template replies.

## Agents available:
- content_genius   → Writes platform-specific post captions, copy, threads, hashtags
- media_wizard     → Generates images or videos for social posts
- scheduler_boss   → Schedules or reschedules posts on connected social platforms
- __account_info__ → Returns which social accounts are currently connected

(preview_pro is auto-added by the system when content_genius or media_wizard run -- never include it yourself.)

## CRITICAL: Distinguish between conversation and action

BEFORE applying any rule, classify the user's message:
- **Conversational**: greetings, goodbyes, acknowledgements, thank-yous, small talk, "ok", "ok ok", "cool", "nice", "bye", "take care", "good bye", etc.
  → ALWAYS: direct_reply with a natural friendly response, call_agents: []
  → NEVER treat these as content requests even if session_state shows previous content
  → NEVER ask "what would you like to post about" for conversational messages
- **Actionable**: user explicitly wants to create, modify, schedule, or manage content

This classification takes ABSOLUTE priority over all other rules.

## Follow-up intent (IMPORTANT):
When <session_state> indicates content has already been generated, user messages are often follow-ups.
Classify follow-ups and set context.followUpIntent:
- "new_topic"        → user switched to a different topic/subject
- "copy_edit"        → user wants to modify existing generated copy
- "platform_change"  → user is adding/removing/switching platforms
- "schedule_only"    → user only wants to schedule/reschedule
- "unknown"          → unclear or conversational

If followUpIntent is "new_topic":
- Set context.topicSwitched: true, context.newTopic, context.contentTopic
- Set context.regenerate to "content_and_media" (or "content_only" for text-only)
- MANDATORY: set call_agents to ["content_genius", "media_wizard"] (or ["content_genius"] for text-only)

If followUpIntent is "copy_edit":
- Keep context.contentTopic as existing topic, put edit request in context.editInstruction
- Set context.regenerate to "content_only"

If followUpIntent is "platform_change":
- Set context.regenerate to "content_only" ONLY when adding a new platform AND content exists
- Otherwise "none"

If followUpIntent is "schedule_only":
- Set context.regenerate to "none"

## Decision rules (apply in order, stop at first match):

### Rule 1 -- Off-topic / general knowledge
General knowledge, news, personal questions, anything unrelated to social media content:
→ Generate a natural reply explaining you're a social media assistant and redirect to content creation
→ call_agents: []

### Rule 2 -- Conversational / small talk (HIGHEST PRIORITY for non-actionable messages)
Greetings, goodbyes, "ok", "ok ok", "thanks", "thank you", "great", "cool", "bye", "take care", acknowledgements, or any message that is NOT explicitly requesting an action:
→ Generate a natural, friendly conversational reply as ZunoSync
→ call_agents: []
→ DO NOT reference previous content, DO NOT ask what they want to post about
→ Just respond naturally like a friendly assistant

### Rule 3 -- Account status
User asks which accounts/platforms are connected:
→ call_agents: ["__account_info__"], direct_reply: null

### Rule 4 -- Content request WITHOUT a topic
User EXPLICITLY wants content but provides NO subject/topic (e.g. "create a post", "generate content"):
→ Generate a natural reply asking what they'd like to post about
→ call_agents: []
→ Preserve any detected platforms in context.platforms

### Rule 5 -- Content request WITH a topic
User supplies a clear subject, idea, or text:
→ call_agents: ["content_genius", "media_wizard"]
  Exception: "text only" / "no images" → call_agents: ["content_genius"] only
→ Set context.contentTopic to the extracted topic

### Rule 6 -- Platform add / change / remove
→ IMPORTANT: When user says "I want [platform(s)]" or "only [platform(s)]" or "just [platform(s)]" — even while also removing others — set platforms to ONLY the explicitly desired platform(s). "Don't want LinkedIn, I want Facebook" means platforms: ["facebook"], NOT all-minus-linkedin. The phrase "I want X" defines the complete desired set.
→ If user ONLY says to remove platform(s) without naming what they want: remove those from the existing list
→ If user says "add [platform]" / "also [platform]": platforms MUST include ALL (existing + new)
→ If user says "change platform" / "different platform" WITHOUT specifying which one: set call_agents: ["content_genius"], platforms: [], direct_reply: null, followUpIntent: "platform_change" — the system will automatically show connected platforms as selectable buttons
→ If content already exists AND changing platforms: call_agents: ["content_genius"], direct_reply: null (must regenerate for new platforms)
→ If only removing a platform (no new ones): call_agents: [], generate natural acknowledgement
→ Exception: if user explicitly requests images/video as part of the platform change (e.g., "facebook with image", "with images"), also include "media_wizard" in call_agents

### Rule 7 -- More images / media variants
→ call_agents: ["media_wizard"]

### Rule 8 -- Scheduling WITH a time
→ call_agents: ["scheduler_boss"]
→ Set context.rawScheduleRequest to the time string

### Rule 9 -- Scheduling WITHOUT a time
User says "schedule it" / "post this" but gives NO time:
→ Generate a natural reply asking when they'd like to schedule
→ call_agents: []

### Rule 10 -- Reschedule
→ call_agents: ["scheduler_boss"], set context.rawScheduleRequest

### Rule 11 -- Skip scheduling
User says "skip", "not now", "maybe later":
→ Generate a natural reply confirming content is saved for later
→ call_agents: []

## Knowledge Base / document references (IMPORTANT):
When the user refers to uploaded data using generic phrases like:
- "the data I provided", "my knowledge", "the info I uploaded"
- "use my content", "based on what I shared", "from my materials"
- "the stuff I uploaded", "what I gave you", "my uploaded files"
- "the document", "the pdf", "this file"
These ALL mean the user wants to use their Knowledge Base (uploaded PDFs, docs, URLs).
Treat these as actionable content requests with the topic being the referenced knowledge.
Set context.contentTopic to a description like "content based on uploaded knowledge base documents".
Set context.useKnowledgeBase to true.
Do NOT treat these as vague or off-topic — the Knowledge Keeper agent will handle retrieval.

## Guided flow & mandatory inputs:

For content creation, these are mandatory:
- Content topic (what to post about) — NOTE: referencing uploaded knowledge/data counts as having a topic
- Platforms (which social networks)
- Content format (text only / text + image / text + video)

Ask ONLY for what is missing. NEVER re-ask for details you already have.
If ALL details are present → set call_agents and generate, do NOT ask more questions.

## Platform extraction:
Extract platforms from message, normalise "twitter" → "x".
Always populate context.platforms with detected platforms even in clarification replies.

## Content type:
Default "image". Set "video" ONLY for explicit video/reel/shorts/TikTok requests.

## Output (STRICT -- return ONLY this JSON, nothing else):
{
  "direct_reply": null,
  "call_agents": [],
  "sub_tasks": [],
  "context": {
    "platforms": [],
    "contentType": "image",
    "contentTopic": null,
    "followUpIntent": "unknown",
    "topicSwitched": false,
    "newTopic": null,
    "editInstruction": null,
    "regenerate": "none",
    "useKnowledgeBase": false,
    "rawScheduleRequest": null,
    "scheduleAt": null,
    "platformSchedule": null
  }
}`;

let cachedLlm = null;
function getRouterLlm() {
  if (!cachedLlm) {
    cachedLlm = new ChatOpenAI({
      model: process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini',
      temperature: 0.2,
      apiKey: config.openai.apiKey,
    });
  }
  return cachedLlm;
}

export async function runOrchestrator(userMessage, stateSummary = '', { signal } = {}) {
  const enrichedMessage = stateSummary ? `${stateSummary}\n${userMessage}` : userMessage;

  console.log('\n[Orchestrator] === PROMPT SENT TO ROUTER LLM ===');
  console.log('[Orchestrator] userMessage length:', userMessage?.length);
  if (stateSummary) console.log('[Orchestrator] stateSummary:\n' + stateSummary);
  console.log('[Orchestrator] Exact content:\n' + userMessage + '\n===============================================');

  const llm = getRouterLlm();
  const response = await llm.invoke(
    [new SystemMessage(ROUTER_SYSTEM), new HumanMessage(enrichedMessage)],
    signal ? { signal } : undefined,
  );
  const text = typeof response.content === 'string' ? response.content : response.content?.[0]?.text || '{}';
  const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
  console.log('[Orchestrator] LLM raw response:', text?.slice(0, 500) + (text?.length > 500 ? '...' : ''));

  try {
    const parsed = JSON.parse(cleaned);
    console.log('[Orchestrator] LLM JSON parsed:', JSON.stringify(parsed, null, 2));

    // LLM sometimes returns direct_reply as an object like {"text": "..."} instead of a string — normalise it.
    let directReply = parsed.direct_reply ?? null;
    if (directReply && typeof directReply === 'object') {
      directReply = directReply.text || directReply.message || JSON.stringify(directReply);
    }
    if (directReply != null) directReply = String(directReply);
    let callAgents = Array.isArray(parsed.call_agents) ? parsed.call_agents : [];
    let context = typeof parsed.context === 'object' ? parsed.context : {};

    const platformAliases = {
      twitter: 'x',
      fb: 'facebook',
      ig: 'instagram',
      yt: 'youtube',
      tiktok_short: 'tiktok',
      linkedin_company: 'linkedin',
    };

    // Normalise platform names to lowercase and resolve aliases (e.g. "Facebook" → "facebook", "twitter" → "x")
    if (Array.isArray(context.platforms)) {
      context.platforms = context.platforms
        .map(p => {
          const lower = String(p || '').toLowerCase().trim();
          return platformAliases[lower] || lower;
        })
        .filter(p => p && p.length > 0);
    }

    if (context.platformSchedule && typeof context.platformSchedule === 'object') {
      const normalized = {};
      for (const [k, v] of Object.entries(context.platformSchedule)) {
        const key = platformAliases[k?.toLowerCase()] || k?.toLowerCase();
        if (key && v != null) normalized[key] = String(v).trim();
      }
      context = { ...context, platformSchedule: normalized };
    }

    const actualUserMessage = userMessage.includes('User now says:')
      ? userMessage.split('User now says:').pop().trim()
      : userMessage;

    // Safety strip: if user explicitly said no images/media, remove media_wizard regardless of LLM output
    const noImagesKeywords = /\b(without|no|don't|do not)\s*(any\s*)?(images?|pictures?|photos?|visuals?|videos?|reels?)|text\s*only|copy\s*only|caption\s*only|no\s*media|only\s*(content|text|copy|caption)|not\s*req\w*\s*(images?|media|pictures?|photos?|visuals?)|don'?t\s*(need|want|require)\s*(any\s*)?(images?|media|pictures?|photos?|visuals?)\b/i;
    if (noImagesKeywords.test(actualUserMessage)) {
      const selectedPlatforms = Array.isArray(context.platforms) ? context.platforms : [];
      const hasYouTube = selectedPlatforms.some(p => p === 'youtube' || p === 'youtube_short');
      const nonYouTubePlatforms = selectedPlatforms.filter(p => p !== 'youtube' && p !== 'youtube_short');
      const isYouTubeOnly = hasYouTube && nonYouTubePlatforms.length === 0;

      if (isYouTubeOnly) {
        // YouTube-only + "text only": still generate video since YouTube is video-first
        if (!callAgents.includes('media_wizard')) callAgents.push('media_wizard');
        context = {
          ...context,
          contentType: 'video',
          skipThumbnails: true,
        };
      } else if (hasYouTube && nonYouTubePlatforms.length > 0) {
        // Multi-platform with YouTube + "text only": remove YouTube from platforms,
        // skip media_wizard, generate text-only for remaining platforms
        callAgents = callAgents.filter(a => a !== 'media_wizard');
        context = {
          ...context,
          platforms: nonYouTubePlatforms,
          youtubeSkippedReason: 'text_only_requested',
          prefaceReply: "ℹ️ YouTube is a video-first platform and doesn't support text-only posts, so I've skipped it. Generating text content for your other platforms.",
        };
      } else {
        // No YouTube: just remove media_wizard
        callAgents = callAgents.filter(a => a !== 'media_wizard');
      }
    }

    // ── YouTube Confirmation Check ────────────────────────────────────────────
    // When YouTube is selected, we always need to ask the user whether they want
    // Short (9:16) or Regular/Full (16:9) video, since the aspect ratios differ.
    const selectedPlatforms = Array.isArray(context.platforms) ? context.platforms : [];
    const hasYouTube = selectedPlatforms.includes('youtube') || selectedPlatforms.includes('youtube_short');
    const nonYouTube = selectedPlatforms.filter(p => p !== 'youtube' && p !== 'youtube_short');
    const isYouTubeOnly = hasYouTube && nonYouTube.length === 0;
    const isImageRequest = context.contentType === 'image' || !context.contentType;
    const agentCallingMediaWizard = callAgents.includes('media_wizard');

    // YouTube-only + image request: warn that images can't be generated, ask for video format
    if (isYouTubeOnly && isImageRequest && agentCallingMediaWizard) {
      console.log('[Orchestrator] YOUTUBE CONFIRMATION REQUIRED: YouTube selected for image content request');

      directReply = "⚠️ I notice you selected YouTube. YouTube doesn't support image generation, but I can generate a video for you instead. Would you like me to generate a video for YouTube, or would you prefer to choose a different platform for images?";
      callAgents = [];

      context = {
        ...context,
        confirmationNeeded: 'youtube_video_only',
        pendingPlatforms: selectedPlatforms,
        pendingContentTopic: context.contentTopic,
      };
    }
    // Multi-platform with YouTube + no youtubeConfig yet: ask Short vs Regular before proceeding
    else if (hasYouTube && !isYouTubeOnly && agentCallingMediaWizard && !context.youtubeConfig) {
      console.log('[Orchestrator] YOUTUBE FORMAT REQUIRED: YouTube in multi-platform — asking Short vs Regular');

      directReply = "I see you've selected YouTube along with other platforms. For YouTube, I need to know the video format:\n\n**Regular Video** (landscape 16:9) — for longer content with a full title, description, and tags.\n**YouTube Short** (vertical 9:16) — for quick, punchy content under 60 seconds.\n\nImages will be generated for your other platforms automatically.";
      callAgents = [];

      context = {
        ...context,
        confirmationNeeded: 'youtube_format_multiplatform',
        pendingPlatforms: selectedPlatforms,
        pendingContentTopic: context.contentTopic,
      };
    }

    return {
      direct_reply: directReply,
      call_agents: callAgents,
      sub_tasks: Array.isArray(parsed.sub_tasks) ? parsed.sub_tasks : [],
      context,
    };
  } catch (e) {
    console.error('[Orchestrator] Parse error:', e.message);
    return {
      direct_reply: "I'm sorry, I couldn't process that. Try asking to create a post, generate an image, or schedule content.",
      call_agents: [], sub_tasks: [], context: {},
    };
  }
}

export default runOrchestrator;

// ─────────────────────────────────────────────────────────────────────────────
// HUMAN-IN-THE-LOOP HELPERS
// ─────────────────────────────────────────────────────────────────────────────

let cachedVisionLlm = null;
function getVisionLlm() {
  if (!cachedVisionLlm) {
    cachedVisionLlm = new ChatOpenAI({
      model: 'gpt-4o', temperature: 0.1,
      apiKey: config.openai.apiKey, maxTokens: 600,
    });
  }
  return cachedVisionLlm;
}

const VISION_SYSTEM = `You are a media analysis specialist for a social media marketing platform.
Analyze the provided image and return a JSON object ONLY (no markdown, no code fences) with this exact shape:
{
  "description": "concise 1-2 sentence description of what the image shows",
  "suggestedActions": ["create_post", "use_as_media"],
  "detectedObjects": ["product", "person", "logo"],
  "brandMatch": true,
  "mediaType": "image",
  "extractedText": null
}
Return ONLY the JSON object.`;

export async function analyzeMediaWithVision(mediaInput, userPrompt = '') {
  if (!mediaInput) return null;

  let effectiveMedia = mediaInput;

  // For video inputs, sample a frame via ffmpeg and analyze that frame as an image.
  if (mediaInput.mimeType?.startsWith('video/')) {
    try {
      if (mediaInput.url) {
        console.log('[Orchestrator] Video input detected -- extracting frame via ffmpeg from URL');
        const frameBuffer = await extractFrame(mediaInput.url, 1);
        effectiveMedia = {
          buffer: frameBuffer,
          mimeType: 'image/jpeg',
        };
      } else {
        console.log('[Orchestrator] Video input has no URL -- skipping frame extraction, falling back to generic video description');
        return {
          description: 'Video file uploaded by user (could not extract a frame from this upload).',
          suggestedActions: ['create_post', 'use_as_media', 'regenerate_media'],
          detectedObjects: [],
          brandMatch: true,
          mediaType: 'video',
          extractedText: null,
        };
      }
    } catch (e) {
      console.warn('[Orchestrator] Video frame extraction failed, falling back to generic video description:', e.message);
      return {
        description: 'Video file uploaded by user (frame extraction failed on the backend).',
        suggestedActions: ['create_post', 'use_as_media', 'regenerate_media'],
        detectedObjects: [],
        brandMatch: true,
        mediaType: 'video',
        extractedText: null,
      };
    }
  }

  console.log('[Orchestrator] analyzeMediaWithVision mediaInput:', { hasBuffer: !!effectiveMedia.buffer, url: effectiveMedia.url ? effectiveMedia.url.slice(0, 120) : undefined, mimeType: effectiveMedia.mimeType });
  let imageUrlBlock;
  try {
    if (effectiveMedia.buffer) {
      const base64 = Buffer.from(effectiveMedia.buffer).toString('base64');
      imageUrlBlock = { type: 'image_url', image_url: { url: `data:${effectiveMedia.mimeType};base64,${base64}`, detail: 'high' } };
    } else if (effectiveMedia.url) {
      if (effectiveMedia.url.startsWith('blob:')) {
        console.log('[Orchestrator] Skipping vision analysis for browser blob: URL');
        return null; // OpenAI cannot fetch local blob URLs
      }
      imageUrlBlock = { type: 'image_url', image_url: { url: effectiveMedia.url, detail: 'high' } };
    } else return null;
  } catch (e) { console.warn('[Orchestrator] Could not prepare media for vision:', e.message); return null; }

  try {
    const visionLlm = getVisionLlm();
    const response = await visionLlm.invoke([
      new SystemMessage(VISION_SYSTEM),
      new HumanMessage({
        content: [
          { type: 'text', text: userPrompt ? `User context: "${String(userPrompt).slice(0, 200)}"` : 'No context.' },
          imageUrlBlock,
        ]
      }),
    ]);
    const text = typeof response.content === 'string' ? response.content : response.content?.[0]?.text || '{}';
    const parsed = JSON.parse(text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim());
    console.log('[Orchestrator] Vision analysis done:', JSON.stringify(parsed).slice(0, 200));
    return parsed;
  } catch (e) {
    console.warn('[Orchestrator] Vision analysis failed:', e.message);
    return { description: 'Media file uploaded by user.', suggestedActions: ['create_post', 'use_as_media'], detectedObjects: [], brandMatch: true, mediaType: 'image', extractedText: null };
  }
}

const CLEAR_INTENT_PATTERNS = [
  /create\s+(a\s+)?(post|caption|thread|content)\s+(for|on|about)/i,
  /generate\s+(an?\s+)?(image|video|reel|carousel)\s+(for|about|of)/i,
  /write\s+(a\s+)?(caption|post|copy|thread)/i,
  /make\s+(a\s+)?(post|caption|image|video)\s+(for|about)/i,
  /schedule\s+(this|the|my|it)\s+(post|content|image)/i,
  /post\s+(this|it|about|on)\s+/i,
];

export function detectIntentClarity(userMessage, orchestratorResult) {
  if (orchestratorResult.direct_reply) return 'clear';
  // If the LLM decided to call agents, trust it -- the LLM is the sole intent classifier.
  // The old word-count check ("add instagram" = 2 words → ambiguous) was blocking valid short commands.
  if (!orchestratorResult.call_agents || orchestratorResult.call_agents.length === 0) return 'ambiguous';
  return 'clear';
}

/**
 * Step 1 of guided flow: ask which platforms to post on.
 * Shows only connected platforms as selectable buttons.
 * @param {string[]} connectedPlatforms - from DB
 */
export function buildPlatformSelectionOptions(connectedPlatforms = []) {
  const ALL_PLATFORMS = [
    { id: 'instagram', label: 'Instagram', icon: 'instagram' },
    { id: 'x', label: 'X (Twitter)', icon: 'x' },
    { id: 'linkedin', label: 'LinkedIn', icon: 'linkedin' },
    { id: 'facebook', label: 'Facebook', icon: 'facebook' },
    { id: 'tiktok', label: 'TikTok', icon: 'tiktok' },
    { id: 'youtube', label: 'YouTube', icon: 'youtube' },
  ];

  // Only include platforms actually connected
  const filtered = connectedPlatforms.length > 0
    ? ALL_PLATFORMS.filter(p => connectedPlatforms.includes(p.id))
    : ALL_PLATFORMS;

  return [
    {
      type: 'platform_multi_select',
      label: 'Select platforms to post on',
      action: 'platforms_selected',  // resume action
      options: filtered.map(p => ({ id: p.id, label: p.label, icon: p.icon })),
      defaultSelected: filtered.map(p => p.id), // pre-select all connected
    },
  ];
}

/**
 * Step 2 of guided flow: ask what type of content to create.
 */
export function buildContentTypeOptions() {
  return [
    { action: 'content_type_selected', label: 'Images + Content', icon: 'image', contentType: 'image', includeMedia: true },
    // Use a unique contentType value so the frontend can default-select this button.
    { action: 'content_type_selected', label: 'Text only', icon: 'text', contentType: 'text_only', includeMedia: false },
    { action: 'content_type_selected', label: 'Video + Content', icon: 'video', contentType: 'video', includeMedia: true },
  ];
}

/**
 * Options shown when user uploads media or intent is ambiguous.
 */
export function buildConfirmationOptions(hasMedia, orchestratorResult, mediaAnalysis, context) {
  const connectedPlatforms = context?.activePlatforms?.length
    ? context.activePlatforms
    : ['instagram', 'x', 'linkedin', 'facebook', 'tiktok', 'youtube'];

  const platformsForDefault =
    Array.isArray(context?.platforms)
      ? context.platforms
      : (Array.isArray(context?.activePlatforms) ? context.activePlatforms : []);
  const isYouTubeOnly = platformsForDefault.length === 1 && platformsForDefault[0] === 'youtube';

  const platformSelector = {
    type: 'platform_selector',
    label: 'Select platforms',
    options: connectedPlatforms,
    defaultSelected: connectedPlatforms,
  };

  // Linked accounts (account holder / page / organization names).
  // Provided by workflow via context.activeAccounts (array of { platform, accountId, accountName, type }).
  // Frontend can render this as a separate section under "Linked accounts".
  const activeAccounts = Array.isArray(context?.activeAccounts) ? context.activeAccounts : [];
  const linkedAccountSelector = activeAccounts.length > 0 ? {
    type: 'linked_account_selector',
    label: 'Linked accounts',
    // Keep it simple & explicit; UI can group by platform.
    options: activeAccounts
      .filter(a => a && a.accountId != null && a.accountName)
      .map(a => ({
        id: String(a.accountId),
        label: String(a.accountName),
        platform: String(a.platform || '').toLowerCase(),
        accountType: a.type ? String(a.type) : undefined,
      })),
    // Default to "all" so existing behaviour stays permissive.
    defaultSelected: activeAccounts
      .filter(a => a && a.accountId != null)
      .map(a => String(a.accountId)),
  } : null;
  const contentTypeSelector = {
    type: 'content_type_selector',
    label: 'Content type',
    options: ['image', 'video', 'carousel'],
    defaultSelected: context?.contentType || 'image',
  };

  /** Buttons: Images + Content, Text only, Video + Content (for confirm panel) */
  const contentTypeButtons = {
    type: 'content_type_buttons',
    label: 'What to create',
    options: buildContentTypeOptions(),
    defaultSelected: isYouTubeOnly ? 'text_only' : (context?.contentType || 'image'),
  };

  if (hasMedia) {
    return [
      { action: 'create_post', label: 'Create a post using this media', icon: 'post' },
      { action: 'use_as_media', label: 'Use this media directly in posts', icon: 'image' },
      { action: 'regenerate_media', label: 'Regenerate a similar image using this as reference', icon: 'regenerate' },
      { action: 'extract_content', label: 'Extract text or ideas from this media', icon: 'extract' },
      { action: 'analyze_brand', label: 'Check brand consistency of this media', icon: 'brand' },
      platformSelector,
      ...(linkedAccountSelector ? [linkedAccountSelector] : []),
      contentTypeSelector,
      contentTypeButtons,
    ];
  }
  return [
    { action: 'create_post', label: 'Create a social media post about this', icon: 'post' },
    { action: 'generate_media', label: 'Generate visual content for this topic', icon: 'image' },
    { action: 'schedule', label: 'Schedule existing content', icon: 'calendar' },
    platformSelector,
    ...(linkedAccountSelector ? [linkedAccountSelector] : []),
    contentTypeSelector,
    contentTypeButtons,
  ];
}

/**
 * Options shown when NO social accounts are connected.
 */
export function buildPlatformConnectionOptions() {
  return [
    { id: 'instagram', label: 'Instagram', icon: 'instagram' },
    { id: 'x', label: 'X (Twitter)', icon: 'x' },
    { id: 'linkedin', label: 'LinkedIn', icon: 'linkedin' },
    { id: 'facebook', label: 'Facebook', icon: 'facebook' },
    { id: 'tiktok', label: 'TikTok', icon: 'tiktok' },
    { id: 'youtube', label: 'YouTube', icon: 'youtube' },
  ].map(p => ({
    action: 'connect_platform',
    platform: p.id,
    label: `Connect ${p.label}`,
    icon: p.icon,
    type: 'platform_connect',
  }));
}

/**
 * Options shown after content is generated -- proactively ask when to schedule.
 * schedule_custom sends scheduleAt: null → frontend shows calendar picker.
 */
export function buildSchedulingOptions() {
  const now = new Date();
  const tomorrowMorning = new Date(now);
  tomorrowMorning.setDate(tomorrowMorning.getDate() + 1);
  tomorrowMorning.setUTCHours(9, 0, 0, 0);
  const tomorrowEvening = new Date(now);
  tomorrowEvening.setDate(tomorrowEvening.getDate() + 1);
  tomorrowEvening.setUTCHours(18, 0, 0, 0);

  return [
    { action: 'schedule_now', label: 'Post now', icon: 'lightning', scheduleAt: now.toISOString() },
    { action: 'schedule_later', label: 'Tomorrow morning (9 AM)', icon: 'sunrise', scheduleAt: tomorrowMorning.toISOString() },
    { action: 'schedule_later', label: 'Tomorrow evening (6 PM)', icon: 'sunset', scheduleAt: tomorrowEvening.toISOString() },
    { action: 'schedule_custom', label: 'Pick a custom time', icon: 'calendar', scheduleAt: null }, // frontend shows calendar picker
    { action: 'skip_schedule', label: 'Skip for now', icon: 'skip', scheduleAt: null },
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// CONTEXTUAL REPLY GENERATOR
// ─────────────────────────────────────────────────────────────────────────────

const CONTEXTUAL_REPLY_SYSTEM = `You are ZunoSync, a friendly and helpful social media content assistant.
Generate a SHORT, conversational, on-brand reply for the given scenario.
Keep replies concise (1-3 sentences), natural, and context-aware.
Never use excessive emojis (max 1 per reply). No markdown unless the scenario requires it.
Be warm, helpful, and direct.`;

/**
 * Generate a contextual LLM reply for workflow scenarios (replaces hardcoded static strings).
 * @param {string} scenario - Description of the situation
 * @param {object} data - Relevant context data
 * @returns {Promise<string|null>} - Generated reply or null on failure
 */
export async function generateContextualReply(scenario, data = {}) {
  try {
    const llm = getRouterLlm();
    const dataStr = Object.keys(data).length > 0 ? `\nContext data: ${JSON.stringify(data)}` : '';
    const prompt = `Scenario: ${scenario}${dataStr}`;
    const response = await llm.invoke([
      new SystemMessage(CONTEXTUAL_REPLY_SYSTEM),
      new HumanMessage(prompt),
    ]);
    const text = typeof response.content === 'string' ? response.content : response.content?.[0]?.text || '';
    return text.trim() || null;
  } catch (e) {
    console.warn('[Orchestrator] generateContextualReply failed:', e.message);
    return null;
  }
}