/**
 * Content Genius Agent
 * Role: Copywriter & Strategist
 * Writes captions, threads, scripts; applies Brand Voice + Knowledge Base; hashtag + emoji optimization.
 * Trigger: Any prompt with text needed.
 * Tools: OpenAI + Knowledge Keeper tool
 */

import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { config } from '../../config/index.js';
import { getBrandKit } from '../tools/brand-kit-tool.js';

// Normalize context platform names to Content Genius supported types
const PLATFORM_ALIAS = {
  instagram_caption: 'instagram',
  instagram_bio: 'instagram',
  youtube_shorts: 'youtube_short', // normalize to youtube_short
};

// Hard character limits per platform (enforced after LLM output)
const PLATFORM_CHAR_LIMITS = {
  linkedin: 3000,
  instagram: 2200,
  x: 280,
  facebook: 63206,
  tiktok: 2200,
  youtube: 5000,
  youtube_short: 5000,
};

const contentGeniusSystemPrompt = `You are Content Genius -- the expert copywriter and platform adapter inside ZunoSync.

CRITICAL -- GROUNDING IN USER'S OWN CONTENT:
- You MUST base all generated content ONLY on the "User's own content" provided below (Knowledge Base excerpts, Brand Voice, and any raw/copied text the user gave).
- Do NOT invent facts, claims, messaging, or tone that are not present in that material. If the user's content is empty or minimal, keep the output short and generic; do not fabricate details.
- Your job is to adapt, rephrase, and optimize THAT content for each platform -- not to create unrelated or generic copy.

You receive:
1. User's own content (Knowledge Base + Brand + any raw text they provided) -- USE THIS AS THE ONLY SOURCE
2. User prompt (what they want: e.g. "create a post about X", "adapt for LinkedIn")
3. List of connected accounts (platform types to generate for)

Your job is to:
- Use ONLY the user's own content as the source; rephrase and optimize it per platform
- If raw/long content is provided ("use this", "adapt this", paste) → summarize and rephrase from it only
- Generate ONE optimized version per connected platform, staying true to the source material

=== STRICT OUTPUT FORMAT ===
You MUST reply with ONLY a valid JSON object (no markdown, no explanation, no extra text):

{
  "platforms": [
    {
      "type": "linkedin",          // exact lowercase platform name
      "content": "Full optimized LinkedIn post text here..."
    },
    {
      "type": "instagram",
      "content": "Optimized Instagram caption here..."
    }
    // ... one object per connected platform
  ]
}

SPECIAL: For "youtube" and "youtube_short" platforms, use this EXTENDED format:
{
  "type": "youtube",
  "title": "Catchy video title (max 100 chars)",
  "description": "Full video description with keywords, links, call to action...",
  "tags": ["tag1", "tag2", "tag3"],
  "content": "title | description"
}
The "content" field for YouTube MUST be: title + " | " + description (for backward compatibility).
IMPORTANT: Do NOT include timestamps in YouTube descriptions. Only generate title, description, and tags.

Supported platform types (use exact lowercase):
linkedin
instagram
x
facebook
tiktok
youtube (regular video -- landscape 16:9)
youtube_short (YouTube Shorts -- vertical 9:16)

=== PLATFORM OPTIMIZATION RULES (2026 best practices) ===
linkedin: 1,200-1,300 chars max, professional, value-first, bullets OK
instagram: 125-150 chars, storytelling + emojis + line breaks
x: ≤280 chars, punchy hook first, max 3 hashtags
facebook: 200-400 chars, conversational + question
tiktok: short hook script (first 3s) + caption ≤150 chars
youtube: title (≤100 chars, curiosity-driven, keywords front-loaded) + description (500-1500 chars, first 2 lines are visible above "Show more", include relevant keywords, call to action, NO timestamps) + 5-10 tags
youtube_short: title (≤100 chars, punchy hook) + description (≤500 chars, short and snappy, 3-5 hashtags, NO timestamps) + 3-5 tags. Content should be optimized for vertical short-form: strong hook in first line, casual/energetic tone

Always maintain 100% brand voice from Brand Kit and facts from Knowledge Base.

If user says "HI" or casual greeting → still return JSON but with a single "welcome" entry for the first platform (you can ignore others).

Examples:

Input platforms: ["linkedin", "instagram", "x"]
Prompt: "Create motivational post about 5x productivity"
Output:
{
  "platforms": [
    { "type": "linkedin", "content": "Long professional version here..." },
    { "type": "instagram", "content": "Short emotional caption with emojis..." },
    { "type": "x", "content": "Punchy 280-char version..." }
  ]
}

Now process the current user prompt and connected accounts list. Return ONLY the JSON object.`;

function extractRequiredPhrases(text) {
  const t = String(text || '');
  const out = [];
  // Double quotes, curly quotes
  const quoted = /["""]([^"""]{2,120})["""]/g;
  let m;
  while ((m = quoted.exec(t)) !== null) out.push(m[1].trim());
  // Backticks
  const backticked = /`([^`]{2,120})`/g;
  while ((m = backticked.exec(t)) !== null) out.push(m[1].trim());
  // Dedupe + keep reasonable strings
  return Array.from(new Set(out)).filter((s) => s && s.length >= 2 && s.length <= 120);
}

function missingPhrases(content, required) {
  const c = String(content || '');
  return (required || []).filter((p) => p && !c.includes(p));
}

export async function runContentGenius(userPrompt, context = {}) {
  const {
    knowledgeChunks = [],
    platforms = [],
    rawContent,
    brandKit: contextBrandKit,
    existingContent,
    originalUserPrompt,
  } = context;
  const modelName = process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini';

  const brandKit = contextBrandKit ?? getBrandKit();
  const connectedAccounts = platforms.map((p) => PLATFORM_ALIAS[p] || p);

  const effectivePromptForConstraints = originalUserPrompt || userPrompt;
  const requiredPhrases = extractRequiredPhrases(effectivePromptForConstraints);

  const brandLine = `Brand voice/tone: ${brandKit?.tone || 'professional, friendly'}.`;
  const knowledgeLines = knowledgeChunks.length ? knowledgeChunks.join('\n\n') : '';
  const rawLines = rawContent ? String(rawContent).trim() : '';

  let mediaLines = '';
  if (context.mediaAnalysis) {
    mediaLines = `[UPLOADED MEDIA CONTEXT] The user uploaded an image/video. Describe or reference this in the post: ${context.mediaAnalysis.description}`;
    if (context.mediaAnalysis.extractedText) mediaLines += `\nText found in media: "${context.mediaAnalysis.extractedText}"`;
    if (context.mediaAnalysis.detectedObjects?.length) mediaLines += `\nDetected elements: ${context.mediaAnalysis.detectedObjects.join(', ')}`;
  }

  // Build existing content block so the LLM can edit rather than regenerate
  let existingContentLines = '';
  if (existingContent && typeof existingContent === 'object') {
    const entries = Object.entries(existingContent).filter(([, v]) => v);
    if (entries.length > 0) {
      existingContentLines = `\n\n=== EXISTING CONTENT (modify this -- do NOT regenerate from scratch) ===\n${entries.map(([platform, content]) => `[${platform}]: ${content}`).join('\n\n')}\n=== END EXISTING CONTENT ===`;
    }
  }

  const userOwnContentParts = [brandLine, mediaLines, knowledgeLines, rawLines].filter(Boolean);
  const userOwnContentBlock =
    userOwnContentParts.length > 0
      ? `\n\n=== USER'S OWN CONTENT (use this as the ONLY source -- do not add facts or tone not present here) ===\n${userOwnContentParts.join('\n\n')}\n=== END USER'S OWN CONTENT ===`
      : '\n\n(No user content provided -- generate only short, generic placeholder copy that does not invent specifics.)';

  const requiredBlock = requiredPhrases.length
    ? `\n\n=== REQUIRED EXACT STRINGS ===\nYou MUST include these exact strings verbatim in the output where relevant:\n- ${requiredPhrases.map((s) => JSON.stringify(s)).join('\n- ')}\n=== END REQUIRED EXACT STRINGS ===`
    : '';

  const disableYouTubeMetadata = context?.disableYouTubeMetadata === true;
  const system =
    contentGeniusSystemPrompt +
    (disableYouTubeMetadata
      ? `\n\nIMPORTANT: Do NOT generate YouTube-specific metadata fields (title/description/tags/timestamps). For youtube/youtube_short, return ONLY { "type": "...", "content": "..." } like other platforms.`
      : '') +
    userOwnContentBlock +
    existingContentLines +
    requiredBlock;
  const userContent = `Connected accounts (platforms to generate for): ${JSON.stringify(connectedAccounts)}.\n\nUser prompt: ${userPrompt}`;

  async function invokeOnce(temperature, extraInstruction = '') {
    const llm = new ChatOpenAI({
      model: modelName,
      temperature,
      apiKey: config.openai.apiKey,
    });
    const response = await llm.invoke([
      new SystemMessage(extraInstruction ? `${system}\n\n${extraInstruction}` : system),
      new HumanMessage(userContent),
    ]);
    const text = typeof response.content === 'string' ? response.content : response.content?.[0]?.text || '';
    const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    console.log('[ContentGenius] LLM raw response:', text?.slice(0, 500) + (text?.length > 500 ? '...' : ''));
    let result = [];
    try {
      const parsed = JSON.parse(cleaned);
      result = Array.isArray(parsed?.platforms) ? parsed.platforms : [];
      console.log('[ContentGenius] LLM JSON parsed:', JSON.stringify(parsed, null, 2));
    } catch (_) {
      result = [{ type: connectedAccounts[0] || 'x', content: text }];
      console.log('[ContentGenius] LLM JSON parse failed, using raw text as single platform');
    }
    return result;
  }

  let platformsOutput = await invokeOnce(0.7);

  // If user asked for exact strings, verify they are present; rerun once with stricter instruction.
  if (requiredPhrases.length) {
    const missingAny = platformsOutput.some((p) => missingPhrases(p.content, requiredPhrases).length > 0);
    if (missingAny) {
      console.log('[ContentGenius] Required phrases missing, retrying with stricter instruction');
      const extra = `CRITICAL: If any required exact string is missing, your answer is WRONG. Include every required exact string verbatim at least once in the appropriate platform content. Do NOT substitute synonyms.`;
      platformsOutput = await invokeOnce(0.3, extra);
    }
  }

  // Post-process YouTube platforms:
  // - default: ensure title/description/tags + backward-compat content
  // - disableYouTubeMetadata: strip metadata and keep only "content"
  platformsOutput = platformsOutput.map((p) => {
    if (p.type === 'youtube' || p.type === 'youtube_short') {
      if (disableYouTubeMetadata) {
        const content =
          typeof p.content === 'string' && p.content.trim()
            ? p.content
            : [p.title, p.description].filter(Boolean).join(' - ');
        return { type: p.type, content: String(content || '').trim() };
      }
      const title = p.title || String(p.content || '').slice(0, 100);
      const description = p.description || p.content || '';
      const tags = Array.isArray(p.tags) ? p.tags : [];
      const content = `${title} | ${description}`;
      return { ...p, title, description, tags, content };
    }
    return p;
  });

  // Enforce per-platform character limits and calculate actual character counts
  platformsOutput = platformsOutput.map((p) => {
    const limit = PLATFORM_CHAR_LIMITS[p.type];
    const originalLength = String(p.content || '').length;
    const truncated = limit && originalLength > limit
      ? String(p.content).slice(0, limit)
      : p.content;
    if (limit && originalLength > limit) {
      console.warn(`[ContentGenius] ${p.type} content truncated: ${originalLength} → ${limit} chars`);
    }
    return { ...p, content: truncated };
  });

  const copy = platformsOutput.length ? platformsOutput.map((p) => p.content).join('\n\n---\n\n') : '';

  const characterCounts = platformsOutput.map((p) => ({
    platform: p.type,
    count: String(p.content || '').length,
    limit: PLATFORM_CHAR_LIMITS[p.type] || null,
    withinLimit: PLATFORM_CHAR_LIMITS[p.type]
      ? String(p.content || '').length <= PLATFORM_CHAR_LIMITS[p.type]
      : true,
  }));

  return {
    copy,
    platforms: platformsOutput,
    characterCounts,
  };
}

export default runContentGenius;
