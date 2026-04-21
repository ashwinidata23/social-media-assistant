/**
 * Media Wizard Agent
 * Role: Designer / Video Maker
 * Generates images, Reels, carousels, GIFs; applies exact Brand Kit (colors/logo/fonts).
 * Trigger: Post/content requests (default) or explicit image requests.
 * Flow: User prompt → LLM restructure → 3 parallel API calls (1:1, 16:9, 9:16).
 * Tools: xAI Grok Imagine API + Brand Kit tool + OpenAI for prompt enhancement
 */

import path from 'path';
import crypto from 'crypto';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { getBrandKit } from '../tools/brand-kit-tool.js';
import { fetchImageBuffer, compositeLogoOnImage } from '../lib/image-variants.js';
import { uploadBufferToS3, isS3Configured } from '../lib/s3-upload.js';
import { config } from '../../config/index.js';
import fs from 'fs/promises';
import { generateVideo, generateVideoForAspectRatio } from '../lib/generate-video.js';
import { getSaliencyScore, scoreImageVariants, formatSaliencyForChat } from '../lib/saliency.js';


const GROK_IMAGINE_URL = config.xai.imagineUrl || 'https://api.x.ai/v1/images/generations';
const XAI_KEY = config.xai.apiKey;

/** System prompt for turning any social media content or topic into a professional, topic-relevant marketing image prompt. */
const PROMPT_RESTRUCTURE_SYSTEM = `You are a professional marketing creative director who specializes in visual advertising. Your job is to read whatever text the user gives (a post caption, a topic, a product description, an instruction) and produce a single, detailed image generation prompt that a photographer or illustrator would use for a polished marketing campaign.

STEP 1 — Understand the topic: Extract the core subject or theme from the user's text. It could be a service, product, event, concept, or industry. Do not get distracted by promotional language — find the actual subject.

STEP 2 — Choose the right visual: Think like a marketing creative director. What image would best represent this topic in a professional advertisement? Be concrete and specific to the real subject. Examples of correct thinking (do NOT copy these, use them as a thinking pattern only):
  - Topic about AI in software → developer at a modern workstation, holographic code, clean office
  - Topic about a new product launch → product hero shot, clean studio background, dramatic lighting
  - Topic about a service → real people benefiting from the service, authentic setting
  - Abstract concepts → metaphorical but recognizable real-world imagery, not random 3D shapes

STEP 3 — Write the prompt: Combine subject + setting + visual style + lighting + composition into one concise prompt.

OUTPUT RULES:
- Output ONLY the final image prompt. No explanation, no JSON, no preamble.
- Style: photorealistic or clean modern digital illustration — whichever fits the topic naturally.
- NEVER use abstract 3D shapes, random geometric objects, or neon colors as the main subject.
- NEVER include text, words, logos, or typography in the image.
- Always add these composition cues for layout stability across all aspect ratios: "centered composition, main subject in center, balanced framing, clean background, professional lighting, high resolution, no text".`;

let cachedPromptLlm = null;

function getPromptLlm() {
  if (!cachedPromptLlm) {
    cachedPromptLlm = new ChatOpenAI({
      model: process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini',
      temperature: 0.3,
      apiKey: config.openai?.apiKey,
    });
  }
  return cachedPromptLlm;
}

/**
 * Restructure user prompt via LLM for stable, layout-aware image generation across aspect ratios.
 * @param {string} userPrompt - Raw user input (e.g. "Generate a futuristic cyberpunk city")
 * @returns {Promise<string>} - Enhanced prompt (e.g. "Futuristic cyberpunk city at night. Neon lights. Ultra detailed. Centered composition...")
 */
export async function restructurePromptForImage(userPrompt) {
  const raw = String(userPrompt ?? '').trim();
  if (!raw) return raw;
  const openaiKey = config.openai?.apiKey;
  if (!openaiKey) {
    console.warn('[MediaWizard] OPENAI_API_KEY not set; using raw user prompt for image generation.');
    return raw;
  }
  try {
    const llm = getPromptLlm();
    const response = await llm.invoke([
      new SystemMessage(PROMPT_RESTRUCTURE_SYSTEM),
      new HumanMessage(raw),
    ]);
    const text = typeof response.content === 'string' ? response.content : response.content?.[0]?.text ?? '';
    const enhanced = text.replace(/^["']|["']$/g, '').trim();
    if (enhanced) {
      console.log('[MediaWizard] Prompt restructured for layout stability:', enhanced.slice(0, 120) + (enhanced.length > 120 ? '...' : ''));
      return enhanced;
    }
  } catch (e) {
    console.warn('[MediaWizard] Prompt restructure failed, using raw prompt:', e?.message || e);
  }
  return raw;
}

/** Generate a single seed for this run (for reproducibility when API supports it). */
function generateSeed() {
  return Math.floor(Math.random() * 1_000_000);
}

/** Recommended aspect ratio per platform (for feed images). Used when platforms are passed to runMediaWizard. */
const PLATFORM_ASPECT_RATIO = {
  x: '1:1',
  instagram: '1:1',
  linkedin: '1:1',
  facebook: '1:1',
  tiktok: '9:16',
  youtube: '16:9',
  youtube_short: '9:16',
  youtube_shorts: '9:16', // legacy
};

function normalizePlatformId(p) {
  const lower = String(p || '').toLowerCase().trim();
  if (lower === 'twitter') return 'x';
  if (lower === 'youtube_shorts') return 'youtube_short';
  return lower;
}

function pickFirst(arr) {
  return Array.isArray(arr) && arr.length ? arr[0] : null;
}

function buildBrandContext(brandKit) {
  if (!brandKit) return '';
  const parts = [];
  // Use tone/mood language — NOT raw hex colors (they cause Grok to render literal neon 3D shapes)
  if (brandKit.tone) parts.push(`Style tone: ${brandKit.tone}`);
  if (brandKit.primaryColor) {
    // Map hex to a descriptive mood hint — keep it subtle so it influences atmosphere, not subject matter
    parts.push(`Color mood: modern, vibrant, professional`);
  }
  const primary = String(brandKit.primaryColor || '').trim();
  const secondary = String(brandKit.secondaryColor || '').trim();
  if (primary || secondary) {
    // Apply both brand colors as palette guidance while preventing hard geometric/logo-like rendering.
    // We still include the actual hex values so DB brand-kit colors are directly represented in generation input.
    const palette = [primary, secondary].filter(Boolean).join(' and ');
    parts.push(`Brand color palette: ${palette}`);
    parts.push('Use these as subtle accent tones in background, props, and wardrobe; keep visuals natural and realistic');
    parts.push('Do not render color swatches, geometric 3D shapes, or isolated abstract color objects');
  }
  if (!parts.length) return '';
  return `Brand context: ${parts.join('. ')}.`;
}

function buildPromptConstraints({ avoidText = true } = {}) {
  const parts = [];
  if (avoidText) {
    parts.push(
      'No text/typography: do not include any words, letters, numbers, captions, subtitles, watermarks, logos, or signatures.'
    );
  }
  if (!parts.length) return '';
  return `Constraints: ${parts.join(' ')}`;
}

/**
 * Get YouTube-specific configuration based on video type and metadata.
 * @param {object} youtubeConfig - { videoType, title, description }
 *   - videoType: 'short' (9:16) or 'full' (16:9), defaults to 'full'
 *   - title: Video title for YouTube metadata (max 100 chars)
 *   - description: Video description for YouTube metadata (max 5000 chars)
 * @returns {object} - { aspectRatio, duration, metadata, videoType }
 */
function getYouTubeVideoConfig(youtubeConfig = {}) {
  let {
    videoType = 'full', // 'short' (YouTube Shorts) or 'full' (regular video)
    title = '',
    description = '',
  } = youtubeConfig;

  // Validate videoType (only 'short' or 'full' allowed)
  if (!['short', 'full'].includes(videoType)) {
    console.warn(`[MediaWizard] Invalid videoType "${videoType}", using default "full"`);
    videoType = 'full';
  }

  // Validate & truncate title (YouTube max: 100 characters)
  title = (title || '').trim();
  if (title.length > 100) {
    console.warn(`[MediaWizard] YouTube title truncated from ${title.length} to 100 characters`);
    title = title.slice(0, 100);
  }

  // Validate & truncate description (YouTube max: 5000 characters)
  description = (description || '').trim();
  if (description.length > 5000) {
    console.warn(`[MediaWizard] YouTube description truncated from ${description.length} to 5000 characters`);
    description = description.slice(0, 5000);
  }

  const isShort = videoType === 'short';
  const aspectRatio = isShort ? '9:16' : '16:9';
  const duration = isShort ? 60 : 10; // Shorts: max 60s, regular: ~10s

  return {
    videoType,
    aspectRatio,
    duration,
    metadata: {
      title,
      description,
      videoType,
    },
  };
}

function buildFullPrompt(userPrompt, { injectBrandKit = false, avoidText = true, brandKit: brandKitOverride } = {}) {
  const base = String(userPrompt ?? '').trim();
  const brandKit = brandKitOverride ?? getBrandKit();
  const brandContext = injectBrandKit && brandKit ? buildBrandContext(brandKit) : '';
  if (brandContext) {
    console.log('[MediaWizard] Brand kit applied to image prompt:', brandContext.slice(0, 120) + (brandContext.length > 120 ? '…' : ''));
  }
  const constraints = buildPromptConstraints({ avoidText });

  const blocks = [base, brandContext, constraints].map((s) => String(s || '').trim()).filter(Boolean);
  return { prompt: blocks.join('\n\n'), brandKit };
}

async function callImagineApi(body) {
  const res = await fetch(GROK_IMAGINE_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${XAI_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    return { error: err, images: [], raw: null, request: body };
  }

  const data = await res.json();
  const images = (data.data || []).map((d) => d.url || d.b64_json).filter(Boolean);
  return { images, raw: data, request: body, error: null };
}

/**
 * Call xAI image-edits API (image-to-image) with a source image URL.
 * Endpoint: POST /v1/images/edits
 * @param {object} body - { model, prompt, image: { url, type }, n, aspect_ratio, ... }
 */
async function callImageEditsApi(body) {
  const editsUrl = GROK_IMAGINE_URL.replace('/images/generations', '/images/edits');
  const res = await fetch(editsUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${XAI_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    return { error: err, images: [], raw: null, request: body };
  }

  const data = await res.json();
  const images = (data.data || []).map((d) => d.url || d.b64_json).filter(Boolean);
  return { images, raw: data, request: body, error: null };
}

/** True if the value looks like raw base64 (no protocol, not a path). */
function isBase64Data(value) {
  if (typeof value !== 'string' || !value) return false;
  const trimmed = value.trim();
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://') || trimmed.startsWith('/')) return false;
  return /^[A-Za-z0-9+/=]+$/.test(trimmed) && trimmed.length > 100;
}

/**
 * When the API returns b64_json, we must not send raw base64 to the client (it shows as "encoded" text).
 * Prefer saving to public/generated and returning file URLs; if aspectRatio missing or save fails (e.g. serverless), use data URI.
 */
async function normalizeImageToUrl(imageValue, aspectRatio, publicDirAbs) {
  if (!imageValue || !isBase64Data(imageValue)) return imageValue;
  if (aspectRatio && publicDirAbs) {
    try {
      const { saveAspectRatioVariantsFromBase64 } = await import('../lib/image-variants.js');
      const { urlsByAspectRatio } = await saveAspectRatioVariantsFromBase64({
        base64: imageValue,
        aspectRatios: [aspectRatio],
        publicDirAbs,
      });
      const url = urlsByAspectRatio[aspectRatio];
      if (url) return url;
    } catch (e) {
      console.warn('[MediaWizard] Could not save base64 to file, using data URI:', e?.message || e);
    }
  }
  return `data:image/png;base64,${imageValue}`;
}

/**
 * Call Grok Imagine API for text-to-image.
 * @param {string} prompt - Image description (Brand Kit can be injected into prompt)
 * @param {object} options - { n, aspectRatio, resolution, responseFormat, injectBrandKit, avoidText, seed }
 */
export async function generateImage(prompt, options = {}) {
  if (!XAI_KEY) {
    return { error: 'XAI_API_KEY not set', images: [], raw: null, request: null };
  }

  const { prompt: fullPrompt } = buildFullPrompt(prompt, {
    injectBrandKit: options.injectBrandKit,
    avoidText: options.avoidText,
    brandKit: options.brandKit,
  });

  const body = {
    model: process.env.GROK_IMAGINE_MODEL || 'grok-imagine-image',
    prompt: fullPrompt,
    n: options.n ?? 1,
    ...(options.aspectRatio ? { aspect_ratio: options.aspectRatio } : {}),
    ...(options.resolution ? { resolution: options.resolution } : {}),
    ...(options.responseFormat ? { response_format: options.responseFormat } : {}),
    // xAI Grok Imagine does not support seed yet; add to body when supported for reproducibility
  };

  const result = await callImagineApi(body);
  if (result.error == null && options.seed != null) result.seed = options.seed;
  return result;
}

// NOTE: We intentionally removed the single-call + local-crop path.
// All image generation now uses generateImageMultiAspect (3 parallel Grok Imagine calls).

/**
 * Generate the same prompt in multiple aspect ratios in parallel (same seed for consistency when API supports it).
 * @param {string} prompt - Image prompt (prefer LLM-restructured for layout stability)
 * @param {object} options - { aspectRatios, n, resolution, responseFormat, injectBrandKit, avoidText, seed }
 */
export async function generateImageMultiAspect(prompt, options = {}) {
  if (!XAI_KEY) {
    return { error: 'XAI_API_KEY not set', imagesByAspectRatio: {}, rawByAspectRatio: {}, requestByAspectRatio: {} };
  }

  const aspectRatios = options.aspectRatios?.length ? options.aspectRatios : ['1:1', '16:9', '9:16'];
  const seed = options.seed != null ? options.seed : generateSeed();

  const { prompt: fullPrompt } = buildFullPrompt(prompt, {
    injectBrandKit: options.injectBrandKit,
    avoidText: options.avoidText,
    brandKit: options.brandKit,
  });

  const logoUrl = options.brandKit?.logoUrl || null;

  // Always use text-to-image (3 parallel Grok calls). Logo is composited locally after generation.
  const results = await Promise.all(
    aspectRatios.map(async (aspectRatio) => {
      const body = {
        model: process.env.GROK_IMAGINE_MODEL || 'grok-imagine-image',
        prompt: fullPrompt,
        n: options.n ?? 1,
        aspect_ratio: aspectRatio,
        ...(options.resolution ? { resolution: options.resolution } : {}),
        ...(options.responseFormat ? { response_format: options.responseFormat } : {}),
      };
      const r = await callImagineApi(body);
      return { aspectRatio, seed, ...r };
    })
  );

  const imagesByAspectRatio = {};
  const rawByAspectRatio = {};
  const requestByAspectRatio = {};
  const errorsByAspectRatio = {};

  const publicDirAbs = path.join(process.cwd(), 'public');
  for (const r of results) {
    const rawImages = r.images || [];
    const normalized = await Promise.all(
      rawImages.map((img) => normalizeImageToUrl(img, r.aspectRatio, publicDirAbs))
    );
    imagesByAspectRatio[r.aspectRatio] = normalized;
    rawByAspectRatio[r.aspectRatio] = r.raw;
    requestByAspectRatio[r.aspectRatio] = r.request;
    if (r.error) errorsByAspectRatio[r.aspectRatio] = r.error;
  }

  // --- Image Post-Processing (Logo + S3/Local Save) ---
  // We now always process images to ensure they are uploaded to S3 (returning pre-signed URLs)
  // or saved locally, instead of returning temporary x.ai URLs.
  try {
    const logoBuffer = logoUrl ? await fetchImageBuffer(logoUrl) : null;
    const userId = options?.userId || null;
    const workspaceId = options?.workspaceId || null;

    for (const aspectRatio of aspectRatios) {
      const urls = imagesByAspectRatio[aspectRatio] || [];
      imagesByAspectRatio[aspectRatio] = await Promise.all(
        urls.map(async (imgUrl) => {
          try {
            // Step 1: Download generated image to buffer
            const imgBuffer = imgUrl.startsWith('data:')
              ? Buffer.from(imgUrl.split(',')[1], 'base64')
              : await fetchImageBuffer(imgUrl);

            // Step 2: Composite logo if present
            let processedBuffer = imgBuffer;
            if (logoBuffer) {
              processedBuffer = await compositeLogoOnImage(imgBuffer, logoBuffer);
              console.log('[MediaWizard] Logo composited on image.');
            }

            // Step 3: Upload to S3 (or fall back to local)
            let finalUrl;
            if (isS3Configured()) {
              finalUrl = await uploadBufferToS3(processedBuffer, {
                contentType: 'image/png',
                userId,
                workspaceId,
                aspectRatio,
              });
              console.log('[MediaWizard] Image uploaded to S3:', finalUrl.slice(0, 100));
            } else {
              // Fallback: save locally if S3 not configured
              console.warn('[MediaWizard] S3 not configured — saving locally as fallback.');
              const slug = aspectRatio.replace(/[^0-9]+/g, '-').replace(/^-+|-+$/g, '');
              const fname = `${crypto.randomUUID()}${logoBuffer ? '_logo' : ''}_${slug}.png`;
              const outDirAbs = path.join(process.cwd(), 'public', 'generated');
              await fs.mkdir(outDirAbs, { recursive: true });
              await fs.writeFile(path.join(outDirAbs, fname), processedBuffer);
              finalUrl = `/generated/${fname}`;
              console.log('[MediaWizard] Image saved locally (fallback):', finalUrl);
            }

            return finalUrl;
          } catch (e) {
            console.warn('[MediaWizard] Image post-processing failed, using original URL:', e?.message || e);
            return imgUrl; // fallback to original Grok URL
          }
        })
      );
    }
  } catch (e) {
    console.warn('[MediaWizard] Post-processing initialization failed:', e?.message || e);
  }

  const allImages = aspectRatios.flatMap((ar) => imagesByAspectRatio[ar] || []);
  const error = Object.keys(errorsByAspectRatio).length ? errorsByAspectRatio : null;

  return { imagesByAspectRatio, allImages, seed, error, errorsByAspectRatio, rawByAspectRatio, requestByAspectRatio };
}

/**
 * Media Wizard: interpret user ask and produce visuals (image/Reel/carousel/GIF).
 * For Reels/video we return a placeholder structure; real video API would be wired here.
 */
export async function runMediaWizard(userPrompt, context = {}) {
  const {
    contentType = 'image',
    count = 1,
    aspectRatios: contextAspectRatios,
    platforms,
    resolution,
    responseFormat,
    injectBrandKit = false,
    avoidText = true,
    brandKit: contextBrandKit,
    userId,
    workspaceId,
  } = context;

  const normalizedPlatforms = Array.isArray(platforms)
    ? platforms.map(normalizePlatformId).filter(Boolean)
    : [];

  // YouTube: Generate ONLY video, NO images (thumbnails)
  // Other platforms: Generate images as needed
  const hasYouTubeFull = normalizedPlatforms.includes('youtube');
  const hasYouTubeShort = normalizedPlatforms.includes('youtube_short');
  const hasAnyYouTube = hasYouTubeFull || hasYouTubeShort;
  const imagePlatforms = hasAnyYouTube
    ? normalizedPlatforms.filter((p) => p !== 'youtube' && p !== 'youtube_short')
    : normalizedPlatforms;

  // For image generation: always request 3 aspect ratios (1:1, 16:9, 9:16) for 3 parallel API calls unless explicitly overridden.
  const DEFAULT_IMAGE_ASPECT_RATIOS = ['1:1', '16:9', '9:16'];
  let aspectRatios = contextAspectRatios;
  if (contentType === 'image') {
    if (!aspectRatios?.length) {
      // When platforms are provided, derive aspect ratios strictly from those platforms.
      // Only fall back to DEFAULT_IMAGE_ASPECT_RATIOS when there is no explicit platform context.
      if (imagePlatforms.length) {
        aspectRatios = [...new Set(imagePlatforms.map((p) => PLATFORM_ASPECT_RATIO[p] || '1:1'))];
      } else {
        aspectRatios = DEFAULT_IMAGE_ASPECT_RATIOS;
      }
    }
  } else if (!aspectRatios?.length && Array.isArray(platforms) && platforms.length) {
    const ratios = [...new Set(platforms.map((p) => PLATFORM_ASPECT_RATIO[p] || '1:1'))];
    aspectRatios = ratios.length ? ratios : ['1:1'];
  }
  if (!aspectRatios?.length) aspectRatios = ['1:1'];

  let result;
  if (contentType === 'image') {
    // YouTube only: Skip image generation, video only
    const youtubeOnly = hasAnyYouTube && imagePlatforms.length === 0;

    if (youtubeOnly) {
      console.log('[MediaWizard] ⚠️ YouTube selected: Cannot generate images for YouTube. Video generation only.');
    }

    if (!youtubeOnly) {
      // Generate images for non-YouTube platforms
      const enhancedPrompt = await restructurePromptForImage(userPrompt);
      const seed = generateSeed();
      result = await generateImageMultiAspect(enhancedPrompt, {
        n: count,
        aspectRatios,
        resolution,
        responseFormat,
        injectBrandKit,
        avoidText,
        brandKit: contextBrandKit,
        seed,
        userId,
        workspaceId,
      });
    } else {
      // YouTube only: No images, only video
      const seed = generateSeed();
      result = {
        imagesByAspectRatio: {},
        allImages: [],
        seed,
        error: null,
        errorsByAspectRatio: {},
        rawByAspectRatio: {},
        requestByAspectRatio: {},
      };
    }

  } else if (['video', 'reel', 'tiktok', 'shorts', 'carousel'].includes(contentType)) {
    console.log('[MediaWizard] Generating video | videoPurpose:', context.videoPurpose);

    const VIDEO_ASPECT_RATIO_MAP = {
      reel: ['9:16'],
      regular: ['16:9'],
      both: ['9:16', '16:9', '1:1'],
    };
    const VIDEO_ASPECT_RATIO_LABELS = {
      '9:16': '9:16 (Reel/Short)',
      '16:9': '16:9 (Landscape)',
      '1:1': '1:1 (Square)',
    };

    const videoPurpose = context.videoPurpose || 'regular';

    // ── Separate YouTube from non-YouTube platforms ──────────────────────
    // When contentType is 'video' but there are both YouTube and non-YouTube
    // platforms, generate video ONLY for YouTube and images for the rest.
    const videoPlatforms = normalizedPlatforms.filter(
      (p) => p === 'youtube' || p === 'youtube_short'
    );
    const nonVideoPlatforms = normalizedPlatforms.filter(
      (p) => p !== 'youtube' && p !== 'youtube_short'
    );
    const hasNonVideoPlatforms = nonVideoPlatforms.length > 0;
    const hasVideoPlatforms = videoPlatforms.length > 0;

    // Determine which aspect ratios to generate video for:
    // - If YouTube + other platforms: video ONLY for YouTube aspect ratios
    // - If YouTube only or no platforms specified: use all platform aspect ratios
    let videoAspectRatios;
    if (hasVideoPlatforms && hasNonVideoPlatforms) {
      // Multi-platform: video only for YouTube, images for others.
      // Respect youtubeConfig.videoType when available (short=9:16, full=16:9, both=both).
      const configVideoType = context.youtubeConfig?.videoType;
      let fromYouTube;
      if (configVideoType === 'short') {
        fromYouTube = ['9:16'];
      } else if (configVideoType === 'both') {
        fromYouTube = ['9:16', '16:9'];
      } else if (configVideoType === 'full') {
        fromYouTube = ['16:9'];
      } else {
        fromYouTube = [...new Set(videoPlatforms.map((p) => PLATFORM_ASPECT_RATIO[p] || '16:9'))];
      }
      videoAspectRatios = fromYouTube.length ? fromYouTube : VIDEO_ASPECT_RATIO_MAP[videoPurpose] || VIDEO_ASPECT_RATIO_MAP.regular;
      console.log('[MediaWizard] Multi-platform video: YouTube video aspect ratios:', videoAspectRatios, '(youtubeConfig:', configVideoType || 'none', ') | Image platforms:', nonVideoPlatforms);
    } else if (hasVideoPlatforms && !hasNonVideoPlatforms) {
      // YouTube only: also respect youtubeConfig.videoType
      const configVideoType = context.youtubeConfig?.videoType;
      if (configVideoType === 'short') {
        videoAspectRatios = ['9:16'];
      } else if (configVideoType === 'both') {
        videoAspectRatios = ['9:16', '16:9'];
      } else if (configVideoType === 'full') {
        videoAspectRatios = ['16:9'];
      } else {
        const fromPlatforms = [...new Set(videoPlatforms.map((p) => PLATFORM_ASPECT_RATIO[p] || '16:9'))];
        videoAspectRatios = fromPlatforms.length ? fromPlatforms : VIDEO_ASPECT_RATIO_MAP[videoPurpose] || VIDEO_ASPECT_RATIO_MAP.regular;
      }
      console.log('[MediaWizard] YouTube-only video: aspect ratios:', videoAspectRatios, '(youtubeConfig:', configVideoType || 'none', ')');
    } else if (Array.isArray(platforms) && platforms.length) {
      const fromPlatforms = [...new Set(platforms.map((p) => PLATFORM_ASPECT_RATIO[p] || '16:9'))];
      videoAspectRatios = fromPlatforms.length ? fromPlatforms : VIDEO_ASPECT_RATIO_MAP[videoPurpose] || VIDEO_ASPECT_RATIO_MAP.regular;
    } else {
      videoAspectRatios = VIDEO_ASPECT_RATIO_MAP[videoPurpose] || VIDEO_ASPECT_RATIO_MAP.regular;
    }

    // ── Step 1: Generate video aspect ratios in parallel ──────────────
    const videoResults = await Promise.allSettled(
      videoAspectRatios.map(ratio =>
        generateVideoForAspectRatio(userPrompt, { aspectRatio: ratio, duration: 6 })
      )
    );

    const videoUrls = [];
    const videoVariants = [];
    const videoErrors = [];

    for (const r of videoResults) {
      if (r.status === 'fulfilled') {
        const { url, error, aspectRatio } = r.value;
        if (url) {
          videoUrls.push(url);
          videoVariants.push({
            url,
            aspectRatio,
            label: VIDEO_ASPECT_RATIO_LABELS[aspectRatio] || aspectRatio,
          });
          console.log(`[MediaWizard] Video ready (${aspectRatio}):`, url.slice(0, 80));
        } else {
          console.error(`[MediaWizard] Video failed (${aspectRatio}):`, error);
          videoErrors.push(`${aspectRatio}: ${error}`);
        }
      } else {
        videoErrors.push(r.reason?.message || 'unknown error');
      }
    }

    // ── Step 2: Generate images for non-YouTube platforms ──────────────
    // When YouTube is combined with other platforms, those other platforms
    // should receive images, not videos.
    let nonVideoImageResult = null;
    if (hasVideoPlatforms && hasNonVideoPlatforms) {
      console.log('[MediaWizard] Generating images for non-YouTube platforms:', nonVideoPlatforms);
      const imageAspectRatios = [...new Set(nonVideoPlatforms.map((p) => PLATFORM_ASPECT_RATIO[p] || '1:1'))];
      const enhancedPrompt = await restructurePromptForImage(userPrompt);
      const seed = generateSeed();
      nonVideoImageResult = await generateImageMultiAspect(enhancedPrompt, {
        n: count,
        aspectRatios: imageAspectRatios,
        injectBrandKit,
        avoidText,
        brandKit: contextBrandKit,
        seed,
        userId,
        workspaceId,
      });
    }

    // Thumbnails skipped — video only.
    const thumbnailUrl = null;
    const thumbnailSaliency = null;

    result = {
      videoUrls,
      videoVariants,
      thumbnailUrl,
      thumbnailSaliency,
      nonVideoImageResult,
      error: videoUrls.length === 0 ? (videoErrors[0] || 'All video variants failed') : null,
      request: { prompt: userPrompt, type: contentType, videoPurpose },
    };
  } else {
    result = await generateImage(userPrompt, {
      n: count,
      resolution,
      responseFormat,
      injectBrandKit,
      avoidText,
      brandKit: contextBrandKit,
    });
    // Single-call path may return base64; normalize so client gets displayable URLs
    if (result.images?.length && !result.allImages) {
      result.images = await Promise.all(
        result.images.map((img) => normalizeImageToUrl(img, null, null))
      );
    }
  }
  const outputs = [];
  const warnings = [];

  if (contentType === 'image') {
    if (result.allImages?.length) {
      // Emit platform-tagged image outputs when we know the platforms.
      // This enables platform removal ("remove twitter") to prune images too.
      if (imagePlatforms.length) {
        for (const p of imagePlatforms) {
          const ratio = PLATFORM_ASPECT_RATIO[p] || '1:1';
          const url =
            pickFirst(result.imagesByAspectRatio?.[ratio]) ||
            pickFirst(result.imagesByAspectRatio?.['1:1']) ||
            pickFirst(result.imagesByAspectRatio?.['9:16']) ||
            pickFirst(result.imagesByAspectRatio?.['16:9']) ||
            pickFirst(result.allImages) ||
            null;
          outputs.push({
            type: 'image',
            platform: p,
            aspectRatio: ratio,
            aspectRatios,
            urlsByAspectRatio: result.imagesByAspectRatio,
            urls: url ? [url] : [],
            ...(result.seed != null ? { seed: result.seed } : {}),
          });
        }
      } else {
        outputs.push({
          type: 'image',
          aspectRatios,
          urlsByAspectRatio: result.imagesByAspectRatio,
          urls: result.allImages,
          ...(result.seed != null ? { seed: result.seed } : {}),
        });
      }
    } else if (result.images?.length) {
      outputs.push({ type: 'image', urls: result.images });
    }

    // YouTube selected: generate video ONLY (no images)
    if (hasAnyYouTube) {
      try {
        console.log('[MediaWizard] YouTube selected — generating video only (no images).');

        const requestedTypes = new Set();

        // youtubeConfig.videoType from the format selection UI takes priority:
        // 'short' → 9:16 only, 'full' → 16:9 only, 'both' → both formats
        const configVideoType = context.youtubeConfig?.videoType;
        if (configVideoType === 'both') {
          requestedTypes.add('full');
          requestedTypes.add('short');
        } else if (configVideoType === 'short') {
          requestedTypes.add('short');
        } else if (configVideoType === 'full') {
          requestedTypes.add('full');
        } else {
          // No explicit config — infer from platform list
          if (hasYouTubeFull) requestedTypes.add('full');
          if (hasYouTubeShort) requestedTypes.add('short');
          // Fallback: default to full
          if (!requestedTypes.size) requestedTypes.add('full');
        }

        for (const videoType of requestedTypes) {
          const youtubeConfig = getYouTubeVideoConfig({ ...(context.youtubeConfig || {}), videoType });
          console.log(`[MediaWizard] YouTube ${youtubeConfig.videoType} video | Aspect: ${youtubeConfig.aspectRatio} | Duration: ${youtubeConfig.duration}s`);

          const ytVideo = await generateVideoForAspectRatio(userPrompt, {
            aspectRatio: youtubeConfig.aspectRatio,
            duration: youtubeConfig.duration,
          });

          if (ytVideo?.url) {
            outputs.push({
              type: 'video',
              platform: youtubeConfig.videoType === 'short' ? 'youtube_short' : 'youtube',
              urls: [ytVideo.url],
              videoType: youtubeConfig.videoType,
              variants: [{
                url: ytVideo.url,
                aspectRatio: youtubeConfig.aspectRatio,
                label: youtubeConfig.videoType === 'short' ? '9:16 (YouTube Shorts)' : '16:9 (YouTube)',
              }],
              thumbnailUrl: null,
              metadata: youtubeConfig.metadata,
            });
          } else {
            const errMsg = ytVideo?.error || 'unknown error';
            console.warn('[MediaWizard] YouTube video generation failed:', errMsg);
            warnings.push({
              type: 'error',
              message: `⚠️ I couldn't generate a video for YouTube${youtubeConfig.videoType === 'short' ? ' Shorts' : ''}. The remaining platform content has been generated successfully. Would you like me to retry the YouTube video?`,
              platform: youtubeConfig.videoType === 'short' ? 'youtube_short' : 'youtube',
              error: errMsg,
            });
          }
        }
      } catch (e) {
        const errMsg = e?.message || String(e);
        console.warn('[MediaWizard] YouTube video generation failed:', errMsg);
        warnings.push({
          type: 'error',
          message: `⚠️ I couldn't generate a video for YouTube due to an error. The remaining platform content has been generated successfully. Would you like me to retry?`,
          platform: 'youtube',
          error: errMsg,
        });
      }
    }
  } else if (['video', 'reel', 'tiktok', 'shorts', 'carousel'].includes(contentType)) {
    if (result.videoUrls?.length) {
      outputs.push({
        type: 'video',
        urls: result.videoUrls,
        variants: result.videoVariants,          // [{ url, aspectRatio, label }]
        thumbnailUrl: result.thumbnailUrl || null,   // AI-generated still at 16:9
        thumbnailSaliency: result.thumbnailSaliency || null,
      });
    }
    // Include image outputs for non-YouTube platforms when video was generated
    // alongside other platforms (YouTube gets video, others get images).
    if (result.nonVideoImageResult?.allImages?.length) {
      const imgResult = result.nonVideoImageResult;
      const nonVideoPlatforms = normalizedPlatforms.filter(
        (p) => p !== 'youtube' && p !== 'youtube_short'
      );
      if (nonVideoPlatforms.length) {
        for (const p of nonVideoPlatforms) {
          const ratio = PLATFORM_ASPECT_RATIO[p] || '1:1';
          const url =
            pickFirst(imgResult.imagesByAspectRatio?.[ratio]) ||
            pickFirst(imgResult.imagesByAspectRatio?.['1:1']) ||
            pickFirst(imgResult.imagesByAspectRatio?.['9:16']) ||
            pickFirst(imgResult.imagesByAspectRatio?.['16:9']) ||
            pickFirst(imgResult.allImages) ||
            null;
          outputs.push({
            type: 'image',
            platform: p,
            aspectRatio: ratio,
            aspectRatios: Object.keys(imgResult.imagesByAspectRatio || {}),
            urlsByAspectRatio: imgResult.imagesByAspectRatio,
            urls: url ? [url] : [],
            ...(imgResult.seed != null ? { seed: imgResult.seed } : {}),
          });
        }
        console.log('[MediaWizard] Added image outputs for non-YouTube platforms:', nonVideoPlatforms);
      }
    }
  } else if (result.images?.length) {
    outputs.push({ type: 'image', urls: result.images });
  }

  // ── Image Saliency Scoring ────────────────────────────────────────────────
  // Score the 16:9 variant as the primary thumbnail candidate (fastest, most common)
  if (contentType === 'image' && result.imagesByAspectRatio) {
    const primaryUrl =
      result.imagesByAspectRatio['16:9']?.[0] ||
      result.imagesByAspectRatio['1:1']?.[0] ||
      result.allImages?.[0] ||
      null;

    if (primaryUrl) {
      try {
        const saliency = await getSaliencyScore(primaryUrl, 'social media post image');
        result.saliency = saliency;
        console.log(`[MediaWizard] Image saliency score: ${saliency?.score}/10`);
      } catch (e) {
        console.warn('[MediaWizard] Image saliency scoring failed:', e?.message || e);
      }
    }
  }

  // Prepare warnings/info messages for user
  const youtubeOnly = hasAnyYouTube && imagePlatforms.length === 0;

  // Inform user if YouTube can't generate images
  if (hasAnyYouTube && youtubeOnly && contentType === 'image') {
    warnings.push({
      type: 'info',
      message: "⚠️ YouTube doesn't support image generation. I can only generate video for YouTube. For thumbnails, use Instagram or another image platform.",
      platform: 'youtube'
    });
  } else if (hasAnyYouTube && imagePlatforms.length > 0 && contentType === 'image') {
    warnings.push({
      type: 'info',
      message: "ℹ️ YouTube platform selected: Generating video only (no images). Images are generated for other selected platforms.",
      platform: 'youtube'
    });
  }

  return {
    outputs,
    brandKit: getBrandKit(),
    error: result.error,
    warnings: warnings.length > 0 ? warnings : null,
    raw: result.raw || result.rawByAspectRatio,
    request: result.request || result.requestByAspectRatio,
    saliency: result.saliency || null,                     // for images
    ...(contentType === 'image' && result.seed != null ? { seed: result.seed } : {}),
  };
}

export default runMediaWizard;
