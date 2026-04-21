/**
 * Preview Pro Agent
 * Role: UI/UX Mockup Expert
 * Renders pixel-perfect previews on IG/Reel/LinkedIn/X/TikTok etc.; shows expected metrics.
 * Trigger: After Content + Media are ready.
 * Tools: Internal component + Brand Kit
 */

import { getBrandKit } from '../tools/brand-kit-tool.js';

// Use the same platform IDs as the rest of the app (workflow/context/db).
const PLATFORMS = ['instagram', 'x', 'linkedin', 'facebook', 'tiktok', 'youtube', 'youtube_short'];

// Media Wizard returns aspect-ratio variants (not per-platform tagged outputs).
// Preview should pick the best-fit aspect ratio per platform.
const PLATFORM_ASPECT_RATIO = {
  instagram: '1:1',
  x: '1:1',
  linkedin: '1:1',
  facebook: '1:1',
  tiktok: '9:16',
  youtube: '16:9',
  youtube_short: '9:16',
};

// Per-platform caption character display limits for preview truncation
const PLATFORM_CAPTION_LIMITS = {
  instagram: 2200,
  x: 280,
  linkedin: 3000,
  facebook: 63206,
  tiktok: 2200,
  youtube: 5000,
  youtube_short: 5000,
};

// Data-driven engagement hints per platform and content type
const PLATFORM_METRICS = {
  instagram: {
    withMedia: { reach: 'High', time: '9-11 AM & 6-9 PM', hint: 'Visual-first content drives 2× more engagement on Instagram.' },
    textOnly: { reach: 'Medium', time: '9-11 AM', hint: 'Adding a carousel or image can significantly boost reach.' },
    video: { reach: 'Very High', time: '6-9 PM', hint: 'Reels receive up to 3× more reach than static posts.' },
  },
  x: {
    withMedia: { reach: 'Medium-High', time: '8-10 AM & 6-9 PM', hint: 'Tweets with images get 150% more retweets.' },
    textOnly: { reach: 'Medium', time: '8-10 AM', hint: 'Hook-first writing and trending hashtags boost visibility.' },
    video: { reach: 'High', time: '7-9 PM', hint: 'Video tweets drive strong engagement in fast-moving feeds.' },
  },
  linkedin: {
    withMedia: { reach: 'High', time: 'Tue-Thu 8-10 AM', hint: 'Document posts and carousels outperform standard updates.' },
    textOnly: { reach: 'High', time: 'Tue-Thu 8-10 AM', hint: 'Long-form professional insights drive strong B2B reach.' },
    video: { reach: 'Medium-High', time: 'Tue-Thu 9 AM', hint: 'Native video gets 5× more reach than shared links.' },
  },
  facebook: {
    withMedia: { reach: 'High', time: '1-4 PM', hint: 'Photo posts receive 2× more engagement than text-only.' },
    textOnly: { reach: 'Medium', time: '1-3 PM', hint: 'Asking a question in your post drives comment engagement.' },
    video: { reach: 'Very High', time: '1-4 PM', hint: 'Facebook video gets 6× more interactions than other post types.' },
  },
  tiktok: {
    withMedia: { reach: 'High', time: '7-9 PM', hint: 'Consistent posting and trending audio drive discoverability.' },
    textOnly: { reach: 'Low', time: '7-9 PM', hint: 'TikTok is video-first -- adding a video will dramatically boost reach.' },
    video: { reach: 'Very High', time: '7-9 PM', hint: 'Short-form video with trending sounds can reach millions organically.' },
  },
  youtube: {
    withMedia: { reach: 'Medium', time: '2-4 PM & 8-11 PM', hint: 'Thumbnails with faces and text overlays improve click-through rates.' },
    textOnly: { reach: 'Low', time: '2-4 PM', hint: 'YouTube is video-first -- a video will dramatically improve reach.' },
    video: { reach: 'High', time: '2-4 PM & 8-11 PM', hint: 'Longer watch time signals boost recommendations in the algorithm.' },
  },
  youtube_short: {
    withMedia: { reach: 'High', time: '7-9 PM', hint: 'YouTube Shorts with trending audio and quick hooks drive massive discoverability.' },
    textOnly: { reach: 'Low', time: '7-9 PM', hint: 'Shorts are video-first -- add a vertical video for best reach.' },
    video: { reach: 'Very High', time: '7-9 PM', hint: 'Shorts under 30 seconds with strong hooks get the most replays and shares.' },
  },
};

function getMetricsTeaser(platform, contentType, hasMedia) {
  const platformMetrics = PLATFORM_METRICS[String(platform).toLowerCase()];
  if (!platformMetrics) {
    return {
      expectedReach: 'Medium',
      bestPostingTime: '9-11 AM',
      engagementHint: 'Consistent posting builds audience over time.',
    };
  }
  let variant;
  if (contentType === 'video') {
    variant = platformMetrics.video;
  } else if (hasMedia) {
    variant = platformMetrics.withMedia;
  } else {
    variant = platformMetrics.textOnly;
  }
  return {
    expectedReach: variant.reach,
    bestPostingTime: variant.time,
    engagementHint: variant.hint,
  };
}

function normalizePlatformId(platform) {
  const lower = String(platform || '').toLowerCase().trim();
  if (lower === 'twitter') return 'x';
  return lower;
}

function pickFirstUrl(value) {
  if (Array.isArray(value) && value.length > 0) return value[0];
  if (typeof value === 'string' && value) return value;
  return null;
}

function pickMediaUrlForPlatform(platform, media) {
  const outputs = Array.isArray(media?.outputs) ? media.outputs : [];
  const p = normalizePlatformId(platform);

  // Prefer a platform-tagged output when available (newer Media Wizard behaviour).
  const platformOutput =
    outputs.find(o => normalizePlatformId(o?.platform) === p) ||
    null;

  // If outputs are not platform-tagged, don't blindly pick outputs[0].
  // It's common to have multiple outputs (e.g. a YouTube video + an Instagram image).
  // For image-first platforms, prefer an image output; for video-first platforms, prefer video.
  const wantsVideo =
    p === 'youtube' ||
    p === 'youtube_short' ||
    p === 'tiktok';

  const byType = (t) => outputs.find(o => String(o?.type || '').toLowerCase() === t) || null;
  const firstVideo = byType('video');
  const firstImage = byType('image');

  const firstOutput =
    platformOutput ||
    (wantsVideo ? (firstVideo || firstImage) : (firstImage || firstVideo)) ||
    outputs[0] ||
    null;
  const type = String(firstOutput?.type || '').toLowerCase();

  // Video: use thumbnail still if present so frontend can render an image preview.
  if (type === 'video') {
    return (
      pickFirstUrl(firstOutput?.thumbnailUrls) ||
      pickFirstUrl(firstOutput?.keyframeUrls) ||
      pickFirstUrl(firstOutput?.thumbnailUrl) ||
      pickFirstUrl(media?.thumbnailUrl) ||
      pickFirstUrl(firstOutput?.urls) ||
      pickFirstUrl(media?.urls) ||
      null
    );
  }

  // Image: prefer urlsByAspectRatio for the platform's recommended ratio.
  const ratio = PLATFORM_ASPECT_RATIO[p] || '1:1';
  const fromRatios =
    pickFirstUrl(firstOutput?.urlsByAspectRatio?.[ratio]) ||
    // Soft fallback order: try common ratios before "any".
    pickFirstUrl(firstOutput?.urlsByAspectRatio?.['1:1']) ||
    pickFirstUrl(firstOutput?.urlsByAspectRatio?.['9:16']) ||
    pickFirstUrl(firstOutput?.urlsByAspectRatio?.['16:9']) ||
    null;

  return (
    fromRatios ||
    pickFirstUrl(firstOutput?.urls) ||
    pickFirstUrl(media?.urls) ||
    pickFirstUrl(media?.keyframeUrls) ||
    null
  );
}

/**
 * Build mockup payload for a platform (for frontend to render).
 */
function buildMockup(platform, content, media, brandKit) {
  const kit = brandKit || getBrandKit();
  const p = normalizePlatformId(platform);
  const platformEntry = Array.isArray(content?.platforms)
    ? content.platforms.find(pl => normalizePlatformId(pl?.type) === p)
    : null;

  const caption = (platformEntry?.content || '') || content?.copy || content?.caption || '';

  // Apply per-platform caption character limit for preview display
  const charLimit = PLATFORM_CAPTION_LIMITS[p] || 500;
  const truncatedCaption = String(caption || '').slice(0, charLimit);

  const mediaUrl = pickMediaUrlForPlatform(platform, media);
  const hasMedia = !!mediaUrl;
  const outputs = Array.isArray(media?.outputs) ? media.outputs : [];
  const firstOutput = outputs[0] || null;
  const contentType = firstOutput?.type === 'video' ? 'video' : (kit?.contentType || 'image');

  const mockup = {
    platform,
    caption: truncatedCaption,
    mediaUrl,
    primaryColor: kit.primaryColor,
    secondaryColor: kit.secondaryColor,
    logoUrl: kit.logoUrl,
    fontHeading: kit.fontHeading,
    fontBody: kit.fontBody,
    metricsTeaser: getMetricsTeaser(p, contentType, hasMedia),
  };

  // Include structured YouTube fields for frontend rendering
  if ((p === 'youtube' || p === 'youtube_short') && platformEntry) {
    if (platformEntry.title) mockup.title = platformEntry.title;
    if (platformEntry.description) mockup.description = platformEntry.description;
    if (Array.isArray(platformEntry.tags)) mockup.tags = platformEntry.tags;
  }

  return mockup;
}

/**
 * Run Preview Pro: generate preview payloads for each requested platform.
 * Uses options.brandKit when provided (from workflow context), else getBrandKit().
 */
export async function runPreviewPro(contentResult, mediaResult, options = {}) {
  const platforms = options.platforms || PLATFORMS;
  const brandKit = options.brandKit || getBrandKit();
  const previews = platforms.map((platform) =>
    buildMockup(platform, contentResult, mediaResult, brandKit)
  );
  return {
    previews,
    platforms,
  };
}

export default runPreviewPro;
