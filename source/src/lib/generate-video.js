
/**
 * ZunoSync Video Generation — xAI Grok
 * Generates a single video for a given aspect ratio.
 * Called in parallel from media-wizard.js for multiple aspect ratios.
 */

const BASE_URL = process.env.XAI_VIDEO_BASE_URL || 'https://api.x.ai/v1';
const MODEL = process.env.XAI_VIDEO_MODEL || 'grok-imagine-video';
const POLL_INTERVAL_MS = 5000;
const MAX_POLL_ATTEMPTS = 60;

/**
 * Generate a single video for one aspect ratio.
 * @param {string} prompt
 * @param {object} opts
 * @param {string} opts.aspectRatio  - '9:16' | '16:9' | '1:1'
 * @param {number} opts.duration     - seconds (default 6)
 * @returns {{ url: string, aspectRatio: string } | { error: string, aspectRatio: string }}
 */
export async function generateVideoForAspectRatio(prompt, { aspectRatio = '16:9', duration = 6 } = {}) {
    console.log(`[VideoGen] Starting | aspectRatio: ${aspectRatio} | prompt length: ${prompt?.length}`);

    const apiKey = process.env.XAI_API_KEY;
    if (!apiKey) return { error: 'XAI_API_KEY not set', aspectRatio };

    // ── Step 1: Submit ─────────────────────────────────────────────────────────
    const submitRes = await fetch(`${BASE_URL}/videos/generations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({
            model: MODEL,
            prompt: String(prompt).trim(),
            duration: Number(duration) || 6,
            aspect_ratio: String(aspectRatio),
        }),
    });

    const submitRaw = await submitRes.json().catch(() => ({}));
    if (!submitRes.ok) {
        const msg = submitRaw?.error?.message || submitRaw?.message || submitRes.statusText;
        console.error(`[VideoGen] Submit failed (${aspectRatio}):`, msg);
        return { error: `Submit failed: ${msg}`, aspectRatio };
    }

    const request_id = submitRaw?.request_id ?? submitRaw?.id;
    if (!request_id) {
        console.error(`[VideoGen] No request_id returned (${aspectRatio}):`, JSON.stringify(submitRaw));
        return { error: 'No request_id returned', aspectRatio };
    }
    console.log(`[VideoGen] Submitted (${aspectRatio}) | request_id: ${request_id}`);

    // ── Step 2: Poll ───────────────────────────────────────────────────────────
    for (let attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
        await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));

        const pollRes = await fetch(`${BASE_URL}/videos/${request_id}`, {
            headers: { 'Authorization': `Bearer ${apiKey}` },
        });

        const raw = await pollRes.json().catch(() => ({}));
        if (!pollRes.ok) {
            console.warn(`[VideoGen] Poll HTTP error (${aspectRatio}): ${pollRes.status}`);
            continue;
        }

        const status = (raw?.status || '').toLowerCase();
        console.log(`[VideoGen] Poll attempt ${attempt} (${aspectRatio}) | status: ${status}`);

        if (status === 'failed' || status === 'error') {
            return { error: raw?.error || raw?.message || 'Generation failed', aspectRatio };
        }

        if (status === 'done' || status === 'completed' || status === 'succeeded') {
            const url =
                raw?.video?.url ||
                raw?.video_url ||
                raw?.url ||
                raw?.output?.url ||
                raw?.output?.video_url ||
                raw?.data?.[0]?.url ||
                raw?.data?.[0]?.video_url ||
                raw?.result?.url ||
                raw?.result?.video_url ||
                null;

            if (url) {
                console.log(`[VideoGen] Done (${aspectRatio}) | url: ${url.slice(0, 80)}`);
                return { url, aspectRatio };
            }
            console.warn(`[VideoGen] Status complete but no URL (${aspectRatio}):`, JSON.stringify(raw).slice(0, 300));
            return { error: 'No video URL in response', aspectRatio };
        }
        // else: still processing, keep polling
    }

    return { error: 'Polling timed out', aspectRatio };
}

/**
 * Backward-compat wrapper — existing code importing generateVideo still works.
 */
export async function generateVideo(prompt, options = {}) {
    const { duration = 6, aspect_ratio = '16:9' } = options;
    const result = await generateVideoForAspectRatio(prompt, { aspectRatio: aspect_ratio, duration });
    if (result.error) return { error: result.error };
    return { videoUrl: result.url, requestId: null };
}

export default generateVideoForAspectRatio;