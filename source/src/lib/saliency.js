/**
 * ZunoSync Saliency Scorer
 * Uses GPT-4o Vision to analyze visual attention and thumbnail readiness.
 * Zero new dependencies — uses the same OpenAI client already in the project.
 */

import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

const SALIENCY_SYSTEM = `You are a visual attention analyst for social media marketing.
Your job is to evaluate how visually attention-grabbing an image is, especially as a social media thumbnail.

Analyze the image and return ONLY a valid JSON object — no markdown, no explanation:
{
  "score": 8.2,
  "dominant_region": "center",
  "focal_points": ["face", "bright color", "high contrast subject"],
  "weaknesses": ["cluttered background", "low contrast edges"],
  "thumbnail_ready": true,
  "suggestions": ["Add bold text overlay", "Crop tighter on subject"],
  "platform_scores": {
    "youtube": 8.5,
    "instagram": 7.8,
    "tiktok": 9.0,
    "linkedin": 6.5
  },
  "emotion": "exciting",
  "color_vibrancy": "high"
}

Scoring guide (score field, 0–10):
  9–10 : Instantly eye-catching, strong focal point, vibrant, professional
  7–8  : Good attention value, minor issues
  5–6  : Average, needs improvement
  3–4  : Weak, low contrast, busy or unclear subject
  0–2  : Very poor, likely to be ignored

dominant_region: where the eye goes first — "center" | "top-left" | "top-right" | "bottom" | "spread"
thumbnail_ready: true if score >= 7 and has a clear focal point
emotion: overall emotional tone — "exciting" | "calm" | "professional" | "fun" | "inspirational" | "neutral"
color_vibrancy: "high" | "medium" | "low"

Return ONLY the JSON object.`;

let cachedSaliencyLlm = null;
function getSaliencyLlm() {
    if (!cachedSaliencyLlm) {
        cachedSaliencyLlm = new ChatOpenAI({
            model: 'gpt-4o',
            temperature: 0.1,
            apiKey: process.env.OPENAI_API_KEY,
            maxTokens: 500,
        });
    }
    return cachedSaliencyLlm;
}

/**
 * Score an image URL for visual saliency / thumbnail readiness.
 * @param {string} imageUrl  - Public or pre-signed S3 URL
 * @param {string} [context] - Optional context ("youtube thumbnail", "instagram post")
 * @returns {Promise<SaliencyResult>}
 */
export async function getSaliencyScore(imageUrl, context = '') {
    if (!imageUrl) return null;
    if (!process.env.OPENAI_API_KEY) {
        console.warn('[Saliency] OPENAI_API_KEY not set — skipping saliency analysis');
        return null;
    }

    try {
        const llm = getSaliencyLlm();
        const userText = context
            ? `Analyze this image for visual saliency. Context: ${context}`
            : 'Analyze this image for visual saliency as a social media thumbnail.';

        const response = await llm.invoke([
            new SystemMessage(SALIENCY_SYSTEM),
            new HumanMessage({
                content: [
                    { type: 'text', text: userText },
                    { type: 'image_url', image_url: { url: imageUrl, detail: 'low' } }, // 'low' = faster + cheaper
                ],
            }),
        ]);

        const text = typeof response.content === 'string'
            ? response.content
            : response.content?.[0]?.text || '{}';

        const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        const parsed = JSON.parse(cleaned);
        console.log(`[Saliency] Score: ${parsed.score}/10 | thumbnail_ready: ${parsed.thumbnail_ready} | emotion: ${parsed.emotion}`);
        return parsed;
    } catch (e) {
        console.warn('[Saliency] Analysis failed:', e?.message || e);
        return null;
    }
}

/**
 * Score multiple images and return ranked results.
 * Useful when you have 3 aspect ratio variants and want to pick the best thumbnail.
 * @param {Array<{ url: string, aspectRatio: string, label: string }>} variants
 * @param {string} [context]
 * @returns {Promise<Array<{ url, aspectRatio, label, saliency }>>}
 */
export async function scoreImageVariants(variants = [], context = '') {
    if (!variants.length) return [];

    const results = await Promise.allSettled(
        variants.map(async (v) => {
            const saliency = await getSaliencyScore(v.url, context);
            return { ...v, saliency };
        })
    );

    return results
        .filter(r => r.status === 'fulfilled')
        .map(r => r.value)
        .sort((a, b) => (b.saliency?.score ?? 0) - (a.saliency?.score ?? 0)); // best first
}

/**
 * Format saliency result into a readable chat message section.
 * @param {SaliencyResult} saliency
 * @param {string} [label] - e.g. "Thumbnail" or "16:9 Image"
 */
export function formatSaliencyForChat(saliency, label = 'Image') {
    if (!saliency) return '';

    const scoreEmoji = saliency.score >= 8 ? '🟢' : saliency.score >= 6 ? '🟡' : '🔴';
    const thumbEmoji = saliency.thumbnail_ready ? '✅' : '⚠️';

    const lines = [
        `📊 **${label} Saliency: ${saliency.score}/10** ${scoreEmoji}`,
        `${thumbEmoji} Thumbnail ready: ${saliency.thumbnail_ready ? 'Yes' : 'Needs improvement'}`,
    ];

    if (saliency.focal_points?.length) {
        lines.push(`👁️ Focal points: ${saliency.focal_points.join(', ')}`);
    }
    if (saliency.emotion) {
        lines.push(`🎭 Tone: ${saliency.emotion} | Vibrancy: ${saliency.color_vibrancy}`);
    }
    if (saliency.weaknesses?.length) {
        lines.push(`⚠️ Weaknesses: ${saliency.weaknesses.join(', ')}`);
    }
    if (saliency.suggestions?.length) {
        lines.push(`💡 Tips: ${saliency.suggestions.join(' • ')}`);
    }

    return lines.join('\n');
}

export default getSaliencyScore;