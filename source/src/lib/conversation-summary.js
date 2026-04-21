/**
 * Conversation end summary: generate summary from recent turns, save to DB and return.
 * Used when user ends chat so they can "continue conversation" later.
 */

import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { config } from '../../config/index.js';
import * as db from './db.js';

const SUMMARY_PROMPT = `You are summarizing the end of a user's chat session with a social media marketing assistant.
Given the recent conversation turns (user and assistant messages), write a short end summary in 2-4 sentences.
Include: what the user asked for (e.g. captions, scheduling, platforms), what was created or scheduled if any, and any decisions or next steps.
Write in third person, past tense. Example: "The user requested captions for LinkedIn and Instagram. Content was generated and scheduled for tomorrow 9am. They chose to post on two platforms."
Reply with ONLY the summary text, no heading or extra wording.`;

let summaryLlm = null;
function getSummaryLlm() {
  if (!summaryLlm) {
    summaryLlm = new ChatOpenAI({
      model: process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini',
      temperature: 0.3,
      apiKey: config.openai?.apiKey,
    });
  }
  return summaryLlm;
}

/**
 * Generate a short end-of-chat summary from conversation turns.
 * @param {Array<{ role: string, content: string }>} turns
 * @returns {Promise<string>}
 */
export async function generateSummary(turns) {
  if (!Array.isArray(turns) || turns.length === 0) return 'Session ended. No summary.';
  const text = turns
    .slice(-30)
    .map((t) => `${t.role}: ${(t.content || '').slice(0, 500)}`)
    .join('\n');
  const llm = getSummaryLlm();
  const response = await llm.invoke([
    new SystemMessage(SUMMARY_PROMPT),
    new HumanMessage(text),
  ]);
  const out = typeof response.content === 'string' ? response.content : response.content?.[0]?.text || '';
  return out.trim() || 'Session ended.';
}

const SHORT_DESCRIPTION_PROMPT = `You are given an end-of-chat summary for a user's conversation with an AI social media marketing assistant.
Write a short, meaningful heading that captures the main intent or outcome of the conversation.
Constraints:
- 3 to 7 words
- Title case (Capitalize Main Words)
- No trailing punctuation
- Do NOT wrap in quotes or add labels.
Reply with ONLY the heading text.`;

async function buildShortDescriptionFromSummary(summary) {
  const s = String(summary || '').replace(/\s+/g, ' ').trim();
  if (!s) return '';

  try {
    const llm = getSummaryLlm();
    const response = await llm.invoke([
      new SystemMessage(SHORT_DESCRIPTION_PROMPT),
      new HumanMessage(s),
    ]);
    const raw =
      typeof response.content === 'string'
        ? response.content
        : response.content?.[0]?.text || '';

    let heading = String(raw || '').replace(/\s+/g, ' ').trim();
    if (!heading) throw new Error('empty heading');

    // Enforce 3–7 words maximum, just in case
    const words = heading.split(' ').filter(Boolean);
    if (words.length > 7) {
      heading = words.slice(0, 7).join(' ');
    }
    return heading;
  } catch (e) {
    console.warn('[conversation-summary] shortDescription generation failed, falling back:', e.message);
    // Fallback: simple first 7 words from summary
    const words = s.split(' ').filter(Boolean);
    return words.slice(0, 7).join(' ');
  }
}

/**
 * Save one conversation turn to DB (conversation_turns). No-op if DB not configured.
 * Optionally persists media URLs if a "media" column exists (e.g. text[] or jsonb).
 */
export async function saveConversationTurn(userId, threadId, role, content, media = null, conversationHistory = null) {
  if (!userId || !role || content == null) return;
  if (!db.isUsingPostgres()) return;
  try {
    const thread = (threadId || 'default').slice(0, 128);
    const safeRole = role === 'assistant' ? 'assistant' : 'user';
    const uid = String(userId).slice(0, 256);
    const textContent = String(content).slice(0, 50000);

    // Normalize media into an array of strings or null
    let mediaUrls = null;
    if (Array.isArray(media)) {
      mediaUrls = media.filter((u) => typeof u === 'string' && u.trim().length > 0);
      if (mediaUrls.length === 0) mediaUrls = null;
    }

    const historyJson = conversationHistory ? JSON.stringify(conversationHistory) : null;

    if (mediaUrls) {
      console.log(
        '[conversation-summary] Saving turn with media',
        '| userId:', uid,
        '| threadId:', thread,
        '| role:', safeRole,
        '| mediaCount:', mediaUrls.length
      );
    }

    await db.query(
      `INSERT INTO public.conversation_turns (user_id, thread_id, role, content, media, conversation_history)
       VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)`,
      [uid, thread, safeRole, textContent, mediaUrls ? JSON.stringify(mediaUrls) : null, historyJson]
    );

  } catch (e) {
    console.warn('[conversation-summary] saveTurn failed:', e.message);
  }
}

/**
 * Get all historical conversation turns for a user and thread from DB.
 * Returns them ordered chronologically.
 * @returns {Promise<Array<{ role: string, content: string }>>}
 */
export async function getHistoricalTurns(userId, threadId = 'default') {
  if (!userId) return [];
  if (!db.isUsingPostgres()) return [];
  try {
    const { rows } = await db.query(
      `SELECT role, content, media, conversation_history FROM public.conversation_turns 
       WHERE user_id = $1 AND thread_id = $2 
       ORDER BY created_at ASC`,
      [String(userId).slice(0, 256), (threadId || 'default').slice(0, 128)]
    );
    return rows;
  } catch (e) {
    console.warn('[conversation-summary] getHistoricalTurns failed:', e.message);
    return [];
  }
}

/**
 * Get the latest summary for a user/thread (optionally scoped by workspace) from DB.
 * @returns {Promise<string|null>}
 */
export async function getLatestSummary(userId, threadId = 'default', workspaceId) {
  if (!userId) return null;
  if (!db.isUsingPostgres()) return null;
  try {
    const uid = String(userId).slice(0, 256);
    const thread = (threadId || 'default').slice(0, 128);

    let query = `SELECT summary
                 FROM public.conversation_summaries
                 WHERE user_id = $1 AND thread_id = $2`;
    const params = [uid, thread];

    // Only filter by workspace_id if explicitly provided.
    if (workspaceId != null) {
      query += ' AND workspace_id = $3';
      params.push(String(workspaceId).slice(0, 256));
    }

    const { rows } = await db.query(`${query}
       ORDER BY updated_at DESC NULLS LAST, created_at DESC
       LIMIT 1`, params);
    return rows[0]?.summary ?? null;
  } catch (e) {
    console.warn('[conversation-summary] getLatestSummary failed:', e.message);
    return null;
  }
}

/**
 * Upsert (insert or update) the thread summary + short_description.
 * This is used for dynamic updates after each user+assistant exchange.
 *
 * NOTE: We intentionally ignore the passed `turns` argument for content,
 * and instead summarize the full conversation history for this user+thread
 * from `conversation_turns` so the summary reflects the whole thread.
 *
 * @returns {Promise<{ summary: string, shortDescription: string }>}
 */
export async function generateAndUpsertThreadSummary(userId, threadId, workspaceId, _turns) {
  const allTurns = await getHistoricalTurns(userId, threadId);
  const turnsForSummary = Array.isArray(allTurns) ? allTurns.slice(-100) : [];
  const summary = await generateSummary(turnsForSummary);
  const shortDescription = await buildShortDescriptionFromSummary(summary);

  const thread = (threadId || 'default').slice(0, 128);
  const uid = String(userId).slice(0, 256);
  const wid = workspaceId != null ? String(workspaceId).slice(0, 256) : null;

  if (db.isUsingPostgres()) {
    try {
      // Find the latest existing summary row for this user + thread (+ workspace when provided)
      let selectQuery = `SELECT id
                         FROM public.conversation_summaries
                         WHERE user_id = $1 AND thread_id = $2`;
      const selectParams = [uid, thread];
      if (wid != null) {
        selectQuery += ' AND workspace_id = $3';
        selectParams.push(wid);
      }
      selectQuery += ' ORDER BY updated_at DESC NULLS LAST, created_at DESC LIMIT 1';

      const { rows } = await db.query(selectQuery, selectParams);

      if (rows[0]?.id) {
        // Update latest row
        await db.query(
          `UPDATE public.conversation_summaries
           SET summary = $1,
               short_description = $2,
               updated_at = NOW()
           WHERE id = $3`,
          [summary, shortDescription, rows[0].id]
        );
        console.log(
          '[conversation-summary] Updated summary',
          '| userId:', uid,
          '| threadId:', thread,
          '| shortDescription:', shortDescription.slice(0, 120),
          '| summarySnippet:', summary.slice(0, 180)
        );
      } else {
        // Insert first summary row for this user/thread, including workspace when provided
        if (wid != null) {
          await db.query(
            `INSERT INTO public.conversation_summaries (user_id, thread_id, workspace_id, summary, short_description, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, NOW(), NOW())`,
            [uid, thread, wid, summary, shortDescription]
          );
        } else {
          await db.query(
            `INSERT INTO public.conversation_summaries (user_id, thread_id, summary, short_description, created_at, updated_at)
             VALUES ($1, $2, $3, $4, NOW(), NOW())`,
            [uid, thread, summary, shortDescription]
          );
        }
        console.log(
          '[conversation-summary] Inserted first summary',
          '| userId:', uid,
          '| threadId:', thread,
          '| shortDescription:', shortDescription.slice(0, 120),
          '| summarySnippet:', summary.slice(0, 180)
        );
      }
    } catch (e) {
      console.warn('[conversation-summary] upsert summary failed:', e.message);
    }
  }

  return { summary, shortDescription };
}

/**
 * Generate end summary from turns, save to conversation_summaries, return summary text.
 * @param {string} userId
 * @param {string} [threadId]
 * @param {Array<{ role: string, content: string }>} turns
 * @returns {Promise<{ summary: string, id?: string }>}
 */
export async function generateAndSaveEndSummary(userId, threadId, workspaceId, turns) {
  const { summary } = await generateAndUpsertThreadSummary(userId, threadId, workspaceId, turns);
  return { summary };
}

export async function getUserThreads(userId) {
  if (!userId) return [];
  if (!db.isUsingPostgres()) return [];
  try {
    const { rows } = await db.query(
      `SELECT t."threadId", t."lastActivity", s.summary, s.short_description AS "shortDescription", s.updated_at AS "summaryUpdatedAt"
       FROM (
         SELECT DISTINCT ON (thread_id)
           thread_id AS "threadId",
           created_at AS "lastActivity"
         FROM public.conversation_turns
         WHERE user_id = $1
         ORDER BY thread_id, created_at DESC
       ) t
       LEFT JOIN LATERAL (
         SELECT summary, short_description, updated_at
         FROM public.conversation_summaries s
         WHERE s.user_id = $1 AND s.thread_id = t."threadId"
         ORDER BY updated_at DESC NULLS LAST, created_at DESC
         LIMIT 1
       ) s ON true
       ORDER BY t."lastActivity" DESC`,
      [String(userId).slice(0, 256)]
    );
    return rows;
  } catch (e) {
    console.warn('[conversation-summary] getUserThreads failed:', e.message);
    return [];
  }
}

export default {
  generateSummary,
  saveConversationTurn,
  getLatestSummary,
  generateAndUpsertThreadSummary,
  generateAndSaveEndSummary,
  getHistoricalTurns,
  getUserThreads
};
