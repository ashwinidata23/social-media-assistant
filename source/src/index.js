/**
 * ZunoSync - Multi-agentic AI marketing co-pilot
 * Entry: run the LangGraph workflow for a single user message.
 * Supports per-user Redis context and chat-end summary for "continue conversation".
 * Supports Human-in-the-Loop: interrupt() + Command resume for media/ambiguous inputs.
 */

import { Command } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { buildGraph } from './graph/workflow.js';
import { setBrandKit, loadBrandKitFromDb } from './tools/brand-kit-tool.js';
import { getUserState, setUserState } from './lib/user-context-store.js';
import {
  generateAndSaveEndSummary,
  getLatestSummary,
  saveConversationTurn,
  getHistoricalTurns,
  generateAndUpsertThreadSummary,
} from './lib/conversation-summary.js';
import { config } from '../config/index.js';

const graph = buildGraph();

// In-memory store for mediaAnalysis bridging interrupt → resume.
// Keyed by threadId; no Redis dependency needed.
const pendingMediaAnalysis = new Map();

// In-memory store for pending context (platforms, topic, etc.) bridging interrupt → resume.
// interrupt() suspends BEFORE the node's return, so state updates are lost.
const pendingInterruptContext = new Map();

const MAX_TAIL = 20;

// Conversation end phrases from the user (triggers chatEnd=true and end summary saving).
// Keep strict to reduce false positives.
const END_CHAT_REGEX = /^\s*(bye[\s\-]*bye|good\s*bye|goodbye|bye|that'?s all|we'?re done|end chat|see you|talk later|thanks,?\s*that'?s it|take\s*care)\b[.!?]?\s*$/i;

/** User is only discussing when to schedule (date/time), not asking for new content or images. */
const SCHEDULING_ONLY_REGEX = /(schedule|post now|calendar|when to post|schedule it|schedule for|for (youtube|x|twitter|instagram|linkedin|facebook|tiktok)|tomorrow|morning|evening|\d{1,2}(:\d{2})?\s*(am|pm)|post at|reschedule)/i;
const REQUESTING_NEW_CONTENT_REGEX = /(create|generate|write|new caption|new post|another (caption|post|image)|make (a |an )?(new )?(caption|post|image))/i;

/**
 * Derive billing/credit metadata from the final state.
 * - generationType: 1 = text only, 2 = text + image, 3 = text + video.
 * - contentCount:   how many content items (platform posts) were generated.
 * - imageCount:     how many image URLs were generated.
 * - videoCount:     how many video URLs were generated.
 */
function computeGenerationStats(finalState) {
  const ctx = finalState?.context || {};
  const mediaResult = finalState?.mediaResult;
  const contentResult = finalState?.contentResult;

  const hasMediaOutputs = !!(mediaResult && Array.isArray(mediaResult.outputs) && mediaResult.outputs.length);

  // ----- generationType -----
  let generationType = 1;
  if (hasMediaOutputs) {
    const contentType = ctx.contentType || 'image';
    generationType = contentType === 'video' ? 3 : 2;
  }

  // ----- contentCount -----
  let contentCount = 0;
  if (contentResult?.platforms?.length) {
    contentCount = contentResult.platforms.length;
  } else if (contentResult?.copy) {
    contentCount = 1;
  }

  // ----- imageCount / videoCount -----
  let imageCount = 0;
  let videoCount = 0;
  if (hasMediaOutputs) {
    for (const out of mediaResult.outputs) {
      if (!out) continue;
      const type = (out.type || '').toLowerCase();
      const urls = Array.isArray(out.urls) ? out.urls : [];
      const n = urls.length || 0;
      if (type === 'video') videoCount += n;
      else imageCount += n;
    }
  }

  return { generationType, contentCount, imageCount, videoCount };
}

/**
 * Process one user message through the 6-agent pipeline.
 * With userId/sessionId: loads Redis state and previous summary for context awareness; saves state after run.
 * With endChat: generates and stores end summary, returns chat-end response.
 * @param {string} userPrompt - User message
 * @param {object} [options] - { userId?, sessionId?, threadId?, endChat?, scheduleAt?, platforms?, ... }
 * @returns {Promise<{ finalReply, buttons, contentResult?, mediaResult?, schedulerResult?, context?, endSummary?, chatEnd? }>}
 */
export async function processUserMessage(userPrompt, options = {}) {
  const userId = options.userId ?? options.sessionId ?? null;
  const companyId = options.companyId ?? null;
  const threadId = options.threadId || 'default';
  const resetContext = options.resetContext === true;
  const workspaceId = options.workspaceId ?? null;
  const signal = options.signal ?? null; // AbortSignal for cancellation

  if (options.resumeWithSelection === true && options.resumeSelection) {
    return resumeFromInterrupt(threadId, options.resumeSelection, userId, companyId, options);
  }

  const trimmedUserPrompt = String(userPrompt || '').trim();
  const endChatRequested = options.endChat === true || END_CHAT_REGEX.test(trimmedUserPrompt);

  const initialContext = {};
  if (userId) initialContext.userId = userId;
  if (companyId) initialContext.companyId = companyId;
  if (options.workspaceId) initialContext.workspaceId = options.workspaceId;
  if (options.scheduleAt != null && String(options.scheduleAt).trim()) {
    const t = new Date(options.scheduleAt);
    if (!Number.isNaN(t.getTime())) initialContext.scheduleAt = t.toISOString();
  }
  if (options.publishAllAtSameTime !== undefined) initialContext.publishAllAtSameTime = !!options.publishAllAtSameTime;
  if (Array.isArray(options.platforms) && options.platforms.length) initialContext.platforms = options.platforms;
  if (Array.isArray(options.accounts) && options.accounts.length) initialContext.accounts = options.accounts;
  if (options.rawContent && typeof options.rawContent === 'string' && options.rawContent.trim()) {
    initialContext.rawContent = options.rawContent.trim();
  }

  // Optionally start with a completely fresh context, ignoring any stored state/history.
  if (resetContext && userId) {
    console.log('[processUserMessage] resetContext=true — clearing stored user state', {
      userId,
      threadId,
      workspaceId,
    });
    await setUserState(userId, {
      // IMPORTANT: user-context-store merges context by default.
      // Use context: null to force a full reset (drop all prior keys).
      context: null,
      contentResult: null,
      mediaResult: null,
      conversationTail: [],
      previousSummary: null,
    });
  }

  let loaded = null;
  // workspaceId already derived above

  if (userId && !resetContext) {
    loaded = await getUserState(userId);
    // Prevent cross-thread context bleed: only reuse stored context if it's from the same thread.
    const lastThreadId = loaded?.context?.lastThreadId || null;
    const sameThread = !!(lastThreadId && String(lastThreadId) === String(threadId));
    if (loaded?.context && sameThread) {
      Object.assign(initialContext, { ...loaded.context, ...initialContext });
    } else if (loaded?.context && !sameThread) {
      console.log('[processUserMessage] New thread detected — ignoring stored Redis context', {
        userId,
        threadId,
        lastThreadId,
      });
      // Also ignore cached content/media from previous threads.
      loaded = null;
    }

    // Load actual DB history for this specific threadId (fixes thread switching)
    const dbTurns = await getHistoricalTurns(userId, threadId);
    if (dbTurns.length > 0) {
      const historyText = dbTurns
        .slice(-20)
        .map(t => `${t.role}: ${t.content}`)
        .join('\n');
      initialContext.previousSummary = `[Previous conversation history]\n${historyText}`;
      console.log(`[processUserMessage] Loaded ${dbTurns.length} turns from DB for threadId: ${threadId}`);
    } else {
      // Fall back to summary if no turns in DB yet (scoped by workspace when available)
      const prevSummary = loaded?.previousSummary || (await getLatestSummary(userId, threadId, workspaceId));
      if (prevSummary) initialContext.previousSummary = prevSummary;
    }
  }

  // --- PRIMARY: Load brand kit from DB using BOTH userId AND workspaceId ---
  if (userId && workspaceId) {
    try {
      const dbBrandKit = await loadBrandKitFromDb(userId, workspaceId);
      if (dbBrandKit) {
        console.log('[processUserMessage] Brand kit loaded from DB | userId:', userId, '| workspaceId:', workspaceId, '| logoUrl:', dbBrandKit.logoUrl?.slice(0, 60) || '(none)');
        initialContext.brandKit = dbBrandKit;
      }
    } catch (e) {
      console.warn('[processUserMessage] Could not load brand kit from DB:', e?.message || e);
    }
  } else {
    console.log('[processUserMessage] Skipping brand kit DB lookup \u2014 need both userId and workspaceId. Got userId:', userId, 'workspaceId:', workspaceId);
  }

  // --- OVERRIDE: request-body brandKit (use only for explicit overrides / dev testing) ---
  if (options.brandKit && typeof options.brandKit === 'object') {
    console.log('[processUserMessage] brandKit override received from request body — overriding DB kit.');
    initialContext.brandKit = options.brandKit;
  }

  if (initialContext.brandKit) setBrandKit(initialContext.brandKit);

  let promptForGraph =
    initialContext.previousSummary && userPrompt
      ? `[Previous conversation history]\n${initialContext.previousSummary.replace(/^\[Previous conversation history\]\n/, '')}\n\nUser now says: ${userPrompt}`
      : userPrompt || '';

  const hasExistingContent = !!(loaded?.contentResult && loaded?.mediaResult);
  const looksLikeScheduling = userPrompt && SCHEDULING_ONLY_REGEX.test(String(userPrompt).trim());
  const notAskingForNewContent = !userPrompt || !REQUESTING_NEW_CONTENT_REGEX.test(String(userPrompt).trim());
  const isSchedulingOnly = looksLikeScheduling && notAskingForNewContent;
  if (hasExistingContent && isSchedulingOnly) {
    promptForGraph = `[We already have generated content and media from the previous turn. User is only specifying when to schedule.] ${promptForGraph}`;
  }

  // END-CHAT: When user says goodbye (or clicks "End chat"), skip the content pipeline
  // and return an LLM-generated farewell + saved end-summary.
  if (endChatRequested) {
    const llm = new ChatOpenAI({
      model: process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini',
      temperature: 0.3,
      apiKey: config.openai?.apiKey,
    });

    const farewellSystem = `You are a social media marketing assistant.
The user is ending the conversation.
Write a single short friendly reply (max 6-8 words).
Do NOT say "goodbye", "bye", "farewell", "take care", or any goodbye words.
Do NOT mention saving summaries or sessions.
Just acknowledge warmly. Example: "Happy to help anytime!"
Return only the message text.`;

    const farewellResponse = await llm.invoke([
      new SystemMessage(farewellSystem),
      new HumanMessage(String(trimmedUserPrompt || userPrompt || '').trim()),
    ]);

    const farewellReply =
      typeof farewellResponse.content === 'string'
        ? farewellResponse.content.trim()
        : String(farewellResponse.content?.[0]?.text || '').trim();

    let endSummary = null;
    let chatEnd = false;

    const conversationTail = [];
    if (loaded?.conversationTail?.length) conversationTail.push(...loaded.conversationTail);
    conversationTail.push({ role: 'user', content: String(trimmedUserPrompt).slice(0, 2000) });
    conversationTail.push({ role: 'assistant', content: String(farewellReply).slice(0, 2000) });
    const trimmedTail = conversationTail.slice(-MAX_TAIL);

    if (userId) {
      const responseData = {
        finalReply: farewellReply || "Got it. I've saved a summary of this session so you can continue from here next time.",
        buttons: [],
        callAgents: [],
        context: initialContext,
        contentResult: null,
        mediaResult: null,
        previewResult: null,
        schedulerResult: null,
        schedulingOnlyReply: false,
        directReply: null,
        endSummary,
        chatEnd,
      };

      await Promise.all([
        saveConversationTurn(userId, threadId, 'user', trimmedUserPrompt),
        // End-of-chat farewell usually has no new media; save without media URLs.
        saveConversationTurn(userId, threadId, 'assistant', responseData.finalReply, null, responseData),
      ]);

      const { summary } = await generateAndSaveEndSummary(userId, threadId, workspaceId, trimmedTail);
      endSummary = summary;
      chatEnd = true;

      await setUserState(userId, {
        context: initialContext,
        contentResult: loaded?.contentResult,
        mediaResult: loaded?.mediaResult,
        conversationTail: [],
        previousSummary: summary,
      });

      responseData.endSummary = endSummary;
      responseData.chatEnd = chatEnd;
      return responseData;
    }

    // Anonymous mode: we can still return the goodbye, but cannot persist summary.
    return {
      finalReply: farewellReply || "Got it. I've saved a summary of this session so you can continue from here next time.",
      buttons: [],
      callAgents: [],
      context: initialContext,
      contentResult: null,
      mediaResult: null,
      previewResult: null,
      schedulerResult: null,
      schedulingOnlyReply: false,
      directReply: null,
      endSummary,
      chatEnd,
    };
  }

  const initialState = {
    userPrompt: promptForGraph,
    rawUserPrompt: String(userPrompt || ''),
    context: initialContext,
    canonicalUserId: userId || undefined,
    canonicalCompanyId: companyId || undefined,
    contentResult: loaded?.contentResult ?? undefined,
    mediaResult: loaded?.mediaResult ?? undefined,
    mediaInput: options.mediaInput || null,
  };

  const lgConfig = { configurable: { thread_id: threadId }, ...(signal ? { signal } : {}) };
  let finalState = null;
  let interruptData = null;

  try {
    const stream = await graph.stream(initialState, { ...lgConfig, streamMode: 'updates' });
    for await (const chunk of stream) {
      // Check for cancellation before processing each chunk
      if (signal?.aborted) {
        console.log('[processUserMessage] Aborted during stream | thread:', threadId);
        return {
          finalReply: '', buttons: [], callAgents: [], context: initialContext,
          contentResult: null, mediaResult: null, previewResult: null,
          schedulerResult: null, schedulingOnlyReply: false, directReply: null,
          cancelled: true,
        };
      }
      if (chunk.__interrupt__) {
        interruptData = chunk.__interrupt__[0]?.value ?? chunk.__interrupt__[0];
        break;
      }
      for (const nodeUpdate of Object.values(chunk)) {
        if (nodeUpdate && typeof nodeUpdate === 'object') {
          finalState = finalState ? { ...finalState, ...nodeUpdate } : { ...nodeUpdate };
        }
      }
    }
  } catch (e) {
    // Cancelled — return empty result
    if (e?.name === 'AbortError' || signal?.aborted) {
      console.log('[processUserMessage] Aborted (caught) | thread:', threadId);
      return {
        finalReply: '', buttons: [], callAgents: [], context: initialContext,
        contentResult: null, mediaResult: null, previewResult: null,
        schedulerResult: null, schedulingOnlyReply: false, directReply: null,
        cancelled: true,
      };
    }
    if (e?.name === 'GraphInterrupt' || e?.constructor?.name === 'GraphInterrupt') {
      interruptData = e.interrupts?.[0]?.value ?? { reason: 'unknown', options: [], message: 'Please select an option.' };
    } else {
      throw e;
    }
  }

  if (interruptData) {
    const responseData = {
      finalReply: interruptData.message || 'Please select an option to continue.',
      // Preserve any buttons the node provided (e.g. quick platform shortcuts)
      buttons: interruptData.buttons || [],
      callAgents: [],
      context: initialContext,
      contentResult: null,
      mediaResult: null,
      previewResult: null,
      schedulerResult: null,
      schedulingOnlyReply: false,
      directReply: null,
      awaitingConfirmation: true,
      confirmationOptions: interruptData.options || [],
      mediaAnalysis: interruptData.mediaAnalysis || null,
    };

    // Persist mediaAnalysis in-memory so resumeFromInterrupt can inject it back
    // into the graph state without relying on Redis.
    if (interruptData.mediaAnalysis) {
      pendingMediaAnalysis.set(threadId, interruptData.mediaAnalysis);
      console.log('[processUserMessage] Stored mediaAnalysis for thread:', threadId, '|', interruptData.mediaAnalysis.description?.slice(0, 80));
    }

    // Persist pending context (platforms, topic, reason) in-memory.
    // interrupt() suspends BEFORE the node return, so these are lost from graph state.
    if (interruptData.pendingContext) {
      pendingInterruptContext.set(threadId, interruptData.pendingContext);
      console.log('[processUserMessage] Stored pendingContext for thread:', threadId, '| platforms:', interruptData.pendingContext.platforms, '| reason:', interruptData.pendingContext.confirmationReason, '| contentType:', interruptData.pendingContext.contentType, '| includeMedia:', interruptData.pendingContext.includeMedia);
    }

    if (userId) {
      try {
        await saveConversationTurn(userId, threadId, 'user', String(userPrompt || '').slice(0, 2000));
        await saveConversationTurn(userId, threadId, 'assistant', responseData.finalReply, null, responseData);
        await setUserState(userId, {
          context: {
            ...initialContext,
            awaitingConfirmationReason: interruptData.reason,
            mediaAnalysis: interruptData.mediaAnalysis || null,
          },
          contentResult: loaded?.contentResult,
          mediaResult: loaded?.mediaResult,
          conversationTail: loaded?.conversationTail || [],
        });
      } catch (e) {
        console.warn('[processUserMessage] Could not save user turn or state on interrupt:', e.message);
      }
    }
    return responseData;
  }

  if (!finalState) {
    try {
      const cp = await graph.getState(lgConfig);
      finalState = cp?.values || {};
    } catch (_) {
      finalState = {};
    }
  }

  const goodbyeNote =
    "Got it. I've saved a summary of this session so you can continue from here next time.";

  let finalReply = finalState.finalReply || finalState.directReply || '';
  if (endChatRequested && finalReply) {
    // Keep the main assistant message LLM-generated, but append the fixed "saved summary" notice.
    // (So the user still gets the goodbye UX.)
    finalReply = `${finalReply}\n\n${goodbyeNote}`;
  } else if (endChatRequested) {
    finalReply = goodbyeNote;
  }

  const conversationTail = [];
  if (loaded?.conversationTail?.length) {
    conversationTail.push(...loaded.conversationTail);
  }
  conversationTail.push({ role: 'user', content: String(userPrompt || '').slice(0, 2000) });
  conversationTail.push({ role: 'assistant', content: String(finalReply).slice(0, 2000) });
  const trimmedTail = conversationTail.slice(-MAX_TAIL);

  // If the LLM returned a direct reply with no agents, this is a normal
  // conversational response — don't leak stale content/media/preview from
  // previous turns. Only include results when agents actually ran this turn.
  const agentsRanThisTurn = Array.isArray(finalState.callAgents) && finalState.callAgents.length > 0;
  const stats = computeGenerationStats(agentsRanThisTurn ? finalState : {});

  let endSummary = null;
  let chatEnd = false;

  const responseData = {
    finalReply,
    buttons: finalState.buttons || [],
    callAgents: finalState.callAgents || [],
    context: finalState.context || {},
    contentResult: agentsRanThisTurn ? finalState.contentResult : null,
    mediaResult: agentsRanThisTurn ? finalState.mediaResult : null,
    previewResult: agentsRanThisTurn ? finalState.previewResult : null,
    schedulerResult: agentsRanThisTurn ? finalState.schedulerResult : null,
    schedulingOnlyReply: finalState.schedulingOnlyReply ?? false,
    directReply: finalState.directReply,
    awaitingConfirmation: finalState.awaitingConfirmation ?? false,
    confirmationOptions: finalState.confirmationOptions ?? undefined,
    mediaAnalysis: undefined,
    generationType: stats.generationType,
    contentCount: stats.contentCount,
    imageCount: stats.imageCount,
    videoCount: stats.videoCount,
  };

  if (userId) {
    try {
      const assistantMediaUrls =
        finalState?.mediaResult?.outputs?.flatMap((o) => Array.isArray(o?.urls) ? o.urls : []) || [];

      if (assistantMediaUrls.length > 0) {
        console.log(
          '[processUserMessage] Assistant media URLs for this turn',
          '| userId:', userId,
          '| threadId:', threadId,
          '| count:', assistantMediaUrls.length
        );
      }

      await Promise.all([
        saveConversationTurn(userId, threadId, 'user', String(userPrompt || '').slice(0, 2000)),
        saveConversationTurn(
          userId,
          threadId,
          'assistant',
          String(finalReply).slice(0, 2000),
          assistantMediaUrls,
          responseData
        ),
      ]);

      // Dynamic summary update (used by frontend thread list / continue conversation UX), scoped by workspace
      const { summary } = await generateAndUpsertThreadSummary(userId, threadId, workspaceId, trimmedTail);
      if (endChatRequested) {
        endSummary = summary;
        chatEnd = true;
      }
    } catch (e) {
      console.warn('[processUserMessage] Could not save conversation turns:', e.message);
    }
  }

  if (userId) {
    try {
      await setUserState(userId, {
        context: {
          ...(finalState.context || {}),
          // Track which thread this Redis snapshot belongs to.
          lastThreadId: threadId,
        },
        contentResult: finalState.contentResult,
        mediaResult: finalState.mediaResult,
        conversationTail: endChatRequested ? [] : trimmedTail,
        ...(endChatRequested ? { previousSummary: endSummary } : {}),
      });
    } catch (e) {
      console.warn('[processUserMessage] Could not save user state:', e.message);
    }
  }

  if (endChatRequested) {
    responseData.endSummary = endSummary;
    responseData.chatEnd = chatEnd;
  }

  return responseData;
}

async function resumeFromInterrupt(threadId, selection, userId, companyId, options) {
  console.log('[processUserMessage] Resuming from interrupt | thread:', threadId, '| selection:', JSON.stringify(selection));

  // Recover mediaAnalysis stored at interrupt time so it's available in graph state.
  const restoredMediaAnalysis = pendingMediaAnalysis.get(threadId) || null;
  if (restoredMediaAnalysis) {
    console.log('[resumeFromInterrupt] Restoring mediaAnalysis for thread:', threadId, '|', restoredMediaAnalysis.description?.slice(0, 80));
    pendingMediaAnalysis.delete(threadId); // consume it — one-shot
  }

  // Recover pending context (platforms, topic, confirmationReason) stored at interrupt time.
  const restoredContext = pendingInterruptContext.get(threadId) || null;
  if (restoredContext) {
    console.log('[resumeFromInterrupt] Restoring pendingContext for thread:', threadId, '| platforms:', restoredContext.platforms, '| reason:', restoredContext.confirmationReason, '| contentType:', restoredContext.contentType, '| includeMedia:', restoredContext.includeMedia);
    pendingInterruptContext.delete(threadId); // consume it — one-shot
  }

  const lgConfig = { configurable: { thread_id: threadId } };
  const resumeCommand = new Command({
    resume: selection,
    update: {
      userSelection: selection,
      awaitingConfirmation: false,
      // Inject mediaAnalysis back into graph state so executeAgentsNode can use it.
      mediaAnalysis: restoredMediaAnalysis,
      // Inject restored context (platforms, topic, confirmationReason) that was lost during interrupt.
      ...(restoredContext ? {
        context: {
          ...(restoredMediaAnalysis ? { mediaAnalysis: restoredMediaAnalysis } : {}),
          ...restoredContext,
        },
        confirmationReason: restoredContext.confirmationReason || null,
      } : {
        context: restoredMediaAnalysis ? { mediaAnalysis: restoredMediaAnalysis } : {},
      }),
    },
  });

  let finalState = null;
  let interruptData = null;

  try {
    const stream = await graph.stream(resumeCommand, { ...lgConfig, streamMode: 'updates' });
    for await (const chunk of stream) {
      if (chunk.__interrupt__) {
        interruptData = chunk.__interrupt__[0]?.value ?? chunk.__interrupt__[0];
        break;
      }
      for (const nodeUpdate of Object.values(chunk)) {
        if (nodeUpdate && typeof nodeUpdate === 'object') {
          finalState = finalState ? { ...finalState, ...nodeUpdate } : { ...nodeUpdate };
        }
      }
    }
  } catch (e) {
    if (e?.name === 'GraphInterrupt' || e?.constructor?.name === 'GraphInterrupt') {
      interruptData = e.interrupts?.[0]?.value ?? { reason: 'unknown', options: [] };
    } else {
      throw e;
    }
  }

  if (interruptData) {
    const interruptMessage = interruptData.message || '';
    const responseData = {
      finalReply: interruptMessage || 'Please select an option to continue.',
      awaitingConfirmation: true,
      confirmationOptions: interruptData.options || [],
      buttons: interruptData.buttons || [],
      callAgents: [],
      contentResult: null, mediaResult: null, previewResult: null, schedulerResult: null,
      schedulingOnlyReply: false, directReply: null,
    };

    // Persist pending context so the next resume can restore platforms/topic/etc.
    if (interruptData.pendingContext) {
      pendingInterruptContext.set(threadId, interruptData.pendingContext);
      console.log('[resumeFromInterrupt] Stored pendingContext for re-interrupt | thread:', threadId, '| reason:', interruptData.pendingContext.confirmationReason);
    }

    if (userId) {
      saveConversationTurn(userId, threadId, 'assistant', responseData.finalReply, null, responseData).catch(e => console.warn(e));
    }

    return responseData;
  }

  if (!finalState) {
    try {
      const cp = await graph.getState(lgConfig);
      finalState = cp?.values || {};
    } catch (_) {
      finalState = {};
    }
  }

  const finalReply = finalState.directReply || finalState.finalReply || '';

  const stats = computeGenerationStats(finalState);

  const responseData = {
    finalReply,
    buttons: finalState.buttons || [],
    callAgents: finalState.callAgents || [],
    context: finalState.context || {},
    contentResult: finalState.contentResult,
    mediaResult: finalState.mediaResult,
    previewResult: finalState.previewResult,
    schedulerResult: finalState.schedulerResult,
    schedulingOnlyReply: finalState.schedulingOnlyReply ?? false,
    directReply: finalState.directReply,
    awaitingConfirmation: finalState.awaitingConfirmation ?? false,
    confirmationOptions: finalState.confirmationOptions ?? undefined,
    mediaAnalysis: undefined,
    generationType: stats.generationType,
    contentCount: stats.contentCount,
    imageCount: stats.imageCount,
    videoCount: stats.videoCount,
  };

  if (userId) {
    const selectionLabel = `[Selected: ${selection.action} for ${(selection.platforms || []).join(', ')}]`;
    try {
      const assistantMediaUrls =
        finalState?.mediaResult?.outputs?.flatMap((o) => Array.isArray(o?.urls) ? o.urls : []) || [];

      if (assistantMediaUrls.length > 0) {
        console.log(
          '[resumeFromInterrupt] Assistant media URLs for resume turn',
          '| userId:', userId,
          '| threadId:', threadId,
          '| count:', assistantMediaUrls.length
        );
      }

      await Promise.all([
        saveConversationTurn(userId, threadId, 'user', selectionLabel),
        saveConversationTurn(
          userId,
          threadId,
          'assistant',
          String(finalReply).slice(0, 2000),
          assistantMediaUrls,
          responseData
        ),
      ]);

      // Dynamic summary update after resume turn as well (scoped by workspace when available)
      const workspaceId = options?.workspaceId;
      const turnsForSummary = [
        { role: 'user', content: selectionLabel },
        { role: 'assistant', content: String(finalReply).slice(0, 2000) },
      ];
      await generateAndUpsertThreadSummary(userId, threadId, workspaceId, turnsForSummary);
    } catch (e) {
      console.warn('[resumeFromInterrupt] Could not save conversation turns:', e.message);
    }
  }

  if (userId) {
    try {
      await setUserState(userId, {
        context: finalState.context,
        contentResult: finalState.contentResult,
        mediaResult: finalState.mediaResult,
      });
    } catch (e) {
      console.warn('[resumeFromInterrupt] Could not save user state:', e.message);
    }
  }

  return responseData;
}

async function main() {
  const prompt = process.argv.slice(2).join(' ') || 'Hi';
  console.log('User:', prompt);
  const result = await processUserMessage(prompt);
  console.log('Call agents:', result.callAgents);
  console.log('Final reply:', result.finalReply);
  console.log('Buttons:', result.buttons);
  if (result.contentResult) {
    if (result.contentResult.platforms?.length) {
      result.contentResult.platforms.forEach((p) => {
        const preview = p.content?.length > 150 ? p.content.slice(0, 150) + '...' : (p.content || '');
        console.log(`[${p.type}]`, preview);
      });
    } else {
      console.log('Content:', result.contentResult.copy?.slice(0, 200) + '...');
    }
  }
  if (result.mediaResult) {
    if (result.mediaResult.outputs?.length) {
      console.log('Media:', result.mediaResult.outputs);
    } else if (result.mediaResult.error) {
      console.log('Media error:', result.mediaResult.error);
    } else {
      console.log('Media: (no outputs)');
    }
  }
  if (result.schedulerResult) {
    console.log('Scheduler:', result.schedulerResult.message);
    if (result.schedulerResult.bestTime) console.log('Scheduler bestTime:', result.schedulerResult.bestTime);
    if (result.schedulerResult.scheduleResult) console.log('Scheduler scheduleResult:', result.schedulerResult.scheduleResult);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});