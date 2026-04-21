/**
 * Knowledge Keeper Agent
 * Role: Researcher / Memory
 * Runs in parallel on every prompt; searches/updates Knowledge Base and injects facts into other agents.
 * Tools: Supabase vector search + pgvector
 */

import { searchKnowledgeBase } from '../tools/knowledge-keeper-tool.js';

export const KNOWLEDGE_KEEPER_DESCRIPTION = `
Knowledge Keeper: Searches and retrieves from the brand Knowledge Base (PDFs, URLs, docs).
Run on every user prompt in parallel; results are injected into Content Genius and other agents as context.
`;

/**
 * Run Knowledge Keeper: query the vector store and return relevant chunks for this turn.
 * @param {string} userPrompt - Current user message
 * @param {object} options - { topK, companyId?, workspaceId? }
 * @returns {Promise<{ chunks: string[], raw: object }>}
 */
export async function runKnowledgeKeeper(userPrompt, options = {}) {
  const topK = options.topK ?? 5;
  const companyId = options.companyId || null;
  const workspaceId = options.workspaceId || null;
  try {
    const result = await searchKnowledgeBase(userPrompt, topK, companyId, workspaceId);
    const chunks = (result.chunks || [])
      .map((c) => (typeof c === 'string' ? c : c?.content))
      .filter((c) => typeof c === 'string' && c.trim().length > 0);
    return { chunks, raw: result };
  } catch (e) {
    console.warn('[KnowledgeKeeper] Search failed, continuing without knowledge context:', e.message);
    return { chunks: [], raw: { error: e.message } };
  }
}

export default runKnowledgeKeeper;
