/**
 * Per-user context store in Redis for context awareness across turns.
 * Stores context, last contentResult, mediaResult, conversation tail, and last summary.
 */

const KEY_PREFIX = 'zunosync:user:';
const DEFAULT_TTL_SEC = 24 * 60 * 60; // 24 hours

let client = null;
let clientUrl = null;

async function getClient() {
  const url = process.env.REDIS_URL || process.env.REDIS_URI;
  if (!url || !url.trim()) return null;

  const normalizedUrl = url.trim();
  const resetClient = () => {
    try {
      if (client) client.disconnect(false);
    } catch (_) {}
    client = null;
    clientUrl = null;
  };

  // If URL changed or client is ended, recreate.
  if (client && clientUrl && clientUrl !== normalizedUrl) resetClient();
  if (client && (client.status === 'end' || client.status === 'close')) resetClient();

  if (!client) {
    try {
      const Redis = (await import('ioredis')).default;
      client = new Redis(normalizedUrl, {
        // Keep retries enabled; do not permanently end the client after a single failure.
        maxRetriesPerRequest: null,
        connectTimeout: 5000,
        enableOfflineQueue: false,
        retryStrategy(times) {
          // Exponential-ish backoff capped at 2s
          return Math.min(times * 200, 2000);
        },
        lazyConnect: true,
      });
      clientUrl = normalizedUrl;
      client.on('error', (err) => console.warn('[Redis]', err.message));
    } catch (e) {
      console.warn('[Redis] Not available:', e.message);
      resetClient();
      return null;
    }
  }

  try {
    // With lazyConnect, connect on demand; this also recovers from "Connection is closed".
    if (client.status !== 'ready') await client.connect();
  } catch (e) {
    console.warn('[Redis] connect failed:', e.message);
    resetClient();
    return null;
  }

  return client;
}

function shouldRetryRedisError(e) {
  const msg = (e?.message || '').toLowerCase();
  return msg.includes('connection is closed') || msg.includes('econnrefused') || msg.includes('etimedout');
}

async function withRedisRetry(fn) {
  try {
    return await fn();
  } catch (e) {
    if (!shouldRetryRedisError(e)) throw e;
    // Force client recreate on next call
    client = null;
    clientUrl = null;
    const redis = await getClient();
    if (!redis) throw e;
    return await fn();
  }
}

/** Quick health check for Redis. */
export async function pingRedis() {
  try {
    const redis = await getClient();
    if (!redis) return { ok: false, status: 'not_configured', message: 'REDIS_URL not set or redis unreachable' };
    const pong = await withRedisRetry(() => redis.ping());
    return { ok: pong === 'PONG', status: redis.status, message: pong };
  } catch (e) {
    return { ok: false, status: 'error', message: e.message || String(e) };
  }
}

/**
 * Get stored state for a user.
 * @param {string} userId - User or session id
 * @returns {Promise<{ context: object, contentResult: object|null, mediaResult: object|null, conversationTail: array, previousSummary: string|null, updatedAt: number }|null>}
 */
export async function getUserState(userId) {
  if (!userId || typeof userId !== 'string') return null;
  const redis = await getClient();
  if (!redis) return null;
  try {
    const key = KEY_PREFIX + userId.trim();
    const raw = await withRedisRetry(() => redis.get(key));
    if (!raw) return null;
    const data = JSON.parse(raw);
    return {
      context: data.context || {},
      contentResult: data.contentResult ?? null,
      mediaResult: data.mediaResult ?? null,
      conversationTail: Array.isArray(data.conversationTail) ? data.conversationTail : [],
      previousSummary: data.previousSummary ?? null,
      updatedAt: data.updatedAt || 0,
    };
  } catch (e) {
    console.warn('[user-context-store] get failed:', e.message);
    return null;
  }
}

/**
 * Save user state to Redis. Merges context; keeps conversationTail and previousSummary when not provided.
 * @param {string} userId
 * @param {object} state - { context?, contentResult?, mediaResult?, conversationTail?, previousSummary? }
 * @param {number} [ttlSec]
 */
export async function setUserState(userId, state, ttlSec = DEFAULT_TTL_SEC) {
  if (!userId || typeof userId !== 'string') return;
  const redis = await getClient();
  if (!redis) return;
  try {
    const key = KEY_PREFIX + userId.trim();
    const existing = await getUserState(userId);
    // By default we merge context so multi-turn sessions persist accumulated state.
    // But allow callers to force a full reset by passing context: null.
    const mergedContext = state.context === null
      ? {}
      : { ...(existing?.context || {}), ...(state.context || {}) };
    const payload = {
      context: mergedContext,
      contentResult: state.contentResult !== undefined ? state.contentResult : existing?.contentResult ?? null,
      mediaResult: state.mediaResult !== undefined ? state.mediaResult : existing?.mediaResult ?? null,
      conversationTail: state.conversationTail !== undefined ? state.conversationTail : existing?.conversationTail ?? [],
      previousSummary: state.previousSummary !== undefined ? state.previousSummary : existing?.previousSummary ?? null,
      updatedAt: Date.now(),
    };
    const serialized = JSON.stringify(payload);
    if (ttlSec > 0) await withRedisRetry(() => redis.setex(key, ttlSec, serialized));
    else await withRedisRetry(() => redis.set(key, serialized));
  } catch (e) {
    console.warn('[user-context-store] set failed:', e.message);
  }
}

export default { getUserState, setUserState, pingRedis };
