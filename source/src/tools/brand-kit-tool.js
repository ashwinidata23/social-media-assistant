import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import * as db from '../lib/db.js';

// Fallback when no kit is provided (e.g. production defaults)
const fallbackBrandKit = {
  primaryColor: '#6366F1',
  secondaryColor: '#8B5CF6',
  accentColor: '#EC4899',
  logoUrl: '',
  fontHeading: 'Inter',
  fontBody: 'Inter',
  tone: 'professional, friendly, concise',
  hashtagStyle: '3-5 relevant hashtags at end',
};

// Local default: mtouch labs (as you provided) — used when no brand kit from DB or request
const localDefaultBrandKit = {
  primaryColor: '#00FF9F',
  secondaryColor: '#00BFFF',
  accentColor: '#BF5FFF',
  logoUrl: 'https://zunosync.s3.ap-south-1.amazonaws.com/invoices/1772436284543-768577139-1772436284787.png',
  fontHeading: 'Space Grotesk',
  fontBody: 'Inter',
  tone: 'professional, friendly, concise',
  hashtagStyle: '3-5 relevant hashtags at end',
};

const defaultBrandKit = localDefaultBrandKit;
let brandKitStore = { ...defaultBrandKit };

/**
 * Resolve color palette from either object { primary, secondary, accent } or array [{ name, colorCode }, ...].
 * Accepts the same shape as DB/frontend: array of { name, colorCode } (e.g. primary, secondary, accent, text, background).
 * Only primary, secondary, accent are used for generation; text/background are accepted but not passed through.
 * @returns {{ primary?: string, secondary?: string, accent?: string }}
 */
function resolvePalette(palette) {
  if (palette == null) return {};
  if (typeof palette === 'string') {
    try {
      palette = JSON.parse(palette);
    } catch {
      return {};
    }
  }
  if (typeof palette !== 'object') return {};
  if (Array.isArray(palette)) {
    const out = {};
    for (const item of palette) {
      if (item && typeof item === 'object' && item.name && item.colorCode) {
        const name = String(item.name).toLowerCase();
        if (name === 'primary') out.primary = item.colorCode;
        else if (name === 'secondary') out.secondary = item.colorCode;
        else if (name === 'accent') out.accent = item.colorCode;
      }
    }
    return out;
  }
  return {
    primary: palette.primary,
    secondary: palette.secondary,
    accent: palette.accent,
  };
}

/**
 * Resolve logo URL from logo field (string URL, or object with .url, or JSON string).
 */
function resolveLogoUrl(logo) {
  if (logo == null) return '';
  if (typeof logo === 'string') {
    if (logo.startsWith('http')) return logo;
    try {
      const parsed = JSON.parse(logo);
      return parsed?.url ?? '';
    } catch {
      return logo;
    }
  }
  if (typeof logo === 'object' && logo !== null && logo.url) return logo.url;
  return '';
}

/**
 * Normalize brand kit from frontend/DB shape to internal shape (primaryColor, fontHeading, etc.).
 * Accepts: { colorPalette: { primary, secondary, accent } or [{ name, colorCode }, ...], headingFont, bodyFont, logo (url or { url, altText, fileName }) } or already-normalized keys.
 */
export function normalizeBrandKit(kit) {
  if (!kit || typeof kit !== 'object') return null;
  const paletteRaw = kit.colorPalette || kit.color_palette;
  const resolved = resolvePalette(paletteRaw);
  const primary = resolved.primary ?? paletteRaw?.primary ?? kit.primaryColor;
  const secondary = resolved.secondary ?? paletteRaw?.secondary ?? kit.secondaryColor;
  const accent = resolved.accent ?? paletteRaw?.accent ?? kit.accentColor;
  const headingFont = kit.headingFont ?? kit.fontHeading;
  const bodyFont = kit.bodyFont ?? kit.fontBody;
  const logoUrl = resolveLogoUrl(kit.logo) || kit.logoUrl || '';
  if (!primary && !secondary && !accent && !headingFont && !bodyFont && !logoUrl && !kit.tone) return null;
  return {
    primaryColor: primary || fallbackBrandKit.primaryColor,
    secondaryColor: secondary || fallbackBrandKit.secondaryColor,
    accentColor: accent || fallbackBrandKit.accentColor,
    fontHeading: headingFont || fallbackBrandKit.fontHeading,
    fontBody: bodyFont || fallbackBrandKit.fontBody,
    logoUrl: logoUrl || fallbackBrandKit.logoUrl,
    tone: kit.tone ?? fallbackBrandKit.tone,
    hashtagStyle: kit.hashtagStyle ?? fallbackBrandKit.hashtagStyle,
  };
}

export function getBrandKit() {
  return { ...brandKitStore };
}

export function setBrandKit(kit) {
  brandKitStore = { ...brandKitStore, ...kit };
}

/**
 * Helper to extract integer from string (e.g. "user-123" -> 123, "3" -> 3).
 * Falls back to null if not parsable.
 */
function parseUserId(userId) {
  if (!userId) return null;
  const num = parseInt(String(userId).replace(/[^0-9]/g, ''), 10);
  return isNaN(num) ? null : num;
}

/**
 * Helper to extract integer workspace ID from string (e.g. "workspace-2" -> 2, "2" -> 2).
 * Returns null if not a valid number (e.g. "workspace-xyz" stays null — no match in DB).
 */
function parseWorkspaceId(workspaceId) {
  if (workspaceId == null) return null;
  const num = parseInt(String(workspaceId).replace(/[^0-9]/g, ''), 10);
  return isNaN(num) ? null : num;
}

/**
 * Load brand kit from public.brand_kits using BOTH userId AND workspaceId.
 * Both are required — a brand kit is always scoped to a specific user + workspace.
 * @param {string} userId
 * @param {string} workspaceId
 * @returns {Promise<object|null>}
 */
export async function loadBrandKitFromDb(userId, workspaceId) {
  if (!userId || !workspaceId) {
    console.log('[BrandKit] Skipping DB lookup — both userId and workspaceId are required. Got userId:', userId, 'workspaceId:', workspaceId);
    return null;
  }
  if (!db.isUsingPostgres()) return null;

  const numericUserId = parseUserId(userId);
  if (numericUserId === null) {
    console.log('[BrandKit] Skipping DB lookup — invalid userId:', userId);
    return null;
  }

  const ws = parseWorkspaceId(workspaceId);
  if (ws === null) {
    console.log('[BrandKit] Skipping DB lookup — invalid workspaceId:', workspaceId);
    return null;
  }

  try {
    const { rows } = await db.query(
      `SELECT logo, "colorPalette", "headingFont", "bodyFont" FROM public.brand_kits WHERE "userId" = $1 AND "workspaceId" = $2 AND status = 'active' ORDER BY "updatedAt" DESC LIMIT 1`,
      [numericUserId, ws]
    );
    const r = rows?.[0];
    if (!r) {
      console.log('[BrandKit] No brand kit found in DB for userId:', numericUserId, 'workspaceId:', ws);
      return null;
    }

    // logo is jsonb: could be a string "https://..." or object {url: "..."}. Extract URL safely.
    const logoUrl = typeof r.logo === 'string' ? r.logo : r.logo?.url ?? r.logo?.logo ?? null;
    console.log('[BrandKit] Found brand kit. logoUrl:', logoUrl?.slice(0, 60), 'palette:', JSON.stringify(r.colorPalette)?.slice(0, 60));
    return normalizeBrandKit({
      logo: logoUrl,
      color_palette: r.colorPalette,
      fontHeading: r.headingFont,
      fontBody: r.bodyFont,
    });
  } catch (e) {
    console.warn('[BrandKit] Load from DB failed:', e?.message || e);
    return null;
  }
}

/**
 * Persist normalized brand kit to public.brand_kits (upsert by user_id + workspace_id).
 * No-op if DB not configured or userId missing. Safe to call from request handler.
 * @param {string} userId
 * @param {string|null} [workspaceId]
 * @param {object} brandKit - normalized shape (primaryColor, fontHeading, etc.)
 * @returns {Promise<boolean>} true if saved
 */
export async function saveBrandKitToDb(userId, workspaceId, brandKit) {
  if (!userId || !brandKit || typeof brandKit !== 'object') {
    if (brandKit && !userId) console.log('[BrandKit] Save skipped: no userId in request');
    return false;
  }
  const numericUserId = parseUserId(userId);
  if (numericUserId === null) {
    console.log('[BrandKit] Save skipped: invalid userId format (expected number or string containing number):', userId);
    return false;
  }

  if (!db.isUsingPostgres()) {
    console.log('[BrandKit] Save skipped: database not configured (set DATABASE_URL or LOCAL_DATABASE_URL in .env)');
    return false;
  }
  const ws = parseWorkspaceId(workspaceId);
  const colorPalette = {
    primary: brandKit.primaryColor ?? null,
    secondary: brandKit.secondaryColor ?? null,
    accent: brandKit.accentColor ?? null,
  };
  try {
    const { rows } = await db.query(
      `SELECT id FROM public.brand_kits WHERE "userId" = $1 AND "workspaceId" = $2 LIMIT 1`,
      [numericUserId, ws]
    );
    if (rows?.length > 0) {
      await db.query(
        `UPDATE public.brand_kits SET logo = $1, "colorPalette" = $2, "headingFont" = $3, "bodyFont" = $4, "updatedAt" = now() WHERE "userId" = $5 AND "workspaceId" = $6`,
        [
          brandKit.logoUrl ?? null,
          JSON.stringify(colorPalette),
          brandKit.fontHeading ?? null,
          brandKit.fontBody ?? null,
          numericUserId,
          ws,
        ]
      );
    } else {
      await db.query(
        `INSERT INTO public.brand_kits ("userId", "workspaceId", name, logo, "colorPalette", "headingFont", "bodyFont", status) VALUES ($1, $2, $3, $4, $5, $6, $7, 'active')`,
        [
          numericUserId,
          ws,
          brandKit.name ?? 'Default',
          brandKit.logoUrl ?? null,
          JSON.stringify(colorPalette),
          brandKit.fontHeading ?? null,
          brandKit.fontBody ?? null,
        ]
      );
    }
    return true;
  } catch (e) {
    console.warn('[BrandKit] Save to DB failed:', e?.message || e);
    return false;
  }
}

const BrandKitReadSchema = z.object({
  keys: z.array(z.string()).optional().describe('Specific keys to read (e.g. primaryColor, fontHeading). Omit for full kit.'),
});

export const brandKitTool = tool(
  async ({ keys }) => {
    const kit = getBrandKit();
    if (keys && keys.length) {
      const out = {};
      keys.forEach((k) => { if (kit[k] !== undefined) out[k] = kit[k]; });
      return JSON.stringify(out);
    }
    return JSON.stringify(kit);
  },
  {
    name: 'brand_kit_read',
    description: 'Read Brand Kit (colors, logo, fonts, tone) for consistent visuals and copy.',
    schema: BrandKitReadSchema,
  }
);

export default brandKitTool;
