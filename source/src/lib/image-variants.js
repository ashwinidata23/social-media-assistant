import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

let canvasModPromise = null;
async function getCanvasMod() {
  if (!canvasModPromise) canvasModPromise = import('@napi-rs/canvas');
  return canvasModPromise;
}

const DEFAULT_SIZES_BY_RATIO = {
  '1:1': { width: 1024, height: 1024 },
  '9:16': { width: 1080, height: 1920 },
  '16:9': { width: 1920, height: 1080 },
};

function drawCover(ctx, img, width, height) {
  const scale = Math.max(width / img.width, height / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  const x = (width - w) / 2;
  const y = (height - h) / 2;
  ctx.drawImage(img, x, y, w, h);
}

function drawContain(ctx, img, width, height) {
  const scale = Math.min(width / img.width, height / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  const x = (width - w) / 2;
  const y = (height - h) / 2;
  ctx.drawImage(img, x, y, w, h);
}

async function renderPaddedVariantPng(imageBuffer, { width, height }) {
  const { createCanvas, loadImage } = await getCanvasMod().catch((e) => {
    const msg =
      e?.message ||
      'Failed to load @napi-rs/canvas (required to derive aspect-ratio variants). Install it with: npm i @napi-rs/canvas';
    const err = new Error(msg);
    err.cause = e;
    throw err;
  });

  const img = await loadImage(imageBuffer);
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // Draw image to fully cover the target size (may crop slightly, but no blur or padding).
  drawCover(ctx, img, width, height);

  return canvas.toBuffer('image/png');
}

function sanitizeRatioForFilename(ratio) {
  return String(ratio).replace(/[^0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

/**
 * Save padded aspect-ratio variants for a base64 image.
 * Returns URLs rooted at `/generated/...` (served from `public/generated`).
 *
 * @param {object} args
 * @param {string} args.base64 - raw base64 image bytes (no data URI prefix)
 * @param {string[]} args.aspectRatios - e.g. ['1:1','9:16','16:9']
 * @param {string} [args.publicDirAbs] - defaults to `<cwd>/public`
 * @param {object} [args.sizesByRatio] - override pixel sizes per ratio
 * @param {string} [args.fileStem] - optional deterministic stem
 */
export async function saveAspectRatioVariantsFromBase64({
  base64,
  aspectRatios,
  publicDirAbs = path.join(process.cwd(), 'public'),
  sizesByRatio = DEFAULT_SIZES_BY_RATIO,
  fileStem,
}) {
  const ratios = Array.isArray(aspectRatios) && aspectRatios.length ? aspectRatios : ['1:1', '9:16', '16:9'];
  const buf = Buffer.from(String(base64 || ''), 'base64');

  const outDirAbs = path.join(publicDirAbs, 'generated');
  await fs.mkdir(outDirAbs, { recursive: true });

  const stem = fileStem || crypto.randomUUID();
  const urlsByAspectRatio = {};

  for (const ratio of ratios) {
    const size = sizesByRatio[ratio];
    if (!size?.width || !size?.height) continue;

    const png = await renderPaddedVariantPng(buf, size);
    const fname = `${stem}_${sanitizeRatioForFilename(ratio)}.png`;
    const outPath = path.join(outDirAbs, fname);
    await fs.writeFile(outPath, png);
    urlsByAspectRatio[ratio] = `/generated/${fname}`;
  }

  return { urlsByAspectRatio, stem };
}

/**
 * Fetch a remote image URL and return it as a Buffer.
 * Works for any HTTP/HTTPS URL (Grok-returned URLs, S3 URLs, etc.)
 * @param {string} url
 * @returns {Promise<Buffer>}
 */
export async function fetchImageBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetchImageBuffer: HTTP ${res.status} for ${url}`);
  const arrayBuf = await res.arrayBuffer();
  return Buffer.from(arrayBuf);
}

/**
 * Composite a logo image on top of a base image (top-right corner).
 * Logo is scaled to ~18% of the shorter canvas dimension with a 2% margin.
 *
 * @param {Buffer} baseImageBuffer - PNG/JPEG buffer of the generated image
 * @param {Buffer} logoBuffer      - PNG/JPEG/WebP buffer of the logo
 * @returns {Promise<Buffer>}      - PNG buffer with logo composited on top
 */
export async function compositeLogoOnImage(baseImageBuffer, logoBuffer) {
  const { createCanvas, loadImage } = await getCanvasMod().catch((e) => {
    const msg = e?.message || 'Failed to load @napi-rs/canvas for logo compositing.';
    throw Object.assign(new Error(msg), { cause: e });
  });

  const base = await loadImage(baseImageBuffer);
  const logo = await loadImage(logoBuffer);

  const W = base.width;
  const H = base.height;

  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');

  // Draw base image to fill canvas exactly
  ctx.drawImage(base, 0, 0, W, H);

  // Constrained fit: logo must fit within 25% wide × 18% tall box (shorter edge as reference).
  // Preserves the logo's own aspect ratio — whichever dimension hits its limit controls the scale.
  const shorter = Math.min(W, H);
  const maxLogoW = Math.round(shorter * 0.25);  // max 25% of shorter edge wide
  const maxLogoH = Math.round(shorter * 0.18);  // max 18% of shorter edge tall
  const logoAspect = logo.width / logo.height;  // logo's own width:height ratio

  let logoTargetW = maxLogoW;
  let logoTargetH = Math.round(logoTargetW / logoAspect);
  if (logoTargetH > maxLogoH) {
    // Logo is too tall — clamp height and recompute width
    logoTargetH = maxLogoH;
    logoTargetW = Math.round(logoTargetH * logoAspect);
  }

  // 2.5% margin from top-right corner
  const margin = Math.round(shorter * 0.025);
  const logoX = W - logoTargetW - margin;  // right edge
  const logoY = margin;                     // top edge

  ctx.drawImage(logo, logoX, logoY, logoTargetW, logoTargetH);

  return canvas.toBuffer('image/png');
}

export default {
  saveAspectRatioVariantsFromBase64,
  compositeLogoOnImage,
};

