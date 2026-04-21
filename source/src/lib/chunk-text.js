/**
 * Chunk text for embedding storage. Splits by paragraph then by size with overlap.
 */

const DEFAULT_CHUNK_SIZE = 600;
const DEFAULT_OVERLAP = 100;

/**
 * Split text into chunks of roughly chunkSize characters with overlap.
 * Tries to split on paragraph (\n\n), then \n, then space.
 * @param {string} text - Full text to chunk
 * @param {{ chunkSize?: number, overlap?: number }} options
 * @returns {string[]} Array of chunk strings
 */
export function chunkText(text, options = {}) {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const overlap = options.overlap ?? DEFAULT_OVERLAP;
  const trimmed = (text || '').trim();
  if (!trimmed) return [];

  const chunks = [];
  const separators = ['\n\n', '\n', '. ', ' '];

  function splitAtSeparator(str, seps) {
    if (str.length <= chunkSize) {
      if (str.length > 0) chunks.push(str);
      return;
    }
    const sep = seps[0];
    const parts = str.split(sep);
    let current = '';
    for (let i = 0; i < parts.length; i++) {
      const next = i < parts.length - 1 ? parts[i] + sep : parts[i];
      if (current.length + next.length <= chunkSize) {
        current += next;
      } else {
        if (current.length > 0) {
          chunks.push(current.trim());
          const overlapStart = Math.max(0, current.length - overlap);
          current = current.slice(overlapStart) + next;
        } else {
          if (next.length <= chunkSize) {
            current = next;
          } else {
            if (seps.length > 1) {
              splitAtSeparator(next, seps.slice(1));
            } else {
              for (let j = 0; j < next.length; j += chunkSize - overlap) {
                const piece = next.slice(j, j + chunkSize).trim();
                if (piece) chunks.push(piece);
              }
            }
          }
        }
      }
    }
    if (current.trim().length > 0) chunks.push(current.trim());
  }

  splitAtSeparator(trimmed, separators);
  return chunks.filter((c) => c.length > 0);
}

export default chunkText;
