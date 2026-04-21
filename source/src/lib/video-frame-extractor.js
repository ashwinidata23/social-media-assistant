import { spawn } from 'child_process';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import ffmpegPath from 'ffmpeg-static';

/**
 * Extract a single frame from a video URL or local file as a JPEG buffer.
 * Requires ffmpeg to be installed and available on the system PATH.
 *
 * @param {string} videoPathOrUrl - Local filesystem path or HTTP(S) URL to the video.
 * @param {number} timestampSeconds - Timestamp (in seconds) at which to grab the frame.
 * @returns {Promise<Buffer>} - JPEG image buffer for the extracted frame.
 */
export async function extractFrame(videoPathOrUrl, timestampSeconds = 1) {
  if (!videoPathOrUrl || typeof videoPathOrUrl !== 'string') {
    throw new Error('extractFrame: videoPathOrUrl must be a non-empty string');
  }

  const outPath = path.join(tmpdir(), `zunosync-frame-${randomUUID()}.jpg`);

  return new Promise((resolve, reject) => {
    const args = [
      '-y',                     // overwrite output without asking
      '-ss', String(timestampSeconds), // seek to timestamp
      '-i', videoPathOrUrl,     // input
      '-frames:v', '1',         // one video frame
      '-q:v', '2',              // quality (lower is better; 2 is very good)
      outPath,
    ];

    const proc = spawn(ffmpegPath || 'ffmpeg', args);

    proc.on('error', (err) => {
      reject(err);
    });

    proc.on('close', async (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}`));
        return;
      }
      try {
        const buf = await fs.readFile(outPath);
        await fs.unlink(outPath).catch(() => {});
        resolve(buf);
      } catch (e) {
        reject(e);
      }
    });
  });
}

