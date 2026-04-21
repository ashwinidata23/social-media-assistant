/**
 * S3 Upload Utility
 * Uploads a Buffer to S3 (private), then returns a pre-signed URL for temporary public access.
 * Uses @aws-sdk/client-s3 + @aws-sdk/s3-request-presigner (AWS SDK v3).
 */

import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import crypto from 'crypto';
import { config } from '../../config/index.js';

let _s3Client = null;

function getS3Client() {
    if (_s3Client) return _s3Client;
    const { accessKeyId, secretAccessKey, region } = config.aws;
    if (!accessKeyId || !secretAccessKey) return null;
    _s3Client = new S3Client({
        region,
        credentials: { accessKeyId, secretAccessKey },
    });
    return _s3Client;
}

/**
 * Check if S3 is configured (credentials + bucket present).
 */
export function isS3Configured() {
    return !!(config.aws?.accessKeyId && config.aws?.secretAccessKey && config.aws?.bucket);
}

/**
 * Upload a Buffer to S3 (private) and return a pre-signed URL for temporary access.
 *
 * @param {Buffer} buffer - Image buffer to upload
 * @param {object} opts
 * @param {string} [opts.key]           - S3 object key (auto-generated if not provided)
 * @param {string} [opts.contentType]   - MIME type, default 'image/png'
 * @param {string} [opts.userId]        - Used in auto-generated key path
 * @param {string} [opts.workspaceId]   - Used in auto-generated key path
 * @param {string} [opts.aspectRatio]   - Used in auto-generated key path (e.g. '1:1')
 * @param {number} [opts.expiresIn]     - Pre-signed URL expiry in seconds (default: 7 days)
 * @returns {Promise<string>} - Pre-signed S3 URL
 */
export async function uploadBufferToS3(buffer, opts = {}) {
    const s3 = getS3Client();
    if (!s3) throw new Error('[S3] AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env');

    const { bucket, prefix } = config.aws;
    const contentType = opts.contentType || 'image/png';
    const expiresIn = opts.expiresIn ?? 604800; // 7 days default

    // Build key: {prefix}/{userId}/{workspaceId}/{uuid}_{aspect}.png
    let key = opts.key;
    if (!key) {
        const uid = opts.userId || 'anon';
        const wid = opts.workspaceId || 'default';
        const slug = opts.aspectRatio ? opts.aspectRatio.replace(/[^0-9]+/g, '-').replace(/^-+|-+$/g, '') : 'img';
        const uuid = crypto.randomUUID();
        key = `${prefix}/${uid}/${wid}/${uuid}_${slug}.png`;
    }

    // Step 1: Upload privately to S3
    await s3.send(new PutObjectCommand({
        Bucket: bucket,
        Key: key,
        Body: buffer,
        ContentType: contentType,
    }));
    console.log('[S3] Uploaded (private):', key);

    // Step 2: Generate pre-signed URL for temporary public access
    const command = new GetObjectCommand({ Bucket: bucket, Key: key });
    const signedUrl = await getSignedUrl(s3, command, { expiresIn });
    console.log('[S3] Pre-signed URL generated (expires in', expiresIn, 'seconds):', signedUrl.slice(0, 100) + '...');
    return signedUrl;
}

/**
 * Upload a document (e.g. PDF) to S3 and return permanent URLs (or S3 metadata) 
 * suitable for database storage in knowledge base context.
 */
export async function uploadDocumentToS3(buffer, opts = {}) {
    const s3 = getS3Client();
    if (!s3) throw new Error('[S3] AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env');

    const { bucket, prefix } = config.aws;
    const region = config.aws.region || 'us-east-1';
    const contentType = opts.contentType || 'application/pdf';

    const uid = opts.userId || 'anon';
    const wid = opts.workspaceId || 'default';
    const filename = (opts.filename || 'document.pdf').replace(/[^a-zA-Z0-9.-]/g, '_');
    const uuid = crypto.randomUUID();
    const key = `documents/${uid}/${wid}/${uuid}_${filename}`;

    await s3.send(new PutObjectCommand({
        Bucket: bucket,
        Key: key,
        Body: buffer,
        ContentType: contentType,
    }));

    // Construct public or raw s3 URL depending on bucket settings.
    const s3Url = `https://${bucket}.s3.${region}.amazonaws.com/${key}`;
    console.log('[S3] Uploaded document (raw URL):', s3Url);

    // Generate pre-signed URL for temporary public access (frontend display)
    const expiresIn = opts.expiresIn ?? 604800; // 7 days default
    const command = new GetObjectCommand({ Bucket: bucket, Key: key });
    const signedUrl = await getSignedUrl(s3, command, { expiresIn });
    console.log('[S3] Document pre-signed URL generated (expires in', expiresIn, 'seconds):', signedUrl.slice(0, 100) + '...');

    return { s3Url: signedUrl, rawUrl: s3Url, key, bucket, contentType };
}

export default { uploadBufferToS3, uploadDocumentToS3, isS3Configured };
