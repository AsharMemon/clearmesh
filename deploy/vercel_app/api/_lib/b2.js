const crypto = require('crypto');

let authCache = null;
let bucketCache = null;

function storageConfig() {
  const keyId = process.env.B2_KEYID || process.env.B2_KEY_ID || process.env.B2_APPLICATION_KEY_ID || '';
  const appKey = process.env.B2_APPKEY || process.env.B2_APP_KEY || process.env.B2_APPLICATION_KEY || '';
  const bucket = process.env.CLEARMESH_STORAGE_B2_BUCKET || process.env.B2_BUCKET || 'clearmesh-pairs';
  const prefix = (process.env.CLEARMESH_STORAGE_B2_PREFIX || 'clearmesh-app').replace(/^\/+|\/+$/g, '');
  return { keyId, appKey, bucket, prefix };
}

function isConfigured() {
  const cfg = storageConfig();
  return Boolean(cfg.keyId && cfg.appKey && cfg.bucket);
}

async function authorize() {
  const cfg = storageConfig();
  if (!isConfigured()) throw new Error('B2 storage is not configured');
  if (authCache && authCache.expiresAt > Date.now() + 60_000) return authCache;
  const basic = Buffer.from(`${cfg.keyId}:${cfg.appKey}`).toString('base64');
  const res = await fetch('https://api.backblazeb2.com/b2api/v4/b2_authorize_account', {
    headers: { Authorization: `Basic ${basic}` },
  });
  if (!res.ok) throw new Error(`B2 authorize failed (${res.status})`);
  const payload = await res.json();
  authCache = { ...payload, expiresAt: Date.now() + 1000 * 60 * 50 };
  return authCache;
}

async function bucketId() {
  const cfg = storageConfig();
  const auth = await authorize();
  const allowed = auth.allowed || auth.apiInfo?.storageApi?.allowed || {};
  if (allowed.bucketId && (!allowed.bucketName || allowed.bucketName === cfg.bucket)) return allowed.bucketId;
  if (bucketCache && bucketCache.name === cfg.bucket) return bucketCache.id;
  const apiUrl = auth.apiUrl || auth.apiInfo?.storageApi?.apiUrl;
  const res = await fetch(`${apiUrl}/b2api/v4/b2_list_buckets`, {
    method: 'POST',
    headers: { Authorization: auth.authorizationToken, 'Content-Type': 'application/json' },
    body: JSON.stringify({ accountId: auth.accountId }),
  });
  if (!res.ok) throw new Error(`B2 list buckets failed (${res.status})`);
  const payload = await res.json();
  const bucket = (payload.buckets || []).find((item) => item.bucketName === cfg.bucket);
  if (!bucket) throw new Error(`B2 bucket not found: ${cfg.bucket}`);
  bucketCache = { name: cfg.bucket, id: bucket.bucketId };
  return bucket.bucketId;
}

function keyPath(...parts) {
  const cfg = storageConfig();
  const clean = parts.flat().filter(Boolean).map((part) => String(part).replace(/^\/+|\/+$/g, '')).filter(Boolean).join('/');
  return cfg.prefix ? `${cfg.prefix}/${clean}` : clean;
}

function encodeFileName(key) {
  return encodeURIComponent(key).replace(/%2F/g, '/');
}

async function uploadBytes(key, bytes, contentType = 'application/octet-stream') {
  const auth = await authorize();
  const id = await bucketId();
  const apiUrl = auth.apiUrl || auth.apiInfo?.storageApi?.apiUrl;
  const uploadUrlRes = await fetch(`${apiUrl}/b2api/v4/b2_get_upload_url`, {
    method: 'POST',
    headers: { Authorization: auth.authorizationToken, 'Content-Type': 'application/json' },
    body: JSON.stringify({ bucketId: id }),
  });
  if (!uploadUrlRes.ok) throw new Error(`B2 get upload URL failed (${uploadUrlRes.status})`);
  const upload = await uploadUrlRes.json();
  const buffer = Buffer.from(bytes);
  const sha1 = crypto.createHash('sha1').update(buffer).digest('hex');
  const res = await fetch(upload.uploadUrl, {
    method: 'POST',
    headers: {
      Authorization: upload.authorizationToken,
      'X-Bz-File-Name': encodeFileName(key),
      'Content-Type': contentType,
      'Content-Length': String(buffer.length),
      'X-Bz-Content-Sha1': sha1,
    },
    body: buffer,
  });
  if (!res.ok) throw new Error(`B2 upload failed (${res.status})`);
  const payload = await res.json();
  return { key, fileId: payload.fileId, size: buffer.length, sha1 };
}

async function uploadJson(key, value) {
  return uploadBytes(key, Buffer.from(JSON.stringify(value, null, 2)), 'application/json; charset=utf-8');
}

async function downloadBytes(key) {
  const cfg = storageConfig();
  const auth = await authorize();
  const downloadUrl = auth.downloadUrl || auth.apiInfo?.storageApi?.downloadUrl;
  const res = await fetch(`${downloadUrl}/file/${encodeURIComponent(cfg.bucket)}/${encodeFileName(key)}`, {
    headers: { Authorization: auth.authorizationToken },
  });
  if (!res.ok) throw new Error(`B2 download failed (${res.status})`);
  return { bytes: Buffer.from(await res.arrayBuffer()), contentType: res.headers.get('content-type') || 'application/octet-stream' };
}

async function downloadJson(key) {
  const { bytes } = await downloadBytes(key);
  return JSON.parse(bytes.toString('utf8'));
}

async function listFileNames(prefix, maxFileCount = 1000) {
  const auth = await authorize();
  const id = await bucketId();
  const apiUrl = auth.apiUrl || auth.apiInfo?.storageApi?.apiUrl;
  const res = await fetch(`${apiUrl}/b2api/v4/b2_list_file_names`, {
    method: 'POST',
    headers: { Authorization: auth.authorizationToken, 'Content-Type': 'application/json' },
    body: JSON.stringify({ bucketId: id, prefix, maxFileCount }),
  });
  if (!res.ok) throw new Error(`B2 list files failed (${res.status})`);
  const payload = await res.json();
  return payload.files || [];
}

module.exports = {
  downloadBytes,
  downloadJson,
  isConfigured,
  keyPath,
  listFileNames,
  storageConfig,
  uploadBytes,
  uploadJson,
};
