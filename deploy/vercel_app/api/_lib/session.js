const crypto = require('crypto');

const COOKIE = 'cm_session';

function secret() {
  return process.env.CLEARMESH_SESSION_SECRET || process.env.CLEARMESH_INFERENCE_TOKEN || 'clearmesh-local-dev';
}

function userIdForEmail(email) {
  return `usr_${crypto.createHash('sha256').update(String(email || '').trim().toLowerCase()).digest('hex').slice(0, 24)}`;
}

function parseCookies(header = '') {
  return Object.fromEntries(header.split(';').map((part) => {
    const [key, ...rest] = part.trim().split('=');
    return [key, rest.join('=')];
  }).filter(([key]) => key));
}

function sign(payload) {
  const body = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const sig = crypto.createHmac('sha256', secret()).update(body).digest('base64url');
  return `${body}.${sig}`;
}

function verify(token) {
  if (!token || !token.includes('.')) return null;
  const [body, sig] = token.split('.', 2);
  const expected = crypto.createHmac('sha256', secret()).update(body).digest('base64url');
  try {
    if (!crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(expected))) return null;
    const payload = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
    if (!payload.exp || payload.exp < Date.now()) return null;
    const email = String(payload.email || '').trim().toLowerCase();
    if (!email || !email.includes('@')) return null;
    return { ...payload, email, user_id: payload.user_id || userIdForEmail(email) };
  } catch (_err) {
    return null;
  }
}

function sessionFromRequest(req) {
  return verify(parseCookies(req.headers.cookie || '')[COOKIE]);
}

function sessionCookie(token, maxAgeSeconds) {
  return `${COOKIE}=${token}; Path=/; Max-Age=${maxAgeSeconds}; HttpOnly; Secure; SameSite=Lax`;
}

function clearSessionCookie() {
  return `${COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax`;
}

function publicUser(session) {
  return {
    id: session.user_id,
    email: session.email,
    name: session.name || session.email.split('@')[0],
  };
}

function json(res, status, payload, headers = {}) {
  res.statusCode = status;
  for (const [key, value] of Object.entries(headers)) res.setHeader(key, value);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(payload));
}

async function readJson(req) {
  if (req.body && typeof req.body === 'object') return req.body;
  if (typeof req.body === 'string' && req.body.trim()) return JSON.parse(req.body);
  const chunks = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString('utf8').trim();
  return raw ? JSON.parse(raw) : {};
}

module.exports = {
  COOKIE,
  clearSessionCookie,
  json,
  parseCookies,
  publicUser,
  readJson,
  sessionCookie,
  sessionFromRequest,
  sign,
  userIdForEmail,
  verify,
};
