const {
  clearSessionCookie,
  json,
  publicUser,
  readJson,
  sessionCookie,
  sessionFromRequest,
  sign,
  userIdForEmail,
} = require('./_lib/session');

const SESSION_TTL_SECONDS = 60 * 60 * 24 * 14;

module.exports = async function handler(req, res) {
  const action = String(req.query.action || 'session');

  if (req.method === 'GET' && action === 'session') {
    const session = sessionFromRequest(req);
    if (!session) return json(res, 401, { authenticated: false });
    return json(res, 200, { authenticated: true, user: publicUser(session) });
  }

  if (req.method === 'POST' && action === 'login') {
    const body = await readJson(req).catch(() => ({}));
    const email = String(body.email || '').trim().toLowerCase();
    const expectedCode = process.env.CLEARMESH_APP_ACCESS_CODE || '';
    if (!email || !email.includes('@')) return json(res, 400, { detail: 'valid email required' });
    if (expectedCode && body.accessCode !== expectedCode) return json(res, 401, { detail: 'invalid access code' });
    const token = sign({ email, user_id: userIdForEmail(email), exp: Date.now() + SESSION_TTL_SECONDS * 1000 });
    const session = { email, user_id: userIdForEmail(email) };
    return json(res, 200, { authenticated: true, user: publicUser(session) }, {
      'Set-Cookie': sessionCookie(token, SESSION_TTL_SECONDS),
    });
  }

  if (req.method === 'POST' && action === 'logout') {
    return json(res, 200, { ok: true }, { 'Set-Cookie': clearSessionCookie() });
  }

  return json(res, 404, { detail: 'auth route not found' });
};
