const crypto = require('crypto');
const { downloadJson, isConfigured, keyPath, uploadJson } = require('./_lib/b2');
const { planForTier, publicPlans, stripePriceForTier } = require('./_lib/plans');
const { json, readJson, sessionFromRequest } = require('./_lib/session');

function originFromRequest(req) {
  const configured = process.env.CLEARMESH_PUBLIC_APP_URL || process.env.VERCEL_PROJECT_PRODUCTION_URL || '';
  if (configured) return configured.startsWith('http') ? configured.replace(/\/+$/, '') : `https://${configured.replace(/\/+$/, '')}`;
  const proto = req.headers['x-forwarded-proto'] || 'https';
  const host = req.headers['x-forwarded-host'] || req.headers.host || 'localhost:3000';
  return `${proto}://${host}`;
}

function subscriptionKey(userId) {
  return keyPath('users', String(userId).replace(/[^a-zA-Z0-9_.:-]/g, '_'), 'subscription.json');
}

async function loadSubscription(session) {
  const fallback = {
    configured: isConfigured(),
    tier: 'free',
    status: 'free',
    credits_used: 0,
    plan: planForTier('free'),
  };
  if (!isConfigured()) return fallback;
  try {
    const subscription = await downloadJson(subscriptionKey(session.user_id));
    const plan = planForTier(subscription.tier);
    return { ...fallback, ...subscription, plan };
  } catch (_err) {
    return fallback;
  }
}

async function saveSubscription(userId, patch) {
  if (!isConfigured()) return { configured: false };
  const next = {
    tier: patch.tier || 'free',
    status: patch.status || 'active',
    credits_used: Number.isFinite(patch.credits_used) ? patch.credits_used : 0,
    stripe_customer_id: patch.stripe_customer_id || patch.customer || null,
    stripe_subscription_id: patch.stripe_subscription_id || patch.subscription || null,
    updated_at: new Date().toISOString(),
  };
  next.plan = planForTier(next.tier);
  await uploadJson(subscriptionKey(userId), next);
  return { configured: true, subscription: next };
}

async function stripePost(path, params) {
  const secret = process.env.STRIPE_SECRET_KEY || '';
  if (!secret) {
    const err = new Error('Stripe is not configured');
    err.setupNeeded = true;
    throw err;
  }
  const body = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') body.append(key, String(value));
  }
  const res = await fetch(`https://api.stripe.com/v1/${path.replace(/^\/+/, '')}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${secret}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body,
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(payload.error?.message || `Stripe request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return payload;
}

async function readRawBody(req) {
  if (Buffer.isBuffer(req.body)) return req.body.toString('utf8');
  if (typeof req.body === 'string') return req.body;
  const chunks = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf8');
}

function verifyStripeSignature(rawBody, header, secret) {
  if (!secret) throw new Error('Stripe webhook secret is not configured');
  const parts = Object.fromEntries(String(header || '').split(',').map((part) => {
    const [key, value] = part.split('=', 2);
    return [key, value];
  }));
  if (!parts.t || !parts.v1) throw new Error('missing Stripe signature');
  const signed = `${parts.t}.${rawBody}`;
  const expected = crypto.createHmac('sha256', secret).update(signed).digest('hex');
  const a = Buffer.from(parts.v1, 'hex');
  const b = Buffer.from(expected, 'hex');
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) throw new Error('invalid Stripe signature');
  const age = Math.abs(Math.floor(Date.now() / 1000) - Number(parts.t));
  if (age > 300) throw new Error('stale Stripe signature');
}

async function handleWebhook(req, res) {
  const rawBody = await readRawBody(req);
  verifyStripeSignature(rawBody, req.headers['stripe-signature'], process.env.STRIPE_WEBHOOK_SECRET || '');
  const event = JSON.parse(rawBody);
  const object = event.data?.object || {};
  const metadata = object.metadata || {};
  const userId = metadata.user_id || object.client_reference_id || null;
  const tier = metadata.tier || 'pro';

  if (event.type === 'checkout.session.completed' && userId) {
    await saveSubscription(userId, {
      tier,
      status: object.payment_status === 'paid' || object.status === 'complete' ? 'active' : object.status || 'active',
      stripe_customer_id: object.customer,
      stripe_subscription_id: object.subscription,
    });
  }

  if ((event.type === 'customer.subscription.updated' || event.type === 'customer.subscription.deleted') && userId) {
    await saveSubscription(userId, {
      tier,
      status: event.type === 'customer.subscription.deleted' ? 'canceled' : object.status || 'active',
      stripe_customer_id: object.customer,
      stripe_subscription_id: object.id,
    });
  }

  return json(res, 200, { received: true });
}

module.exports = async function handler(req, res) {
  const action = String(req.query.action || 'subscription');
  try {
    if (action === 'webhook' && req.method === 'POST') return await handleWebhook(req, res);

    if (req.method === 'GET' && action === 'plans') return json(res, 200, { plans: publicPlans() });

    const session = sessionFromRequest(req);
    if (!session) return json(res, 401, { detail: 'sign in required' });

    if (req.method === 'GET' && action === 'subscription') {
      return json(res, 200, await loadSubscription(session));
    }

    if (req.method === 'POST' && action === 'checkout') {
      const body = await readJson(req).catch(() => ({}));
      const tier = String(body.tier || 'pro').toLowerCase();
      if (tier === 'free') return json(res, 200, { checkout_url: '/app', tier: 'free' });
      const priceId = stripePriceForTier(tier);
      if (!priceId || !process.env.STRIPE_SECRET_KEY) {
        return json(res, 200, {
          status: 'setup_needed',
          detail: 'Stripe price IDs and STRIPE_SECRET_KEY are needed before paid checkout can go live.',
          missing: {
            stripe_secret: !process.env.STRIPE_SECRET_KEY,
            price_id: !priceId,
          },
        });
      }
      const origin = originFromRequest(req);
      const checkout = await stripePost('checkout/sessions', {
        mode: 'subscription',
        client_reference_id: session.user_id,
        customer_email: session.email,
        'line_items[0][price]': priceId,
        'line_items[0][quantity]': 1,
        success_url: `${origin}/app?checkout=success`,
        cancel_url: `${origin}/pricing?checkout=cancelled`,
        allow_promotion_codes: true,
        'metadata[user_id]': session.user_id,
        'metadata[tier]': tier,
        'subscription_data[metadata][user_id]': session.user_id,
        'subscription_data[metadata][tier]': tier,
      });
      return json(res, 200, { checkout_url: checkout.url, checkout_id: checkout.id });
    }

    if (req.method === 'POST' && action === 'portal') {
      const subscription = await loadSubscription(session);
      if (!subscription.stripe_customer_id || !process.env.STRIPE_SECRET_KEY) {
        return json(res, 200, { status: 'setup_needed', detail: 'No Stripe customer is linked to this account yet.' });
      }
      const origin = originFromRequest(req);
      const portal = await stripePost('billing_portal/sessions', {
        customer: subscription.stripe_customer_id,
        return_url: `${origin}/app`,
      });
      return json(res, 200, { portal_url: portal.url });
    }

    return json(res, 404, { detail: 'billing route not found' });
  } catch (err) {
    const status = err.setupNeeded ? 200 : (err.status || 502);
    return json(res, status, { status: err.setupNeeded ? 'setup_needed' : 'error', detail: err.message || 'billing request failed' });
  }
};
