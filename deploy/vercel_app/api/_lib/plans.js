const PLANS = {
  free: {
    tier: 'free',
    name: 'Free',
    price: '$0',
    interval: 'month',
    monthly_credits: 50,
    features: ['Text to mesh', 'Image to mesh', 'GLB export', 'Personal use'],
  },
  pro: {
    tier: 'pro',
    name: 'Pro',
    price: '$24',
    interval: 'month',
    monthly_credits: 500,
    features: ['Commercial license', 'Priority queue', 'Image conditioning', 'Full library storage'],
  },
  studio: {
    tier: 'studio',
    name: 'Studio',
    price: '$89',
    interval: 'seat / month',
    monthly_credits: 2500,
    features: ['Team workspace', 'Parallel jobs', 'API keys', 'Retention controls'],
  },
};

function publicPlans() {
  return ['free', 'pro', 'studio'].map((tier) => PLANS[tier]);
}

function planForTier(tier = 'free') {
  return PLANS[tier] || PLANS.free;
}

function stripePriceForTier(tier) {
  const key = `CLEARMESH_STRIPE_PRICE_${String(tier || '').toUpperCase()}`;
  return process.env[key] || '';
}

module.exports = { PLANS, publicPlans, planForTier, stripePriceForTier };
