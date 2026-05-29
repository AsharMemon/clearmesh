function bridgeBase() {
  return (process.env.CLEARMESH_INFERENCE_BASE_URL || '').replace(/\/+$/, '');
}

function bridgeHeaders(extra = {}) {
  const headers = { Accept: 'application/json', ...extra };
  if (process.env.CLEARMESH_INFERENCE_TOKEN) {
    headers.Authorization = `Bearer ${process.env.CLEARMESH_INFERENCE_TOKEN}`;
  }
  return headers;
}

async function bridgeFetch(path, init = {}) {
  const base = bridgeBase();
  if (!base) {
    const err = new Error('remote inference bridge is not connected');
    err.status = 404;
    throw err;
  }
  const url = `${base}/${String(path || '').replace(/^\/+/, '')}`;
  return fetch(url, init);
}

module.exports = { bridgeBase, bridgeFetch, bridgeHeaders };
