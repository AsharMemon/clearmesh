const { artifactForMesh, listMeshes, persistMeshFromJob } = require('./_lib/library');
const { bridgeFetch, bridgeHeaders } = require('./_lib/inference-bridge');
const { json, readJson, sessionFromRequest } = require('./_lib/session');

function extensionFor(contentType = '', name = 'mesh') {
  if (contentType.includes('model/gltf-binary')) return 'glb';
  if (contentType.includes('model/gltf+json')) return 'gltf';
  if (contentType.includes('png')) return 'png';
  if (contentType.includes('jpeg')) return 'jpg';
  if (contentType.includes('webp')) return 'webp';
  return name.includes('image') ? 'png' : 'bin';
}

module.exports = async function handler(req, res) {
  const action = String(req.query.action || 'list');
  try {
    const session = sessionFromRequest(req);
    if (!session) return json(res, 401, { detail: 'sign in required' });

    if (req.method === 'GET' && action === 'list') {
      return json(res, 200, await listMeshes(session));
    }

    if (req.method === 'POST' && action === 'store') {
      const body = await readJson(req).catch(() => ({}));
      const jobId = String(body.jobId || body.job_id || '').trim();
      if (!jobId) return json(res, 400, { detail: 'jobId is required' });
      const upstream = await bridgeFetch(`/v1/pipeline/jobs/${encodeURIComponent(jobId)}`, { headers: bridgeHeaders() });
      const job = await upstream.json().catch(() => ({}));
      if (!upstream.ok) return json(res, upstream.status, job || { detail: 'job not found' });
      const stored = await persistMeshFromJob(session, job, bridgeFetch);
      return json(res, stored?.stored ? 200 : 202, stored);
    }

    if (req.method === 'GET' && action === 'artifact') {
      const meshId = String(req.query.id || req.query.mesh_id || '').trim();
      const name = String(req.query.name || 'final_mesh').trim();
      if (!meshId) return json(res, 400, { detail: 'mesh id is required' });
      const artifact = await artifactForMesh(session, meshId, name);
      const ext = extensionFor(artifact.contentType, name);
      const disposition = req.query.download ? 'attachment' : 'inline';
      res.statusCode = 200;
      res.setHeader('Content-Type', artifact.contentType || 'application/octet-stream');
      res.setHeader('Cache-Control', 'private, max-age=300');
      res.setHeader('Content-Disposition', `${disposition}; filename="${meshId}-${name}.${ext}"`);
      res.end(artifact.bytes);
      return;
    }

    return json(res, 404, { detail: 'library route not found' });
  } catch (err) {
    return json(res, err.status || 502, { detail: err.message || 'library request failed' });
  }
};
