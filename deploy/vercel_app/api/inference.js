const DEFAULT_PIPELINE = 'Text/Image -> Trellis -> FACE-Q -> Texture/UV -> Easy3E-ready';
const { json, readJson, sessionFromRequest } = require('./_lib/session');
const { bridgeBase, bridgeFetch, bridgeHeaders } = require('./_lib/inference-bridge');
const { persistMeshFromJob } = require('./_lib/library');

function joinPath(value) {
  if (Array.isArray(value)) return value.join('/');
  return String(value || '').replace(/^\/+|\/+$/g, '');
}

function artifactUrls(job, basePath) {
  const artifacts = job.artifacts || {};
  const urls = {};
  for (const name of Object.keys(artifacts)) {
    urls[name] = `${basePath}/${encodeURIComponent(name)}`;
  }
  return urls;
}

function mergeArtifactUrls(primary = {}, secondary = {}) {
  return { ...secondary, ...primary };
}

function extensionFor(contentType = '', name = 'artifact') {
  if (contentType.includes('model/gltf-binary')) return 'glb';
  if (contentType.includes('model/gltf+json')) return 'gltf';
  if (contentType.includes('png')) return 'png';
  if (contentType.includes('jpeg')) return 'jpg';
  if (contentType.includes('webp')) return 'webp';
  if (contentType.includes('json')) return 'json';
  return name.includes('mesh') ? 'glb' : 'bin';
}

module.exports = async function handler(req, res) {
  const path = joinPath(req.query.path);
  const base = bridgeBase();

  try {
    const session = sessionFromRequest(req);
    if (!session) return json(res, 401, { detail: 'sign in required' });

    if (req.method === 'GET' && path === 'config') {
      if (!base) {
        return json(res, 200, {
          available: false,
          stage: 'local_preview',
          pipeline: {
            text_to_image: 'configurable open-source model',
            trellis: 'microsoft/TRELLIS.2-4B',
            faceq: 'faceq-1p4b-65k1024-effective100k',
            texture_uv: true,
            easy3e: true,
          },
        });
      }
      const upstream = await bridgeFetch('/v1/pipeline/config', { headers: bridgeHeaders() });
      const payload = await upstream.json().catch(() => ({}));
      return json(res, upstream.ok ? 200 : 502, {
        available: upstream.ok,
        stage: upstream.ok ? 'remote' : 'bridge_error',
        pipeline: {
          text_to_image: payload.text_to_image_model || 'configurable open-source model',
          text_to_image_backend: payload.text_to_image_backend || 'unknown',
          trellis: payload.trellis_model || 'microsoft/TRELLIS.2-4B',
          faceq: payload.faceq_checkpoint_exists ? 'faceq-1p4b-65k1024-effective100k' : 'checkpoint missing',
          texture_uv: Boolean(payload.textures?.postprocess_enabled),
          easy3e: Boolean(payload.easy3e_enabled),
        },
        bridge: payload,
      });
    }

    if (req.method === 'POST' && path === 'demo') {
      const body = await readJson(req);
      if (!base) {
        return json(res, 202, {
          ok: true,
          status: 'local_preview',
          job_id: `demo_${Date.now().toString(36)}`,
          message: 'Remote inference GPU is not connected yet; showing local preview only.',
          pipeline: DEFAULT_PIPELINE,
        });
      }
      const upstream = await bridgeFetch('/v1/pipeline/jobs', {
        method: 'POST',
        headers: bridgeHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(body),
      });
      const payload = await upstream.json().catch(() => ({}));
      payload.ok = upstream.ok;
      payload.pipeline = payload.pipeline || DEFAULT_PIPELINE;
      return json(res, upstream.ok ? 202 : 502, payload);
    }

    const jobMatch = path.match(/^jobs\/([^/]+)(?:\/artifacts\/([^/]+))?$/);
    if (req.method === 'GET' && jobMatch) {
      const jobId = decodeURIComponent(jobMatch[1]);
      const artifactName = jobMatch[2] ? decodeURIComponent(jobMatch[2]) : null;
      if (!artifactName) {
        const upstream = await bridgeFetch(`/v1/pipeline/jobs/${encodeURIComponent(jobId)}`, { headers: bridgeHeaders() });
        const payload = await upstream.json().catch(() => ({}));
        if (upstream.ok) {
          payload.artifact_urls = artifactUrls(payload, `/v1/inference/jobs/${encodeURIComponent(jobId)}/artifacts`);
          try {
            const stored = await persistMeshFromJob(session, payload, bridgeFetch);
            if (stored?.stored && stored.mesh) {
              payload.library_mesh = stored.mesh;
              payload.artifact_urls = mergeArtifactUrls(payload.artifact_urls, stored.mesh.artifact_urls);
            }
          } catch (storageErr) {
            payload.storage_warning = storageErr.message || 'mesh storage failed';
          }
        }
        return json(res, upstream.ok ? 200 : upstream.status, payload);
      }

      const upstream = await bridgeFetch(
        `/v1/pipeline/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(artifactName)}`,
        { headers: bridgeHeaders({ Accept: '*/*' }) }
      );
      const buffer = Buffer.from(await upstream.arrayBuffer());
      const contentType = upstream.headers.get('content-type') || 'application/octet-stream';
      const disposition = req.query.download ? 'attachment' : 'inline';
      res.statusCode = upstream.status;
      res.setHeader('Content-Type', contentType);
      res.setHeader('Cache-Control', 'private, max-age=0, must-revalidate');
      res.setHeader('Content-Disposition', `${disposition}; filename="${jobId}-${artifactName}.${extensionFor(contentType, artifactName)}"`);
      res.end(buffer);
      return;
    }

    return json(res, 404, { detail: 'inference route not found' });
  } catch (err) {
    return json(res, err.status || 502, { detail: err.message || 'inference proxy failed' });
  }
};
