const {
  downloadBytes,
  downloadJson,
  isConfigured,
  keyPath,
  listFileNames,
  uploadBytes,
  uploadJson,
} = require('./b2');
const { publicUser } = require('./session');

const KNOWN_ARTIFACTS = [
  'final_mesh',
  'faceq_mesh',
  'faceq_rejected_mesh',
  'faceq_retry_mesh',
  'faceq_retry_rejected_mesh',
  'trellis_mesh',
  'trellis_faceq_proxy_mesh',
  'textured_mesh',
  'uv_mesh',
  'texture_uv_report',
  'preview_image',
  'trellis_preview_image',
  'final_preview_image',
  'input_image',
  'reference_image',
  'edited_mesh',
];
const MESH_EXTENSIONS = new Set(['glb', 'gltf', 'obj', 'stl', 'fbx', 'ply', 'usdz']);

function safeId(value) {
  return String(value || '').replace(/[^a-zA-Z0-9_.:-]/g, '_').slice(0, 160);
}

function fileExtension(contentType = '', artifactName = '') {
  if (artifactName.includes('image')) return contentType.includes('jpeg') ? 'jpg' : 'png';
  if (contentType.includes('model/gltf-binary')) return 'glb';
  if (contentType.includes('model/gltf+json')) return 'gltf';
  if (contentType.includes('json')) return 'json';
  if (contentType.includes('jpeg')) return 'jpg';
  if (contentType.includes('png')) return 'png';
  if (contentType.includes('webp')) return 'webp';
  const suffix = String(artifactName).split('.').pop();
  return suffix && suffix.length <= 5 ? suffix : 'bin';
}

function meshPrefix(userId) {
  return keyPath('users', safeId(userId), 'meshes');
}

function metadataKey(userId, meshId) {
  return `${meshPrefix(userId)}/${safeId(meshId)}/metadata.json`;
}

function artifactKey(userId, meshId, name, contentType = '') {
  const ext = fileExtension(contentType, name);
  return `${meshPrefix(userId)}/${safeId(meshId)}/artifacts/${safeId(name)}.${ext}`;
}

function artifactUrl(meshId, name) {
  return `/api/library?action=artifact&id=${encodeURIComponent(meshId)}&name=${encodeURIComponent(name)}`;
}

function publicMetadata(meta) {
  const artifacts = {};
  for (const [name, artifact] of Object.entries(meta.artifacts || {})) {
    artifacts[name] = {
      content_type: artifact.content_type,
      size: artifact.size,
      stored_at: artifact.stored_at,
    };
  }
  return {
    ...meta,
    user_id: undefined,
    artifacts,
    artifact_urls: meta.artifact_urls || Object.fromEntries(Object.keys(meta.artifacts || {}).map((name) => [name, artifactUrl(meta.id, name)])),
  };
}

async function loadMetadata(userId, meshId) {
  return downloadJson(metadataKey(userId, meshId));
}

async function listMeshes(session) {
  if (!isConfigured()) return { configured: false, meshes: [] };
  const prefix = `${meshPrefix(session.user_id)}/`;
  const files = await listFileNames(prefix, 10000);
  const metadataFiles = files.filter((item) => item.fileName.endsWith('/metadata.json'));
  const meshes = [];
  for (const file of metadataFiles) {
    try {
      meshes.push(publicMetadata(await downloadJson(file.fileName)));
    } catch (_err) {
      // A partially uploaded mesh should not break the whole library view.
    }
  }
  meshes.sort((a, b) => String(b.created_at || b.stored_at || '').localeCompare(String(a.created_at || a.stored_at || '')));
  return { configured: true, meshes };
}

function candidateArtifacts(job) {
  const names = new Set(KNOWN_ARTIFACTS);
  for (const name of Object.keys(job.artifacts || {})) names.add(name);
  for (const name of Object.keys(job.artifact_urls || {})) names.add(name);
  return Array.from(names).filter(Boolean);
}

function isUsefulArtifact(name, contentType, bytes) {
  if (!bytes?.length) return false;
  if (name === 'final_mesh') return true;
  if (name === 'texture_uv_report' && contentType.includes('json')) return true;
  if (contentType.startsWith('image/')) return true;
  const ext = fileExtension(contentType, name).toLowerCase();
  return MESH_EXTENSIONS.has(ext) || name.includes('mesh');
}

async function fetchArtifactFromBridge(jobId, name, bridgeFetch, accept = '*/*') {
  const upstream = await bridgeFetch(`/v1/pipeline/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(name)}`, {
    headers: { Accept: accept, ...(process.env.CLEARMESH_INFERENCE_TOKEN ? { Authorization: `Bearer ${process.env.CLEARMESH_INFERENCE_TOKEN}` } : {}) },
  });
  if (!upstream.ok) return null;
  const contentType = upstream.headers.get('content-type') || 'application/octet-stream';
  const bytes = Buffer.from(await upstream.arrayBuffer());
  if (!isUsefulArtifact(name, contentType, bytes)) return null;
  return { bytes, contentType };
}

async function persistMeshFromJob(session, job, bridgeFetch) {
  if (!isConfigured()) return { configured: false, stored: false };
  const jobId = job.id || job.job_id || job.remote_job_id;
  if (!jobId || job.status !== 'succeeded') return { configured: true, stored: false };
  const meshId = safeId(jobId);
  try {
    return { configured: true, stored: true, mesh: publicMetadata(await loadMetadata(session.user_id, meshId)) };
  } catch (_err) {
    // Missing metadata means this is a first-time persistence attempt.
  }

  const artifacts = {};
  const artifactUrls = {};
  for (const name of candidateArtifacts(job)) {
    const fetched = await fetchArtifactFromBridge(jobId, name, bridgeFetch);
    if (!fetched) continue;
    const key = artifactKey(session.user_id, meshId, name, fetched.contentType);
    const upload = await uploadBytes(key, fetched.bytes, fetched.contentType);
    artifacts[name] = {
      key,
      content_type: fetched.contentType,
      size: upload.size,
      sha1: upload.sha1,
      stored_at: new Date().toISOString(),
    };
    artifactUrls[name] = artifactUrl(meshId, name);
  }

  if (!artifacts.final_mesh && !artifacts.faceq_mesh && !artifacts.trellis_mesh) {
    return { configured: true, stored: false, reason: 'no mesh artifact available' };
  }

  if (!artifacts.final_mesh) {
    const fallback = artifacts.faceq_mesh ? 'faceq_mesh' : 'trellis_mesh';
    artifacts.final_mesh = artifacts[fallback];
    artifactUrls.final_mesh = artifactUrl(meshId, fallback);
  }

  const request = job.request || job.metadata || {};
  const meta = {
    id: meshId,
    job_id: jobId,
    user_id: session.user_id,
    owner: publicUser(session),
    title: request.prompt || job.prompt || 'Untitled mesh',
    prompt: request.prompt || job.prompt || '',
    mode: request.mode || job.mode || 'text_to_3d',
    status: 'ready',
    created_at: job.created_at || job.createdAt || new Date().toISOString(),
    stored_at: new Date().toISOString(),
    source_pipeline: job.pipeline || 'Clearmesh inference bridge',
    quality_report: job.quality_report || null,
    artifacts,
    artifact_urls: artifactUrls,
  };
  await uploadJson(metadataKey(session.user_id, meshId), meta);
  return { configured: true, stored: true, mesh: publicMetadata(meta) };
}

async function artifactForMesh(session, meshId, name) {
  if (!isConfigured()) {
    const err = new Error('B2 mesh storage is not configured');
    err.status = 503;
    throw err;
  }
  const meta = await loadMetadata(session.user_id, safeId(meshId));
  const artifact = meta.artifacts?.[name];
  if (!artifact?.key) {
    const err = new Error('artifact not found');
    err.status = 404;
    throw err;
  }
  const result = await downloadBytes(artifact.key);
  return { ...result, meta: publicMetadata(meta), artifact };
}

module.exports = {
  artifactForMesh,
  artifactUrl,
  listMeshes,
  loadMetadata,
  metadataKey,
  meshPrefix,
  persistMeshFromJob,
  publicMetadata,
};
