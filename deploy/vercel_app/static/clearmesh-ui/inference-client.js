(() => {
  const STORAGE_KEY = 'clearmesh.jobs.v2';
  const SESSION_KEY = 'clearmesh.session.profile';
  const STAGE_COPY = {
    queued: 'Queued',
    preflight: 'Preparing',
    text_to_image: 'Drawing reference image',
    reference_qc: 'Checking reference image',
    trellis: 'Building 3D proxy',
    faceq: 'Cleaning topology',
    faceq_qc: 'Checking final mesh',
    texture_uv: 'UV/texturing',
    easy3e: 'Applying edit',
    complete: 'Ready',
    failed: 'Failed',
    dry_run: 'Preview',
  };

  const state = {
    config: { available: false, stage: 'offline' },
    busy: false,
    pendingInput: null,
    jobs: [],
    selectedJobId: null,
    newDraft: false,
    profile: null,
    subscription: null,
    libraryConfigured: false,
  };

  const qs = (selector, root = document) => root.querySelector(selector);
  const qsa = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function nowIso() { return new Date().toISOString(); }
  function shortId() { return `local_${Math.random().toString(16).slice(2, 10)}`; }
  function stageLabel(stage) { return STAGE_COPY[stage] || (stage || 'Working'); }
  function qualityCopy() {
    return { label: 'Preview + FACE-Q', estimate: 'HiDream + Trellis preview first; FACE-Q refines after' };
  }
  function updateQualityControls() {
    const copy = qualityCopy();
    const button = qs('#preset-quality');
    if (button) button.innerHTML = `${copy.label} <span class="caret">›</span>`;
    const estimate = qs('#run-estimate');
    if (estimate) estimate.textContent = copy.estimate;
  }
  function meshJobForPanel() {
    return state.jobs.find((item) => item.localId === state.selectedJobId && item.remote_job_id && item.artifact_urls?.final_mesh);
  }
  function jobTitle(job) {
    if (job.title) return job.title;
    const prompt = (job.prompt || '').trim();
    if (prompt) return prompt.slice(0, 42) + (prompt.length > 42 ? '...' : '');
    if (job.mode === 'image_to_3d') return 'Image to mesh';
    if (job.mode?.startsWith('edit')) return 'Mesh edit';
    return 'Untitled mesh';
  }

  function previewUrlForJob(job) {
    if (!job?.artifact_urls) return null;
    if (job.status === 'succeeded') {
      if (faceqReport(job)?.accepted === true) {
        return job.artifact_urls.final_mesh || job.artifact_urls.faceq_mesh || job.artifact_urls.textured_mesh || job.artifact_urls.trellis_mesh || null;
      }
      return job.artifact_urls.textured_mesh || job.artifact_urls.final_mesh || job.artifact_urls.faceq_mesh || job.artifact_urls.trellis_mesh || null;
    }
    return job.artifact_urls.textured_mesh || job.artifact_urls.trellis_mesh || job.artifact_urls.faceq_mesh || job.artifact_urls.final_mesh || null;
  }

  function imagePreviewUrlForJob(job) {
    if (!job?.artifact_urls) return null;
    return job.artifact_urls.preview_image
      || job.artifact_urls.final_preview_image
      || job.artifact_urls.trellis_preview_image
      || job.artifact_urls.input_image
      || null;
  }

  function downloadUrlForJob(job) {
    if (!job?.artifact_urls) return null;
    if (job.status === 'succeeded' && faceqReport(job)?.accepted === true) {
      return job.artifact_urls.final_mesh || job.artifact_urls.faceq_mesh || job.artifact_urls.textured_mesh || job.artifact_urls.trellis_mesh || null;
    }
    return job.artifact_urls.textured_mesh || job.artifact_urls.final_mesh || job.artifact_urls.faceq_mesh || job.artifact_urls.trellis_mesh || null;
  }

  function faceqReport(job) {
    return job?.quality_report?.faceq || job?.library_mesh?.quality_report?.faceq || null;
  }

  function faceqRetryReport(job) {
    return job?.quality_report?.faceq_retry || job?.library_mesh?.quality_report?.faceq_retry || null;
  }

  function faceqWasRejected(job) {
    const retry = faceqRetryReport(job);
    if (retry) return retry.accepted === false;
    const report = faceqReport(job);
    return Boolean(report && report.accepted === false);
  }

  function easy3eUnavailableReason() {
    const easy3e = state.config?.bridge?.easy3e;
    const reason = String(easy3e?.reason || '').toLowerCase();
    if (reason.includes('shape_enc_next_dc_f16c32')) {
      return 'Mesh editing is paused: this TRELLIS snapshot is missing the shape encoder Easy3E needs.';
    }
    if (state.config?.pipeline?.easy3e) return '';
    return 'Mesh editing is not enabled on the current GPU.';
  }

  function withDownloadParam(url) {
    if (!url) return url;
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}download=1`;
  }

  function attachCanvasModelPreview(canvas, job, handle) {
    const url = previewUrlForJob(job);
    const fallback = setCanvasImageFallback(canvas, job, Boolean(imagePreviewUrlForJob(job)));
    if (!url || !handle?.loadModel) return;
    handle.loadModel(url)
      .then(() => { if (fallback) fallback.hidden = true; })
      .catch(() => { if (fallback) fallback.hidden = false; });
  }

  function setCanvasImageFallback(canvas, job, visible) {
    const imageUrl = imagePreviewUrlForJob(job);
    const parent = canvas?.parentElement;
    if (!parent || !imageUrl) return null;
    let img = parent.querySelector(':scope > .cm-preview-image-fallback');
    if (!img) {
      img = document.createElement('img');
      img.className = 'cm-preview-image-fallback';
      img.alt = `${jobTitle(job)} preview`;
      parent.appendChild(img);
    }
    if (img.src !== imageUrl) img.src = imageUrl;
    img.hidden = !visible;
    return img;
  }

  function setViewerImageFallback(job, visible) {
    const imageUrl = imagePreviewUrlForJob(job);
    const viewer = qs('.viewer');
    if (!viewer || !imageUrl) {
      const existing = viewer?.querySelector('.cm-viewer-image-fallback');
      if (existing) existing.hidden = true;
      return null;
    }
    let img = viewer.querySelector('.cm-viewer-image-fallback');
    if (!img) {
      img = document.createElement('img');
      img.className = 'cm-viewer-image-fallback';
      viewer.insertBefore(img, viewer.firstChild);
    }
    img.alt = `${jobTitle(job)} preview`;
    if (img.src !== imageUrl) img.src = imageUrl;
    img.hidden = !visible;
    return img;
  }

  function saveJobs() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.jobs.slice(0, 50)));
  }

  function loadJobs() {
    try { state.jobs = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch (_err) { state.jobs = []; }
    state.selectedJobId = state.jobs[0]?.localId || null;
  }

  function setGpuChip(message, active = false) {
    const chip = qs('.topbar .chip');
    if (!chip) return;
    chip.textContent = message;
    chip.classList.toggle('is-online', Boolean(active));
  }

  function setAvatar() {
    const avatar = qs('.avatar');
    if (!avatar) return;
    const email = state.profile?.email || 'CM';
    avatar.textContent = email.split('@')[0].slice(0, 2).toUpperCase();
    avatar.title = email;
  }

  function updateSubscriptionUi() {
    const credits = qs('.sidebar .credits');
    const upgrade = qs('.sidebar .upgrade');
    const plan = state.subscription?.plan;
    if (credits && plan) {
      const used = Number(state.subscription?.credits_used || 0);
      credits.innerHTML = `<b>${used.toLocaleString()}</b> / ${Number(plan.monthly_credits || 0).toLocaleString()} credits this month`;
      credits.title = `${plan.name} plan`;
    }
    if (upgrade && state.subscription) {
      upgrade.textContent = state.subscription.tier === 'free' ? 'Upgrade' : 'Manage plan';
    }
  }

  async function authFetch(url, options = {}) {
    return fetch(url, { credentials: 'same-origin', ...options, headers: { Accept: 'application/json', ...(options.headers || {}) } });
  }

  async function loadSession() {
    try {
      const res = await authFetch('/api/auth?action=session');
      if (res.status === 404) {
        state.profile = JSON.parse(localStorage.getItem(SESSION_KEY) || 'null') || { email: 'local@clearmesh.test' };
        return true;
      }
      if (!res.ok) return false;
      const payload = await res.json();
      state.profile = payload.user || null;
      return Boolean(state.profile);
    } catch (_err) {
      state.profile = JSON.parse(localStorage.getItem(SESSION_KEY) || 'null');
      return Boolean(state.profile);
    }
  }

  function showAuthGate() {
    let gate = qs('.cm-auth-gate');
    if (!gate) {
      gate = document.createElement('div');
      gate.className = 'cm-auth-gate';
      gate.innerHTML = `
        <form class="cm-auth-card">
          <div class="cm-auth-brand">Clearmesh</div>
          <label>Email</label>
          <input name="email" type="email" autocomplete="email" placeholder="you@example.com" required>
          <label>Access code</label>
          <input name="accessCode" type="password" autocomplete="current-password" placeholder="Optional for this test deployment">
          <button type="submit">Sign in</button>
          <p data-auth-error></p>
        </form>`;
      document.body.appendChild(gate);
      gate.querySelector('form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const form = event.currentTarget;
        const error = qs('[data-auth-error]', form);
        const email = form.email.value.trim();
        const accessCode = form.accessCode.value;
        error.textContent = '';
        try {
          const res = await authFetch('/api/auth?action=login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, accessCode }),
          });
          if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Sign in failed');
          const payload = await res.json();
          state.profile = payload.user || { email };
          localStorage.setItem(SESSION_KEY, JSON.stringify(state.profile));
          gate.remove();
          setAvatar();
          Promise.all([loadConfig(), loadBilling(), loadLibrary()]).catch(() => {});
        } catch (err) {
          error.textContent = err.message || 'Sign in failed';
        }
      });
    }
  }

  async function ensureAuth() {
    const ok = await loadSession();
    if (!ok) showAuthGate();
    setAvatar();
    return ok;
  }

  function meshFromLibraryItem(item) {
      return {
        localId: `stored_${item.id}`,
        remote_job_id: item.job_id || item.id,
        title: item.title,
        prompt: item.prompt || item.title || 'Stored mesh',
      mode: item.mode || 'text_to_3d',
      status: 'succeeded',
      stage: 'complete',
      createdAt: item.created_at || item.stored_at || nowIso(),
        updatedAt: item.stored_at || nowIso(),
        artifact_urls: item.artifact_urls || null,
        quality_report: item.quality_report || null,
        library_mesh: item,
      };
    }

  function mergeLibraryMeshes(meshes = []) {
    let changed = false;
    meshes.forEach((item) => {
      const existing = state.jobs.find((job) => job.remote_job_id === item.job_id || job.localId === `stored_${item.id}`);
      if (existing) {
        existing.title = existing.title || item.title;
        existing.artifact_urls = item.artifact_urls || existing.artifact_urls;
        existing.quality_report = item.quality_report || existing.quality_report;
        existing.library_mesh = item;
        existing.status = existing.status === 'failed' ? 'failed' : 'succeeded';
        existing.stage = existing.status === 'succeeded' ? 'complete' : existing.stage;
        changed = true;
      } else {
        state.jobs.push(meshFromLibraryItem(item));
        changed = true;
      }
    });
    if (changed) {
      state.jobs.sort((a, b) => String(b.createdAt || '').localeCompare(String(a.createdAt || '')));
      if (!state.selectedJobId) state.selectedJobId = state.jobs[0]?.localId || null;
      renderAll();
    }
  }

  async function loadLibrary() {
    try {
      const res = await authFetch('/api/library?action=list');
      if (!res.ok) return;
      const payload = await res.json();
      state.libraryConfigured = Boolean(payload.configured);
      mergeLibraryMeshes(payload.meshes || []);
    } catch (_err) {
      state.libraryConfigured = false;
    }
  }

  async function loadBilling() {
    try {
      const res = await authFetch('/api/billing?action=subscription');
      if (!res.ok) return;
      state.subscription = await res.json();
      updateSubscriptionUi();
    } catch (_err) {
      state.subscription = null;
    }
  }

  function setupTopbar() {
    const historyBtn = qs('.topbar .btn-ghost');
    if (historyBtn) {
      historyBtn.textContent = 'Sign out';
      historyBtn.addEventListener('click', async () => {
        await authFetch('/api/auth?action=logout', { method: 'POST' }).catch(() => {});
        localStorage.removeItem(SESSION_KEY);
        state.profile = null;
        showAuthGate();
      });
    }
  }

  async function loadConfig() {
    try {
      const res = await authFetch('/v1/inference/config');
      if (res.status === 401) { showAuthGate(); return; }
      if (!res.ok) throw new Error(`config ${res.status}`);
      state.config = await res.json();
      setGpuChip(state.config.available ? 'GPU connected' : 'Preview mode', Boolean(state.config.available));
    } catch (_err) {
      setGpuChip('GPU unavailable', false);
    }
  }

  function pipelineForGenerate() {
    return {
      text_to_image: true,
      trellis: true,
      faceq: true,
      textures: true,
      uv_mapping: true,
      easy3e: false,
    };
  }

  function clearStaticDemo() {
    const inner = qs('.thread-inner');
    if (inner) inner.innerHTML = '';
    const scroller = qs('#history-scroller');
    if (scroller) scroller.innerHTML = '';
  }

  function renderHistory() {
    const scroller = qs('#history-scroller');
    if (!scroller) return;
    const query = (qs('.sidebar .search input')?.value || '').toLowerCase();
    const jobs = state.jobs.filter((job) => jobTitle(job).toLowerCase().includes(query));
    if (!jobs.length) {
      scroller.innerHTML = '<div class="cm-empty-list">Generated meshes will appear here.</div>';
      return;
    }
    scroller.innerHTML = '<div class="group-label">Meshes</div><div class="cm-history-list"></div>';
    const list = qs('.cm-history-list', scroller);
    jobs.forEach((job) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = `cm-history-row${state.selectedJobId === job.localId ? ' active' : ''}`;
      row.innerHTML = `
        <span class="cm-history-thumb"><canvas data-mini-preview="${job.localId}"></canvas></span>
        <span class="cm-history-text"><b></b><small></small></span>`;
      qs('b', row).textContent = jobTitle(job);
      qs('small', row).textContent = `${stageLabel(job.stage)} · ${new Date(job.createdAt || nowIso()).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
      row.addEventListener('click', () => selectJob(job.localId));
      list.appendChild(row);
      const canvas = qs('canvas', row);
      if (canvas && window.ClearmeshPreview?.makePreview) {
        const handle = window.ClearmeshPreview.makePreview(canvas, { kind: job.status === 'succeeded' ? 'vase' : 'blob', autoRotate: job.status !== 'succeeded', cameraDist: 3.2 });
        attachCanvasModelPreview(canvas, job, handle);
      }
    });
  }

  function renderThread() {
    const inner = qs('.thread-inner');
    if (!inner) return;
    inner.innerHTML = '';
    if (!state.jobs.length || state.newDraft) {
      const empty = document.createElement('div');
      empty.className = 'cm-empty-thread';
      empty.innerHTML = '<b>Create a mesh</b><p>Describe an object, attach an image URL, or attach a mesh URL for editing.</p>';
      inner.appendChild(empty);
      return;
    }
    state.jobs.slice().reverse().forEach((job) => {
      const user = document.createElement('div');
      user.className = 'msg-user';
      user.textContent = job.prompt || job.attachmentLabel || jobTitle(job);
      inner.appendChild(user);

      const turn = document.createElement('div');
      turn.className = 'turn-assistant clearmesh-live-turn';
      turn.dataset.localId = job.localId;
      const ready = job.status === 'succeeded' && job.artifact_urls?.final_mesh;
      turn.innerHTML = `
        <div class="assistant-byline"><span>${stageLabel(job.stage)}</span><span class="cm-job-time"></span></div>
        <div class="mesh-card selected" data-kind="vase" data-name="${escapeHtml(jobTitle(job))}">
          <div class="preview clearmesh-pipeline-status">
            <canvas data-preview="job-${job.localId}"></canvas>
            <span class="badge"></span>
            <div class="preview-stats">
              <span><b></b></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>`;
      updateTurnDom(turn, job);
      turn.addEventListener('click', () => selectJob(job.localId));
      inner.appendChild(turn);
      const canvas = qs('canvas', turn);
      if (canvas && window.ClearmeshPreview?.makePreview) {
        const handle = window.ClearmeshPreview.makePreview(canvas, {
          kind: ready ? 'vase' : 'blob',
          autoRotate: !ready,
          rotateSpeed: 0.25,
          cameraDist: 3.2,
        });
        attachCanvasModelPreview(canvas, job, handle);
      }
    });
    const thread = qs('#thread');
    if (thread) thread.scrollTop = thread.scrollHeight;
  }

  function updateTurnDom(turn, job) {
    const statusText = job.error ? job.error : statusSentence(job);
    qs('.assistant-byline span', turn).textContent = stageLabel(job.stage);
    qs('.cm-job-time', turn).textContent = elapsedText(job.createdAt);
    qs('.badge', turn).textContent = job.status === 'succeeded' ? 'Ready' : stageLabel(job.stage);
    const stats = qsa('.preview-stats span', turn);
    if (stats[0]) stats[0].querySelector('b').textContent = job.status === 'failed' ? 'Stopped' : statusText;
    if (stats[1]) stats[1].textContent = job.remote_job_id ? `Job ${job.remote_job_id}` : 'Submitting';
    if (stats[2]) stats[2].textContent = job.status === 'succeeded' ? completionHint(job) : statusHint(job);
    const downloadUrl = downloadUrlForJob(job);
    if (downloadUrl) {
      let link = qs('[data-final-mesh-link]', turn);
      if (!link) {
        link = document.createElement('a');
        link.dataset.finalMeshLink = 'true';
        link.className = 'clearmesh-final-link';
        link.textContent = 'Download GLB';
        link.target = '_blank';
        link.rel = 'noreferrer';
        qs('.preview-stats', turn).appendChild(link);
      }
      link.textContent = job.artifact_urls?.textured_mesh ? 'Download textured GLB' : 'Download GLB';
      link.href = withDownloadParam(downloadUrl);
      link.setAttribute('download', `${job.remote_job_id || job.localId || 'clearmesh'}-final.glb`);
    }
  }

  function statusSentence(job) {
    if (job.status === 'succeeded') return 'Complete';
    if (job.stage === 'text_to_image') return 'Creating reference';
    if (job.stage === 'reference_qc') return 'Validating single-object reference';
    if (job.stage === 'trellis') return 'Generating mesh';
    if (job.stage === 'faceq') return job.artifact_urls?.textured_mesh ? 'FACE-Q refining preview' : 'FACE-Q final topology pass';
    if (job.stage === 'texture_uv') return 'Preparing UVs and textures';
    if (job.stage === 'easy3e') return 'Editing mesh';
    if (job.status === 'failed') return 'Failed';
    return 'Working';
  }

  function completionHint(job) {
    if (faceqReport(job)?.accepted === false && faceqRetryReport(job)?.accepted === true) return 'FACE-Q retry accepted';
    if (faceqWasRejected(job)) return 'FACE-Q rejected; Trellis mesh kept';
    if (job.artifact_urls?.textured_mesh) return 'Textured GLB ready';
    return 'Open the side panel to inspect';
  }

  function statusHint(job) {
    if (job.stage === 'text_to_image' || job.stage === 'reference_qc') return 'Making a single clean reference image';
    if (job.stage === 'trellis') return 'Building the proxy mesh';
    if (job.stage === 'faceq') return job.artifact_urls?.textured_mesh ? 'Textured preview is ready; final topology is still running' : 'Final mesh pass can take 20+ min on the test GPU';
    if (job.stage === 'texture_uv') return 'Exporting the textured/UV-ready GLB';
    if (job.stage === 'easy3e') return 'Applying the edit to the selected mesh';
    return 'Running on the inference GPU';
  }

  function elapsedText(iso) {
    if (!iso) return '';
    const seconds = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  }

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
  }

  function renderAll() {
    saveJobs();
    renderHistory();
    renderThread();
    renderSelectedPanel();
    updateSubscriptionUi();
  }

  function selectJob(localId) {
    state.selectedJobId = localId;
    state.newDraft = false;
    renderAll();
  }

  function setMeshStats(stats = null) {
    const fmt = (value) => Number.isFinite(value) ? value.toLocaleString() : '--';
    const triangles = stats?.triangles;
    const vertices = stats?.vertices;
    const uvs = stats?.uvSets;
    const materials = stats?.materials;
    if (qs('#stat-tris')) qs('#stat-tris').textContent = fmt(triangles);
    if (qs('#stat-verts')) qs('#stat-verts').textContent = fmt(vertices);
    if (qs('#stat-size')) qs('#stat-size').textContent = stats ? 'GLB loaded' : 'waiting for mesh';
    if (qs('#topo-tris')) qs('#topo-tris').textContent = fmt(triangles);
    if (qs('#topo-verts')) qs('#topo-verts').textContent = fmt(vertices);
    if (qs('#topo-uvs')) qs('#topo-uvs').textContent = Number.isFinite(uvs) ? String(uvs) : '--';
    if (qs('#topo-materials')) qs('#topo-materials').textContent = Number.isFinite(materials) ? String(materials) : '--';
    if (qs('#topo-manifold')) qs('#topo-manifold').textContent = stats ? 'not measured' : 'unknown';
    if (qs('#topo-rig')) qs('#topo-rig').textContent = 'not wired';
  }

  function textureReport(job) {
    return job?.quality_report?.texture_uv || job?.library_mesh?.quality_report?.texture_uv || null;
  }

  function setTextureStatus(job = null) {
    const report = textureReport(job) || {};
    const hasTexture = Boolean(job?.artifact_urls?.textured_mesh);
    const mode = report.mode || (hasTexture ? 'available' : 'waiting');
    if (qs('#texture-mode')) qs('#texture-mode').textContent = mode;
    if (qs('#texture-status')) {
      qs('#texture-status').textContent = hasTexture
        ? (report.fallback ? 'Trellis PBR' : report.ai_textured ? 'AI texture' : 'ready')
        : job?.status === 'succeeded' ? 'not generated' : 'pending';
    }
    if (qs('#uv-status')) {
      qs('#uv-status').textContent = report.uv_ready
        ? report.fallback ? 'from Trellis GLB' : 'generated'
        : hasTexture && report.fallback ? 'from source mesh' : 'not measured';
    }
    if (qs('#texture-source')) {
      if (!job) qs('#texture-source').textContent = 'pending';
      else if (report.reference_mesh && report.fallback) qs('#texture-source').textContent = 'Trellis GLB';
      else if (report.ai_textured) qs('#texture-source').textContent = 'texture model';
      else if (hasTexture) qs('#texture-source').textContent = 'mesh artifact';
      else qs('#texture-source').textContent = stageLabel(job.stage);
    }
  }

  function renderSelectedPanel() {
    const job = state.jobs.find((item) => item.localId === state.selectedJobId);
    if (!job) {
      const app = qs('#app');
      if (app) app.classList.remove('panel-open');
      setMeshStats(null);
      setTextureStatus(null);
      if (window.ClearmeshStudioViewer?.updateViewerForKind) {
        window.ClearmeshStudioViewer.updateViewerForKind('blob', 'New mesh', 'Describe an object to begin');
      }
      return;
    }
    const app = qs('#app');
    if (app) app.classList.add('panel-open');
    setTextureStatus(job);
    qs('#rp-title').textContent = jobTitle(job);
    qs('#rp-sub').textContent = job.status === 'succeeded'
      ? (job.artifact_urls?.textured_mesh ? 'Textured GLB · geometry artifact preserved' : faceqWasRejected(job) ? 'Trellis final · FACE-Q candidate kept for audit' : 'FACE-Q final mesh')
      : (job.artifact_urls?.textured_mesh && job.stage === 'faceq' ? `Textured preview ready · FACE-Q running ${elapsedText(job.createdAt)}` : `${stageLabel(job.stage)} · ${elapsedText(job.createdAt)}`);
    if (job.status === 'succeeded' && faceqReport(job)?.accepted === false && faceqRetryReport(job)?.accepted === true) {
      qs('#rp-sub').textContent = 'FACE-Q final mesh · accepted on retry';
    }
    const download = qs('.download');
    const downloadUrl = downloadUrlForJob(job);
    if (download) {
      download.disabled = !downloadUrl;
      download.onclick = () => {
        if (!downloadUrl) return;
        const a = document.createElement('a');
        a.href = withDownloadParam(downloadUrl);
        a.download = `${job.remote_job_id || job.localId || 'clearmesh'}-final.glb`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      };
    }
    const previewUrl = previewUrlForJob(job);
    const viewerFallback = setViewerImageFallback(job, Boolean(imagePreviewUrlForJob(job)));
    if (previewUrl && window.ClearmeshStudioViewer?.viewer?.loadModel) {
      const canvas = qs('#viewer-canvas');
      if (!canvas || canvas.dataset.meshUrl !== previewUrl) {
        if (canvas) canvas.dataset.meshUrl = previewUrl;
        requestAnimationFrame(() => {
          window.ClearmeshStudioViewer.viewer.loadModel(previewUrl).then((result) => {
            if (viewerFallback) viewerFallback.hidden = true;
            setMeshStats(result?.stats || null);
            if (!job.artifact_urls?.final_mesh && job.artifact_urls?.trellis_mesh && qs('#rp-sub')) {
              qs('#rp-sub').textContent = 'Previewing Trellis while FACE-Q finishes';
            }
          }).catch((err) => {
            console.warn('ClearMesh mesh preview failed', err);
            if (canvas) delete canvas.dataset.meshUrl;
            if (viewerFallback) viewerFallback.hidden = false;
            setMeshStats(null);
            if (!viewerFallback) window.ClearmeshStudioViewer.updateViewerForKind('vase', jobTitle(job), 'Preview unavailable');
          });
        });
      }
    } else if (window.ClearmeshStudioViewer?.updateViewerForKind) {
      const canvas = qs('#viewer-canvas');
      if (canvas) delete canvas.dataset.meshUrl;
      setMeshStats(null);
      if (!viewerFallback) {
        window.ClearmeshStudioViewer.updateViewerForKind(job.status === 'failed' ? 'crystal' : 'blob', jobTitle(job), stageLabel(job.stage));
      }
    }
    const note = qs('#easy3e-note');
    const textButton = qs('#easy3e-text-edit');
    const imageButton = qs('#easy3e-image-edit');
    const instructionInput = qs('#easy3e-instruction');
    const editReady = Boolean(state.config?.pipeline?.easy3e && job.artifact_urls?.final_mesh);
    [textButton, imageButton].forEach((button) => {
      if (button) button.disabled = !editReady;
    });
    if (instructionInput) instructionInput.disabled = !editReady;
    if (note) {
      if (!state.config?.pipeline?.easy3e) note.textContent = easy3eUnavailableReason();
      else if (job.artifact_urls?.final_mesh) note.textContent = 'Easy3E will edit this selected mesh.';
      else note.textContent = 'Wait for the mesh to finish before editing.';
    }
    const texture = qs('#easy3e-texture');
    if (texture) {
      const textureAvailable = Boolean(state.config?.bridge?.textures?.easy3e_ctrl_adapter);
      texture.disabled = !textureAvailable;
      texture.title = textureAvailable ? 'Use Easy3E texture adapter' : 'Texture adapter is not installed on this GPU yet';
      if (!textureAvailable) texture.checked = false;
    }
  }

  function describePendingInput() {
    if (!state.pendingInput) return null;
    if (state.pendingInput.kind === 'image') return `Image: ${state.pendingInput.uri}`;
    if (state.pendingInput.editImageUri) return `Mesh edit: ${state.pendingInput.uri}`;
    return `Mesh: ${state.pendingInput.uri}`;
  }

  async function submitPrompt(prompt) {
    const body = {
      prompt,
      mode: 'text_to_3d',
      quality_tier: 'production',
      pipeline: pipelineForGenerate(),
      metadata: {},
    };
    if (state.pendingInput?.kind === 'image') {
      body.mode = 'image_to_3d';
      body.input_uri = state.pendingInput.uri;
      body.pipeline.text_to_image = false;
    } else if (state.pendingInput?.kind === 'mesh') {
      body.input_uri = state.pendingInput.uri;
      body.pipeline = { text_to_image: false, trellis: false, faceq: false, easy3e: true };
      if (state.pendingInput.editImageUri) {
        body.mode = 'edit_image';
        body.metadata.edit_image_uri = state.pendingInput.editImageUri;
      } else {
        body.mode = 'edit_text';
        body.prompt = prompt || 'clean this mesh while preserving the original design';
      }
    }
    const res = await authFetch('/v1/inference/demo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const payload = await res.json().catch(() => ({}));
    if (res.status === 401) { showAuthGate(); throw new Error('Sign in required'); }
    if (!res.ok) throw new Error(payload.detail || `submit failed (${res.status})`);
    return payload;
  }

  async function submitEasy3EEdit({ instruction, editImageUri }) {
    const source = meshJobForPanel();
    if (!source) throw new Error('Select a completed mesh first');
    const body = {
      prompt: instruction || 'clean this mesh while preserving the original design',
      mode: editImageUri ? 'edit_image' : 'edit_text',
      quality_tier: 'edit',
      pipeline: {
        text_to_image: false,
        trellis: false,
        faceq: false,
        easy3e: true,
        easy3e_options: {
          enable_repair: qs('#easy3e-repair')?.checked !== false,
          enable_texture: Boolean(qs('#easy3e-texture')?.checked),
        },
      },
      metadata: {
        source_job_id: source.remote_job_id,
        source_artifact: 'final_mesh',
      },
    };
    if (editImageUri) body.metadata.edit_image_uri = editImageUri;
    const res = await authFetch('/v1/inference/demo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.detail || `edit failed (${res.status})`);
    return payload;
  }

  async function pollJob(localId) {
    const getJob = () => state.jobs.find((item) => item.localId === localId);
    for (let attempt = 0; attempt < 1440; attempt += 1) {
      const job = getJob();
      if (!job?.remote_job_id) return;
      await new Promise((resolve) => setTimeout(resolve, attempt < 8 ? 2500 : 10000));
      const res = await authFetch(`/v1/inference/jobs/${encodeURIComponent(job.remote_job_id)}`);
      if (!res.ok) continue;
      const payload = await res.json();
      Object.assign(job, {
        status: payload.status || job.status,
        stage: payload.stage || job.stage,
        updatedAt: nowIso(),
        artifact_urls: payload.library_mesh?.artifact_urls || payload.artifact_urls || job.artifact_urls,
        library_mesh: payload.library_mesh || job.library_mesh,
        quality_report: payload.quality_report || payload.library_mesh?.quality_report || job.quality_report,
        error: payload.error || null,
        events: payload.events || job.events,
      });
      renderAll();
      if (job.status === 'succeeded' || job.status === 'failed') return;
    }
  }

  function installComposer() {
    const form = qs('#composer');
    const textarea = qs('#prompt');
    if (!form || !textarea) return;
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (state.busy) return;
      const prompt = textarea.value.trim();
      if (!prompt && !state.pendingInput) return;
      const localId = shortId();
      const job = {
        localId,
        prompt,
        attachmentLabel: describePendingInput(),
        mode: state.pendingInput?.kind === 'image' ? 'image_to_3d' : state.pendingInput?.kind === 'mesh' ? 'edit_text' : 'text_to_3d',
        status: 'queued',
        stage: 'queued',
        createdAt: nowIso(),
        updatedAt: nowIso(),
        artifact_urls: null,
      };
      state.jobs.unshift(job);
      state.selectedJobId = localId;
      state.newDraft = false;
      const pending = state.pendingInput;
      textarea.value = '';
      textarea.style.height = 'auto';
      state.busy = true;
      renderAll();
      try {
        state.pendingInput = pending;
        const payload = await submitPrompt(prompt);
        Object.assign(job, {
          remote_job_id: payload.remote_job_id || payload.job_id || payload.id,
          status: payload.status || 'queued',
          stage: payload.stage || 'queued',
          updatedAt: nowIso(),
        });
        state.pendingInput = null;
        renderAll();
        pollJob(localId).catch(() => {});
      } catch (err) {
        job.status = 'failed';
        job.stage = 'failed';
        job.error = err.message || 'Request failed';
        renderAll();
      } finally {
        state.busy = false;
      }
    });

    qsa('.new-mesh').forEach((button) => {
      button.addEventListener('click', () => {
        state.pendingInput = null;
        state.selectedJobId = null;
        state.newDraft = true;
        state.busy = false;
        textarea.value = '';
        textarea.style.height = 'auto';
        textarea.placeholder = 'Describe a mesh, or attach an image...';
        renderAll();
        textarea.focus();
      });
    });

    const [imageButton, meshButton] = qsa('.composer .tool-btn');
    if (imageButton) {
      imageButton.addEventListener('click', () => {
        const uri = window.prompt('Paste a public image URL:');
        if (!uri) return;
        state.pendingInput = { kind: 'image', uri: uri.trim() };
        textarea.placeholder = 'Optional style/material guidance for this image...';
        textarea.focus();
      });
    }
    if (meshButton) {
      meshButton.addEventListener('click', () => {
        const uri = window.prompt('Paste a source mesh URL:');
        if (!uri) return;
        const editImageUri = window.prompt('Optional edited/reference image URL. Leave blank for text-guided edit.');
        state.pendingInput = { kind: 'mesh', uri: uri.trim(), editImageUri: editImageUri ? editImageUri.trim() : null };
        textarea.placeholder = editImageUri ? 'Optional edit note...' : 'Describe the mesh edit...';
        textarea.focus();
      });
    }

    const search = qs('.sidebar .search input');
    if (search) search.addEventListener('input', renderHistory);

    const quality = qs('#preset-quality');
    if (quality) {
      quality.setAttribute('aria-disabled', 'true');
      quality.title = 'FACE-Q is always enabled for generated meshes.';
    }
  }

  function installEasy3EControls() {
    const textButton = qs('#easy3e-text-edit');
    const imageButton = qs('#easy3e-image-edit');
    const instructionInput = qs('#easy3e-instruction');
    async function runEdit(editImageUri = null) {
      if (!state.config?.pipeline?.easy3e) {
        const note = qs('#easy3e-note');
        if (note) note.textContent = easy3eUnavailableReason();
        return;
      }
      const source = meshJobForPanel();
      if (!source) {
        const note = qs('#easy3e-note');
        if (note) note.textContent = 'Select a completed mesh first.';
        return;
      }
      const localId = shortId();
      const instruction = instructionInput?.value.trim() || '';
      const job = {
        localId,
        prompt: instruction || (editImageUri ? 'Image-guided mesh edit' : 'Mesh cleanup edit'),
        attachmentLabel: `Editing ${jobTitle(source)}`,
        mode: editImageUri ? 'edit_image' : 'edit_text',
        status: 'queued',
        stage: 'queued',
        createdAt: nowIso(),
        updatedAt: nowIso(),
        artifact_urls: null,
      };
      state.jobs.unshift(job);
      state.selectedJobId = localId;
      renderAll();
      try {
        const payload = await submitEasy3EEdit({ instruction, editImageUri });
        Object.assign(job, {
          remote_job_id: payload.remote_job_id || payload.job_id || payload.id,
          status: payload.status || 'queued',
          stage: payload.stage || 'queued',
          updatedAt: nowIso(),
        });
        renderAll();
        pollJob(localId).catch(() => {});
      } catch (err) {
        job.status = 'failed';
        job.stage = 'failed';
        job.error = err.message || 'Edit failed';
        renderAll();
      }
    }
    if (textButton) textButton.addEventListener('click', () => runEdit(null));
    if (imageButton) {
      imageButton.addEventListener('click', () => {
        const editImageUri = window.prompt('Paste a public edit/reference image URL:');
        if (editImageUri) runEdit(editImageUri.trim());
      });
    }
  }

  function refreshClocks() {
    qsa('.clearmesh-live-turn').forEach((turn) => {
      const job = state.jobs.find((item) => item.localId === turn.dataset.localId);
      if (job) updateTurnDom(turn, job);
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    setupTopbar();
    clearStaticDemo();
    loadJobs();
    updateQualityControls();
    installComposer();
    installEasy3EControls();
    renderAll();
    if (await ensureAuth()) {
      await Promise.all([loadConfig(), loadBilling(), loadLibrary()]);
    }
    setInterval(refreshClocks, 1000);
    state.jobs.filter((job) => job.remote_job_id && !['succeeded', 'failed'].includes(job.status)).forEach((job) => pollJob(job.localId));
  });
})();
