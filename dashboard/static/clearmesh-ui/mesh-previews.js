// mesh-previews.js — three.js helpers for Clearmesh
// Renders rotating mesh previews into canvases + provides an interactive viewer
// with orbit controls for the right panel.

(function () {
  const T = window.THREE;
  if (!T) { console.warn('three.js not loaded'); return; }

  // ─── geometry library ────────────────────────────────────────
  // Each entry returns a THREE.BufferGeometry. Designed to LOOK like distinct
  // generated meshes when in fact they are clever combinations of primitives.

  function buildVase() {
    // Lathe geometry — curved profile of a vase
    const pts = [];
    for (let i = 0; i <= 32; i++) {
      const t = i / 32;
      const y = t * 2 - 1; // -1 .. 1
      // hourglass-ish curve
      const r = 0.55 + 0.32 * Math.sin(t * Math.PI * 1.6) + 0.05 * Math.sin(t * 14);
      pts.push(new T.Vector2(Math.max(0.05, r), y));
    }
    return new T.LatheGeometry(pts, 48);
  }
  function buildFox() {
    // Composite "low-poly creature" — body + head + ears (icosahedrons + cones)
    const g = new T.BufferGeometry();
    const meshes = [];
    const body = new T.IcosahedronGeometry(0.7, 1);
    body.scale(1.1, 0.85, 1.4);
    body.translate(0, -0.1, 0);
    meshes.push(body);
    const head = new T.IcosahedronGeometry(0.5, 1);
    head.scale(1, 1, 1.1);
    head.translate(0, 0.55, 0.7);
    meshes.push(head);
    const snout = new T.ConeGeometry(0.22, 0.45, 8);
    snout.rotateX(Math.PI / 2);
    snout.translate(0, 0.5, 1.15);
    meshes.push(snout);
    const earL = new T.ConeGeometry(0.16, 0.36, 6);
    earL.translate(-0.28, 0.95, 0.55);
    meshes.push(earL);
    const earR = new T.ConeGeometry(0.16, 0.36, 6);
    earR.translate(0.28, 0.95, 0.55);
    meshes.push(earR);
    return mergeBufferGeometries(meshes);
  }
  function buildChair() {
    const meshes = [];
    const seat = new T.BoxGeometry(1.2, 0.14, 1.1);
    seat.translate(0, 0, 0);
    meshes.push(seat);
    const back = new T.BoxGeometry(1.2, 1.1, 0.12);
    back.translate(0, 0.6, -0.5);
    meshes.push(back);
    for (const [x, z] of [[-0.5,-0.45],[0.5,-0.45],[-0.5,0.45],[0.5,0.45]]) {
      const leg = new T.BoxGeometry(0.12, 0.9, 0.12);
      leg.translate(x, -0.55, z);
      meshes.push(leg);
    }
    return mergeBufferGeometries(meshes);
  }
  function buildKnot() {
    return new T.TorusKnotGeometry(0.7, 0.22, 140, 18, 2, 3);
  }
  function buildCrystal() {
    const meshes = [];
    const main = new T.OctahedronGeometry(0.9, 0);
    main.scale(0.7, 1.3, 0.7);
    meshes.push(main);
    const small1 = new T.OctahedronGeometry(0.42, 0);
    small1.scale(0.55, 0.9, 0.55);
    small1.translate(0.5, -0.2, 0.1);
    meshes.push(small1);
    const small2 = new T.OctahedronGeometry(0.5, 0);
    small2.scale(0.5, 1.0, 0.5);
    small2.translate(-0.45, -0.1, -0.15);
    meshes.push(small2);
    return mergeBufferGeometries(meshes);
  }
  function buildBlob() {
    // Subdivided icosahedron with vertex noise
    const g = new T.IcosahedronGeometry(0.85, 3);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      const n = 0.12 * Math.sin(3*x + 1.1) * Math.cos(2.4*y) + 0.08 * Math.sin(4*z + 2.0);
      const l = Math.sqrt(x*x+y*y+z*z) || 1;
      p.setXYZ(i, x + x/l*n, y + y/l*n, z + z/l*n);
    }
    g.computeVertexNormals();
    return g;
  }
  function buildLamp() {
    const pts = [];
    for (let i = 0; i <= 30; i++) {
      const t = i / 30;
      let r;
      if (t < 0.18) r = 0.42 - t * 0.6;
      else if (t < 0.6) r = 0.32 + (t - 0.18) * 0.8;
      else r = 0.66 - (t - 0.6) * 0.4;
      pts.push(new T.Vector2(Math.max(0.04, r), t * 2 - 1));
    }
    return new T.LatheGeometry(pts, 40);
  }
  function buildBuilding() {
    const meshes = [];
    const base = new T.BoxGeometry(1.2, 0.15, 1.2);
    base.translate(0, -0.85, 0);
    meshes.push(base);
    const mid = new T.BoxGeometry(0.95, 0.9, 0.95);
    mid.translate(0, -0.32, 0);
    meshes.push(mid);
    const top = new T.BoxGeometry(0.7, 0.7, 0.7);
    top.translate(0.05, 0.32, 0.05);
    meshes.push(top);
    const spire = new T.ConeGeometry(0.28, 0.7, 4);
    spire.translate(0.05, 1.02, 0.05);
    spire.rotateY(Math.PI / 4);
    meshes.push(spire);
    return mergeBufferGeometries(meshes);
  }
  function buildTerrain() {
    const g = new T.PlaneGeometry(1.8, 1.8, 32, 32);
    g.rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), z = p.getZ(i);
      const h =
        0.18 * Math.sin(2.2 * x) * Math.cos(1.8 * z) +
        0.10 * Math.sin(4.5 * x + 0.7) +
        0.08 * Math.cos(3.6 * z + 1.2);
      p.setY(i, h);
    }
    g.computeVertexNormals();
    return g;
  }
  function buildShelving() {
    const meshes = [];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 2; col++) {
        const box = new T.BoxGeometry(0.7, 0.5, 0.6);
        box.translate(col * 0.78 - 0.39, row * 0.55 - 0.55, 0);
        meshes.push(box);
      }
    }
    // dividers
    const wall = new T.BoxGeometry(0.04, 1.7, 0.6);
    wall.translate(0, 0, 0);
    meshes.push(wall);
    return mergeBufferGeometries(meshes);
  }
  function buildCup() {
    const pts = [];
    for (let i = 0; i <= 30; i++) {
      const t = i / 30;
      let r = 0.55;
      if (t < 0.1) r = 0.55 - (0.1 - t) * 3;
      else if (t < 0.92) r = 0.45 + t * 0.12;
      else r = 0.0;
      pts.push(new T.Vector2(Math.max(0.0, r), t * 1.5 - 0.7));
    }
    return new T.LatheGeometry(pts, 40);
  }

  // Manual mergeBufferGeometries that handles attributes uniformly (no addons)
  function mergeBufferGeometries(geos) {
    const merged = new T.BufferGeometry();
    let totalVerts = 0;
    geos.forEach(g => { g.computeVertexNormals(); totalVerts += g.attributes.position.count; });
    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    let offset = 0;
    geos.forEach(g => {
      const p = g.attributes.position.array;
      const n = g.attributes.normal.array;
      positions.set(p, offset * 3);
      normals.set(n, offset * 3);
      offset += g.attributes.position.count;
    });
    merged.setAttribute('position', new T.BufferAttribute(positions, 3));
    merged.setAttribute('normal', new T.BufferAttribute(normals, 3));
    return merged;
  }

  const KINDS = {
    vase: { build: buildVase, label: 'lathe', tris: '4.2k', verts: '2.1k' },
    fox: { build: buildFox, label: 'composite', tris: '1.8k', verts: '0.9k' },
    chair: { build: buildChair, label: 'hard-surface', tris: '0.6k', verts: '0.3k' },
    knot: { build: buildKnot, label: 'parametric', tris: '12.4k', verts: '6.2k' },
    crystal: { build: buildCrystal, label: 'low-poly', tris: '0.4k', verts: '0.2k' },
    blob: { build: buildBlob, label: 'organic', tris: '5.1k', verts: '2.6k' },
    lamp: { build: buildLamp, label: 'lathe', tris: '3.2k', verts: '1.6k' },
    building: { build: buildBuilding, label: 'isometric', tris: '0.5k', verts: '0.3k' },
    terrain: { build: buildTerrain, label: 'displaced', tris: '2.0k', verts: '1.1k' },
    shelving: { build: buildShelving, label: 'modular', tris: '1.0k', verts: '0.5k' },
    cup: { build: buildCup, label: 'lathe', tris: '2.8k', verts: '1.4k' },
  };

  window.ClearmeshKinds = KINDS;

  // ─── renderer factory ───────────────────────────────────────
  // Lightweight rotating preview for cards / thumbs.
  function makePreview(canvas, options) {
    const opts = Object.assign({
      kind: 'vase',
      autoRotate: true,
      rotateSpeed: 0.4,
      wireframe: false,
      cameraDist: 3.0,
      color: 0xBFC9DC, // cool soft material
      bg: null,
      pixelRatio: window.devicePixelRatio || 1,
    }, options || {});

    const renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(opts.pixelRatio, 2));
    renderer.outputEncoding = T.sRGBEncoding;

    const scene = new T.Scene();
    if (opts.bg) scene.background = new T.Color(opts.bg);

    const camera = new T.PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.set(opts.cameraDist * 0.7, opts.cameraDist * 0.5, opts.cameraDist);
    camera.lookAt(0, 0, 0);

    // lighting — warm key + cool fill, mimicking a soft studio
    const key = new T.DirectionalLight(0xFFF7E6, 1.15);
    key.position.set(2, 3, 2);
    scene.add(key);
    const fill = new T.DirectionalLight(0xB8C5DD, 0.55);
    fill.position.set(-2, 1, -1);
    scene.add(fill);
    const amb = new T.AmbientLight(0xFFFFFF, 0.35);
    scene.add(amb);

    const buildFn = KINDS[opts.kind] ? KINDS[opts.kind].build : KINDS.vase.build;
    const geom = buildFn();
    const mat = new T.MeshStandardMaterial({
      color: opts.color,
      roughness: 0.55,
      metalness: 0.06,
      flatShading: opts.kind === 'fox' || opts.kind === 'chair' || opts.kind === 'crystal' || opts.kind === 'building',
    });
    const mesh = new T.Mesh(geom, mat);
    scene.add(mesh);

    // wireframe overlay (separate edges mesh, toggleable)
    const wireGeo = new T.EdgesGeometry(geom, 25);
    const wireMat = new T.LineBasicMaterial({ color: 0x15202E, transparent: true, opacity: 0.55 });
    const wire = new T.LineSegments(wireGeo, wireMat);
    wire.visible = !!opts.wireframe;
    mesh.add(wire);
    let loadedModel = null;

    // ground contact shadow as a darkened ellipse (cheap)
    const shadowMat = new T.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.10 });
    const shadow = new T.Mesh(new T.CircleGeometry(0.95, 32), shadowMat);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = -1.05;
    scene.add(shadow);

    function disposeObject(obj) {
      obj.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.filter(Boolean).forEach((m) => {
          if (m.map) m.map.dispose();
          if (m.normalMap) m.normalMap.dispose();
          if (m.roughnessMap) m.roughnessMap.dispose();
          if (m.metalnessMap) m.metalnessMap.dispose();
          m.dispose();
        });
      });
    }

    function clearLoadedModel() {
      if (!loadedModel) return;
      scene.remove(loadedModel);
      disposeObject(loadedModel);
      loadedModel = null;
      mesh.visible = true;
      shadow.visible = true;
    }

    function fitLoadedObject(obj) {
      const box = new T.Box3().setFromObject(obj);
      const size = new T.Vector3();
      const center = new T.Vector3();
      box.getSize(size);
      box.getCenter(center);
      const maxAxis = Math.max(size.x, size.y, size.z) || 1;
      obj.position.sub(center);
      obj.scale.multiplyScalar(1.75 / maxAxis);
      camera.lookAt(0, 0, 0);
    }

    let raf = 0;
    let running = true;
    let manualRotX = 0, manualRotY = 0;

    function resize() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      camera.lookAt(0, 0, 0);
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const start = performance.now();
    function tick() {
      if (!running) return;
      const t = (performance.now() - start) / 1000;
      if (opts.autoRotate) {
        mesh.rotation.y = t * opts.rotateSpeed + manualRotY;
        mesh.rotation.x = Math.sin(t * 0.3) * 0.08 + manualRotX;
        if (loadedModel) {
          loadedModel.rotation.y = t * opts.rotateSpeed + manualRotY;
          loadedModel.rotation.x = Math.sin(t * 0.3) * 0.05 + manualRotX;
        }
      }
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    }
    tick();

    return {
      setWireframe(v) { wire.visible = !!v; },
      setColor(c) { mat.color.setHex(c); },
      setKind(k) {
        clearLoadedModel();
        const newGeom = (KINDS[k] || KINDS.vase).build();
        mesh.geometry.dispose();
        mesh.geometry = newGeom;
        wireGeo.dispose();
        wire.geometry = new T.EdgesGeometry(newGeom, 25);
      },
      loadModel(url) {
        if (!T.GLTFLoader) return Promise.reject(new Error('GLTFLoader is not available'));
        return new Promise((resolve, reject) => {
          const loader = new T.GLTFLoader();
          if (loader.setWithCredentials) loader.setWithCredentials(true);
          loader.load(url, (gltf) => {
            clearLoadedModel();
            loadedModel = gltf.scene;
            loadedModel.traverse((child) => {
              if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
              }
            });
            fitLoadedObject(loadedModel);
            mesh.visible = false;
            shadow.visible = false;
            scene.add(loadedModel);
            resolve({ model: loadedModel });
          }, undefined, (err) => {
            clearLoadedModel();
            reject(err);
          });
        });
      },
      dispose() {
        running = false;
        cancelAnimationFrame(raf);
        ro.disconnect();
        clearLoadedModel();
        renderer.dispose();
        mesh.geometry.dispose();
        wireGeo.dispose();
      },
      get mesh() { return mesh; },
      get camera() { return camera; },
      get renderer() { return renderer; },
      get scene() { return scene; },
    };
  }

  // ─── interactive viewer (right panel) ───────────────────────
  // Mouse-orbit + zoom, no external addon
  function makeViewer(canvas, options) {
    const opts = Object.assign({
      kind: 'vase',
      wireframe: false,
      showGrid: true,
      cameraDist: 3.3,
    }, options || {});

    const renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputEncoding = T.sRGBEncoding;

    const scene = new T.Scene();

    const camera = new T.PerspectiveCamera(38, 1, 0.1, 100);
    let radius = opts.cameraDist;
    let theta = Math.PI / 4; // around y
    let phi = Math.PI / 3;   // from y axis
    let target = new T.Vector3(0, -0.05, 0);
    function applyCamera() {
      camera.position.x = target.x + radius * Math.sin(phi) * Math.cos(theta);
      camera.position.z = target.z + radius * Math.sin(phi) * Math.sin(theta);
      camera.position.y = target.y + radius * Math.cos(phi);
      camera.lookAt(target);
    }
    applyCamera();

    // lights
    scene.add(new T.AmbientLight(0xFFFFFF, 0.35));
    const key = new T.DirectionalLight(0xFFF7E6, 1.1);
    key.position.set(3, 4, 3);
    scene.add(key);
    const fill = new T.DirectionalLight(0xB8C5DD, 0.5);
    fill.position.set(-3, 1, -2);
    scene.add(fill);

    // grid (subtle)
    const grid = new T.GridHelper(8, 16, 0x6E6557, 0xC4BDAE);
    grid.material.transparent = true;
    grid.material.opacity = 0.45;
    grid.position.y = -1.05;
    grid.visible = opts.showGrid;
    scene.add(grid);

    // mesh
    const buildFn = KINDS[opts.kind] ? KINDS[opts.kind].build : KINDS.vase.build;
    let geom = buildFn();
    const flat = opts.kind === 'fox' || opts.kind === 'chair' || opts.kind === 'crystal' || opts.kind === 'building';
    const mat = new T.MeshStandardMaterial({
      color: 0xBFC9DC, roughness: 0.55, metalness: 0.06, flatShading: flat,
    });
    let mesh = new T.Mesh(geom, mat);
    scene.add(mesh);
    let wire = new T.LineSegments(new T.EdgesGeometry(geom, 25),
      new T.LineBasicMaterial({ color: 0x15202E, transparent: true, opacity: 0.6 }));
    wire.visible = !!opts.wireframe;
    mesh.add(wire);
    let loadedModel = null;

    function disposeObject(obj) {
      obj.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.filter(Boolean).forEach((m) => {
          if (m.map) m.map.dispose();
          if (m.normalMap) m.normalMap.dispose();
          if (m.roughnessMap) m.roughnessMap.dispose();
          if (m.metalnessMap) m.metalnessMap.dispose();
          m.dispose();
        });
      });
    }

    function clearLoadedModel() {
      if (!loadedModel) return;
      scene.remove(loadedModel);
      disposeObject(loadedModel);
      loadedModel = null;
    }

    function fitObject(obj) {
      const box = new T.Box3().setFromObject(obj);
      const size = new T.Vector3();
      const center = new T.Vector3();
      box.getSize(size);
      box.getCenter(center);
      const maxAxis = Math.max(size.x, size.y, size.z) || 1;
      const scale = 2.2 / maxAxis;
      obj.scale.setScalar(scale);
      obj.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
      target.set(0, 0, 0);
      radius = 3.3;
      applyCamera();
    }

    // resize
    function resize() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    // input
    let isDragging = false, dragMode = 'orbit'; // 'orbit' | 'pan'
    let lastX = 0, lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      isDragging = true;
      lastX = e.clientX; lastY = e.clientY;
      dragMode = (e.button === 1 || e.shiftKey) ? 'pan' : 'orbit';
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (dragMode === 'orbit') {
        theta -= dx * 0.008;
        phi = Math.max(0.15, Math.min(Math.PI - 0.15, phi - dy * 0.008));
      } else {
        const factor = radius * 0.0015;
        const right = new T.Vector3().subVectors(camera.position, target).cross(new T.Vector3(0,1,0)).normalize();
        const up = new T.Vector3(0,1,0);
        target.addScaledVector(right, -dx * factor);
        target.addScaledVector(up, dy * factor);
      }
      applyCamera();
    });
    canvas.addEventListener('pointerup', () => { isDragging = false; });
    canvas.addEventListener('pointerleave', () => { isDragging = false; });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      radius = Math.max(1.4, Math.min(8, radius * (1 + e.deltaY * 0.001)));
      applyCamera();
    }, { passive: false });

    let auto = false;
    let running = true;
    function tick() {
      if (!running) return;
      if (auto) {
        theta += 0.003;
        applyCamera();
      }
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }
    tick();

    return {
      setKind(k) {
        clearLoadedModel();
        mesh.visible = true;
        const newGeom = (KINDS[k] || KINDS.vase).build();
        const newFlat = k === 'fox' || k === 'chair' || k === 'crystal' || k === 'building';
        mat.flatShading = newFlat;
        mat.needsUpdate = true;
        mesh.geometry.dispose();
        mesh.geometry = newGeom;
        wire.geometry.dispose();
        wire.geometry = new T.EdgesGeometry(newGeom, 25);
      },
      setWireframe(v) { wire.visible = !!v; },
      setGrid(v) { grid.visible = !!v; },
      setAutoRotate(v) { auto = !!v; },
      loadModel(url) {
        if (!T.GLTFLoader) return Promise.reject(new Error('GLTFLoader is not available'));
        return new Promise((resolve, reject) => {
          const loader = new T.GLTFLoader();
          if (loader.setWithCredentials) loader.setWithCredentials(true);
          loader.load(url, (gltf) => {
            clearLoadedModel();
            mesh.visible = false;
            loadedModel = gltf.scene;
            loadedModel.traverse((child) => {
              if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
              }
            });
            fitObject(loadedModel);
            scene.add(loadedModel);
            const stats = { vertices: 0, triangles: 0, materials: 0, uvSets: 0 };
            const materials = new Set();
            loadedModel.traverse((child) => {
              if (!child.isMesh || !child.geometry) return;
              const geometry = child.geometry;
              stats.vertices += geometry.attributes.position ? geometry.attributes.position.count : 0;
              stats.triangles += geometry.index ? Math.floor(geometry.index.count / 3) : Math.floor((geometry.attributes.position?.count || 0) / 3);
              if (geometry.attributes.uv) stats.uvSets += 1;
              (Array.isArray(child.material) ? child.material : [child.material]).filter(Boolean).forEach((material) => materials.add(material.uuid));
            });
            stats.materials = materials.size;
            resolve({ model: loadedModel, stats });
          }, undefined, reject);
        });
      },
      resetView() {
        theta = Math.PI / 4; phi = Math.PI / 3; radius = opts.cameraDist;
        target.set(0, -0.05, 0);
        applyCamera();
      },
      dispose() {
        running = false;
        ro.disconnect();
        clearLoadedModel();
        renderer.dispose();
        mesh.geometry.dispose();
        wire.geometry.dispose();
      },
    };
  }

  window.ClearmeshPreview = { makePreview, makeViewer, KINDS };
})();
