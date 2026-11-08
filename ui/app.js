/* Q-Scientist mission control. Vanilla JS + inline SVG, no build step.
 * Data: /api/runs, /api/runs/<id>/{events,calibration,budget,experiments/<exp>}. See CONTRACTS.md.
 */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const SVGNS = 'http://www.w3.org/2000/svg';
const qs = new URLSearchParams(location.search);

function h(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) n.append(c);
  return n;
}
function s(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const fmt = (x, d = 3) => (typeof x === 'number' && isFinite(x) ? x.toFixed(d) : '—');
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ================================================================ state */
const S = {
  runs: [], runId: null, cal: null, calIndex: null,
  events: [], cursor: -1, model: null,
  playing: false, speed: Number(qs.get('speed')) || 1, mode: 'replay',
  timer: null, pollTimer: null, waiting: false, finished: false,
  selected: null,        // exp_id pinned by the user (null = follow the current experiment)
  expCache: new Map(), token: 0, stick: true, typing: null,
};

/* ================================================================ model */
function emptyModel() {
  return {
    run: null,
    budget: { llm_calls: 0, max_llm_calls: null, qpu_seconds: 0, max_qpu_seconds: null },
    exps: new Map(), order: [], current: null,
    best: { sim: null, qpu: null },
    feed: [], finished: null, firstTs: null, lastTs: null, lastType: null,
  };
}
const modeOf = (e) => (e.result && e.result.mode) || (e.spec && e.spec.mode) || 'sim';
const isCertified = (d) => d && !d.error && typeof d.ci_low === 'number' && d.ci_low > 0.5;

function applyEvent(m, ev) {
  const d = ev.data || {};
  if (!m.firstTs && ev.ts) m.firstTs = ev.ts;
  if (ev.ts) m.lastTs = ev.ts;
  m.lastType = ev.type;
  switch (ev.type) {
    case 'run_started':
      m.run = d;
      if (d.max_llm_calls != null) m.budget.max_llm_calls = d.max_llm_calls;
      if (d.max_qpu_seconds != null) m.budget.max_qpu_seconds = d.max_qpu_seconds;
      m.feed.push({ kind: 'start', seq: ev.seq, ts: ev.ts, d });
      break;
    case 'thought':
      m.feed.push({ kind: 'thought', seq: ev.seq, ts: ev.ts, text: d.text || '' });
      break;
    case 'experiment_started': {
      const e = { id: d.exp_id, spec: d.spec || {}, result: null, startTs: ev.ts, startSeq: ev.seq };
      m.exps.set(d.exp_id, e); m.order.push(d.exp_id); m.current = d.exp_id;
      m.feed.push({ kind: 'exp', seq: ev.seq, ts: ev.ts, id: d.exp_id });
      break;
    }
    case 'experiment_result': {
      let e = m.exps.get(d.exp_id);
      if (!e) {
        e = { id: d.exp_id, spec: { n_qubits: d.n_qubits, mode: d.mode }, result: null, startTs: ev.ts };
        m.exps.set(d.exp_id, e); m.order.push(d.exp_id);
        m.feed.push({ kind: 'exp', seq: ev.seq, ts: ev.ts, id: d.exp_id });
      }
      e.result = d; e.resultSeq = ev.seq; e.endTs = ev.ts;
      m.current = d.exp_id;
      recomputeBest(m);
      break;
    }
    case 'budget':
      Object.assign(m.budget, d);
      break;
    case 'milestone':
      m.feed.push({ kind: 'milestone', seq: ev.seq, ts: ev.ts, d });
      break;
    case 'error':
      m.feed.push({ kind: 'error', seq: ev.seq, ts: ev.ts, text: d.text || 'error' });
      break;
    case 'run_finished':
      m.finished = d;
      m.feed.push({ kind: 'finish', seq: ev.seq, ts: ev.ts, d });
      break;
    default:
      break;
  }
}
function modelAt(cursor) {
  const m = emptyModel();
  for (let i = 0; i <= cursor && i < S.events.length; i++) applyEvent(m, S.events[i]);
  return m;
}

/* ================================================================ pacing */
// Milliseconds to dwell on an event at 1× ("narrated pace"); divided by the speed factor.
function typeMs(text) { return clamp(text.length * 17, 700, 4600); }
function dwell(ev) {
  const d = ev.data || {};
  switch (ev.type) {
    case 'run_started': return 1600;
    case 'thought': return typeMs(d.text || '') + 1100;
    case 'experiment_started': {
      const e = S.model && S.model.exps.get(d.exp_id);
      // look ahead to the result to know the tree depth (the started event doesn't carry the layout)
      const depth = lookaheadDepth(d.exp_id);
      return Math.max(2200, depth * stepMs(1) + 1400) + (e && e.spec.mode === 'qpu' ? 900 : 0);
    }
    case 'experiment_result': return d.error ? 2600 : 2900;
    case 'budget': return 260;
    case 'milestone': return 3400;
    case 'error': return 2400;
    case 'run_finished': return 1000;
    default: return 500;
  }
}
function stepMs(speed) { return clamp(300 / speed, 18, 300); }
function effSpeed() {
  if (S.mode === 'live') {
    const backlog = S.events.length - 1 - S.cursor;
    if (backlog > 6) return Math.max(S.speed, 40);
  }
  return S.speed;
}
function lookaheadDepth(expId) {
  for (let i = S.cursor + 1; i < S.events.length && i < S.cursor + 40; i++) {
    const ev = S.events[i];
    if (ev.type === 'experiment_result' && ev.data && ev.data.exp_id === expId) {
      return schedule(ev.data.root, ev.data.tree_edges || []).depth;
    }
  }
  return 8;
}

/* ================================================================ GHZ tree schedule */
// Greedy CNOT layering: once a qubit holds the GHZ state, it entangles one child per layer,
// largest subtree first. Returns {layer: Map(child -> layer), depth}.
function schedule(root, edges) {
  const kids = new Map();
  for (const [p, c] of edges) { if (!kids.has(p)) kids.set(p, []); kids.get(p).push(c); }
  const size = new Map();
  const sz = (v) => {
    if (size.has(v)) return size.get(v);
    let n = 1; for (const c of kids.get(v) || []) n += sz(c);
    size.set(v, n); return n;
  };
  const layer = new Map();
  if (root == null) return { layer, depth: 0 };
  layer.set(root, 0);
  const stack = [root]; let depth = 0; let guard = 0;
  while (stack.length && guard++ < 5000) {
    const v = stack.pop();
    const cs = (kids.get(v) || []).slice().sort((a, b) => sz(b) - sz(a));
    cs.forEach((c, i) => {
      if (layer.has(c)) return;
      const t = layer.get(v) + 1 + i;
      layer.set(c, t); depth = Math.max(depth, t); stack.push(c);
    });
  }
  return { layer, depth };
}

// Best certified result per mode. A result whose pre-registered confirmation re-run (spec.confirmation_of)
// failed is withdrawn, exactly as in the paper.
function recomputeBest(m) {
  const failed = new Set(), passed = new Set();
  for (const e of m.exps.values()) {
    const of = e.spec && e.spec.confirmation_of;
    if (of && e.result && !e.result.error) (isCertified(e.result) ? passed : failed).add(of);
  }
  const best = { sim: null, qpu: null };
  for (const e of m.exps.values()) {
    const d = e.result;
    if (!d || !isCertified(d) || (failed.has(e.id) && !passed.has(e.id))) continue;
    const k = modeOf(e) === 'qpu' ? 'qpu' : 'sim';
    const n = d.n_qubits ?? (d.physical_qubits || []).length;
    const b = best[k];
    if (!b || n > b.n || (n === b.n && d.fidelity > b.fidelity)) best[k] = { n, fidelity: d.fidelity, ci_low: d.ci_low, ci_high: d.ci_high, id: d.exp_id };
  }
  m.best = best;
}

/* ================================================================ API */
async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}
// Static mode (Vercel snapshot built by scripts/export_static.py): same data, read from pre-exported files.
const STATIC = !!window.QS_STATIC;
const POLL_MS = STATIC ? 60000 : 2000;  // static hosts rate-limit hammering; new results arrive by redeploy anyway
const E = encodeURIComponent;
const api = STATIC ? {
  runs: () => getJSON('data/runs.json'),
  events: async (id, since = -1) => {
    const r = await getJSON(`data/${E(id)}/events.json`);
    return { ...r, events: (r.events || []).filter((e) => e.seq > since) };
  },
  cal: (id) => getJSON(`data/${E(id)}/calibration.json`),
  exp: (id, exp) => getJSON(`data/${E(id)}/experiments/${E(exp)}.json`),
  paperUrl: (id) => `data/${E(id)}/paper/paper.html`,
} : {
  runs: () => getJSON('/api/runs'),
  events: (id, since = -1) => getJSON(`/api/runs/${E(id)}/events?since=${since}`),
  cal: (id) => getJSON(`/api/runs/${E(id)}/calibration`),
  exp: (id, exp) => getJSON(`/api/runs/${E(id)}/experiments/${E(exp)}`),
  paperUrl: (id) => `/api/runs/${E(id)}/paper/paper.html`,
};

/* ================================================================ chip map */
const chip = { svg: $('#chipSvg'), gTree: null, gX: null, pos: new Map(), shownId: null, shownState: null };

function errColor(err) {
  if (err == null) return '#2a333d';
  if (err >= 0.5) return '#5a3330';
  const t = clamp((Math.log10(err) - Math.log10(1e-3)) / (Math.log10(3e-2) - Math.log10(1e-3)), 0, 1);
  // quiet slate (good) -> warm rust (bad)
  const stops = [[51, 66, 79], [111, 90, 85], [200, 109, 92]];
  const seg = t < 0.5 ? 0 : 1, u = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const c = stops[seg].map((a, i) => Math.round(a + (stops[seg + 1][i] - a) * u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function drawChip(cal) {
  const svg = chip.svg;
  svg.innerHTML = '';
  chip.pos.clear(); chip.shownId = null; chip.shownState = null;
  $('#chipEmpty').hidden = !!cal;
  if (!cal) return;
  const qs_ = cal.qubits.filter((q) => q.x != null && q.y != null);
  for (const q of qs_) chip.pos.set(q.id, [q.x, q.y]);
  const xs = qs_.map((q) => q.x), ys = qs_.map((q) => q.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pad = 0.8;
  const padTop = 2.0; // room for the caption overlay
  svg.setAttribute('viewBox', `${x0 - pad} ${y0 - padTop} ${x1 - x0 + 2 * pad} ${y1 - y0 + pad + padTop}`);
  // edges: draw good ones first so bad ones sit on top
  const gE = s('g', { class: 'edges' }, svg);
  const edges = (cal.edges || []).filter((e) => chip.pos.has(e.a) && chip.pos.has(e.b))
    .sort((a, b) => (a.error ?? 0) - (b.error ?? 0));
  for (const e of edges) {
    const [ax, ay] = chip.pos.get(e.a), [bx, by] = chip.pos.get(e.b);
    s('line', {
      x1: ax, y1: ay, x2: bx, y2: by, class: 'c-edge' + (e.error >= 0.5 ? ' dead' : ''),
      stroke: errColor(e.error), 'stroke-width': e.error >= 0.5 ? 0.07 : 0.1,
    }, gE);
  }
  chip.gTree = s('g', { class: 'tree' }, svg);
  const gN = s('g', { class: 'nodes' }, svg);
  chip.gTop = s('g', { class: 'tree-top' }, svg);
  chip.gX = s('g', { class: 'excl' }, svg);
  chip.calIndex = new Map(cal.qubits.map((q) => [q.id, q]));
  for (const q of qs_) {
    const c = s('circle', { cx: q.x, cy: q.y, r: 0.24, class: 'c-node' }, gN);
    c.addEventListener('mouseenter', (ev) => showTip(ev, qubitTip(q)));
    c.addEventListener('mousemove', moveTip);
    c.addEventListener('mouseleave', hideTip);
  }
  $('#chipName').textContent = `${cal.backend || ''} · ${cal.num_qubits || qs_.length} qubits · ${edges.length} couplers`;
}

function qubitTip(q) {
  const deg = (S.cal.edges || []).filter((e) => e.a === q.id || e.b === q.id);
  const worst = deg.reduce((m, e) => Math.max(m, e.error ?? 0), 0);
  return `<b>Qubit ${q.id}</b>
    <div class="tt-row"><span class="tt-k">T1</span><span>${fmt(q.t1_us, 1)} µs</span></div>
    <div class="tt-row"><span class="tt-k">T2</span><span>${fmt(q.t2_us, 1)} µs</span></div>
    <div class="tt-row"><span class="tt-k">readout err</span><span>${q.readout_error != null ? q.readout_error.toExponential(1) : '—'}</span></div>
    <div class="tt-row"><span class="tt-k">worst CZ</span><span>${worst ? worst.toExponential(1) : '—'}</span></div>`;
}

function expLayout(e) {
  const r = e && e.result;
  if (!r || !r.tree_edges) return null;
  return { root: r.root, edges: r.tree_edges, qubits: r.physical_qubits || [] };
}

// state: 'running' | 'ok' | 'bad' | 'err'
function treeState(e) {
  if (!e || !e.result) return 'running';
  if (e.result.error) return 'err';
  return isCertified(e.result) ? 'ok' : 'bad';
}

function renderTree(e, { animate = false, layoutOverride = null } = {}) {
  if (!chip.gTree) return;
  const state = treeState(e);
  const lay = layoutOverride || expLayout(e);
  const key = e ? e.id : null;
  // same experiment, layout already drawn -> just switch the outcome class (colour transition)
  if (key && chip.shownId === key && chip.hasLayout && lay) {
    setTreeClass(state, false);
    chip.shownState = state;
    renderCaption(e);
    return;
  }
  chip.gTree.innerHTML = ''; chip.gTop.innerHTML = ''; chip.gX.innerHTML = '';
  chip.shownId = key; chip.shownState = state; chip.hasLayout = !!lay;
  renderCaption(e);
  if (!e) return;
  for (const q of (e.spec && e.spec.exclude_qubits) || []) {
    const p = chip.pos.get(q); if (!p) continue;
    const d = 0.17;
    s('path', { d: `M${p[0] - d} ${p[1] - d}L${p[0] + d} ${p[1] + d}M${p[0] + d} ${p[1] - d}L${p[0] - d} ${p[1] + d}`, class: 'c-x' }, chip.gX);
  }
  if (!lay) { setTreeClass(state, true); return; }
  const { layer, depth } = schedule(lay.root, lay.edges);
  const step = stepMs(effSpeed());
  const anim = animate && !reducedMotion;
  const halo = s('g', {}, chip.gTree);
  const gEdge = s('g', {}, chip.gTree);
  const gNode = chip.gTop;
  for (const [p, c] of lay.edges) {
    const a = chip.pos.get(p), b = chip.pos.get(c);
    if (!a || !b) continue;
    s('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: 't-halo' }, halo);
    const ln = s('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: 't-edge', pathLength: 1 }, gEdge);
    const L = layer.get(c) ?? 1;
    if (anim) {
      ln.animate([{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }],
        { duration: step * 0.95, delay: (L - 1) * step, fill: 'both', easing: 'cubic-bezier(.3,.6,.4,1)' });
    }
  }
  const nodes = new Set(lay.qubits.length ? lay.qubits : [lay.root, ...lay.edges.map((x) => x[1])]);
  for (const q of nodes) {
    const p = chip.pos.get(q); if (!p) continue;
    const isRoot = q === lay.root;
    const c = s('circle', { cx: p[0], cy: p[1], r: isRoot ? 0.34 : 0.25, class: 't-node' + (isRoot ? ' root' : '') }, gNode);
    if (anim && !isRoot) {
      const L = layer.get(q) ?? 1;
      c.animate([{ transform: 'scale(0)', opacity: 0 }, { transform: 'scale(1.55)', opacity: 1, offset: 0.55 }, { transform: 'scale(1)', opacity: 1 }],
        { duration: step * 1.6, delay: L * step - step * 0.25, fill: 'both', easing: 'ease-out' });
    }
  }
  const rp = chip.pos.get(lay.root);
  if (rp) {
    s('circle', { cx: rp[0], cy: rp[1], r: 0.34, class: 't-root-ring' }, gNode);
    const lbl = s('text', { x: rp[0], y: rp[1] + 0.01, class: 't-root-label' }, gNode);
    lbl.textContent = lay.root;
  }
  setTreeClass(state, true);
  // after the spread finishes, nodes "breathe" while the job runs
  clearTimeout(chip.waitTimer);
  if (state === 'running') {
    chip.waitTimer = setTimeout(() => { if (chip.shownState === 'running') chip.gTree.parentNode && setTreeClass('running', false, true); },
      anim ? (depth + 1) * step : 0);
  }
}
function setTreeClass(state, reset, waiting = false) {
  const cls = `tree ${state}` + (waiting ? ' waiting' : '');
  chip.gTree.setAttribute('class', cls);
  chip.gTop.setAttribute('class', cls);
}

function renderCaption(e) {
  const cap = $('#chipCaption');
  if (!e) { cap.hidden = true; return; }
  const spec = e.spec || {}, r = e.result;
  const mode = modeOf(e);
  const n = spec.n_qubits ?? (r && r.n_qubits);
  let status;
  if (!r) status = `<span class="badge run">${mode === 'qpu' ? 'RUNNING ON QPU' : 'SIMULATING'}</span>`;
  else if (r.error) status = `<span class="badge bad">JOB FAILED</span>`;
  else if (isCertified(r)) status = `<span class="badge ok">CERTIFIED</span>`;
  else status = `<span class="badge bad">NOT CERTIFIED</span>`;
  const lay = expLayout(e);
  const depth = lay ? schedule(lay.root, lay.edges).depth : null;
  const sub = r && !r.error
    ? `F = <b style="color:var(--text)">${fmt(r.fidelity)}</b> · 95% CI [${fmt(r.ci_low)}, ${fmt(r.ci_high)}]${depth ? ` · CNOT depth ${depth}` : ''}`
    : r && r.error ? escapeHtml(r.error).slice(0, 110) : escapeHtml(spec.hypothesis || '').slice(0, 120);
  cap.innerHTML = `<div class="cap-top"><span>${e.id}</span><span class="badge ${mode}">${mode.toUpperCase()}</span><span>N = ${n}</span>${status}${S.selected ? '<span class="muted" style="font-size:12px">pinned · Esc to follow</span>' : ''}</div><div class="cap-sub">${sub}</div>`;
  cap.hidden = false;
}

/* ================================================================ fidelity chart */
function plotSize(svg) {
  const r = svg.parentNode.getBoundingClientRect();
  const w = Math.max(200, r.width - 16), h = Math.max(120, r.height - 10);
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  return [w, h];
}

function renderChart(newId = null) {
  const svg = $('#fidSvg');
  svg.innerHTML = '';
  const [W, H] = plotSize(svg);
  const M = { l: 46, r: 14, t: 26, b: 36 };
  const m = S.model;
  const pts = [];
  for (const id of m.order) {
    const e = m.exps.get(id);
    const r = e && e.result;
    if (!r || r.error || typeof r.fidelity !== 'number') continue;
    pts.push({ id, n: r.n_qubits ?? (r.physical_qubits || []).length, f: r.fidelity, lo: r.ci_low, hi: r.ci_high, mode: modeOf(e), e });
  }
  const maxN = Math.max(8, ...pts.map((p) => p.n));
  const xMax = Math.max(48, Math.ceil((maxN + 4) / 8) * 8);
  const minLo = Math.min(0.5, ...pts.map((p) => (p.lo ?? p.f)));
  const yMin = Math.min(0.3, Math.floor((minLo - 0.04) * 10) / 10);
  const yMax = 1.0;
  const X = (n) => M.l + (n / xMax) * (W - M.l - M.r);
  const Y = (f) => M.t + (1 - (f - yMin) / (yMax - yMin)) * (H - M.t - M.b);

  const gA = s('g', { class: 'axis' }, svg);
  s('rect', { x: M.l, y: Y(0.5), width: W - M.l - M.r, height: Y(yMin) - Y(0.5), class: 'thr-zone' }, gA);
  const yStep = (H - M.t - M.b) / ((yMax - yMin) * 10) < 22 ? 0.2 : 0.1;
  for (let f = yMax; f >= yMin - 1e-9; f -= yStep) {
    const y = Y(f);
    s('line', { x1: M.l, x2: W - M.r, y1: y, y2: y, class: 'grid-line' }, gA);
    s('text', { x: M.l - 8, y: y + 4, 'text-anchor': 'end' }, gA).textContent = f.toFixed(1);
  }
  const xStep = xMax > 64 ? 16 : 8;
  for (let n = 0; n <= xMax; n += xStep) {
    s('text', { x: X(n), y: H - M.b + 18, 'text-anchor': 'middle' }, gA).textContent = n;
    s('line', { x1: X(n), x2: X(n), y1: H - M.b, y2: H - M.b + 4, class: 'grid-line' }, gA);
  }
  s('text', { x: W - M.r, y: H - 4, 'text-anchor': 'end', class: 'axis-title' }, gA).textContent = 'qubits in GHZ state (N)';
  s('text', { x: 8, y: 12, class: 'axis-title' }, gA).textContent = 'fidelity';
  // threshold
  s('line', { x1: M.l, x2: W - M.r, y1: Y(0.5), y2: Y(0.5), class: 'thr-line' }, svg);
  s('text', { x: M.l + 8, y: Y(0.5) - 8, class: 'thr-label' }, svg)
    .textContent = 'F > 0.5  =  genuine multipartite entanglement';
  s('text', { x: M.l + 8, y: Y(0.5) + 17, class: 'p-label' }, svg).textContent = 'not certifiable';

  // trend lines: best fidelity per N, per mode
  for (const mode of ['sim', 'qpu']) {
    const byN = new Map();
    for (const p of pts) if (p.mode === mode && (!byN.has(p.n) || byN.get(p.n).f < p.f)) byN.set(p.n, p);
    const line = [...byN.values()].sort((a, b) => a.n - b.n);
    if (line.length > 1) s('polyline', { points: line.map((p) => `${X(p.n)},${Y(p.f)}`).join(' '), class: `trend ${mode}` }, svg);
  }
  // sim first, qpu on top
  const ordered = pts.slice().sort((a, b) => (a.mode === b.mode ? 0 : a.mode === 'qpu' ? 1 : -1));
  const fidSel = focusId(m); const selId = fidSel && m.exps.get(fidSel) && m.exps.get(fidSel).result ? fidSel : null;
  for (const p of ordered) {
    const x = X(p.n), y = Y(p.f);
    const g = s('g', {}, svg);
    if (typeof p.lo === 'number' && typeof p.hi === 'number') {
      const cw = 4;
      s('path', { d: `M${x} ${Y(p.lo)}V${Y(p.hi)}M${x - cw} ${Y(p.lo)}h${2 * cw}M${x - cw} ${Y(p.hi)}h${2 * cw}`, class: `ci ${p.mode}` }, g);
    }
    if (p.id === selId) s('circle', { cx: x, cy: y, r: p.mode === 'qpu' ? 10 : 9, class: 'pt-sel' }, g);
    const dot = s('circle', { cx: x, cy: y, r: p.mode === 'qpu' ? 6 : 4.8, class: `pt ${p.mode}` }, g);
    if (p.id === newId && !reducedMotion) {
      dot.classList.add('pt-new');
      s('circle', { cx: x, cy: y, r: 5, class: `ripple ${p.mode}` }, g);
    }
    const hit = s('circle', { cx: x, cy: y, r: 11, class: 'pt-hit' }, g);
    hit.addEventListener('mouseenter', (ev) => showTip(ev, pointTip(p)));
    hit.addEventListener('mousemove', moveTip);
    hit.addEventListener('mouseleave', hideTip);
    hit.addEventListener('click', () => select(p.id));
  }
}
function pointTip(p) {
  const r = p.e.result, sp = p.e.spec || {};
  const cert = isCertified(r) ? '<span style="color:var(--accent)">certified</span>' : '<span style="color:var(--fail)">not certified</span>';
  return `<b>${p.id}</b> · ${p.mode === 'qpu' ? 'real hardware' : 'simulation'} · ${cert}
    <div class="tt-row"><span class="tt-k">N</span><span>${p.n} qubits</span></div>
    <div class="tt-row"><span class="tt-k">fidelity</span><span>${fmt(p.f)}</span></div>
    <div class="tt-row"><span class="tt-k">95% CI</span><span>[${fmt(p.lo)}, ${fmt(p.hi)}]</span></div>
    <div class="tt-row"><span class="tt-k">layout</span><span>${sp.layout_strategy || '—'}</span></div>
    <div class="tt-row"><span class="tt-k">shots</span><span>${sp.shots ?? '—'}</span></div>
    ${r.qpu_seconds ? `<div class="tt-row"><span class="tt-k">QPU time</span><span>${fmt(r.qpu_seconds, 1)} s</span></div>` : ''}`;
}

/* ================================================================ parity plot */
let parityToken = 0;
async function renderParity() {
  const m = S.model;
  const fid = focusId(m), fe = fid && m.exps.get(fid);
  const id = (fe && fe.result && !fe.result.error) ? fid : S.selected || [...m.order].reverse().find((x) => { const e = m.exps.get(x); return e.result && !e.result.error; });
  const svg = $('#parSvg');
  const e = id && m.exps.get(id);
  if (!e || !e.result) { svg.innerHTML = ''; $('#parEmpty').hidden = false; $('#parityTitle').textContent = ''; $('#parityMeta').innerHTML = ''; return; }
  const tok = ++parityToken;
  let full = S.expCache.get(id);
  if (!full) {
    try { full = await api.exp(S.runId, id); S.expCache.set(id, full); } catch { full = null; }
    if (tok !== parityToken) return;
  }
  const mode = modeOf(e);
  $('#parityTitle').textContent = `${id} · ${mode.toUpperCase()} · N = ${e.result.n_qubits}`;
  const curve = full && full.parity_curve;
  if (!curve || !curve.length) {
    svg.innerHTML = ''; $('#parEmpty').hidden = false;
    $('#parEmpty').textContent = e.result.error ? 'Job failed — no data' : 'No parity data for this experiment';
    $('#parityMeta').innerHTML = '';
    return;
  }
  $('#parEmpty').hidden = true;
  const N = e.result.n_qubits || (full.physical_qubits || []).length;
  // coherence per CONTRACTS.md: C = 2·|Σ_j e^{−iNφ_j} P_j| / (2N+2)
  const fourier = () => {
    let re = 0, im = 0;
    for (const p of curve) { re += Math.cos(N * p.phi) * p.parity; im -= Math.sin(N * p.phi) * p.parity; }
    return 2 * Math.hypot(re, im) / curve.length;
  };
  const Cc = typeof full.coherence === 'number' ? full.coherence : fourier();
  $('#parityMeta').innerHTML = `P = <b>${fmt(full.populations, 2)}</b> · C = <b>${fmt(Cc, 2)}</b> · F = (P+C)/2 = <b>${fmt(full.fidelity)}</b>`;
  svg.innerHTML = '';
  const [W, H] = plotSize(svg);
  const M = { l: 46, r: 14, t: 24, b: 30 };
  const phiMax = Math.max(2 * Math.PI, ...curve.map((p) => p.phi));
  const X = (phi) => M.l + (phi / phiMax) * (W - M.l - M.r);
  // readout mitigation can push noisy points past ±1: grow the axis instead of drawing outside the panel
  const lim = Math.max(1, Math.ceil(Math.max(...curve.map((p) => Math.abs(p.parity))) * 2) / 2);
  const Y = (v) => M.t + (1 - (v + lim) / (2 * lim)) * (H - M.t - M.b);
  const gA = s('g', { class: 'axis' }, svg);
  for (const v of [-lim, -lim / 2, 0, lim / 2, lim]) {
    s('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), class: v === 0 ? 'zero' : 'grid-line' }, gA);
    s('text', { x: M.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, gA).textContent = v.toFixed(1);
  }
  const ticks = [[0, '0'], [Math.PI / 2, 'π/2'], [Math.PI, 'π'], [3 * Math.PI / 2, '3π/2'], [2 * Math.PI, '2π']];
  for (const [phi, lab] of ticks) if (phi <= phiMax + 1e-6) s('text', { x: X(phi), y: H - M.b + 17, 'text-anchor': 'middle' }, gA).textContent = lab;
  s('text', { x: W - M.r, y: H - 2, 'text-anchor': 'end', class: 'axis-title' }, gA).textContent = 'phase φ  (Rz on every qubit, then X-basis readout)';
  s('text', { x: 8, y: 11, class: 'axis-title' }, gA).textContent = 'parity';
  for (const v of [Cc, -Cc]) s('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), class: 'env' }, svg);
  s('text', { x: W - M.r - 4, y: Y(Cc) - 6, 'text-anchor': 'end', class: 'env-label' }, svg).textContent = `±C = ${fmt(Cc, 2)}`;
  // fitted ideal oscillation C·cos(Nφ + δ), phase δ from the Fourier component at frequency N
  let re = 0, im = 0;
  for (const p of curve) { re += Math.cos(N * p.phi) * p.parity; im -= Math.sin(N * p.phi) * p.parity; }
  const delta = Math.atan2(im, re);
  const npts = Math.min(1600, Math.max(200, N * 24));
  let dpath = '';
  for (let k = 0; k <= npts; k++) {
    const phi = (k / npts) * phiMax;
    dpath += `${k ? 'L' : 'M'}${X(phi).toFixed(1)} ${Y(Cc * Math.cos(N * phi + delta)).toFixed(1)}`;
  }
  s('path', { d: dpath, class: `fit ${mode}` }, svg);
  const g = s('g', {}, svg);
  const dense = curve.length > 60;
  curve.forEach((p, i) => {
    const x = X(p.phi), y = Y(p.parity);
    const stem = s('line', { x1: x, x2: x, y1: Y(0), y2: y, class: `stem ${mode}`, 'stroke-width': dense ? 1.5 : 2 }, g);
    const dot = s('circle', { cx: x, cy: y, r: dense ? 2 : 2.8, class: `stem-dot ${mode}` }, g);
    if (!reducedMotion) {
      const o = { duration: 380, delay: i * 9, fill: 'both', easing: 'ease-out' };
      stem.animate([{ opacity: 0 }, { opacity: 1 }], o);
      dot.animate([{ opacity: 0 }, { opacity: 1 }], o);
    }
  });
}

/* ================================================================ hero + budget */
function countTo(node, target) {
  const from = Number(node.dataset.v || 0);
  node.dataset.v = target;
  if (reducedMotion || from === target) { node.textContent = target; return; }
  const t0 = performance.now(), dur = 900;
  const tick = (t) => {
    const u = clamp((t - t0) / dur, 0, 1), e = 1 - Math.pow(1 - u, 3);
    node.textContent = Math.round(from + (target - from) * e);
    if (u < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  node.classList.remove('bump'); void node.offsetWidth; node.classList.add('bump');
}
function renderHero(animate = false) {
  const m = S.model;
  for (const [k, numEl, subEl] of [['qpu', $('#heroQpu'), $('#heroQpuSub')], ['sim', $('#heroSim'), $('#heroSimSub')]]) {
    const b = m.best[k];
    if (!b) {
      numEl.textContent = '—'; numEl.dataset.v = 0; numEl.classList.add('is-empty');
      subEl.textContent = k === 'qpu' ? 'awaiting a certified hardware run' : 'no certified state yet';
      continue;
    }
    numEl.classList.remove('is-empty');
    if (animate && Number(numEl.dataset.v) !== b.n) countTo(numEl, b.n);
    else { numEl.textContent = b.n; numEl.dataset.v = b.n; }
    subEl.innerHTML = `${b.id} · F = <b>${fmt(b.fidelity)}</b> [${fmt(b.ci_low)}, ${fmt(b.ci_high)}]`;
  }
  const be = m.run && m.run.backend;
  $('#heroQpuBackend').textContent = `${be || (S.cal && S.cal.backend) || 'ibm'} · ${S.cal ? S.cal.num_qubits : 156}-qubit chip`;
}
function renderBudget() {
  const b = S.model.budget;
  const q = b.qpu_seconds || 0, qm = b.max_qpu_seconds, l = b.llm_calls || 0, lm = b.max_llm_calls;
  $('#qpuUsed').textContent = q.toFixed(1);
  $('#qpuMax').textContent = qm != null ? Math.round(qm) : '—';
  $('#llmUsed').textContent = l;
  $('#llmMax').textContent = lm != null ? lm : '—';
  const qp = qm ? clamp(q / qm, 0, 1) : 0, lp = lm ? clamp(l / lm, 0, 1) : 0;
  $('#qpuBar').style.width = `${qp * 100}%`;
  $('#llmBar').style.width = `${lp * 100}%`;
  $('#qpuBar').parentNode.classList.toggle('warn', qp > 0.9);
  $('#qpuFoot').textContent = qm ? `${Math.max(0, qm - q).toFixed(0)} s of real quantum hardware left` : 'the scarce 10 minutes';
}
function renderNow() {
  const m = S.model, nl = $('#nowLine');
  const e = m.current && m.exps.get(m.current);
  let txt = 'idle', active = false;
  if (m.finished) txt = `run finished · best certified N = ${m.finished.best_certified_n ?? '—'}`;
  else if (e && !e.result) { active = true; txt = `${e.id} ${modeOf(e) === 'qpu' ? 'running on ' + ((m.run && m.run.backend) || 'QPU') : 'in simulation'} · N = ${e.spec.n_qubits}`; }
  else if (m.lastType === 'thought') { active = true; txt = 'agent is reasoning…'; }
  else if (m.run) txt = 'agent is planning the next experiment';
  nl.textContent = txt; nl.classList.toggle('active', active);
}

/* ================================================================ feed */
const feedEl = $('#feed');
function clock(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}
function escapeHtml(t) { return String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

function feedNode(item, { typewriter = false, fresh = false } = {}) {
  const m = S.model;
  switch (item.kind) {
    case 'start':
      return h('div', { class: 'f-item f-system', 'data-seq': item.seq },
        h('div', { class: 'f-meta' }, h('span', { text: clock(item.ts) }), h('span', { text: 'RUN STARTED' })),
        htmlSpan(`<b>${escapeHtml(item.d.backend || '')}</b> · ${escapeHtml(item.d.mode || '')} mode — ${escapeHtml(item.d.goal || '')}`));
    case 'thought': {
      const t = h('div', { class: 'f-text' });
      const n = h('div', { class: 'f-item f-thought', 'data-seq': item.seq },
        h('div', { class: 'f-meta' }, h('span', { text: clock(item.ts) }), h('span', { text: 'AGENT' })), t);
      if (typewriter && !reducedMotion) typewrite(n, t, item.text, typeMs(item.text) / effSpeed());
      else t.textContent = item.text;
      return n;
    }
    case 'exp': return expCard(m.exps.get(item.id), item);
    case 'milestone':
      return h('div', { class: 'f-item f-milestone' + (fresh ? ' fresh' : ''), 'data-seq': item.seq },
        h('div', { class: 'f-meta' }, h('span', { text: clock(item.ts) }), h('span', { text: '★ MILESTONE' })),
        h('div', { text: item.d.text || '' }));
    case 'error':
      return h('div', { class: 'f-item f-error', 'data-seq': item.seq },
        h('div', { class: 'f-meta' }, h('span', { text: clock(item.ts) }), h('span', { text: 'ERROR' })),
        h('div', { text: item.text }));
    case 'finish':
      return h('div', { class: 'f-item f-system finish', 'data-seq': item.seq },
        h('div', { class: 'f-meta' }, h('span', { text: clock(item.ts) }), h('span', { text: 'RUN FINISHED' })),
        htmlSpan(`<b>Best certified: ${escapeHtml(item.d.best_certified_n ?? '—')} qubits</b> (${escapeHtml(item.d.best_exp_id || '')}). ${escapeHtml(item.d.summary || '')}`));
    default: return h('div');
  }
}
function htmlSpan(html) { const d = document.createElement('div'); d.innerHTML = html; return d; }

function expCard(e, item) {
  if (!e) return h('div');
  const sp = e.spec || {}, r = e.result, mode = modeOf(e);
  const card = h('div', { class: `f-item f-card ${mode}-card`, 'data-exp': e.id, 'data-seq': item.seq, onclick: () => r && select(e.id) });
  fillCard(card, e);
  void sp;
  return card;
}
function fillCard(card, e) {
  const sp = e.spec || {}, r = e.result, mode = modeOf(e);
  const ok = r && isCertified(r);
  card.classList.toggle('is-ok', !!ok);
  card.classList.toggle('selected', S.selected === e.id);
  card.title = r ? 'Click to inspect' : '';
  let res;
  if (!r) res = h('div', { class: 'runbar' }, h('i'));
  else if (r.error) res = htmlSpan(`<div class="res"><span class="badge bad">FAILED</span><span class="err">${escapeHtml(r.error)}</span></div>`);
  else res = htmlSpan(`<div class="res"><span class="badge ${ok ? 'ok' : 'bad'}">${ok ? 'CERTIFIED' : 'NOT CERTIFIED'}</span><span>F = <b>${fmt(r.fidelity)}</b></span><span>[${fmt(r.ci_low)}, ${fmt(r.ci_high)}]</span>${r.qpu_seconds ? `<span>${fmt(r.qpu_seconds, 1)} QPU-s</span>` : ''}</div>`);
  const strat = [sp.layout_strategy, sp.shots ? `${sp.shots} shots` : null, (sp.exclude_qubits || []).length ? `excl ${sp.exclude_qubits.length}` : null].filter(Boolean).join(' · ');
  card.replaceChildren(
    h('div', { class: 'row' },
      h('span', { class: 'exp-id', text: e.id }),
      h('span', { class: `badge ${mode}`, text: mode === 'qpu' ? 'QPU' : 'SIM' }),
      h('span', { class: 'exp-n', text: `N = ${sp.n_qubits ?? (r && r.n_qubits) ?? '?'}` }),
      h('span', { class: 'spacer' }),
      h('span', { class: 'exp-strat', text: strat })),
    sp.hypothesis ? h('div', { class: 'hyp', text: sp.hypothesis }) : null,
    res);
}
function updateCard(id) {
  const card = feedEl.querySelector(`[data-exp="${CSS.escape(id)}"]`);
  const e = S.model.exps.get(id);
  if (card && e) fillCard(card, e);
}

function typewrite(wrap, textEl, text, ms) {
  if (S.typing) S.typing.finish();
  wrap.classList.add('typing');
  const t0 = performance.now();
  let raf = 0, done = false;
  const finish = () => { if (done) return; done = true; cancelAnimationFrame(raf); textEl.textContent = text; wrap.classList.remove('typing'); if (S.typing && S.typing.wrap === wrap) S.typing = null; stickScroll(); };
  const tick = (t) => {
    if (done) return;
    const u = clamp((t - t0) / Math.max(ms, 1), 0, 1);
    textEl.textContent = text.slice(0, Math.ceil(text.length * u));
    stickScroll();
    if (u >= 1) finish(); else raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  S.typing = { wrap, finish };
}

function renderFeed() {
  if (S.typing) { S.typing.finish(); S.typing = null; }
  feedEl.replaceChildren(...S.model.feed.map((it) => feedNode(it)));
  $('#feedCount').textContent = `${S.model.order.length} experiments`;
  S.stick = true; stickScroll(true);
}
function appendFeed(item, opts) {
  feedEl.append(feedNode(item, opts));
  $('#feedCount').textContent = `${S.model.order.length} experiments`;
  stickScroll();
}
function stickScroll(force = false) {
  if (force || S.stick) feedEl.scrollTop = feedEl.scrollHeight;
  $('#feedJump').hidden = S.stick;
}
feedEl.addEventListener('scroll', () => {
  S.stick = feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight < 48;
  $('#feedJump').hidden = S.stick;
});
$('#feedJump').addEventListener('click', () => { S.stick = true; stickScroll(true); });

/* ================================================================ selection */
// What the map/parity show when the user hasn't pinned anything: the live experiment,
// or, once the run is finished, the record-setting one.
function focusId(m = S.model) {
  if (S.selected) return S.selected;
  const b = m.finished && m.finished.best_exp_id;
  if (b && m.exps.has(b)) return b;
  return m.current;
}
function select(id) {
  S.selected = (S.selected === id) ? null : id;
  for (const c of feedEl.querySelectorAll('.f-card')) c.classList.toggle('selected', c.dataset.exp === S.selected);
  const m = S.model;
  const e = m.exps.get(focusId(m));
  renderTree(e, { animate: false });
  renderChart();
  renderParity();
}

/* ================================================================ transport */
const scrub = $('#scrub');
function renderTransport() {
  const n = S.events.length;
  scrub.max = Math.max(0, n - 1);
  scrub.value = Math.max(0, S.cursor);
  const p = n > 1 ? (Math.max(0, S.cursor) / (n - 1)) * 100 : 0;
  scrub.style.setProperty('--p', `${p}%`);
  $('#playBtn').classList.toggle('playing', S.playing);
  $('#playBtn').classList.toggle('attn', !S.playing && S.cursor < n - 1 && S.mode === 'replay');
  const m = S.model;
  $('#clockTime').textContent = m.lastTs ? new Date(m.lastTs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) : '--:--:--';
  let el = '';
  if (m.firstTs && m.lastTs) {
    const sec = Math.max(0, (new Date(m.lastTs) - new Date(m.firstTs)) / 1000);
    el = `T+${Math.floor(sec / 3600)}h${String(Math.floor((sec % 3600) / 60)).padStart(2, '0')}m · `;
  }
  $('#clockSub').textContent = `${el}event ${Math.max(0, S.cursor) + 1} / ${n}`;
  for (const t of $('#ticks').children) t.classList.toggle('past', Number(t.dataset.i) <= S.cursor);
  const pill = $('#statusPill');
  const live = S.mode === 'live' && !S.finished;
  pill.className = 'pill' + (live ? ' live' : '');
  pill.textContent = S.mode === 'live' ? (S.finished ? 'LIVE · FINISHED' : 'LIVE') : 'REPLAY';
}
function renderTicks() {
  const box = $('#ticks');
  box.replaceChildren();
  const n = S.events.length;
  if (n < 2) return;
  S.events.forEach((ev, i) => {
    let cls = null;
    if (ev.type === 'milestone') cls = 'ms';
    else if (ev.type === 'experiment_result') cls = ev.data.error || !isCertified(ev.data) ? 'bad' : (ev.data.mode === 'qpu' ? 'qpu' : '');
    if (cls === null) return;
    const t = h('i', { class: cls });
    t.dataset.i = i;
    t.style.left = `${(i / (n - 1)) * 100}%`;
    box.append(t);
  });
}

function renderAll() {
  const m = S.model;
  renderHero(false); renderBudget(); renderNow(); renderFeed();
  const e = m.exps.get(focusId(m));
  chip.shownId = null;
  renderTree(e, { animate: false, layoutOverride: e && !e.result ? peekLayout(e.id) : null });
  if (e && !e.result) setTreeClass('running', false, true);
  renderChart(); renderParity(); renderTransport();
}

function step() {
  S.timer = null;
  if (S.cursor >= S.events.length - 1) { S.waiting = true; if (S.mode !== 'live') pause(); renderTransport(); return; }
  S.cursor++;
  const ev = S.events[S.cursor];
  const m = S.model;
  const prevBest = { ...m.best };
  applyEvent(m, ev);
  const d = ev.data || {};
  switch (ev.type) {
    case 'run_started': case 'thought': case 'error': case 'run_finished':
      appendFeed(m.feed[m.feed.length - 1], { typewriter: ev.type === 'thought' });
      if (ev.type === 'run_finished') {
        S.finished = true;
        if (!S.selected) { renderTree(m.exps.get(focusId(m))); renderChart(); renderParity(); }
      }
      break;
    case 'milestone':
      appendFeed(m.feed[m.feed.length - 1], { fresh: true });
      break;
    case 'experiment_started':
      appendFeed(m.feed[m.feed.length - 1]);
      if (!S.selected) {
        const e = m.exps.get(d.exp_id);
        // the start event has no layout; peek ahead at the result so the tree can grow now
        renderTree(e, { animate: true, layoutOverride: peekLayout(d.exp_id) });
      }
      break;
    case 'experiment_result':
      if (!m.feed.some((f) => f.kind === 'exp' && f.id === d.exp_id) || !feedEl.querySelector(`[data-exp="${CSS.escape(d.exp_id)}"]`)) {
        appendFeed(m.feed.find((f) => f.kind === 'exp' && f.id === d.exp_id) || m.feed[m.feed.length - 1]);
      }
      updateCard(d.exp_id);
      if (!S.selected) renderTree(m.exps.get(d.exp_id), { animate: chip.shownId !== d.exp_id });
      renderChart(d.exp_id);
      if (!S.selected) renderParity();
      renderHero(prevBest.qpu !== m.best.qpu || prevBest.sim !== m.best.sim);
      break;
    case 'budget':
      break;
  }
  renderBudget(); renderNow(); renderTransport();
  if (S.playing || S.mode === 'live') schedule_(dwell(ev) / effSpeed());
}
function peekLayout(expId) {
  for (let i = S.cursor + 1; i < S.events.length && i < S.cursor + 60; i++) {
    const ev = S.events[i];
    if (ev.type === 'experiment_result' && ev.data && ev.data.exp_id === expId && ev.data.tree_edges) {
      return { root: ev.data.root, edges: ev.data.tree_edges, qubits: ev.data.physical_qubits || [] };
    }
  }
  return null; // live mode: layout appears with the result
}
function schedule_(ms) { clearTimeout(S.timer); S.waiting = false; S.timer = setTimeout(step, ms); }

function play() {
  if (S.cursor >= S.events.length - 1 && S.mode === 'replay') seek(-1);
  S.playing = true; renderTransport(); schedule_(150);
}
function pause() { S.playing = false; clearTimeout(S.timer); S.timer = null; renderTransport(); }
function seek(i) {
  const was = S.playing;
  clearTimeout(S.timer); S.timer = null;
  S.cursor = clamp(i, -1, S.events.length - 1);
  S.model = modelAt(S.cursor);
  renderAll();
  if (was) schedule_(400);
}

/* ================================================================ run loading + live */
async function loadRuns() {
  try { S.runs = (await api.runs()).runs || []; } catch { S.runs = []; }
  const sel = $('#runSelect');
  sel.replaceChildren(...S.runs.map((r) => h('option', { value: r.run_id, text:
    `${r.run_id}  ·  ${String(r.mode || '').toUpperCase()}  ·  best ${r.best_certified_qpu || r.best_certified_n || 0}q  ·  ${r.finished ? 'done' : 'running'}` })));
  if (!S.runs.length) sel.append(h('option', { text: 'no runs in runs/ — try: uv run python -m scripts.make_demo_run' }));
}

async function loadRun(id, { mode = null, at = null, autoplay = false } = {}) {
  const tok = ++S.token;
  pause(); stopPolling();
  S.runId = id; S.selected = null; S.expCache.clear(); S.finished = false;
  $('#runSelect').value = id;
  const info = S.runs.find((r) => r.run_id === id) || {};
  const paper = $('#paperLink');
  paper.hidden = !info.has_paper;
  paper.href = api.paperUrl(id);
  let cal = null, evs = [];
  try { cal = await api.cal(id); } catch { cal = null; }
  try { const r = await api.events(id, -1); evs = r.events || []; S.finished = !!r.finished; } catch { evs = []; }
  if (tok !== S.token) return;
  S.cal = cal; S.events = evs;
  drawChip(cal);
  renderTicks();
  setMode(mode || (info.finished === false && !STATIC ? 'live' : 'replay'), { reload: false });  // static snapshot: replay, no polling
  let cursor;
  if (S.mode === 'live' || at === 'end' || (STATIC && (at == null || at === ''))) cursor = evs.length - 1;  // land on the result; play restarts
  else if (at != null && at !== '' && !isNaN(Number(at))) cursor = Number(at);
  else cursor = Math.min(0, evs.length - 1);
  seek(cursor);
  const url = new URL(location.href); url.searchParams.set('run', id); history.replaceState(null, '', url);
  if (S.mode === 'live') { S.playing = true; startPolling(); renderTransport(); }
  else if (autoplay) play();
}

function setMode(mode, { reload = true } = {}) {
  S.mode = mode;
  for (const b of document.querySelectorAll('[data-mode]')) b.classList.toggle('on', b.dataset.mode === mode);
  if (!reload) return;
  if (mode === 'live') { seek(S.events.length - 1); S.playing = true; startPolling(); }
  else { stopPolling(); pause(); }
  renderTransport();
}

function startPolling() {
  stopPolling();
  const tick = async () => {
    const tok = S.token;
    const last = S.events.length ? S.events[S.events.length - 1].seq : -1;
    try {
      const r = await api.events(S.runId, last);
      if (tok !== S.token || S.mode !== 'live') return;
      if (r.events && r.events.length) {
        const known = new Set(S.events.map((e) => e.seq));
        for (const ev of r.events) if (!known.has(ev.seq)) S.events.push(ev);
        renderTicks();
        if (S.playing && !S.timer) schedule_(80);
      }
      S.finished = !!r.finished;
      renderTransport();
    } catch { /* server hiccup: keep polling */ }
    if (tok === S.token && S.mode === 'live') S.pollTimer = setTimeout(tick, POLL_MS);
  };
  S.pollTimer = setTimeout(tick, POLL_MS);
}
function stopPolling() { clearTimeout(S.pollTimer); S.pollTimer = null; }

/* ================================================================ tooltip */
const tip = $('#tooltip');
function showTip(ev, html) { tip.innerHTML = html; tip.hidden = false; moveTip(ev); }
function moveTip(ev) {
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; }

/* ================================================================ wiring */
$('#playBtn').addEventListener('click', () => (S.playing ? pause() : play()));
for (const b of document.querySelectorAll('[data-speed]')) {
  b.classList.toggle('on', Number(b.dataset.speed) === S.speed);
  b.addEventListener('click', () => {
    S.speed = Number(b.dataset.speed);
    for (const o of document.querySelectorAll('[data-speed]')) o.classList.toggle('on', o === b);
    if (S.typing) S.typing.finish();
    if (S.playing && S.timer) schedule_(Math.min(400, dwell(S.events[S.cursor] || {}) / effSpeed()));
  });
}
for (const b of document.querySelectorAll('[data-mode]')) b.addEventListener('click', () => setMode(b.dataset.mode));
scrub.addEventListener('input', () => {
  if (S.mode === 'live') setMode('replay');
  seek(Number(scrub.value));
});
$('#runSelect').addEventListener('change', (e) => loadRun(e.target.value));
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.key === ' ') { e.preventDefault(); S.playing ? pause() : play(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); seek(S.cursor + 1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); seek(S.cursor - 1); }
  else if (e.key === 'End') seek(S.events.length - 1);
  else if (e.key === 'Home') seek(0);
  else if (e.key === 'Escape' && S.selected) select(S.selected);
});
let rsz = 0;
new ResizeObserver(() => { cancelAnimationFrame(rsz); rsz = requestAnimationFrame(() => { if (S.model) { renderChart(); renderParity(); } }); })
  .observe($('#fidPlot'));

(async function init() {
  S.model = emptyModel();
  await loadRuns();
  const want = qs.get('run');
  const id = (want && S.runs.some((r) => r.run_id === want)) ? want : (S.runs[0] && S.runs[0].run_id);
  if (!id) { renderAll(); return; }
  await loadRun(id, { mode: qs.get('mode'), at: qs.get('at'), autoplay: qs.get('autoplay') === '1' });
})();
