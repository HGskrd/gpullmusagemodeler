/* Planner app: per-tab session plumbing (X-Tab-ID), HTMX wiring, ECharts
 * rendering (5 modes), pickers, scenario import/export, live sliders,
 * scroll/focus restore, toasts. Shared by base.html and use_cases.html. */
let chart = null;
let lastChartData = null;
const colorSchemeQuery = matchMedia('(prefers-color-scheme:dark)');
const FAST_HTMX_MS = 10;
if (window.htmx) {
  htmx.config.defaultSettleDelay = FAST_HTMX_MS;
}

/* Keep page scroll stable across HTMX swaps of #main-content.
 * Without this, replacing the whole config section re-anchors the scroll
 * near the top of that container and jumps the user around as they
 * tweak sliders. */
let _savedScrollY = null;
let _focusRestore = null;
function rememberScroll() { _savedScrollY = window.scrollY; }
function restoreScroll() {
  if (_savedScrollY == null) return;
  const y = _savedScrollY;
  _savedScrollY = null;
  window.scrollTo({ top: y, left: 0, behavior: 'auto' });
}
const TAB_PARAM = 'tab_id';
const TAB_STORAGE_KEY = 'planner_tab_id';

function newTabId() {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  const bytes = new Uint32Array(4);
  if (window.crypto && window.crypto.getRandomValues) {
    window.crypto.getRandomValues(bytes);
    return Array.from(bytes, n => n.toString(16).padStart(8, '0')).join('-');
  }
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

function storedTabId() {
  try {
    return window.sessionStorage.getItem(TAB_STORAGE_KEY);
  } catch {
    return null;
  }
}

function setStoredTabId(tabId) {
  try {
    window.sessionStorage.setItem(TAB_STORAGE_KEY, tabId);
  } catch {}
}

function removeTabIdFromUrl() {
  const current = new URL(window.location.href);
  if (!current.searchParams.has(TAB_PARAM)) return;
  current.searchParams.delete(TAB_PARAM);
  const clean = `${current.pathname}${current.search}${current.hash}`;
  window.history.replaceState(null, '', clean || '/');
}

function initTabId() {
  const params = new URLSearchParams(window.location.search);
  const legacyUrlTabId = params.get(TAB_PARAM);
  const tabId = storedTabId() || legacyUrlTabId || newTabId();
  setStoredTabId(tabId);
  removeTabIdFromUrl();
  return tabId;
}

const currentTabIdValue = initTabId();

function currentTabId() {
  return currentTabIdValue;
}

function withTabUrl(url) {
  const next = new URL(url, window.location.origin);
  next.searchParams.delete(TAB_PARAM);
  return `${next.pathname}${next.search}`;
}

function requestHeaders(extra = {}) {
  return {
    'X-Tab-ID': currentTabId(),
    ...extra,
  };
}

function disposeChart() {
  if (chart) {
    chart.dispose();
    chart = null;
  }
}

function chartTheme() {
  const dk = colorSchemeQuery.matches;
  const cssSans = getComputedStyle(document.documentElement).getPropertyValue('--sans').trim();
  return {
    dk,
    gc: dk ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.04)',
    tc: dk ? 'rgba(255,255,255,.5)' : 'rgba(0,0,0,.4)',
    bg: dk ? '#1a1a18' : '#ffffff',
    border: dk ? 'rgba(255,255,255,.12)' : 'rgba(0,0,0,.1)',
    text: dk ? '#e4e4e0' : '#1a1a18',
    fontFamily: cssSans || "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
  };
}

function normalizeDatasets(datasets, dk) {
  return (datasets || []).map(ds => {
    if (!ds._isAggregate) return ds;
    return {
      ...ds,
      borderColor: dk ? '#ddd' : '#222',
      backgroundColor: dk ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)',
    };
  });
}

function dashStyle(borderDash) {
  return borderDash && borderDash.length ? 'dashed' : 'solid';
}

function makeSeries(ds, showPoints) {
  const lw = Number(ds.borderWidth || 2);
  return {
    name: ds.label,
    type: 'line',
    data: (ds.data || []).map(point => ({
      value: [Number(point.x), point.y == null ? null : Number(point.y)],
      raw: point,
      hardware: ds.hardware || '',
      specDisclosure: ds.spec_disclosure || '',
    })),
    connectNulls: Boolean(ds.spanGaps),
    showSymbol: Boolean(showPoints),
    symbol: 'circle',
    symbolSize: 5,
    smooth: Number(ds.tension || 0),
    lineStyle: {
      color: ds.borderColor,
      width: lw,
      type: dashStyle(ds.borderDash),
    },
    itemStyle: {
      color: ds.borderColor,
      borderColor: 'rgba(255,255,255,0.75)',
      borderWidth: 1.5,
    },
    areaStyle: ds.fill ? { color: ds.backgroundColor || ds.borderColor } : undefined,
    emphasis: {
      focus: 'series',
      symbolSize: 8,
      lineStyle: { width: lw + 1.5 },
      itemStyle: { borderWidth: 2 },
    },
    blur: {
      lineStyle: { opacity: 0.1 },
      itemStyle: { opacity: 0.1 },
    },
  };
}

function specDisclosure(p) {
  const raw = p?.data?.raw || {};
  if (raw.spec_method && raw.spec_k) {
    const autoOff = raw.spec_auto && raw.spec_beneficial === false;
    const auto = autoOff ? `Auto→off (best ${raw.spec_method} k=${raw.spec_k})` : raw.spec_auto ? `Auto→${raw.spec_k}` : raw.spec_k;
    const alpha = raw.spec_alpha == null ? '' : ` · α ${(Number(raw.spec_alpha) * 100).toFixed(0)}%`;
    const speedup = raw.spec_speedup == null ? '' : ` · ${Number(raw.spec_speedup).toFixed(2)}×`;
    return `${raw.spec_method} · k ${auto}${alpha}${speedup}`;
  }
  return p?.data?.specDisclosure || '';
}

function specDisclosureHtml(p, theme) {
  const disclosure = specDisclosure(p);
  return disclosure
    ? `<div style="margin:0 0 3px 17px;color:${theme.tc};font-size:9px;">Spec: ${disclosure}</div>`
    : '';
}

function escapeTooltipHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function hardwareDisclosureHtml(p, theme) {
  const hardware = p?.data?.hardware;
  return hardware
    ? `<div style="margin:0 0 2px 17px;color:${theme.tc};font-family:${theme.fontFamily};font-size:8px;line-height:1.25;">${escapeTooltipHtml(hardware)}</div>`
    : '';
}

function axisCommon(theme, title, formatter, extra = {}) {
  return {
    scale: true,
    name: title,
    nameLocation: 'middle',
    nameTextStyle: {
      color: theme.tc,
      fontFamily: theme.fontFamily,
      fontSize: 11,
    },
    splitLine: { lineStyle: { color: theme.gc } },
    axisLine: { lineStyle: { color: theme.gc } },
    axisTick: { lineStyle: { color: theme.gc } },
    axisLabel: {
      color: theme.tc,
      fontFamily: theme.fontFamily,
      fontSize: 10,
      formatter: value => formatter(Number(value)),
    },
    ...extra,
  };
}

function tooltipBase(theme, trigger, formatter) {
  const shadow = theme.dk ? 'rgba(0,0,0,.55)' : 'rgba(0,0,0,.14)';
  return {
    trigger,
    renderMode: 'html',
    backgroundColor: theme.bg,
    borderColor: theme.border,
    borderWidth: 1,
    padding: [9, 13],
    confine: true,
    extraCssText: `box-shadow:0 6px 24px ${shadow};border-radius:7px;font-family:${theme.fontFamily};line-height:1.35;max-width:min(460px,calc(100vw - 24px));white-space:normal;overflow-wrap:anywhere;`,
    textStyle: { color: theme.text, fontFamily: theme.fontFamily, fontSize: 11 },
    axisPointer: trigger === 'axis' ? {
      type: 'cross',
      snap: false,
      crossStyle: { color: theme.tc, opacity: 0.55, width: 1 },
      lineStyle: { color: theme.tc, opacity: 0.35, type: 'dashed', width: 1 },
      label: { show: false },
    } : undefined,
    formatter,
  };
}

function baseOption(theme, tooltipFormatter, trigger) {
  return {
    aria: { enabled: true, decal: { show: true } },
    animation: true,
    animationDuration: 280,
    animationEasing: 'cubicOut',
    textStyle: { fontFamily: theme.fontFamily },
    grid: { top: 18, right: 18, bottom: 48, left: 22, containLabel: true },
    legend: { show: false },
    tooltip: tooltipBase(theme, trigger || 'item', tooltipFormatter),
  };
}

function axisRow(p, theme, yStr) {
  const oom = yStr === 'OOM';
  return `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
    `<span style="display:inline-block;width:10px;height:2.5px;border-radius:2px;background:${p.color};flex-shrink:0;"></span>` +
    `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;color:${theme.text};">${escapeTooltipHtml(p.seriesName)}</span>` +
    `<span style="font-weight:600;margin-left:4px;color:${oom ? 'var(--red)' : theme.text};">${yStr}</span>` +
    `</div>`;
}

function axisHeader(theme, label, value) {
  return `<div style="font-size:9px;color:${theme.tc};text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid ${theme.border};">${label}: <b style="color:${theme.text};">${value}</b></div>`;
}

function hydrateHtmx(root) {
  if (root && window.htmx) htmx.process(root);
}

/* Live slider-to-display mirroring. Sliders POST on `change` (release); while
 * dragging, the `input` event updates the element at `data-live-target` so the
 * user sees the value move without a round-trip per frame.
 *
 * Attributes on the slider:
 *   data-live-target   CSS selector for the element to update
 *   data-live-scale    Multiplier applied to slider value before display (default 1)
 *   data-live-decimals Decimal places (default 0)
 *   data-live-suffix   Appended to the formatted number (default "")
 *   data-live-format   "fmt_num" or "log2-tok" — special formatters
 */
function formatLiveValue(slider, raw) {
  const fmt = slider.dataset.liveFormat;
  if (fmt === 'log2-tok') {
    return Math.round(Math.pow(2, raw)).toLocaleString();
  }
  if (fmt === 'fmt_num') {
    // Mirror of the server-side fmt_num Jinja filter — keep units and
    // decimals identical so live slider labels match re-rendered HTML.
    const v = Number(raw);
    if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'k';
    return String(Math.trunc(v));
  }
  const scale = Number(slider.dataset.liveScale || 1);
  const decimals = Number(slider.dataset.liveDecimals || 0);
  const suffix = slider.dataset.liveSuffix || '';
  return (raw * scale).toFixed(decimals) + suffix;
}

function writeLiveValue(target, text) {
  if (!target) return;
  if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
    target.value = text;
  } else {
    target.textContent = text;
  }
}

function bindLiveSliders(root) {
  (root || document).querySelectorAll('input[type="range"][data-live-target]').forEach(slider => {
    if (slider._liveBound) return;
    slider._liveBound = true;
    slider.addEventListener('input', () => {
      const target = document.querySelector(slider.dataset.liveTarget);
      writeLiveValue(target, formatLiveValue(slider, Number(slider.value)));
    });
  });
}

function showToast(message) {
  const toast = document.getElementById('toast');
  if (!toast || !message) return;
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => { toast.hidden = true; }, 3200);
}

function markFastRemove(source) {
  const trigger = source?.closest?.('[data-fast-remove]');
  const card = trigger?.closest?.('[data-fast-card]');
  if (!card) return;
  card.dataset.fastRemoving = 'true';
  card.dataset.fastDisplay = card.style.display || '';
  card.style.display = 'none';
  card.style.pointerEvents = 'none';
}

function restoreFastRemove(source) {
  const trigger = source?.closest?.('[data-fast-remove]');
  const card = trigger?.closest?.('[data-fast-card]');
  if (!card || card.dataset.fastRemoving !== 'true') return;
  delete card.dataset.fastRemoving;
  card.style.display = card.dataset.fastDisplay || '';
  delete card.dataset.fastDisplay;
  card.style.pointerEvents = '';
}

function applyOobSwaps(fragment) {
  fragment.querySelectorAll('[hx-swap-oob]').forEach(node => {
    const id = node.id;
    const swap = node.getAttribute('hx-swap-oob');
    const current = id ? document.getElementById(id) : null;
    const replacement = node.cloneNode(true);
    replacement.removeAttribute('hx-swap-oob');
    node.remove();
    if (!current || swap !== 'outerHTML') return;
    current.replaceWith(replacement);
    hydrateHtmx(replacement);
  });
}

function replaceHtml(target, html) {
  if (!target) return;
  rememberScroll();
  const tpl = document.createElement('template');
  tpl.innerHTML = html;
  applyOobSwaps(tpl.content);
  target.replaceChildren(tpl.content);
  hydrateHtmx(target);
  bindLiveSliders(target);
  restoreScroll();
}

function syncCurrentTabState() {
  return fetch(withTabUrl('/session/sync'), {
    headers: requestHeaders({ 'HX-Request': 'true' }),
  }).then(async r => {
    const text = await r.text();
    if (!r.ok) throw new Error(text || `Request failed (${r.status})`);
    replaceHtml(document.getElementById('main-content'), text);
  });
}

function refreshChart() {
  if (!document.getElementById('mc')) return;
  fetch(withTabUrl('/api/chart-data'), { headers: requestHeaders() })
    .then(r => r.json())
    .then(data => renderChart(data))
    .catch(console.error);
}

function updateChartAccessibleSummary(data) {
  const root = document.getElementById('chartAccessibleSummary');
  const chartEl = document.getElementById('mc');
  if (!root || !chartEl) return;
  const modeLabels = {
    userpareto: 'User Pareto', usercapacity: 'User Capacity', processingpareto: 'Processing Pareto',
    decision: 'Decision response time', embedquality: 'Embedding Quality', asrquality: 'ASR Quality', realtime: 'Realtime capacity',
  };
  const modeLabel = modeLabels[data?.mode] || 'Performance estimate';
  const datasets = Array.isArray(data?.datasets) ? data.datasets : [];
  chartEl.setAttribute('aria-label', `${modeLabel} chart with ${datasets.length} model series. A text summary follows.`);
  root.replaceChildren();
  const heading = document.createElement('h3');
  heading.textContent = `${modeLabel} chart summary`;
  root.appendChild(heading);
  if (!datasets.length) {
    const empty = document.createElement('p');
    empty.textContent = data.capacity_only?.length
      ? data.capacity_only.map(row => `${row.label}: ${Number(row.capacity).toLocaleString()} ${row.unit}; quality unavailable.`).join(' ')
      : 'No deployed model data is available for this chart.';
    root.appendChild(empty);
    return;
  }
  const list = document.createElement('ul');
  datasets.forEach(ds => {
    const points = (ds.data || []).filter(point => Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y)));
    const item = document.createElement('li');
    if (!points.length) {
      item.textContent = `${ds.label}: no feasible points.`;
    } else {
      const xs = points.map(point => Number(point.x));
      const ys = points.map(point => Number(point.y));
      item.textContent = `${ds.label}: ${points.length} points; x range ${Math.min(...xs).toLocaleString()} to ${Math.max(...xs).toLocaleString()}, y range ${Math.min(...ys).toLocaleString()} to ${Math.max(...ys).toLocaleString()}.`;
    }
    list.appendChild(item);
  });
  root.appendChild(list);
}

function setChartCaption(text) {
  const el = document.getElementById('chartCaption');
  if (!el) return;
  el.textContent = text || '';
  el.hidden = !text;
}

/* Decision models: mean response time against arrival rate. Each model is a
 * band spanning a range of input lengths, with a solid line at one reference
 * length. The server sends, per model and length, capacity (decisions/s) and
 * the time one request takes alone; the curve comes from a queue model:
 * M/D/c, with c = capacity × single-request time (1 when batching adds no
 * throughput), Erlang-C wait halved for deterministic service. */
const decisionView = { off: new Set(), lo: null, hi: null, ref: null, target: null };
try { decisionView.target = Number(localStorage.getItem('decisionTargetMs')) || null; } catch (e) { /* storage unavailable */ }

function showDecisionExtras(show) {
  ['chartControls', 'chartReadout'].forEach(id => {
    const node = document.getElementById(id);
    if (!node) return;
    node.hidden = !show;
    if (!show) node.replaceChildren();
  });
}

function decisionResponseMs(mu, serviceMs, lam) {
  if (!(mu > 0) || !(serviceMs > 0) || lam >= mu) return null;
  const s = serviceMs / 1000;
  const c = Math.max(1, Math.round(mu * s));
  const mu1 = mu / c;
  const a = lam / mu1;
  let sum = 0, term = 1;
  for (let k = 0; k < c; k++) { sum += term; term *= a / (k + 1); }
  const top = term * c / (c - a);
  const wait = (top / (sum + top)) / (c * mu1 - lam) / 2;
  return (s + wait) * 1000;
}

function decisionLenLabel(len) {
  return len >= 1024 && len % 1024 === 0 ? `${len / 1024}k` : String(len);
}

function renderDecisionChart(data, el, theme) {
  const css = getComputedStyle(document.documentElement);
  const red = css.getPropertyValue('--red').trim() || '#b42e2e';
  const mono = css.getPropertyValue('--mono').trim() || 'monospace';
  const fmt = v => v >= 100 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(1) : v.toFixed(2);
  const YCAP = 1500;
  const useCases = Array.isArray(data.use_cases) ? data.use_cases : [];
  const models = (data.datasets || []).map(ds => ({
    ...ds,
    byLen: new Map((ds.data || []).filter(p => p.y != null && p.service_ms != null).map(p => [Number(p.x), p])),
  }));
  const lengths = [...new Set((data.datasets || []).flatMap(ds => (ds.data || []).map(p => Number(p.x))))].sort((a, b) => a - b);
  const view = decisionView;
  const ucLens = useCases.map(u => u.len);
  if (!lengths.includes(view.ref)) view.ref = ucLens[0] ?? (lengths.includes(512) ? 512 : lengths[0]);
  if (!lengths.includes(view.lo) || !lengths.includes(view.hi)) {
    const lo = ucLens.length ? Math.min(...ucLens) : 512;
    const hi = ucLens.length ? Math.max(...ucLens) : 512;
    view.lo = lengths.filter(l => l <= lo / 2).pop() ?? lengths[0];
    view.hi = lengths.find(l => l >= hi * 2) ?? lengths[lengths.length - 1];
  }

  const sample = (mu, serviceMs, xMin) => {
    const pts = [];
    const steps = 140;
    for (let i = 0; i <= steps; i++) {
      const lam = xMin * Math.pow(mu * 0.995 / xMin, i / steps);
      const t = decisionResponseMs(mu, serviceMs, lam);
      if (t == null || t > YCAP) break;
      pts.push([lam, t]);
    }
    pts.push([mu, YCAP]);
    return pts;
  };

  const visible = models.filter(m => !view.off.has(m.label));
  const peaks = useCases.map(u => u.peak_rps).filter(v => v > 0);
  const mus = visible.flatMap(m => [...m.byLen.values()].map(p => p.y));
  const floors = visible.flatMap(m => [...m.byLen.values()].map(p => p.service_ms));
  const xMin = Math.pow(10, Math.floor(Math.log10(Math.min(0.1, ...peaks.map(v => v / 3)))));
  const xMax = Math.pow(10, Math.ceil(Math.log10(Math.max(10, ...mus, ...peaks) * 1.5)));
  const yMin = Math.pow(10, Math.floor(Math.log10(Math.max(0.01, Math.min(1, ...floors) * 0.8))));

  const series = [];
  visible.forEach(m => {
    const color = m.borderColor || m.color;
    const lo = m.byLen.get(view.lo);
    const hiLen = [...m.byLen.keys()].filter(l => l <= view.hi && l >= view.lo).pop();
    const hi = hiLen != null ? m.byLen.get(hiLen) : null;
    if (lo && hi && hiLen !== view.lo) {
      const poly = sample(lo.y, lo.service_ms, xMin).concat(sample(hi.y, hi.service_ms, xMin).reverse());
      series.push({
        type: 'custom', name: `${m.label} band`, silent: true, z: 1, data: [0],
        renderItem: (params, api) => ({
          type: 'polygon',
          shape: { points: poly.map(pt => api.coord(pt)) },
          style: { fill: color, opacity: 0.1, stroke: color, lineWidth: 0.8 },
        }),
      });
    }
    const ref = m.byLen.get(view.ref);
    if (ref) {
      series.push({
        type: 'line', name: m.label, z: 3, showSymbol: false,
        data: sample(ref.y, ref.service_ms, xMin).map(pt => ({ value: pt, hardware: m.hardware || '' })),
        lineStyle: { color, width: 2.2 }, itemStyle: { color },
        endLabel: { show: true, distance: 4, formatter: `${fmt(ref.y)}/s`, color, fontFamily: mono, fontSize: 10 },
        emphasis: { disabled: true },
      });
    }
  });
  const marks = useCases.filter(u => u.peak_rps > 0).map(u => ({
    xAxis: u.peak_rps,
    label: {
      formatter: `${u.name} · ${decisionLenLabel(u.len)} tok · ${fmt(u.peak_rps)}/s${u.batch ? ' mean' : ' peak'}`,
      position: 'insideEndTop', color: theme.text, fontFamily: theme.fontFamily, fontSize: 10,
    },
    lineStyle: { color: theme.text, type: u.batch ? 'dashed' : 'solid', width: 1, opacity: 0.7 },
  }));
  if (view.target) {
    marks.push({
      yAxis: view.target,
      label: { formatter: `target ${fmt(view.target)} ms`, position: 'insideStartTop', color: red, fontFamily: mono, fontSize: 10 },
      lineStyle: { color: red, type: 'dashed', width: 1 },
    });
  }
  if (marks.length) series.push({ type: 'line', data: [], silent: true, markLine: { symbol: 'none', silent: true, data: marks } });

  chart.setOption({
    ...baseOption(theme, params => {
      // Values are computed at the pointer's arrival rate, not read off the
      // sampled curve, so they are exact for any x.
      const lam = Number(Array.isArray(params) ? params[0]?.axisValue : params?.value?.[0]);
      if (!(lam > 0)) return '';
      const cell = p => {
        if (!p) return '<span style="color:' + theme.tc + ';">—</span>';
        const t = decisionResponseMs(p.y, p.service_ms, lam);
        return t == null ? `<b style="color:${red};">over capacity</b>` : `<b>${fmt(t)} ms</b>`;
      };
      const near = useCases.filter(u => u.peak_rps > 0 && Math.abs(Math.log10(u.peak_rps / lam)) < 0.04);
      let html = axisHeader(theme, 'Arriving', `${fmt(lam)} decisions/s`);
      if (near.length) {
        html += near.map(u => `<div style="color:${theme.tc};font-size:10px;margin:-2px 0 5px;">${escapeTooltipHtml(u.name)} · ${u.batch ? 'mean' : 'peak'}</div>`).join('');
      }
      html += `<table style="border-collapse:collapse;font-size:11px;font-family:${theme.fontFamily};">` +
        `<tr style="color:${theme.tc};font-size:10px;"><td></td>` +
        `<td style="padding:0 0 3px 10px;text-align:right;">${decisionLenLabel(view.ref)} tok</td>` +
        `<td style="padding:0 0 3px 10px;text-align:right;">busy</td>` +
        `<td style="padding:0 0 3px 10px;text-align:right;">${decisionLenLabel(view.lo)} tok</td>` +
        `<td style="padding:0 0 3px 10px;text-align:right;">${decisionLenLabel(view.hi)} tok</td></tr>`;
      visible.forEach(m => {
        const ref = m.byLen.get(view.ref);
        const busy = ref ? Math.min(lam / ref.y, 1) : null;
        html += `<tr><td style="white-space:nowrap;padding:1px 0;"><span style="display:inline-block;width:10px;height:2.5px;background:${m.borderColor || m.color};margin-right:6px;vertical-align:middle;"></span>${escapeTooltipHtml(m.short_label || m.label)}</td>` +
          `<td style="padding:1px 0 1px 10px;text-align:right;white-space:nowrap;">${cell(ref)}</td>` +
          `<td style="padding:1px 0 1px 10px;text-align:right;color:${theme.tc};">${busy == null ? '—' : Math.round(busy * 100) + '%'}</td>` +
          `<td style="padding:1px 0 1px 10px;text-align:right;white-space:nowrap;">${cell(m.byLen.get(view.lo))}</td>` +
          `<td style="padding:1px 0 1px 10px;text-align:right;white-space:nowrap;">${cell(m.byLen.get(view.hi))}</td></tr>`;
      });
      return html + '</table>' +
        `<div style="color:${theme.tc};font-size:9.5px;margin-top:5px;">mean response time · busy = share of capacity at the reference length</div>`;
    }, 'axis'),
    grid: { top: 30, right: 64, bottom: 48, left: 22, containLabel: true },
    xAxis: axisCommon(theme, 'Decisions per second arriving', fmt, { type: 'log', min: xMin, max: xMax, nameGap: 30 }),
    yAxis: axisCommon(theme, 'Mean response time, ms', fmt, {
      type: 'log', min: yMin, max: 2000,
      nameLocation: 'end', nameGap: 10,
      nameTextStyle: { color: theme.tc, fontFamily: theme.fontFamily, fontSize: 11.5, align: 'left' },
    }),
    series,
  });

  renderDecisionControls(models, lengths, useCases, data);
  renderDecisionReadout(visible, useCases, fmt);
  setChartCaption(models.length
    ? 'Band: input lengths in the chosen range · line: reference length · vertical lines: each decision use case at its busiest hour (batch: mean rate). Queue model, not measured.'
    : 'No decision model is deployed.');
}

function renderDecisionControls(models, lengths, useCases, data) {
  const root = document.getElementById('chartControls');
  if (!root) return;
  const view = decisionView;
  const opts = sel => lengths.map(l => {
    const uc = useCases.find(u => u.len === l);
    return `<option value="${l}"${l === sel ? ' selected' : ''}>${decisionLenLabel(l)}${uc ? ` · ${escapeTooltipHtml(uc.name)}` : ''}</option>`;
  }).join('');
  const pick = key => `<span class="pick"><span class="lbl">${decisionLenLabel(view[key])}</span><span class="car">▾</span><select data-dc="${key}" aria-label="${key === 'ref' ? 'Reference length' : key === 'lo' ? 'Shortest length' : 'Longest length'}">${opts(view[key])}</select></span>`;
  const chips = models.map(m => `<button type="button" class="dc-chip" data-dc-model="${escapeTooltipHtml(m.label)}" aria-pressed="${!view.off.has(m.label)}" title="${escapeTooltipHtml(m.hardware || '')}"><i style="background:${m.borderColor || m.color}"></i>${escapeTooltipHtml(m.short_label || m.label)}</button>`).join('');
  root.innerHTML =
    `<span class="dc-chips">${chips}</span>` +
    `<label>Lengths ${pick('lo')} to ${pick('hi')} tok</label>` +
    `<label>Line at ${pick('ref')} tok</label>` +
    `<label>Target <input type="number" min="0" step="5" data-dc="target" value="${view.target || ''}" placeholder="none"> ms</label>`;
  root.hidden = false;
  root.onclick = e => {
    const chip = e.target.closest('[data-dc-model]');
    if (!chip) return;
    const label = chip.dataset.dcModel;
    if (view.off.has(label)) view.off.delete(label); else view.off.add(label);
    renderChart(data);
  };
  root.onchange = e => {
    const key = e.target.dataset?.dc;
    if (!key) return;
    const value = Number(e.target.value);
    if (key === 'target') {
      view.target = value > 0 ? value : null;
      try { localStorage.setItem('decisionTargetMs', view.target ? String(view.target) : ''); } catch (err) { /* storage unavailable */ }
    } else {
      view[key] = value;
      if (view.lo > view.hi) [view.lo, view.hi] = [view.hi, view.lo];
    }
    renderChart(data);
  };
}

function renderDecisionReadout(visible, useCases, fmt) {
  const root = document.getElementById('chartReadout');
  if (!root) return;
  if (!useCases.length || !visible.length) {
    root.innerHTML = visible.length ? '<p class="T">No decision use case in the workload. Add one in step 2 to see where its peak falls.</p>' : '';
    root.hidden = !root.innerHTML;
    return;
  }
  const target = decisionView.target;
  const head = '<tr><th>Model</th>' + useCases.map(u =>
    `<th colspan="3">${escapeTooltipHtml(u.name)} · ${decisionLenLabel(u.len)} tok · ${fmt(u.peak_rps)}/s ${u.batch ? 'mean' : 'peak'}</th>`).join('') + '</tr>' +
    '<tr><th></th>' + useCases.map(() => '<th>capacity</th><th>response</th><th>headroom</th>').join('') + '</tr>';
  const rows = visible.map(m => `<tr><td><i style="background:${m.borderColor || m.color}"></i>${escapeTooltipHtml(m.short_label || m.label)}</td>` +
    useCases.map(u => {
      const p = m.byLen.get(u.len);
      if (!p) return '<td class="dim" colspan="3">input too long or does not fit</td>';
      const t = decisionResponseMs(p.y, p.service_ms, u.peak_rps);
      const late = t != null && target && t > target && !u.batch;
      const response = t == null ? '<td class="bad">over capacity</td>' : `<td class="${late ? 'bad' : ''}">${fmt(t)} ms</td>`;
      const room = u.peak_rps > 0 ? p.y / u.peak_rps : null;
      return `<td>${fmt(p.y)}/s</td>${response}<td class="${room != null && room < 1.4 ? 'bad' : ''}">${room == null ? '—' : fmt(room) + '×'}</td>`;
    }).join('') + '</tr>').join('');
  root.innerHTML = `<table class="dc-read"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
  root.hidden = false;
}

/* Per-user speed (User Pareto) and per-model capacity strips (User
 * Capacity). Both read the same series. User Pareto has two view toggles,
 * remembered per browser: end labels on the chart or as a table below, and
 * an optional per-user speed target. */
const userView = { labels: 'chart', targetOn: false, target: 50 };
try {
  userView.labels = localStorage.getItem('userChartLabels') === 'table' ? 'table' : 'chart';
  userView.targetOn = localStorage.getItem('userChartTargetOn') === '1';
  userView.target = Number(localStorage.getItem('userChartTarget')) || 50;
} catch (e) { /* storage unavailable */ }
// The label column is laid out in pixels, so a resize must redraw it.
let chartRerenderOnResize = false;

function saveUserView() {
  try {
    localStorage.setItem('userChartLabels', userView.labels);
    localStorage.setItem('userChartTargetOn', userView.targetOn ? '1' : '');
    localStorage.setItem('userChartTarget', String(userView.target));
  } catch (e) { /* storage unavailable */ }
}

function userTooltipRow(p, theme, fmtUserTok) {
  const raw = p.data?.raw;
  const yStr = raw?.y != null ? `${fmtUserTok(Number(raw.y))} tok/s/user` : 'OOM';
  const detail = raw?.y != null ? ` <span style="color:${theme.tc};font-size:10px;">total ${Number(raw.total_tps).toLocaleString()} tok/s · ${Number(raw.lat).toFixed(1)}ms/tok</span>` : '';
  const oom = raw?.y == null;
  return `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
    `<span style="display:inline-block;width:10px;height:2.5px;border-radius:2px;background:${p.data?.lineColor || p.color};flex-shrink:0;"></span>` +
    `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:140px;color:${theme.text};">${escapeTooltipHtml(p.seriesName)}</span>` +
    `<span style="margin-left:4px;text-align:right;font-weight:600;color:${oom ? 'var(--red)' : theme.text};">${yStr}${detail}</span>` +
    `</div>` + hardwareDisclosureHtml(p, theme) + specDisclosureHtml(p, theme);
}

function userAxisTooltip(theme, fmtBatch, fmtUserTok) {
  return paramsArr => {
    if (!Array.isArray(paramsArr) || !paramsArr.length) return '';
    const rows = paramsArr.filter(p => p.data?.raw);
    if (!rows.length) return '';
    const xVal = Number(rows[0].axisValue ?? rows[0].data?.value?.[0]);
    let html = axisHeader(theme, 'Concurrent users', fmtBatch(xVal));
    [...rows]
      .sort((a, b) => (b.data?.raw?.y ?? -Infinity) - (a.data?.raw?.y ?? -Infinity))
      .forEach(p => { html += userTooltipRow(p, theme, fmtUserTok); });
    return html;
  };
}

function userPoints(ds) { return (ds.data || []).filter(p => p.y != null); }

// Most users the line reaches and whether it stopped because the KV cache filled.
function userEnd(ds, xMax) {
  const pts = userPoints(ds);
  if (!pts.length) return null;
  const last = pts[pts.length - 1];
  return { x: Number(last.x), y: Number(last.y), raw: last, kvLimited: xMax && Number(last.x) < Number(xMax) };
}

// Line with a bar at its end when the KV cache is full there.
function userSeries(ds, xMax, opts = {}) {
  const s = makeSeries(ds, true);
  const end = userEnd(ds, xMax);
  if (end?.kvLimited) {
    s.data = s.data.map(d => d.raw === end.raw
      ? { ...d, symbol: 'rect', symbolSize: [3, 14], itemStyle: { color: ds.borderColor, borderWidth: 0 } }
      : d);
  }
  if (opts.solid) s.lineStyle = { ...s.lineStyle, type: 'solid' };
  if (opts.grid != null) { s.xAxisIndex = opts.grid; s.yAxisIndex = opts.grid; }
  return s;
}

// Log-2 y range cropped to the data, so no band of the plot sits empty.
function userYRange(datasets) {
  const ys = datasets.flatMap(ds => userPoints(ds).map(p => Number(p.y))).filter(v => v > 0);
  if (!ys.length) return {};
  return { min: 2 ** Math.floor(Math.log2(Math.min(...ys))), max: 2 ** Math.ceil(Math.log2(Math.max(...ys))) };
}

function userEndText(ds, end, fmtBatch, fmtUserTok) {
  return end.kvLimited
    ? `full load: ${fmtBatch(end.x)} users × ${fmtUserTok(end.y)} tok/s`
    : `${fmtBatch(end.x)} users: ${fmtUserTok(end.y)} tok/s`;
}

// End labels stacked in a column right of the grid, joined to their line
// ends by dotted leaders, so they never overlap each other or the lines.
function userLabelColumn(theme, datasets, xMax, gridIndex, f) {
  const rect = chart.getModel().getComponent('grid', gridIndex).coordinateSystem.getRect();
  const colX = rect.x + rect.width + 22;
  const gap = 31;
  const items = datasets.map(ds => {
    const end = userEnd(ds, xMax);
    if (!end) return null;
    const [px, py] = chart.convertToPixel({ gridIndex }, [end.x, end.y]);
    return { ds, end, px, py, ly: py };
  }).filter(Boolean).sort((a, b) => a.py - b.py);
  if (!items.length) return [];
  for (let i = 1; i < items.length; i++) items[i].ly = Math.max(items[i].ly, items[i - 1].ly + gap);
  const over = items[items.length - 1].ly - (rect.y + rect.height);
  if (over > 0) items.forEach(it => { it.ly -= over; });
  for (let i = items.length - 2; i >= 0; i--) items[i].ly = Math.min(items[i].ly, items[i + 1].ly - gap);
  return items.flatMap(({ ds, end, px, py, ly }) => [
    {
      type: 'polyline', silent: true, z: 1,
      shape: { points: [[px + 5, py], [colX - 14, py], [colX - 4, ly]] },
      style: { stroke: ds.borderColor, lineWidth: 1, lineDash: [2, 3], fill: 'none', opacity: 0.6 },
    },
    {
      type: 'text', silent: true, x: colX, y: ly - 14,
      style: {
        text: `{n|${ds.short_label || ds.label}}\n{s|${userEndText(ds, end, f.fmtBatch, f.fmtUserTok)}}`,
        rich: {
          n: { fill: ds.borderColor, fontFamily: theme.fontFamily, fontSize: 11.5, fontWeight: 600, lineHeight: 15 },
          s: { fill: theme.tc, fontFamily: theme.fontFamily, fontSize: 10.5, lineHeight: 13 },
        },
      },
    },
  ]);
}

// Users a line holds before per-user speed drops below `target`
// (log-log interpolation between sampled points).
function userAtTarget(ds, target, xMax) {
  const pts = userPoints(ds).map(p => [Number(p.x), Number(p.y)]);
  if (!pts.length || pts[0][1] < target) return { users: 0, limit: 'too slow with one user' };
  for (let i = 1; i < pts.length; i++) {
    if (pts[i][1] < target) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const t = (Math.log(y0) - Math.log(target)) / (Math.log(y0) - Math.log(y1));
      return { users: Math.exp(Math.log(x0) + t * (Math.log(x1) - Math.log(x0))), y: target, limit: 'speed' };
    }
  }
  const end = userEnd(ds, xMax);
  return { users: end.x, y: end.y, limit: end.kvLimited ? 'memory full' : 'end of axis' };
}

function renderUserControls(data) {
  const root = document.getElementById('chartControls');
  if (!root) return;
  const seg = (value, label) =>
    `<button type="button" class="dc-chip uv" data-uv-labels="${value}" aria-pressed="${userView.labels === value}">${label}</button>`;
  root.innerHTML =
    `<label>Labels <span class="dc-chips">${seg('chart', 'On chart')}${seg('table', 'Table')}</span></label>` +
    `<label><button type="button" class="dc-chip uv" data-uv-target-on aria-pressed="${userView.targetOn}">Speed target</button>` +
    (userView.targetOn ? ` each user gets at least <input type="number" min="1" step="5" data-uv-target value="${userView.target}" aria-label="Target tok/s per user"> tok/s` : '') +
    `</label>`;
  root.hidden = false;
  root.onclick = e => {
    const labels = e.target.closest('[data-uv-labels]');
    if (labels) userView.labels = labels.dataset.uvLabels;
    else if (e.target.closest('[data-uv-target-on]')) userView.targetOn = !userView.targetOn;
    else return;
    saveUserView();
    renderChart(data);
  };
  root.onchange = e => {
    if (!e.target.matches('[data-uv-target]')) return;
    const v = Number(e.target.value);
    if (!(v > 0)) return;
    userView.target = v;
    saveUserView();
    renderChart(data);
  };
}

function renderUserReadout(html) {
  const root = document.getElementById('chartReadout');
  if (!root) return;
  root.innerHTML = html;
  root.hidden = !html;
}

function bindUserReadoutHover(datasets) {
  const root = document.getElementById('chartReadout');
  root?.querySelectorAll('tr[data-ux]').forEach(tr => {
    const name = datasets[Number(tr.dataset.ux)]?.label;
    const on = () => chart?.dispatchAction({ type: 'highlight', seriesName: name });
    const off = () => chart?.dispatchAction({ type: 'downplay', seriesName: name });
    tr.addEventListener('mouseenter', on);
    tr.addEventListener('mouseleave', off);
  });
}

function renderUserChart(data, el, theme, datasets, f) {
  if (data.mode === 'usercapacity') {
    renderUserCapacityChart(data, theme, datasets, f);
    return;
  }
  const { fmtBatch, fmtUserTok, narrowChart } = f;
  const xMax = data.x_max;
  const { labels, targetOn, target } = userView;
  const sideLabels = labels === 'chart' && !narrowChart;
  renderUserControls(data);
  renderUserReadout('');
  chartRerenderOnResize = sideLabels;

  const axisFmt = v => Math.round(Number(v)).toLocaleString();
  const yRange = userYRange(datasets);
  const yMin = Math.min(yRange.min ?? Infinity, targetOn ? 2 ** Math.floor(Math.log2(target)) : Infinity);
  const yMax = Math.max(yRange.max ?? 0, targetOn ? 2 ** Math.ceil(Math.log2(target)) : 0);
  const hits = targetOn ? datasets.map((ds, i) => ({ ds, i, ...userAtTarget(ds, target, xMax) })) : [];

  const targetSeries = targetOn ? [
    {
      type: 'line', data: [], silent: true, tooltip: { show: false },
      markArea: {
        silent: true,
        itemStyle: { color: theme.dk ? 'rgba(255,255,255,.04)' : 'rgba(0,0,0,.04)' },
        label: { show: true, position: 'insideBottomLeft', color: theme.tc, fontFamily: theme.fontFamily, fontSize: 10.5, formatter: 'slower than target' },
        data: [[{ yAxis: yMin }, { yAxis: target }]],
      },
      markLine: {
        silent: true, symbol: 'none',
        lineStyle: { color: theme.tc, type: 'solid', width: 1 },
        label: { position: 'insideEndTop', color: theme.text, fontFamily: theme.fontFamily, fontSize: 11, fontWeight: 600, formatter: `${fmtUserTok(target)} tok/s` },
        data: [{ yAxis: target }],
      },
    },
    {
      type: 'scatter', z: 5, symbolSize: 9, silent: true, tooltip: { show: false },
      labelLayout: { hideOverlap: true },
      data: hits.filter(h => h.users >= 1).map(h => ({
        value: [h.users, h.y],
        itemStyle: { color: h.ds.borderColor, borderColor: theme.bg, borderWidth: 2 },
        label: { show: true, position: 'bottom', distance: 4, color: h.ds.borderColor, fontFamily: theme.fontFamily, fontSize: 10.5, fontWeight: 600, formatter: String(fmtBatch(h.users)) },
      })),
    },
  ] : [];

  chart.setOption({
    ...baseOption(theme, userAxisTooltip(theme, fmtBatch, fmtUserTok), 'axis'),
    // Labels: a column right of the plot, a table below it, or (narrow
    // screens, where the column would squeeze the plot) a legend.
    grid: { top: narrowChart ? 58 : 34, right: sideLabels ? 236 : 18, bottom: 48, left: 22, containLabel: true },
    legend: narrowChart && labels === 'chart' ? {
      show: true, type: 'scroll', top: 0, left: 8, right: 8, icon: 'roundRect', itemWidth: 14, itemHeight: 5,
      textStyle: { color: theme.tc, fontFamily: theme.fontFamily, fontSize: 11 },
      formatter: name => (datasets.find(d => d.label === name)?.short_label) || name,
    } : { show: false },
    xAxis: axisCommon(theme, 'Concurrent users', v => fmtBatch(Number(v)), {
      type: 'log', min: 1, max: xMax, nameGap: 30,
    }),
    yAxis: axisCommon(theme, 'Decode tok/s each user sees', axisFmt, {
      type: 'log', logBase: 2,
      min: Number.isFinite(yMin) ? yMin : undefined,
      max: yMax > 0 ? yMax : undefined,
      nameLocation: 'end',
      nameGap: 12,
      nameTextStyle: { color: theme.tc, fontFamily: theme.fontFamily, fontSize: 11.5, align: 'left' },
    }),
    series: [...datasets.map(ds => userSeries(ds, xMax)), ...targetSeries],
  });
  if (sideLabels) chart.setOption({ graphic: userLabelColumn(theme, datasets, xMax, 0, f) });
  setChartCaption('Bar at a line end: the KV cache is full, so that is the most users the deployment holds at once.');

  if (labels === 'table') {
    const byIndex = new Map(hits.map(h => [h.i, h]));
    const order = datasets.map((ds, i) => ({ ds, i, end: userEnd(ds, xMax) })).filter(r => r.end)
      .sort((a, b) => b.end.x - a.end.x);
    const targetHead = targetOn ? `<th>Users at ≥ ${fmtUserTok(target)} tok/s</th><th>Limited by</th>` : '';
    const rows = order.map(({ ds, i, end }) => {
      const alone = userPoints(ds)[0];
      const h = byIndex.get(i);
      const targetCells = h ? `<td>${h.users >= 1 ? fmtBatch(h.users) : '0'}</td><td>${userLimitText(h, target, fmtUserTok)}</td>` : '';
      return `<tr data-ux="${i}"><td><i style="background:${ds.borderColor}"></i>${escapeTooltipHtml(ds.short_label || ds.label)}</td>` +
        `<td>${escapeTooltipHtml(ds.hardware || '')}</td>` +
        `<td>${fmtUserTok(Number(alone.y))}</td>` +
        `<td>${fmtBatch(end.x)}</td><td>${fmtUserTok(end.y)}</td>` +
        `<td>${Number(end.raw.total_tps).toLocaleString()}</td>` +
        `<td>${end.kvLimited ? 'KV cache full' : 'end of axis'}</td>${targetCells}</tr>`;
    }).join('');
    renderUserReadout(`<table class="dc-read"><thead><tr><th>Model</th><th>Hardware</th><th>Alone tok/s</th><th>Most users</th><th>tok/s each there</th><th>Total tok/s there</th><th>Stops because</th>${targetHead}</tr></thead><tbody>${rows}</tbody></table>`);
    bindUserReadoutHover(datasets);
  } else if (targetOn) {
    const rows = [...hits].sort((a, b) => b.users - a.users).map(h =>
      `<tr data-ux="${h.i}"><td><i style="background:${h.ds.borderColor}"></i>${escapeTooltipHtml(h.ds.short_label || h.ds.label)}</td>` +
      `<td>${h.users >= 1 ? fmtBatch(h.users) : '0'}</td>` +
      `<td>${h.users >= 1 ? fmtUserTok(h.y) : '—'}</td>` +
      `<td>${h.users >= 1 ? Math.round(h.users * h.y).toLocaleString() : '—'}</td>` +
      `<td>${userLimitText(h, target, fmtUserTok)}</td></tr>`).join('');
    renderUserReadout(`<table class="dc-read"><thead><tr><th>Model</th><th>Users at ≥ ${fmtUserTok(target)} tok/s</th><th>tok/s each</th><th>Total tok/s</th><th>Limited by</th></tr></thead><tbody>${rows}</tbody></table>`);
    bindUserReadoutHover(datasets);
  }
}

function userLimitText(h, target, fmtUserTok) {
  return h.limit === 'speed' ? `drops below ${fmtUserTok(target)} tok/s` : h.limit;
}

// One strip per model from 1 user to the most it holds; each segment's
// opacity shows the per-user speed over that stretch of load.
function renderUserCapacityChart(data, theme, datasets, f) {
  const { fmtBatch, fmtUserTok, narrowChart } = f;
  const xMax = data.x_max;
  const yRange = userYRange(datasets);
  const order = datasets.map(ds => ({ ds, end: userEnd(ds, xMax) })).filter(r => r.end)
    .sort((a, b) => b.end.x - a.end.x);
  const cats = order.map(r => r.ds.short_label || r.ds.label);
  const lo = Math.log(yRange.min || 1), hi = Math.log(yRange.max || 2);
  const alpha = y => 0.14 + 0.86 * Math.min(1, Math.max(0, (Math.log(y) - lo) / (hi - lo || 1)));
  const segData = (ds, c) => {
    const pts = userPoints(ds);
    return pts.map((p, k) => ({
      value: [k ? Number(pts[k - 1].x) : Number(p.x), Number(p.x), c, Number(p.y)],
      raw: p, hardware: ds.hardware || '', specDisclosure: ds.spec_disclosure || '', lineColor: ds.borderColor,
      itemStyle: { color: echarts.color.modifyAlpha(ds.borderColor, alpha(Number(p.y))) },
    })).filter(d => d.value[1] > d.value[0]);
  };
  chart.setOption({
    ...baseOption(theme, p => p.data?.raw
      ? axisHeader(theme, 'Concurrent users', fmtBatch(Number(p.data.raw.x))) + userTooltipRow(p, theme, fmtUserTok)
      : '', 'item'),
    grid: { top: 18, right: narrowChart ? 18 : 190, bottom: 48, left: 22, containLabel: true },
    xAxis: axisCommon(theme, 'Concurrent users', v => fmtBatch(Number(v)), {
      type: 'log', min: 1, max: xMax, nameGap: 30,
    }),
    yAxis: {
      type: 'category', data: cats, inverse: true,
      axisLine: { show: false }, axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { color: theme.text, fontFamily: theme.fontFamily, fontSize: 11 },
    },
    series: [
      ...order.map(({ ds }, c) => ({
        name: ds.label, type: 'custom',
        renderItem: (params, api) => {
          const a = api.coord([api.value(0), api.value(2)]);
          const b = api.coord([api.value(1), api.value(2)]);
          const h = Math.min(22, api.size([0, 1])[1] * 0.6);
          return { type: 'rect', shape: { x: a[0], y: a[1] - h / 2, width: Math.max(1, b[0] - a[0] + 0.5), height: h }, style: { fill: api.visual('color') } };
        },
        encode: { x: [0, 1], y: 2 },
        data: segData(ds, c),
      })),
      {
        type: 'scatter', silent: true, tooltip: { show: false }, z: 4,
        data: order.flatMap(({ ds, end }, c) => {
          const first = userPoints(ds)[0];
          return [
            {
              value: [end.x, c], symbol: end.kvLimited ? 'rect' : 'none', symbolSize: [3, 26],
              itemStyle: { color: theme.text },
              label: {
                show: !narrowChart, position: 'right', distance: 8,
                formatter: `{b|${fmtBatch(end.x)} users} × ${fmtUserTok(end.y)} tok/s`,
                color: theme.tc, fontFamily: theme.fontFamily, fontSize: 10.5,
                rich: { b: { color: theme.text, fontWeight: 600, fontFamily: theme.fontFamily, fontSize: 10.5 } },
              },
            },
            {
              value: [Number(first.x), c], symbol: 'circle', symbolSize: 0,
              label: { show: true, position: 'right', distance: 4, formatter: `${fmtUserTok(Number(first.y))} alone`, color: theme.text, fontFamily: theme.fontFamily, fontSize: 10, fontWeight: 600 },
            },
          ];
        }),
      },
    ],
  });
  setChartCaption('Each strip runs from 1 user to the most the KV cache holds (bar). Darker means faster per user. Hover a stretch for its numbers.');
}

function renderChart(data) {
  lastChartData = data;
  setChartCaption('');
  showDecisionExtras(false);
  updateChartAccessibleSummary(data);
  disposeChart();
  chartRerenderOnResize = false;
  const el = document.getElementById('mc');
  if (!el || !window.echarts) return;

  const theme = chartTheme();
  const datasets = normalizeDatasets(data.datasets, theme.dk);
  chart = echarts.init(el);

  const fmtBatch = v => v >= 1e3 ? (v/1e3).toFixed(v >= 1e4 ? 0 : 1).replace('.0', '')+'k' : Math.round(v);
  const fmtUserTok = v => v >= 100 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(1) : v.toFixed(2);
  const fmtRate = v => v >= 100 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(1) : v.toFixed(2);

  const narrowChart = el.clientWidth < 600;

  // Keep unmeasured quality off the quality axes. Capacity remains available
  // as a separate text readout, or as bars when no rated series are selected.
  const capacityOnly = data.capacity_only || [];
  const capacityRoot = document.getElementById('chartCapacityOnly');
  if (capacityRoot) {
    capacityRoot.replaceChildren();
    capacityRoot.hidden = !capacityOnly.length || !datasets.length;
    capacityOnly.forEach(row => {
      const item = document.createElement('p');
      item.textContent = `${row.label}: ${Number(row.capacity).toLocaleString()} ${row.unit}. Quality unavailable; capacity estimate only.`;
      capacityRoot.appendChild(item);
    });
  }
  if (!datasets.length && capacityOnly.length) {
    chart.setOption({
      ...baseOption(theme, params => {
        const row = capacityOnly[params.dataIndex];
        return `${escapeTooltipHtml(row.label)}<br>${Number(row.capacity).toLocaleString()} ${escapeTooltipHtml(row.unit)}<br>${escapeTooltipHtml(row.hardware)}<br>${escapeTooltipHtml(row.note)}`;
      }, 'item'),
      grid: { top: 30, right: 36, bottom: 48, left: 22, containLabel: true },
      xAxis: axisCommon(theme, capacityOnly[0].unit, fmtRate, { type: 'value', min: 0 }),
      yAxis: axisCommon(theme, '', v => v, {
        type: 'category', data: capacityOnly.map(row => row.label),
      }),
      series: [{ type: 'bar', barMaxWidth: 28,
        data: capacityOnly.map(row => ({ value: row.capacity, itemStyle: { color: row.color } })),
      }],
    });
    setChartCaption('Capacity estimates only. No compatible published quality score is available for these checkpoints.');
    return;
  }

  if (data.mode === 'userpareto' || data.mode === 'usercapacity') {
    renderUserChart(data, el, theme, datasets, { fmtBatch, fmtUserTok, narrowChart });
    return;
  }

  if (data.mode === 'decision') {
    renderDecisionChart(data, el, theme);
    return;
  }

  if (data.mode === 'processingpareto') {
    chart.setOption({
      ...baseOption(theme, paramsArr => {
        if (!Array.isArray(paramsArr) || !paramsArr.length) return '';
        const xVal = Number(paramsArr[0].axisValue ?? paramsArr[0].data?.value?.[0]);
        let html = axisHeader(theme, 'Batch size', fmtBatch(xVal));
        [...paramsArr]
          .sort((a, b) => (b.data?.raw?.y ?? -Infinity) - (a.data?.raw?.y ?? -Infinity))
          .forEach(p => {
            const raw = p.data?.raw;
            const yStr = raw?.y != null ? `${fmtRate(Number(raw.y))} req/s` : 'OOM';
            const detail = raw?.y != null ? ` <span style="color:${theme.tc};font-size:10px;">avg in ${Number(raw.in_len).toLocaleString()} / out ${Number(raw.out_len).toLocaleString()} · ${Number(raw.tps).toLocaleString()} tok/s</span>` : '';
            const oom = raw?.y == null;
            html += `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
              `<span style="display:inline-block;width:10px;height:2.5px;border-radius:2px;background:${p.color};flex-shrink:0;"></span>` +
              `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:140px;color:${theme.text};">${escapeTooltipHtml(p.seriesName)}</span>` +
              `<span style="margin-left:4px;text-align:right;font-weight:600;color:${oom ? 'var(--red)' : theme.text};">${yStr}${detail}</span>` +
              `</div>`;
            html += hardwareDisclosureHtml(p, theme);
            html += specDisclosureHtml(p, theme);
          });
        return html;
      }, 'axis'),
      grid: { top: 58, right: 18, bottom: 48, left: 22, containLabel: true },
      legend: {
        show: true,
        type: 'scroll',
        top: 0,
        left: 8,
        right: 8,
        height: 36,
        icon: 'roundRect',
        itemWidth: 16,
        itemHeight: 5,
        textStyle: { color: theme.tc, fontFamily: theme.fontFamily, fontSize: 10 },
        pageIconColor: theme.tc,
        pageTextStyle: { color: theme.tc, fontFamily: theme.fontFamily, fontSize: 10 },
      },
      xAxis: axisCommon(theme, 'Batch size', v => fmtBatch(Number(v)), {
        type: 'log',
        min: 1,
        max: data.x_max,
        nameGap: 30,
      }),
      yAxis: axisCommon(theme, 'Requests/sec', fmtRate, {
        type: 'log',
        nameGap: 52,
      }),
      series: datasets.map(ds => makeSeries(ds, true)),
    });
    return;
  }

  if (data.mode === 'realtime') {
    chart.setOption({
      ...baseOption(theme, paramsArr => {
        if (!Array.isArray(paramsArr) || !paramsArr.length) return '';
        const xVal = Number(paramsArr[0].axisValue ?? paramsArr[0].data?.value?.[0]);
        let html = axisHeader(theme, 'Concurrent streams', fmtBatch(xVal));
        [...paramsArr]
          .sort((a, b) => (b.data?.raw?.y ?? -Infinity) - (a.data?.raw?.y ?? -Infinity))
          .forEach(p => {
            const raw = p.data?.raw;
            const ok = raw?.y != null && Number(raw.y) >= 1;
            const yStr = raw?.y != null ? `${Number(raw.y).toFixed(2)}× real-time` : 'OOM';
            const detail = raw?.y != null
              ? ` <span style="color:${theme.tc};font-size:10px;">max ${fmtBatch(Number(raw.max_users || 0))} users · ${Number(raw.per_user_tps || 0).toFixed(1)} tok/s/user · ${Number(raw.step_ms || 0).toFixed(1)}ms step</span>`
              : '';
            html += `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
              `<span style="display:inline-block;width:10px;height:2.5px;border-radius:2px;background:${p.color};flex-shrink:0;"></span>` +
              `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:140px;color:${theme.text};">${escapeTooltipHtml(p.seriesName)}</span>` +
              `<span style="margin-left:4px;text-align:right;font-weight:600;color:${raw?.y == null ? 'var(--red)' : (ok ? theme.text : 'var(--amber)')};">${yStr}${detail}</span>` +
              `</div>`;
            html += hardwareDisclosureHtml(p, theme);
          });
        return html;
      }, 'axis'),
      xAxis: axisCommon(theme, 'Concurrent realtime users', v => fmtBatch(Number(v)), {
        type: 'log',
        min: 1,
        max: data.x_max,
        nameGap: 30,
      }),
      yAxis: axisCommon(theme, 'Real-time factor', v => v >= 10 ? v.toFixed(0) + '×' : v.toFixed(2) + '×', {
        type: 'log',
        min: 0.1,
        nameGap: 54,
      }),
      series: datasets.map((ds, idx) => ({
        ...makeSeries(ds, true),
        markLine: idx === 0 ? {
          silent: true,
          symbol: 'none',
          label: { color: theme.tc, formatter: '1× realtime' },
          lineStyle: { color: theme.tc, width: 1, type: 'dashed', opacity: 0.55 },
          data: [{ yAxis: 1 }],
        } : undefined,
      })),
    });
    return;
  }

  if (data.mode === 'asrquality') {
    const fmtWer = v => `${Number(v).toFixed(1)}%`;
    chart.setOption({
      ...baseOption(theme, params => {
        const raw = params.data?.raw;
        if (!raw) return '';
        const isStreaming = raw.asr_mode !== 'non-streaming';
        const markerStyle = isStreaming
          ? `border-radius:50%;background:${params.color};`
          : `transform:rotate(45deg);background:${params.color};`;
        let html = axisHeader(theme, 'WER', `${fmtWer(raw.wer)} · ${raw.language || 'Benchmark'}`);
        html += `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
          `<span style="display:inline-block;width:10px;height:10px;${markerStyle}flex-shrink:0;"></span>` +
          `<span style="flex:1;min-width:0;color:${theme.text};">${escapeTooltipHtml(params.seriesName)}</span>` +
          `<span style="margin-left:8px;font-weight:600;color:${theme.text};">${fmtBatch(Number(raw.max_users))} streams</span>` +
          `</div>`;
        html += hardwareDisclosureHtml(params, theme);
        html += `<div style="margin-top:5px;font-size:10px;color:${theme.tc};">${isStreaming ? 'Streaming ASR' : 'Non-streaming ASR'}</div>`;
        if (raw.source) html += `<div style="margin-top:5px;font-size:10px;color:${theme.tc};">${raw.source}</div>`;
        if (raw.placeholder) html += `<div style="margin-top:5px;font-size:10px;color:var(--amber);">Placeholder WER - not yet sourced</div>`;
        return html;
      }, 'item'),
      legend: {
        show: true, type: 'scroll', top: 0, left: 8, right: 8, height: 24,
        icon: 'roundRect', itemWidth: 16, itemHeight: 5,
        textStyle: { color: theme.tc, fontFamily: theme.fontFamily, fontSize: 10 },
        pageIconColor: theme.tc,
      },
      grid: { top: 34, right: 22, bottom: 48, left: 22, containLabel: true },
      xAxis: axisCommon(theme, 'WER % (lower is better)', fmtWer, {
        type: 'value',
        min: 0,
        nameGap: 30,
      }),
      yAxis: axisCommon(theme, 'Max concurrent streams @ ≥1× realtime', v => fmtBatch(Number(v)), {
        type: 'log',
        min: 1,
        nameGap: 52,
      }),
      series: (data.datasets || []).map((ds, idx) => {
        const placeholder = Boolean(ds._placeholder);
        const isStreaming = ds._asrStreaming !== false;
        return {
          id: ds._seriesId || `asrquality-${idx}`,
          name: ds.label,
          type: 'line',
          showSymbol: true,
          symbol: isStreaming ? 'circle' : 'diamond',
          symbolSize: isStreaming ? 11 : 12,
          data: (ds.data || []).map(p => ({
            value: [Number(p.x), Number(p.y)],
            raw: p,
            hardware: ds.hardware || '',
          })),
          lineStyle: { color: ds.borderColor, width: Number(ds.borderWidth || 2), type: dashStyle(ds.borderDash), opacity: 0.55 },
          itemStyle: placeholder
            ? { color: 'transparent', borderColor: ds.borderColor, borderWidth: 2 }
            : { color: ds.borderColor, borderColor: 'rgba(255,255,255,0.75)', borderWidth: 1.5 },
          emphasis: { focus: 'series', symbolSize: 14 },
          blur: { lineStyle: { opacity: 0.08 }, itemStyle: { opacity: 0.15 } },
        };
      }),
    });
    return;
  }

  if (data.mode === 'embedquality') {
    const fmtQuality = v => Number(v).toFixed(2);
    const qualityMin = Number.isFinite(Number(data.y_min)) ? Number(data.y_min) : 0;
    const qualityMax = Number.isFinite(Number(data.y_max)) ? Number(data.y_max) : 1;
    const fmtBytes = v => {
      const n = Number(v);
      if (!isFinite(n) || n <= 0) return '0 B';
      if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
      if (n >= 1e6) return (n / 1e6).toFixed(2) + ' MB';
      if (n >= 1e3) return (n / 1e3).toFixed(1) + ' kB';
      return Math.round(n) + ' B';
    };
    // Dot size encodes bytes-per-doc. Sqrt scaling so area scales with cost
    // without late-interaction dots dwarfing single-vector ones off-screen.
    const dotSize = bytes => {
      const n = Math.max(Number(bytes) || 0, 0);
      return Math.max(8, Math.min(32, 8 + Math.sqrt(n / 100)));
    };
    chart.setOption({
      ...baseOption(theme, params => {
        const raw = params.data?.raw;
        if (!raw) return '';
        let html = axisHeader(theme, 'Peak throughput', `${fmtRate(Number(raw.docs_per_second))} docs/s`);
        html += `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
          `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${params.color};flex-shrink:0;"></span>` +
          `<span style="flex:1;min-width:0;color:${theme.text};">${escapeTooltipHtml(params.seriesName)}</span>` +
          `<span style="margin-left:8px;font-weight:600;color:${theme.text};">${raw.quality_metric || 'quality'} ${fmtQuality(raw.quality)}</span>` +
          `</div>`;
        html += hardwareDisclosureHtml(params, theme);
        const vpi = Number(raw.vectors_per_input || 1);
        const multiVec = vpi > 1 ? ` · ${vpi.toLocaleString()} vec/doc → ${fmtRate(Number(raw.vectors_per_second || 0))} vec/s` : '';
        html += `<div style="margin-top:4px;font-size:10px;color:${theme.tc};">` +
          `${fmtBytes(raw.bytes_per_doc)} / doc · ${Number(raw.output_mb_s || 0).toFixed(1)} MB/s output${multiVec}` +
          `</div>`;
        html += `<div style="font-size:10px;color:${theme.tc};">` +
          `peak batch ${fmtBatch(Number(raw.peak_batch))} · seq ${Number(raw.seq_len || 0).toLocaleString()} tok · ${raw.mode || ''}` +
          `</div>`;
        if (raw.source) html += `<div style="margin-top:5px;font-size:10px;color:${theme.tc};">${raw.source}</div>`;
        if (raw.uses_decontaminated_beir) {
          if (Number.isFinite(Number(raw.published_quality))) {
            html += `<div style="margin-top:5px;font-size:10px;color:${theme.tc};">Published retrieval comparison ${fmtQuality(raw.published_quality)}</div>`;
          }
        } else {
          html += `<div style="margin-top:5px;font-size:10px;color:${theme.tc};">Decontaminated BEIR not sourced</div>`;
        }
        if (raw.placeholder) html += `<div style="margin-top:5px;font-size:10px;color:var(--amber);">Placeholder quality - not yet sourced</div>`;
        return html;
      }, 'item'),
      legend: {
        show: true, type: 'scroll', top: 0, left: 8, right: 8, height: 24,
        icon: 'circle', itemWidth: 10, itemHeight: 10,
        textStyle: { color: theme.tc, fontFamily: theme.fontFamily, fontSize: 10 },
        pageIconColor: theme.tc,
      },
      grid: { top: 34, right: 22, bottom: 48, left: 22, containLabel: true },
      xAxis: axisCommon(theme, 'Peak docs/sec (higher is better)', fmtRate, {
        type: 'log',
        nameGap: 30,
      }),
      yAxis: axisCommon(theme, 'Retrieval quality (higher is better)', fmtQuality, {
        type: 'value',
        min: qualityMin,
        max: qualityMax,
        nameGap: 38,
      }),
      series: (() => {
        const modelSeries = (data.datasets || []).map(ds => {
          const placeholder = Boolean(ds._placeholder);
          return {
            name: ds.label,
            type: 'scatter',
            symbol: 'circle',
            data: (ds.data || []).map(p => ({
              value: [Number(p.x), Number(p.y)],
              raw: p,
              hardware: ds.hardware || '',
              symbolSize: dotSize(p.bytes_per_doc),
            })),
            itemStyle: placeholder
              ? { color: 'transparent', borderColor: ds.borderColor, borderWidth: 2 }
              : { color: ds.borderColor, borderColor: 'rgba(255,255,255,0.75)', borderWidth: 1.5 },
            emphasis: { focus: 'series', scale: 1.2 },
            blur: { itemStyle: { opacity: 0.15 } },
          };
        });
        // Pareto frontier: a point (x, y) dominates another if it has both higher
        // docs/s and higher quality. Sort by x descending, keep points whose y
        // strictly exceeds the running max — those are non-dominated.
        const allPoints = [];
        (data.datasets || []).forEach(ds => (ds.data || []).forEach(p => {
          const x = Number(p.x), y = Number(p.y);
          if (Number.isFinite(x) && Number.isFinite(y)) allPoints.push([x, y]);
        }));
        allPoints.sort((a, b) => b[0] - a[0]);
        const frontier = [];
        let bestY = -Infinity;
        for (const pt of allPoints) {
          if (pt[1] > bestY) { frontier.push(pt); bestY = pt[1]; }
        }
        frontier.reverse(); // draw left-to-right
        if (frontier.length >= 2) {
          modelSeries.push({
            name: 'Pareto frontier',
            type: 'line',
            data: frontier.map(([x, y]) => ({ value: [x, y] })),
            showSymbol: false,
            lineStyle: { color: theme.tc, width: 1.25, type: 'dashed', opacity: 0.7 },
            tooltip: { show: false },
            silent: true,
            z: 1,
          });
        }
        return modelSeries;
      })(),
    });
    return;
  }
}

/* Picker toggle (only JS needed for dropdowns) */
let openPicker = null;
function closePicker() {
  if (!openPicker) return;
  const trigger = openPicker.parentElement?.querySelector('[data-picker]');
  if (trigger) trigger.setAttribute('aria-expanded', 'false');
  openPicker.remove();
  openPicker = null;
}
document.addEventListener('click', e => {
  if (!e.target.closest('.picker') && !e.target.closest('[data-picker]')) closePicker();
});

function positionPicker(picker) {
  if (!picker?.isConnected) return;
  picker.style.transform = '';
  const bounds = picker.getBoundingClientRect();
  const shift = Math.max(12 - bounds.left, Math.min(0, window.innerWidth - 12 - bounds.right));
  picker.style.transform = `translateX(${shift}px)`;
}

function togglePicker(btn, url) {
  const wrap = btn.closest('.picker-wrap');
  const shouldCloseOnly = openPicker && openPicker.parentElement === wrap;
  closePicker();
  if (shouldCloseOnly) return;
  const pk = document.createElement('div');
  pk.className = 'picker';
  pk.innerHTML = '<div style="padding:10px;text-align:center;" class="T">Loading...</div>';
  wrap.appendChild(pk);
  positionPicker(pk);
  openPicker = pk;
  btn.setAttribute('aria-expanded', 'true');
  pk.setAttribute('role', 'dialog');
  pk.setAttribute('aria-label', `${btn.textContent.trim()} options`);
  fetch(withTabUrl(url), { headers: requestHeaders() }).then(r => r.text()).then(html => {
    replaceHtml(pk, html);
    initGpuPicker(pk);
    initModelPicker(pk);
    positionPicker(pk);
  });
}

function postPartial(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: requestHeaders({
      'Content-Type': 'application/x-www-form-urlencoded',
      'HX-Request': 'true',
    }),
    body: body || '',
  }).then(async r => {
    const text = await r.text();
    if (r.ok) return text;
    let message = text || `Request failed (${r.status})`;
    try {
      const data = JSON.parse(text);
      if (data && data.error) message = data.error;
    } catch {}
    throw new Error(message);
  });
}

function normalizedPickerGpuCount(trigger) {
  let count = Number(window.selectedGpuCount ?? 8) || 8;
  const minCount = Number(trigger?.dataset?.minCount ?? 1) || 1;
  const multiple = Number(trigger?.dataset?.countMultiple ?? 1) || 1;
  count = Math.max(count, minCount);
  if (multiple > 1) count = Math.ceil(count / multiple) * multiple;
  return count;
}

function addGpuFromPicker(panel, gpuType, trigger) {
  const count = normalizedPickerGpuCount(trigger);
  const body = `panel=${encodeURIComponent(panel)}&gpu_type=${encodeURIComponent(gpuType)}&count=${encodeURIComponent(count)}`;
  postPartial('/gpu/add', body)
    .then(html => {
      replaceHtml(document.getElementById('main-content'), html);
      closePicker();
      document.body.dispatchEvent(new Event('refreshChart'));
    })
    .catch(err => showToast(err.message));
}

function exportUseCases(panel) {
  const url = withTabUrl(`/project/export?panel=${encodeURIComponent(panel)}`);
  fetch(url, { headers: requestHeaders() })
    .then(async r => {
      if (r.ok) return r.blob();
      let message = await r.text();
      try {
        const data = JSON.parse(message);
        if (data && data.error) message = data.error;
      } catch {}
      throw new Error(message || `Export failed (${r.status})`);
    })
    .then(blob => {
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = `use-cases-${panel.toLowerCase()}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objectUrl);
    })
    .catch(err => showToast(err.message));
}

function importUseCases(panel, input) {
  const file = input && input.files && input.files[0];
  if (!file) return;
  file.text()
    .then(text => {
      const body = new URLSearchParams({ panel, json: text }).toString();
      return postPartial('/project/import', body);
    })
    .then(html => {
      replaceHtml(document.getElementById('main-content'), html);
      document.body.dispatchEvent(new Event('refreshChart'));
    })
    .catch(err => showToast(err.message))
    .finally(() => { input.value = ''; });
}

function exportScenario() {
  fetch(withTabUrl('/scenario/export'), { headers: requestHeaders() })
    .then(async response => {
      if (response.ok) return response.blob();
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Export failed (${response.status})`);
    })
    .then(blob => {
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = 'gpu-llm-scenario.json';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(objectUrl);
      showToast('Scenario exported.');
    })
    .catch(error => showToast(error.message || 'Could not export scenario.'));
}

function importScenario(input) {
  const file = input?.files?.[0];
  if (!file) return;
  file.text()
    .then(json => postPartial('/scenario/import', new URLSearchParams({ json }).toString()))
    .then(html => {
      replaceHtml(document.getElementById('main-content'), html);
      document.body.dispatchEvent(new Event('refreshChart'));
      showToast('Scenario imported.');
      document.getElementById('main-content')?.focus();
    })
    .catch(error => showToast(error.message || 'Could not import scenario.'))
    .finally(() => { input.value = ''; });
}

function writeClipboardText(text) {
  const fallback = () => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    ta.style.top = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      const ok = document.execCommand('copy');
      if (!ok) throw new Error('Copy command failed');
      return Promise.resolve();
    } catch (err) {
      return Promise.reject(err);
    } finally {
      ta.remove();
    }
  };
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text).catch(fallback);
  }
  return fallback();
}

function copyProjectionReport(button) {
  const original = button ? button.textContent : '';
  if (button) {
    button.disabled = true;
    button.textContent = 'Copying...';
  }
  fetch(withTabUrl('/api/projection-report'), { headers: requestHeaders() })
    .then(async r => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || `Report failed (${r.status})`);
      return data.text || '';
    })
    .then(text => writeClipboardText(text))
    .then(() => {
      if (button) button.textContent = 'Copied';
      showToast('Report copied to clipboard.');
    })
    .catch(err => showToast(err.message || 'Could not copy report.'))
    .finally(() => {
      setTimeout(() => {
        if (!button) return;
        button.disabled = false;
        button.textContent = original || 'Copy report';
      }, 900);
    });
}

function deleteStoredScenarios(button) {
  const original = button?.textContent || 'Delete my stored scenarios';
  if (button) { button.disabled = true; button.textContent = 'Deleting...'; }
  fetch(withTabUrl('/session/data'), {
    method: 'DELETE',
    headers: requestHeaders({ 'Accept': 'application/json' }),
  }).then(async response => {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Delete failed (${response.status})`);
    try { window.sessionStorage.removeItem(TAB_STORAGE_KEY); } catch {}
    showToast(data.message || 'Stored scenarios deleted. Reloading a fresh example.');
    setTimeout(() => window.location.assign('/'), 650);
  }).catch(error => {
    showToast(error.message || 'Could not delete stored scenarios.');
    if (button) { button.disabled = false; button.textContent = original; }
  });
}

/* Vendor and quantity are remembered while the user adds pools. */
window.selectedGpuCount = 8;
const gpuPickerCounts = { Apple: 1 };
let gpuPickerVendor = 'NVIDIA';
function syncGpuCountSelector(root) {
  if (!root?.querySelector('[data-gpu-picker]')) return;
  const count = gpuPickerCounts[gpuPickerVendor] ?? 8;
  window.selectedGpuCount = count;
  root.querySelectorAll('[data-gcnt]').forEach(button => {
    const selected = Number(button.dataset.gcnt) === count;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  root.querySelectorAll('[data-gpu-key]').forEach(button => {
    const actual = normalizedPickerGpuCount(button);
    button.title = `Add ${actual} × ${button.dataset.gpuName}. ${button.dataset.hardwareDetail}`;
  });
}
function filterGpuPicker(root, allowSwitch = false) {
  const query = root.querySelector('[data-hardware-search]').value.trim().toLocaleLowerCase();
  const words = query.split(/\s+/).filter(Boolean);
  const matches = new Map();
  root.querySelectorAll('[data-gpu-vendor-panel]').forEach(panel => {
    let total = 0;
    panel.querySelectorAll('.gpu-picker-family').forEach(family => {
      let visible = 0;
      family.querySelectorAll('[data-gpu-key]').forEach(button => {
        const match = words.every(word => button.dataset.hardwareMatch.toLocaleLowerCase().includes(word));
        button.hidden = !match;
        if (match) visible++;
      });
      family.hidden = visible === 0;
      total += visible;
    });
    matches.set(panel.dataset.gpuVendorPanel, total);
  });
  if (allowSwitch && !matches.get(gpuPickerVendor)) {
    const first = [...matches].find(([, count]) => count > 0);
    if (first) gpuPickerVendor = first[0];
  }
  root.querySelectorAll('[data-gpu-vendor]').forEach(button => {
    const selected = button.dataset.gpuVendor === gpuPickerVendor;
    button.classList.toggle('acc', selected);
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  root.querySelectorAll('[data-gpu-vendor-panel]').forEach(panel => {
    panel.hidden = panel.dataset.gpuVendorPanel !== gpuPickerVendor;
  });
  root.querySelector('[data-hardware-empty]').hidden = !!matches.get(gpuPickerVendor);
  syncGpuCountSelector(root);
}
function initGpuPicker(root) {
  if (!root.querySelector('[data-gpu-picker]')) return;
  filterGpuPicker(root);
  root.querySelector('[data-hardware-search]').addEventListener('input', () => filterGpuPicker(root, true));
  root.addEventListener('click', event => {
    const vendor = event.target.closest('[data-gpu-vendor]');
    const quantity = event.target.closest('[data-gcnt]');
    if (vendor) {
      gpuPickerVendor = vendor.dataset.gpuVendor;
      filterGpuPicker(root);
    } else if (quantity) {
      gpuPickerCounts[gpuPickerVendor] = Number(quantity.dataset.gcnt);
      syncGpuCountSelector(root);
    }
  });
  root.addEventListener('keydown', event => {
    const vendor = event.target.closest('[data-gpu-vendor]');
    if (event.key === 'Escape') {
      const trigger = root.parentElement.querySelector('[data-picker]');
      closePicker();
      trigger?.focus();
    } else if (vendor && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const tabs = [...root.querySelectorAll('[data-gpu-vendor]')];
      const offset = event.key === 'ArrowRight' ? 1 : -1;
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (tabs.indexOf(vendor) + offset + tabs.length) % tabs.length;
      tabs[index].click();
      tabs[index].focus();
    }
  });
}

/* Model picker: rows carry server-computed fit/quality flags (data-pass-*).
   JS only searches, dims rows failing an active filter, and sorts passing rows
   first. Query and chip state persist across kind-tab swaps. */
let modelPickerQuery = '';
// null = not chosen yet: take the server's default (fit filter on only when
// some GPUs are free), then keep the viewer's choice across tab swaps.
const modelPickerFilters = { fit: null, quality: null };
function filterModelPicker(root) {
  const list = root.querySelector('[data-model-list]');
  if (!list) return;
  root.querySelectorAll('[data-model-filter]').forEach(chip => {
    if (modelPickerFilters[chip.dataset.modelFilter] === null) {
      modelPickerFilters[chip.dataset.modelFilter] = chip.getAttribute('aria-pressed') === 'true';
    }
  });
  const words = modelPickerQuery.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const active = [...root.querySelectorAll('[data-model-filter]')]
    .filter(chip => !chip.disabled && modelPickerFilters[chip.dataset.modelFilter])
    .map(chip => chip.dataset.modelFilter);
  root.querySelectorAll('[data-model-filter]').forEach(chip => {
    chip.setAttribute('aria-pressed', String(!chip.disabled && !!modelPickerFilters[chip.dataset.modelFilter]));
  });
  const passes = row => active.every(name => row.dataset[`pass${name[0].toUpperCase()}${name.slice(1)}`] === '1');
  const order = rows => rows
    .map((row, index) => ({ row, index, pass: passes(row) }))
    .sort((a, b) => (b.pass - a.pass) || (a.index - b.index));
  const groups = root._modelGroups;
  let visible = 0;
  const place = (container, entries) => entries.forEach(({ row, pass }) => {
    const match = words.every(word => row.dataset.modelMatch.includes(word));
    row.hidden = !match;
    row.classList.toggle('dim', !pass);
    if (match) visible++;
    container.appendChild(row);
  });
  if (words.length) {
    // Searching: one flat list, passing rows first, catalog order otherwise.
    groups.forEach(group => { group.el.hidden = true; });
    place(list, order(groups.flatMap(group => group.rows)));
  } else {
    groups.forEach(group => {
      list.appendChild(group.el);
      place(group.el, order(group.rows));
      group.el.hidden = !group.rows.length;
    });
  }
  root.querySelector('[data-model-empty]').hidden = visible > 0;
}
function firstPassingModelRow(root) {
  return [...root.querySelectorAll('[data-model-list] [data-model-row]')]
    .find(row => !row.hidden && !row.classList.contains('dim'));
}
function initModelPicker(root) {
  const head = root?.querySelector('[data-model-picker]');
  if (!head) return;
  root._modelGroups = [...root.querySelectorAll('[data-model-group]')].map(el => ({
    el,
    rows: [...el.querySelectorAll('[data-model-row]')],
  }));
  const search = head.querySelector('[data-model-search]');
  search.value = modelPickerQuery;
  root.style.setProperty('--model-picker-head', `${head.offsetHeight}px`);
  filterModelPicker(root);
  search.addEventListener('input', () => {
    modelPickerQuery = search.value;
    filterModelPicker(root);
  });
  head.addEventListener('click', event => {
    const chip = event.target.closest('[data-model-filter]');
    if (!chip || chip.disabled) return;
    modelPickerFilters[chip.dataset.modelFilter] = !modelPickerFilters[chip.dataset.modelFilter];
    filterModelPicker(root);
  });
  search.addEventListener('keydown', event => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    firstPassingModelRow(root)?.click();
  });
  if (!root._modelPickerEscape) {
    // The .picker element survives kind-tab swaps; bind Escape once.
    root._modelPickerEscape = true;
    root.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      const trigger = root.parentElement?.querySelector('[data-picker]');
      closePicker();
      trigger?.focus();
    });
  }
  search.focus({ preventScroll: true });
}
document.body.addEventListener('htmx:afterSwap', e => {
  if (e.detail.target?.classList?.contains('picker')) initModelPicker(e.detail.target);
});

/* After HTMX swap, refresh chart */
document.body.addEventListener('refreshChart', () => {
  refreshChart();
});

document.body.addEventListener('htmx:configRequest', e => {
  e.detail.headers['X-Tab-ID'] = currentTabId();
});

document.body.addEventListener('htmx:beforeRequest', e => {
  const source = e.detail.elt;
  if (source && source.matches?.('button,input,select,a')) {
    _focusRestore = {
      id: source.id || '',
      ariaLabel: source.getAttribute('aria-label') || '',
      post: source.getAttribute('hx-post') || '',
      text: (source.textContent || '').trim(),
    };
  }
  markFastRemove(source);
});

document.body.addEventListener('htmx:beforeSwap', e => {
  if (e.detail.target && e.detail.target.id === 'main-content') {
    rememberScroll();
  }
});

document.body.addEventListener('htmx:afterSwap', e => {
  if (e.detail.target.id === 'main-content') {
    closePicker();
    restoreScroll();
    if (_focusRestore) {
      const controls = Array.from(document.querySelectorAll('button,input,select,a'));
      const match = (_focusRestore.id && document.getElementById(_focusRestore.id)) || controls.find(control => {
        const sameAction = !_focusRestore.post || control.getAttribute('hx-post') === _focusRestore.post;
        const sameName = _focusRestore.ariaLabel
          ? control.getAttribute('aria-label') === _focusRestore.ariaLabel
          : (control.textContent || '').trim() === _focusRestore.text;
        return sameAction && sameName;
      });
      match?.focus();
      _focusRestore = null;
    }
  }
  bindLiveSliders(e.detail.target || document);
});

document.body.addEventListener('htmx:responseError', e => {
  restoreFastRemove(e.detail.elt);
  let message = `Request failed (${e.detail.xhr.status})`;
  try {
    const data = JSON.parse(e.detail.xhr.responseText);
    if (data && data.error) message = data.error;
  } catch {}
  showToast(message);
});

document.body.addEventListener('htmx:sendError', e => {
  restoreFastRemove(e.detail.elt);
});

document.body.addEventListener('htmx:timeout', e => {
  restoreFastRemove(e.detail.elt);
});

function onColorSchemeChange() {
  if (lastChartData) renderChart(lastChartData);
}
if (colorSchemeQuery.addEventListener) {
  colorSchemeQuery.addEventListener('change', onColorSchemeChange);
} else if (colorSchemeQuery.addListener) {
  colorSchemeQuery.addListener(onColorSchemeChange);
}

window.addEventListener('resize', () => {
  positionPicker(openPicker);
  if (chart && chartRerenderOnResize && lastChartData) renderChart(lastChartData);
  else if (chart) chart.resize();
});

/* Initial tab-state load */
document.addEventListener('DOMContentLoaded', () => {
  bindLiveSliders(document);
  /* Pages without the calculator config panel (e.g. use cases) load this
   * file for shared wiring only — skip the tab-state sync and chart fetch. */
  if (!document.getElementById('main-content')) return;
  syncCurrentTabState()
    .catch(console.error)
    .finally(refreshChart);
});

/* ── Disclosures that survive re-renders ─────────────────────────────────
 * Every edit re-renders #main-content, which would snap open <details> shut
 * mid-task (e.g. "Change setup" on a model card). Remember them by key. */
const _openDetails = new Set();
document.addEventListener('click', e => {
  if (!e.target.closest('a[href="#economic-impact"]')) return;
  const fold = document.querySelector('details.econ-fold');
  if (fold) fold.open = true;
});
document.addEventListener('toggle', e => {
  const el = e.target;
  if (!(el instanceof HTMLDetailsElement) || !el.dataset.persistKey) return;
  if (el.open) _openDetails.add(el.dataset.persistKey);
  else _openDetails.delete(el.dataset.persistKey);
}, true);
document.body.addEventListener('htmx:afterSwap', () => {
  document.querySelectorAll('details[data-persist-key]').forEach(el => {
    if (_openDetails.has(el.dataset.persistKey)) el.open = true;
  });
});

/* ── Model row details (▸) ───────────────────────────────────────────────
 * Each row's full-width detail line opens in place; which ones are open is
 * remembered across the HTMX re-render that follows every edit. */
const _openRows = new Set();
document.addEventListener('click', e => {
  const btn = e.target.closest('.mr-more');
  if (!btn) return;
  const id = btn.dataset.more;
  const row = document.getElementById(id);
  if (!row) return;
  row.hidden = !row.hidden;
  btn.setAttribute('aria-expanded', String(!row.hidden));
  btn.textContent = row.hidden ? '▸' : '▾';
  if (row.hidden) _openRows.delete(id); else _openRows.add(id);
});
document.body.addEventListener('htmx:afterSwap', () => {
  _openRows.forEach(id => {
    const row = document.getElementById(id);
    const btn = document.querySelector(`.mr-more[data-more="${id}"]`);
    if (row && btn) { row.hidden = false; btn.setAttribute('aria-expanded', 'true'); btn.textContent = '▾'; }
  });
});

/* Layout dropdown (models with many layouts): post the chosen tp/pp/dp to the
 * same /model/strat route the layout buttons use. */
document.addEventListener('change', e => {
  const sel = e.target.closest('select[data-layout-select]');
  if (!sel || !window.htmx) return;
  const [tp, pp, dp] = sel.value.split(',');
  htmx.ajax('POST', '/model/strat', {
    target: '#main-content',
    values: { panel: sel.dataset.panel, uid: sel.dataset.uid, phase: sel.dataset.phase, tp, pp, dp },
  });
});
