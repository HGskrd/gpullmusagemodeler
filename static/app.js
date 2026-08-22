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
    extraCssText: `box-shadow:0 6px 24px ${shadow};border-radius:7px;font-family:${theme.fontFamily};line-height:1.35;max-width:min(460px,calc(100vw - 24px));overflow-wrap:anywhere;`,
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
  return `<div style="display:flex;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
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
    userpareto: 'User Pareto', processingpareto: 'Processing Pareto',
    embedquality: 'Embedding Quality', asrquality: 'ASR Quality', realtime: 'Realtime capacity',
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
    empty.textContent = 'No deployed model data is available for this chart.';
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

function renderChart(data) {
  lastChartData = data;
  updateChartAccessibleSummary(data);
  disposeChart();
  const el = document.getElementById('mc');
  if (!el || !window.echarts) return;

  const theme = chartTheme();
  const datasets = normalizeDatasets(data.datasets, theme.dk);
  chart = echarts.init(el);

  const fmtBatch = v => v >= 1e3 ? (v/1e3).toFixed(v >= 1e4 ? 0 : 1).replace('.0', '')+'k' : Math.round(v);
  const fmtUserTok = v => v >= 100 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(1) : v.toFixed(2);
  const fmtRate = v => v >= 100 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(1) : v.toFixed(2);

  if (data.mode === 'userpareto') {
    chart.setOption({
      ...baseOption(theme, paramsArr => {
        if (!Array.isArray(paramsArr) || !paramsArr.length) return '';
        const xVal = Number(paramsArr[0].axisValue ?? paramsArr[0].data?.value?.[0]);
        let html = axisHeader(theme, 'Concurrent users', fmtBatch(xVal));
        [...paramsArr]
          .sort((a, b) => (b.data?.raw?.y ?? -Infinity) - (a.data?.raw?.y ?? -Infinity))
          .forEach(p => {
            const raw = p.data?.raw;
            const yStr = raw?.y != null ? `${fmtUserTok(Number(raw.y))} tok/s/user` : 'OOM';
            const detail = raw?.y != null ? ` <span style="color:${theme.tc};font-size:10px;">total ${Number(raw.total_tps).toLocaleString()} tok/s · ${Number(raw.lat).toFixed(1)}ms/tok</span>` : '';
            const oom = raw?.y == null;
            html += `<div style="display:flex;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
              `<span style="display:inline-block;width:10px;height:2.5px;border-radius:2px;background:${p.color};flex-shrink:0;"></span>` +
              `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:140px;color:${theme.text};">${escapeTooltipHtml(p.seriesName)}</span>` +
              `<span style="margin-left:4px;text-align:right;font-weight:600;color:${oom ? 'var(--red)' : theme.text};">${yStr}${detail}</span>` +
              `</div>`;
            html += hardwareDisclosureHtml(p, theme);
            html += specDisclosureHtml(p, theme);
          });
        return html;
      }, 'axis'),
      xAxis: axisCommon(theme, 'Concurrent users', v => fmtBatch(Number(v)), {
        type: 'log',
        min: 1,
        max: data.x_max,
        nameGap: 30,
      }),
      yAxis: axisCommon(theme, 'tok/s/user', fmtUserTok, {
        type: 'log',
        nameGap: 52,
      }),
      series: datasets.map(ds => makeSeries(ds, true)),
    });
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
            html += `<div style="display:flex;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
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
            html += `<div style="display:flex;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
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
        html += `<div style="display:flex;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
          `<span style="display:inline-block;width:10px;height:10px;${markerStyle}flex-shrink:0;"></span>` +
          `<span style="flex:1;color:${theme.text};">${escapeTooltipHtml(params.seriesName)}</span>` +
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
        html += `<div style="display:flex;align-items:center;gap:7px;padding:2px 0;font-family:${theme.fontFamily};font-size:11px;">` +
          `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${params.color};flex-shrink:0;"></span>` +
          `<span style="flex:1;color:${theme.text};">${escapeTooltipHtml(params.seriesName)}</span>` +
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

function togglePicker(btn, url) {
  const wrap = btn.closest('.picker-wrap');
  const shouldCloseOnly = openPicker && openPicker.parentElement === wrap;
  closePicker();
  if (shouldCloseOnly) return;
  const pk = document.createElement('div');
  pk.className = 'picker';
  pk.innerHTML = '<div style="padding:10px;text-align:center;" class="T">Loading...</div>';
  wrap.appendChild(pk);
  openPicker = pk;
  btn.setAttribute('aria-expanded', 'true');
  pk.setAttribute('role', 'dialog');
  pk.setAttribute('aria-label', `${btn.textContent.trim()} options`);
  fetch(withTabUrl(url), { headers: requestHeaders() }).then(r => r.text()).then(html => {
    replaceHtml(pk, html);
    syncGpuCountSelector(pk);
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

/* GPU count selector in picker */
window.selectedGpuCount = 8;
function syncGpuCountSelector(root) {
  const count = Number(window.selectedGpuCount ?? 8) || 8;
  const scope = root || document;
  const groups = [];
  if (scope.classList && scope.classList.contains('gpu-count-sel')) groups.push(scope);
  scope.querySelectorAll?.('.gpu-count-sel').forEach(group => groups.push(group));
  groups.forEach(group => {
    group.querySelectorAll('[data-gcnt]').forEach(button => {
      const selected = Number(button.dataset.gcnt) === count;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
  });
}

document.addEventListener('click', e => {
  const gcb = e.target.closest('[data-gcnt]');
  if (gcb) {
    window.selectedGpuCount = parseInt(gcb.dataset.gcnt, 10);
    syncGpuCountSelector(gcb.closest('.gpu-count-sel'));
  }
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
  if (chart) chart.resize();
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
