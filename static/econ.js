/* Economics section charts and interactions.
 *
 * Chart containers carry data-chart="…" and live inside a [data-econ-panel]
 * block that embeds a <script type="application/json" class="econ-payload">.
 * initEcon() is idempotent (data-inited) so it can run on page load and after
 * every HTMX (OOB) swap, when inline scripts would not execute.
 */
(function () {
  'use strict';

  var css = function (n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); };
  var COLORS = null;
  function colors() {
    if (!COLORS) {
      COLORS = {
        total: css('--acc') || '#2255aa',
        served: css('--acc2') || '#1a7a55',
        margin: '#2ea36b',
        cost: css('--fg2') || '#5f5f58',
        spilled: css('--amber') || '#a06800',
        leaked: css('--red') || '#b42e2e',
        destroyed: '#7f1d1d',
        uc: '#8a8a82',
        fg: css('--fg0') || '#1a1a18',
        fg2: css('--fg2') || '#5f5f58',
        b1: css('--b1') || 'rgba(0,0,0,0.13)'
      };
    }
    return COLORS;
  }
  // Re-resolve on theme change.
  var schemeQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  if (schemeQuery && schemeQuery.addEventListener) {
    schemeQuery.addEventListener('change', function () { COLORS = null; });
  }

  var money = function (v) {
    var s = v < 0 ? '-' : ''; v = Math.abs(v);
    if (v >= 1e6) return s + '$' + (v / 1e6).toFixed(2) + 'M';
    if (v >= 1e3) return s + '$' + (v / 1e3).toFixed(1) + 'k';
    return s + '$' + v.toFixed(v >= 100 ? 0 : 2);
  };
  var num = function (v) {
    if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'k';
    return String(Math.round(v));
  };
  var nodeColor = function (n) { return n.color || colors()[n.c] || '#888'; };

  function sankeyOption(data, fmt) {
    return {
      textStyle: { color: colors().fg, fontFamily: 'inherit' },
      tooltip: {
        trigger: 'item',
        formatter: function (x) {
          if (x.dataType === 'edge') return x.data.source + ' → ' + x.data.target + '<br><b>' + fmt(x.data.value) + '/day</b>';
          return '<b>' + x.name.replace(/^(m|uc\d+)::/, '') + '</b>';
        }
      },
      series: [{
        type: 'sankey',
        animation: false,
        data: data.nodes.map(function (n) { return { name: n.name, depth: n.depth, itemStyle: { color: nodeColor(n) } }; }),
        links: data.links,
        left: 20, right: 150, top: 12, bottom: 12,
        nodeWidth: 14, nodeGap: 14,
        // layoutIterations 0 keeps input order within each layer, so children
        // stack directly beside their parent and bands never cross.
        layoutIterations: 0,
        label: {
          color: colors().fg, fontSize: 11,
          formatter: function (x) { return x.name.replace(/^(m|uc\d+)::/, ''); }
        },
        lineStyle: { color: 'gradient', curveness: 0.5, opacity: 0.32 },
        emphasis: { focus: 'adjacency' }
      }]
    };
  }

  function bridgeOption(v) {
    return {
      textStyle: { color: colors().fg, fontFamily: 'inherit' },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
        formatter: function (xs) { var x = xs[1] || xs[0]; return x.name + '<br><b>' + v.fmt(x.value) + '/day</b>'; } },
      grid: { left: 8, right: 8, top: 32, bottom: 8, containLabel: true },
      xAxis: { type: 'category', data: [v.totalName, 'To cloud', 'Captured on-prem', 'Destroyed'],
        axisLabel: { color: colors().fg2, interval: 0 }, axisLine: { lineStyle: { color: colors().b1 } } },
      yAxis: { type: 'value', axisLabel: { color: colors().fg2, formatter: v.fmt }, splitLine: { lineStyle: { color: colors().b1 } } },
      series: [
        { type: 'bar', stack: 'w', itemStyle: { color: 'transparent' }, tooltip: { show: false },
          data: [0, v.total - v.cloud, v.total - v.cloud - v.captured, 0] },
        { type: 'bar', stack: 'w', barWidth: '55%',
          data: [
            { value: v.total, itemStyle: { color: colors().total } },
            { value: v.cloud, itemStyle: { color: colors().spilled } },
            { value: v.captured, itemStyle: { color: colors().served } },
            { value: v.destroyed, itemStyle: { color: colors().destroyed } }
          ],
          label: { show: true, position: 'top', color: colors().fg, fontSize: 11,
            formatter: function (x) { return v.fmt(x.value); } }
        }
      ]
    };
  }

  function stackOption(rows) {
    var segs = [
      ['served_pct', 'Served', colors().served],
      ['spilled_pct', 'Spilled (capacity)', colors().spilled],
      ['leaked_pct', 'Leaked (fit/price)', colors().leaked],
      ['destroyed_pct', 'Destroyed', colors().destroyed]
    ];
    return {
      textStyle: { color: colors().fg, fontFamily: 'inherit' },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
        formatter: function (xs) {
          var i = xs[0].dataIndex, r = rows[i];
          var out = '<b>' + r.name + '</b><br>' + num(r.tokens_day) + ' tok/day';
          xs.forEach(function (x) { out += '<br>' + x.marker + ' ' + x.seriesName + ': ' + x.value.toFixed(0) + '%'; });
          return out;
        } },
      legend: { bottom: 0, textStyle: { color: colors().fg2, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
      grid: { left: 8, right: 30, top: 8, bottom: 30, containLabel: true },
      xAxis: { type: 'value', max: 100, axisLabel: { color: colors().fg2, formatter: '{value}%' },
        splitLine: { lineStyle: { color: colors().b1 } } },
      yAxis: { type: 'category', data: rows.map(function (r) { return r.name; }), inverse: true,
        axisLabel: { color: colors().fg, fontSize: 11 }, axisLine: { lineStyle: { color: colors().b1 } } },
      series: segs.map(function (s) {
        return {
          name: s[1], type: 'bar', stack: 'f', barWidth: 20,
          itemStyle: { color: s[2] },
          label: { show: true, color: '#fff', fontSize: 10,
            formatter: function (x) { return x.value >= 8 ? x.value.toFixed(0) + '%' : ''; } },
          data: rows.map(function (r) { return r[s[0]]; })
        };
      })
    };
  }

  function payloadFor(el) {
    var host = el.closest('[data-econ-panel]');
    if (!host) return null;
    var tag = host.querySelector('script.econ-payload');
    if (!tag) return null;
    try { return JSON.parse(tag.textContent); } catch (e) { return null; }
  }

  var RENDERERS = {
    'sankey-money': function (el, data) { return sankeyOption(data.money, money); },
    'sankey-tokens': function (el, data) { return sankeyOption(data.tokens, function (v) { return num(v) + ' tok'; }); },
    'alloc': function (el, data) { return sankeyOption(data.alloc, function (v) { return num(v) + ' tok'; }); },
    'stack': function (el, data) { return stackOption(data.stack); },
    'bridge': function (el, data) {
      el._bridgeViews = {
        money: Object.assign({ fmt: money }, data.bridge.money),
        tokens: Object.assign({ fmt: function (v) { return num(v) + ' tok'; } }, data.bridge.tokens)
      };
      return bridgeOption(el._bridgeViews.money);
    }
  };

  function initEcon(root) {
    if (!root || !root.querySelectorAll) return;
    var charts = [];
    if (root.matches && root.matches('.econ-chart[data-chart]')) charts.push(root);
    root.querySelectorAll('.econ-chart[data-chart]').forEach(function (el) { charts.push(el); });
    charts.forEach(function (el) {
      if (el.dataset.inited) return;
      var data = payloadFor(el);
      var render = RENDERERS[el.dataset.chart];
      if (!data || !render) return;
      var chart = echarts.init(el);
      var opt = render(el, data);
      if (opt) chart.setOption(opt);
      el._econChart = chart;
      el.dataset.inited = '1';
    });
  }

  window.addEventListener('resize', function () {
    document.querySelectorAll('.econ-chart[data-inited]').forEach(function (el) {
      if (el._econChart && el.isConnected) el._econChart.resize();
    });
  });

  /* ── Delegated interactions (survive HTMX swaps) ─────────────────────── */
  document.addEventListener('click', function (e) {
    // View tabs within an economics section.
    var tabBtn = e.target.closest('.econ-tab-btn');
    if (tabBtn) {
      var tabs = tabBtn.closest('.econ-tabs');
      tabs.querySelectorAll('.econ-tab-btn').forEach(function (b) { b.classList.remove('on'); });
      tabBtn.classList.add('on');
      tabs.querySelectorAll('.econ-tab-panel').forEach(function (panel) {
        panel.hidden = panel.dataset.tab !== tabBtn.dataset.tab;
      });
      // Charts in a freshly revealed panel need a resize to pick up real width.
      tabs.querySelectorAll('.econ-tab-panel:not([hidden]) .econ-chart[data-inited]').forEach(function (el) {
        if (el._econChart) el._econChart.resize();
      });
      return;
    }

    // Unit/view toggles (bridge $/tokens, swap arrows/delta).
    var toggleBtn = e.target.closest('.view-toggle button');
    if (toggleBtn) {
      var toggle = toggleBtn.closest('.view-toggle');
      toggle.querySelectorAll('button').forEach(function (b) { b.classList.remove('on'); });
      toggleBtn.classList.add('on');
      if (toggle.dataset.toggle === 'bridge') {
        var chartEl = toggle.closest('[data-econ-panel], .econ-card').querySelector('.econ-chart[data-chart="bridge"]');
        if (chartEl && chartEl._econChart && chartEl._bridgeViews) {
          chartEl._econChart.setOption(bridgeOption(chartEl._bridgeViews[toggleBtn.dataset.view]));
        }
      } else if (toggle.dataset.toggle === 'swap') {
        var card = toggle.closest('.econ-swap-views');
        if (card) {
          card.querySelector('#' + toggle.dataset.prefix + 'Arrows').hidden = toggleBtn.dataset.view !== 'arrows';
          card.querySelector('#' + toggle.dataset.prefix + 'Delta').hidden = toggleBtn.dataset.view !== 'delta';
        }
      }
      return;
    }

    // Sortable tables: click cycles desc → asc → original (text: asc → desc → original).
    var th = e.target.closest('.sortable-table th[data-col]');
    if (th) {
      var table = th.closest('table');
      var tbody = table.tBodies[0];
      if (!table._originalRows) table._originalRows = Array.from(tbody.rows);
      var col = +th.dataset.col, type = th.dataset.type;
      if (table._sortCol !== col) { table._sortCol = col; table._sortDir = 1; }
      else { table._sortDir = table._sortDir === 1 ? -1 : (table._sortDir === -1 ? 0 : 1); }
      table.querySelectorAll('th').forEach(function (h) { h.classList.remove('sort-asc', 'sort-desc'); });
      if (table._sortDir === 0) {
        table._originalRows.forEach(function (r) { tbody.appendChild(r); });
        return;
      }
      var isDesc = (type === 'text') ? (table._sortDir === -1) : (table._sortDir === 1);
      th.classList.add(isDesc ? 'sort-desc' : 'sort-asc');
      var rows = Array.from(tbody.rows);
      rows.sort(function (a, b) {
        var va = a.cells[col].dataset.v, vb = b.cells[col].dataset.v;
        var cmp = type === 'text' ? va.localeCompare(vb) : ((+va) - (+vb));
        return isDesc ? -cmp : cmp;
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    }
  });

  document.addEventListener('DOMContentLoaded', function () { initEcon(document); });
  // OOB swap event targets vary (the response target, not the swapped-in
  // fragment), so scan the whole document — init is idempotent. The delayed
  // pass covers content inserted after the event fires.
  var scheduleInit = function () {
    initEcon(document);
    setTimeout(function () { initEcon(document); }, 50);
  };
  document.body.addEventListener('htmx:afterSwap', scheduleInit);
  document.body.addEventListener('htmx:oobAfterSwap', scheduleInit);
  // The app's manual replaceHtml path (session sync, scenario import) re-renders
  // the economics section without htmx events, but always fires refreshChart.
  document.body.addEventListener('refreshChart', scheduleInit);
})();
