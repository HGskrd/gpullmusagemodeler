/* Shared helpers for the /admin prototypes: persona rules, per-visitor
 * features, filter bar, nav and tooltip. Depends on data.js (window.ADMIN). */
(function () {
  "use strict";
  const A = window.ADMIN;
  const { GPU, MODEL, COUNTRY, sizeClass } = A.catalog;

  // ---------- personas ----------
  const PERSONAS = {
    homelab: { label: "Homelab builder", short: "Homelab", blurb: "Sizing consumer GPUs, Macs, Spark/Strix boxes for local models; quantizes hard." },
    buyer: { label: "Infra buyer", short: "Infra buyer", blurb: "Datacenter nodes, A/B hardware compare, cloud-vs-own economics, exports a report." },
    evaluator: { label: "Frontier-model evaluator", short: "Evaluator", blurb: "Tries many large MoE models, parallelism splits and context settings." },
    speech: { label: "Speech & retrieval builder", short: "Speech/RAG", blurb: "Lives in the ASR / embedding tabs; concurrency and streams over params." },
    solutions: { label: "Use-case planner", short: "Use-case", blurb: "Starts from a business workload (use-cases, projections) not a model." },
    agent: { label: "AI agent / API", short: "Agent", blurb: "Assistants and scripts calling the stateless calculator API on someone's behalf." },
    other: { label: "Unclassified explorer", short: "Unclassified", blurb: "Configured something but matched no rule — candidates for a new persona." },
    bouncer: { label: "Bounced", short: "Bounced", blurb: "Opened the default page and left without touching anything." },
    internal: { label: "Internal (you)", short: "Internal", blurb: "Admin-cookie visitor; excluded from every chart by default." },
  };
  const PERSONA_ORDER = ["homelab", "buyer", "evaluator", "speech", "solutions", "agent", "other", "bouncer"];
  const pColor = (k) => `var(--p-${k})`;
  const pchip = (k, txt) => `<span class="pchip"><i style="background:${pColor(k)}"></i>${txt || PERSONAS[k].label}</span>`;

  const CONFIG_TYPES = new Set(["gpu_add", "gpu_qty", "model_add", "model_prec", "model_remove", "mode", "compare_duplicate", "compare_gpu", "econ_view",
    "use_case_add", "projection_toggle", "dist_slide", "settings_mu", "gpu_cost", "copy_link", "export_report", "scenario_export", "api_call"]);

  // Per-visitor features + rule-based persona with human-readable evidence.
  function features(v) {
    const f = { events: 0, config: 0, gpus: {}, models: {}, frontier: 0, nonLlm: 0, llm: 0, compare: 0, econ: 0, useCases: 0, exports: 0, cost: 0, dist: 0, modes: new Set(),
      api: 0, searches: [], misses: 0, oom: 0, recovered: 0, sessions: v.sessions.length, durMs: 0, classes: {}, maxVram: 0, firstModelMs: null, resets: 0 };
    for (const s of v.sessions) {
      f.durMs += s.end - s.start;
      let lastFail = false;
      for (const e of s.events) {
        f.events++;
        if (CONFIG_TYPES.has(e.type)) f.config++;
        switch (e.type) {
          case "gpu_add": case "gpu_qty": {
            const g = GPU[e.gpu]; f.gpus[e.gpu] = Math.max(f.gpus[e.gpu] || 0, e.qty);
            f.classes[g.cls] = (f.classes[g.cls] || 0) + 1; f.maxVram = Math.max(f.maxVram, g.vram * e.qty); break;
          }
          case "model_add": {
            const m = MODEL[e.model]; f.models[e.model] = (f.models[e.model] || 0) + 1;
            if (m.b >= 400) f.frontier++;
            if (m.kind === "llm") f.llm++; else f.nonLlm++;
            if (f.firstModelMs == null) f.firstModelMs = e.t - s.start;
            break;
          }
          case "result": if (!e.fits) { f.oom++; lastFail = true; } else if (lastFail) { f.recovered++; lastFail = false; } break;
          case "compare_duplicate": f.compare++; break;
          case "econ_view": f.econ++; break;
          case "use_case_add": case "projection_toggle": f.useCases++; break;
          case "copy_link": case "export_report": case "scenario_export": f.exports++; break;
          case "gpu_cost": f.cost++; break;
          case "dist_slide": case "settings_mu": f.dist++; break;
          case "mode": f.modes.add(e.mode); break;
          case "api_call": f.api++; if (e.model) f.models[e.model] = (f.models[e.model] || 0) + 1; if (e.gpu) f.gpus[e.gpu] = e.qty; break;
          case "search": f.searches.push(e.term); if (!e.results) f.misses++; break;
          case "session_reset": f.resets++; break;
        }
      }
    }
    f.nModels = Object.keys(f.models).length;
    const cls = f.classes;
    f.domClass = Object.keys(cls).sort((a, b) => cls[b] - cls[a])[0] || null;
    return f;
  }

  function classify(v, f) {
    if (v.internal) return { key: "internal", why: ["Signed-in admin cookie"] };
    if (v.device === "agent" || f.api > 0) return { key: "agent", why: [`${f.api} API call${f.api === 1 ? "" : "s"}`, `UA ${v.ua || "unknown"}`] };
    if (f.config === 0) return { key: "bouncer", why: ["No configuration change", f.misses ? `searched “${f.searches[0]}” → 0 results` : "left on the default scenario"] };
    const why = [];
    if (f.modes.has("asr") || f.modes.has("embedding") || (f.nonLlm > 0 && f.nonLlm >= f.llm)) {
      if (f.modes.size) why.push(`switched to ${[...f.modes].join(" + ")} tab`);
      why.push(`${f.nonLlm} speech/embedding model${f.nonLlm === 1 ? "" : "s"}`);
      return { key: "speech", why };
    }
    if (f.useCases >= 1 || v.entry === "/use-cases") {
      why.push(f.useCases ? `${f.useCases} use-case/projection edit${f.useCases === 1 ? "" : "s"}` : "entered on /use-cases");
      if (f.econ) why.push("opened economics");
      return { key: "solutions", why };
    }
    const dc = f.domClass === "datacenter" || f.domClass === "workstation";
    if (dc && (f.compare || f.econ || f.cost || f.exports) && f.nModels <= 3) {
      why.push(`${f.domClass} hardware (up to ${Math.round(f.maxVram)} GB)`);
      if (f.compare) why.push("A/B compare");
      if (f.econ) why.push("economics view");
      if (f.cost) why.push("entered $/hr");
      if (f.exports) why.push("exported/shared");
      return { key: "buyer", why };
    }
    if (f.nModels >= 4 || f.frontier >= 2 || (f.dist && f.frontier)) {
      why.push(`${f.nModels} distinct models`);
      if (f.frontier) why.push(`${f.frontier} frontier (400B+) adds`);
      if (f.dist) why.push("tuned parallelism / context");
      return { key: "evaluator", why };
    }
    if (f.domClass === "consumer" || f.domClass === "unified") {
      why.push(`${f.domClass === "unified" ? "unified-memory box" : "consumer GPU"} (${Math.round(f.maxVram)} GB)`);
      if (f.oom) why.push(`${f.oom} doesn’t-fit result${f.oom === 1 ? "" : "s"}`);
      return { key: "homelab", why };
    }
    return { key: "other", why: ["configured, no rule matched"] };
  }

  for (const v of A.visitors) {
    v.f = features(v);
    const c = classify(v, v.f);
    v.persona = c.key;
    v.why = c.why;
  }

  // ---------- formatting ----------
  const nf = new Intl.NumberFormat("en-US");
  const fmt = (n) => nf.format(Math.round(n));
  const pct = (n, d = 0) => `${(n * 100).toFixed(d)}%`;
  const flag = (iso) => iso && iso.length === 2 ? String.fromCodePoint(...[...iso.toUpperCase()].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)) : "🌐";
  const countryName = (iso) => (COUNTRY[iso] ? COUNTRY[iso].name : iso);
  const dfmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
  const dtfmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
  const date = (t) => dfmt.format(new Date(t));
  const datetime = (t) => dtfmt.format(new Date(t)) + " UTC";
  function ago(t) {
    const s = (A.meta.now - t) / 1000;
    if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  }
  function dur(ms) {
    const s = Math.round(ms / 1000);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  }
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const shortGpu = (k) => GPU[k].name.replace(/ (Blackwell )?(Server Edition|Workstation)/, (m) => (m.includes("Server") ? " Server" : " WS")).replace(/ HGX\/DGX| SXM| PCIe/g, "");
  const shortModel = (n) => n.replace(/ \+ 196B Engram| \+51B N-gram| 1017| 0731| 0813| 05-2026/g, "");

  // ---------- filter state ----------
  const store = {
    get(k, d) { try { const v = localStorage.getItem("adm:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("adm:" + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
  };
  const filter = { range: store.get("range", 90), internal: store.get("internal", false), agents: store.get("agents", true) };

  function inRange(v) { return v.last >= A.meta.now - filter.range * 86400000; }
  function visible() {
    return A.visitors.filter((v) => inRange(v) && (filter.internal || !v.internal) && (filter.agents || v.persona !== "agent"));
  }

  function filterBar(el, onChange, extra) {
    el.innerHTML = `
      <span class="fl-label">Range</span>
      <span class="seg" data-k="range">${[7, 30, 90].map((d) => `<button data-v="${d}" class="${filter.range === d ? "on" : ""}">${d}d</button>`).join("")}</span>
      <label class="chk"><input type="checkbox" data-k="agents" ${filter.agents ? "checked" : ""}> AI agents</label>
      <label class="chk"><input type="checkbox" data-k="internal" ${filter.internal ? "checked" : ""}> Internal traffic (you)</label>
      ${extra || ""}
      <span class="spacer"></span>
      <span class="muted" id="fl-count"></span>`;
    el.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
      filter.range = +b.dataset.v; store.set("range", filter.range);
      el.querySelectorAll(".seg[data-k=range] button").forEach((x) => x.classList.toggle("on", x === b));
      onChange();
    }));
    el.querySelectorAll("input[type=checkbox][data-k]").forEach((c) => c.addEventListener("change", () => {
      filter[c.dataset.k] = c.checked; store.set(c.dataset.k, c.checked); onChange();
    }));
  }
  function setCount(n) { const c = document.getElementById("fl-count"); if (c) c.textContent = `${fmt(n)} visitors in view`; }

  // ---------- nav ----------
  const PAGES = [
    ["index.html", "", "Overview"],
    ["1-audience.html", "1", "Audience & geo"],
    ["2-personas.html", "2", "Personas"],
    ["3-people.html", "3", "People"],
    ["4-journeys.html", "4", "Journeys & friction"],
    ["5-demand.html", "5", "Demand & gaps"],
  ];
  function nav(el, title, sub) {
    const here = location.pathname.split("/").pop() || "index.html";
    el.innerHTML = `
      <div><h1>${title}</h1><div class="sub">${sub}</div></div>
      <span class="spacer"></span>
      <nav class="nav">${PAGES.map(([h, n, l]) => `<a href="${h}" class="${h === here ? "on" : ""}">${n ? `<span class="n">${n}</span>` : ""}${l}</a>`).join("")}</nav>
      <button class="btn" id="theme-btn" title="Toggle theme">◐</button>`;
    const saved = store.get("theme", null);
    if (saved) document.documentElement.dataset.theme = saved;
    el.querySelector("#theme-btn").addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next; store.set("theme", next);
      window.dispatchEvent(new Event("themechange"));
    });
  }

  // ---------- tooltip ----------
  const tipEl = document.createElement("div");
  tipEl.className = "tip";
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tipEl));
  function tip(html, ev) {
    if (!html) { tipEl.classList.remove("show"); return; }
    tipEl.innerHTML = html;
    tipEl.classList.add("show");
    const pad = 14, w = tipEl.offsetWidth, h = tipEl.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
    tipEl.style.left = x + "px"; tipEl.style.top = y + "px";
  }
  // Delegated: any element with data-tip shows it.
  document.addEventListener("mousemove", (ev) => {
    const t = ev.target.closest && ev.target.closest("[data-tip]");
    if (t) tip(t.getAttribute("data-tip"), ev); else if (!ev.target.closest || !ev.target.closest("[data-tip-own]")) tip(null);
  });

  // ---------- small chart helpers ----------
  // Horizontal bar list: rows [{label, value, color?, sub?, tip?}]
  function hbars(rows, opts = {}) {
    const max = opts.max || Math.max(1, ...rows.map((r) => r.value));
    const fmtV = opts.fmt || fmt;
    return `<div style="display:grid;grid-template-columns:minmax(90px,${opts.labelW || "38%"}) 1fr auto;gap:5px 10px;align-items:center;font-size:12px">` +
      rows.map((r) => `
        <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(r.plain || "")}">${r.label}</div>
        <div class="bar-track" data-tip="${esc(r.tip || `<b>${r.plain || ""}</b><div>${fmtV(r.value)}</div>`)}" style="height:14px;display:flex;align-items:center">
          <div class="bar" style="width:${Math.max(0.5, (r.value / max) * 100)}%;background:${r.color || "var(--accent)"};height:${opts.h || 10}px"></div>
        </div>
        <div class="num" style="text-align:right;color:var(--text-2)">${fmtV(r.value)}${r.sub ? ` <span class="muted">${r.sub}</span>` : ""}</div>`).join("") + `</div>`;
  }
  function countBy(arr, keyFn) {
    const m = new Map();
    for (const x of arr) { const k = keyFn(x); if (k == null) continue; m.set(k, (m.get(k) || 0) + 1); }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }
  const engaged = (v) => v.persona !== "bouncer";

  window.UI = { A, PERSONAS, PERSONA_ORDER, pColor, pchip, fmt, pct, flag, countryName, date, datetime, ago, dur, esc, shortGpu, shortModel, filter, visible, filterBar, setCount, nav, tip, hbars, countBy, engaged, sizeClass, store };
})();
