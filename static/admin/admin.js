/* /admin insight pages: shared runtime.
 *
 * Loads /admin/data.json (visitors -> sessions -> normalized events, built by
 * web/analytics.py), derives per-visitor features and a rule-based persona with
 * human-readable evidence, and exposes formatting, filter, tooltip and small
 * chart helpers as window.UI. Each page calls UI.start(render).
 */
(function () {
  "use strict";

  // ---------- personas ----------
  const PERSONAS = {
    homelab: { label: "Homelab builder", short: "Homelab", blurb: "Sizing consumer GPUs, Macs, Spark/Strix boxes for local models." },
    buyer: { label: "Infra buyer", short: "Infra buyer", blurb: "Datacenter or workstation nodes, A/B compare, economics, exports." },
    evaluator: { label: "Frontier-model evaluator", short: "Evaluator", blurb: "Tries many or very large models, parallelism and context settings." },
    speech: { label: "Speech & retrieval builder", short: "Speech/RAG", blurb: "ASR / embedding modes and models; streams and documents over params." },
    solutions: { label: "Use-case planner", short: "Use-case", blurb: "Starts from a business workload (use-cases, projections), not a model." },
    agent: { label: "AI agent / API", short: "Agent", blurb: "Assistants and scripts calling the calculator on someone's behalf." },
    other: { label: "Unclassified explorer", short: "Unclassified", blurb: "Configured something but matched no rule: candidates for a new persona." },
    bouncer: { label: "Bounced", short: "Bounced", blurb: "Opened a page and left without changing anything." },
    internal: { label: "Internal (you)", short: "Internal", blurb: "Browser with the admin login; excluded by default." },
  };
  const PERSONA_ORDER = ["homelab", "buyer", "evaluator", "speech", "solutions", "agent", "other", "bouncer"];
  const RULES = {
    homelab: "consumer GPU or unified-memory box (Mac, Spark, Strix) dominates",
    buyer: "datacenter/workstation hardware with ≤3 models (compare, economics, $/hr or export make it certain)",
    evaluator: "≥4 distinct models, or ≥2 frontier (400B+) models, or tuning with a frontier model",
    speech: "ASR/embedding mode, or speech/embedding models ≥ LLMs",
    solutions: "edited use-cases / projections, or opened the use-case library",
    agent: "agent-API call or an assistant/script User-Agent",
    other: "changed something, no rule matched",
    bouncer: "no input changed in any visit",
  };
  const pColor = (k) => `var(--p-${k})`;
  const pchip = (k, txt) => `<span class="pchip"><i style="background:${pColor(k)}"></i>${txt || PERSONAS[k].label}</span>`;

  const CONFIG_TYPES = new Set(["gpu_add", "gpu_qty", "gpu_remove", "gpu_cost", "compare_gpu", "model_add", "model_rejected", "model_remove", "model_prec",
    "model_count", "model_tune", "mode", "workload", "settings", "projection", "use_case_add", "use_case_edit", "compare_duplicate", "scenario_import",
    "export_report", "scenario_export", "use_case_export", "api_call"]);
  const EXPORT_TYPES = new Set(["export_report", "scenario_export", "use_case_export"]);

  // ---------- catalog ----------
  let CAT = { gpus: {}, models: {} };
  const GPU = (k) => CAT.gpus[k] || { name: k || "unknown GPU", cls: "datacenter", vram: 0, vendor: "" };
  const MODEL = (k) => CAT.models[k] || { name: k || "unknown model", kind: "llm", b: 0 };
  const gpuName = (k) => GPU(k).name;
  const shortGpu = (k) => gpuName(k).replace(/ (Blackwell )?(Server Edition|Workstation)/, (m) => (m.includes("Server") ? " Server" : " WS")).replace(/ HGX\/DGX| SXM\d?| PCIe/g, "");
  const modelName = (k) => MODEL(k).name;
  const shortModel = (k) => modelName(k).replace(/ \+ \d+B Engram| \+\d+B N-gram| \d{4}(?= )| \d{2}-\d{4}$/g, "");
  const sizeClass = (b) => (b < 4 ? "tiny <4B" : b < 16 ? "small 4–16B" : b < 80 ? "mid 16–80B" : b < 400 ? "large 80–400B" : "frontier 400B+");
  const SIZE_CLASSES = ["tiny <4B", "small 4–16B", "mid 16–80B", "large 80–400B", "frontier 400B+"];
  const PREC_LABEL = { bf16: "BF16", fp8: "FP8", nvfp4: "NVFP4", mxfp4: "MXFP4" };
  const precLabel = (p) => PREC_LABEL[p] || (p ? String(p).toUpperCase() : "");
  const GPU_CLASS_LABEL = { consumer: "Consumer GPU", unified: "Unified memory (Mac/Spark/Strix)", workstation: "Workstation", datacenter: "Datacenter" };

  // ---------- features & persona ----------
  function features(v) {
    const f = { events: 0, config: 0, gpus: {}, models: {}, frontier: 0, nonLlm: 0, llm: 0, compare: 0, econ: 0, useCases: 0, exports: 0, cost: 0, dist: 0,
      modes: new Set(), api: 0, apiErr: 0, rejected: [], noGpu: 0, oom: 0, recovered: 0, sessions: v.sessions.length, durMs: 0, classes: {}, maxVram: 0, firstModelMs: null, resets: 0 };
    for (const s of v.sessions) {
      f.durMs += s.end - s.start;
      let lastFail = false;
      for (const e of s.events) {
        f.events++;
        if (CONFIG_TYPES.has(e.type)) f.config++;
        if (EXPORT_TYPES.has(e.type)) f.exports++;
        switch (e.type) {
          case "gpu_add": case "gpu_qty": {
            if (!e.gpu) break;
            const g = GPU(e.gpu);
            f.gpus[e.gpu] = Math.max(f.gpus[e.gpu] || 0, e.qty || 0);
            f.classes[g.cls] = (f.classes[g.cls] || 0) + 1;
            f.maxVram = Math.max(f.maxVram, g.vram * (e.qty || 0));
            break;
          }
          case "model_add": {
            if (!e.model) break;
            const m = MODEL(e.model);
            f.models[e.model] = (f.models[e.model] || 0) + 1;
            if (m.b >= 400) f.frontier++;
            if (m.kind === "llm" || m.kind === "decision") f.llm++; else f.nonLlm++;
            if (f.firstModelMs == null) f.firstModelMs = e.t - s.start;
            break;
          }
          case "model_rejected":
            if (e.reason === "no_gpu") f.noGpu++;
            else if (e.reason === "fits_nowhere") { f.rejected.push(e.model); f.oom++; }
            break;
          case "result": if (!e.fits) { f.oom++; lastFail = true; } else if (lastFail) { f.recovered++; lastFail = false; } break;
          case "compare_duplicate": case "compare_gpu": f.compare++; break;
          case "econ_view": f.econ++; break;
          case "use_case_add": case "use_case_edit": case "projection": case "use_cases_view": f.useCases++; break;
          case "gpu_cost": f.cost++; break;
          case "workload": case "settings": case "model_tune": f.dist++; break;
          case "mode": if (e.mode) f.modes.add(e.mode); break;
          case "api_call":
            f.api++;
            if (e.status >= 400) f.apiErr++;
            if (e.model) {
              f.models[e.model] = (f.models[e.model] || 0) + 1;
              if (f.firstModelMs == null) f.firstModelMs = e.t - s.start;
            }
            if (e.gpu) { f.gpus[e.gpu] = e.qty || 1; const g = GPU(e.gpu); f.classes[g.cls] = (f.classes[g.cls] || 0) + 1; }
            break;
          case "session_reset": f.resets++; break;
        }
      }
    }
    f.nModels = Object.keys(f.models).length;
    const cls = f.classes;
    f.domClass = Object.keys(cls).sort((a, b) => cls[b] - cls[a])[0] || null;
    return f;
  }

  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  function classify(v, f) {
    if (v.internal) return { key: "internal", why: ["Signed-in admin browser"] };
    if (v.device === "agent" || f.api > 0) return { key: "agent", why: [plural(f.api, "API call"), `UA ${v.ua || "unknown"}`] };
    if (f.config === 0) return { key: "bouncer", why: ["No input changed", f.econ ? "only read the economics page" : f.useCases ? "only read the use-case library" : "left on the default scenario"] };
    const why = [];
    const speechModes = [...f.modes].filter((m) => m === "asrquality" || m === "embedquality");
    if (speechModes.length || (f.nonLlm > 0 && f.nonLlm >= f.llm)) {
      if (speechModes.length) why.push(`used ${speechModes.map((m) => (m === "asrquality" ? "ASR" : "embedding")).join(" + ")} mode`);
      if (f.nonLlm) why.push(plural(f.nonLlm, "speech/embedding model"));
      return { key: "speech", why };
    }
    if (f.useCases >= 1 || v.entry === "/use-cases") {
      why.push(f.useCases ? plural(f.useCases, "use-case action") : "entered on /use-cases");
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
      why.push(plural(f.nModels, "distinct model"));
      if (f.frontier) why.push(`${f.frontier} frontier (400B+) adds`);
      if (f.dist) why.push("tuned workload / parallelism");
      return { key: "evaluator", why };
    }
    if (dc && f.nModels <= 3) {
      return { key: "buyer", why: [`${f.domClass} hardware (up to ${Math.round(f.maxVram)} GB)`, f.nModels ? plural(f.nModels, "model") : "no model added yet"] };
    }
    if (f.domClass === "consumer" || f.domClass === "unified") {
      why.push(`${f.domClass === "unified" ? "unified-memory box" : "consumer GPU"} (${Math.round(f.maxVram)} GB)`);
      if (f.oom) why.push(plural(f.oom, "doesn’t-fit result"));
      return { key: "homelab", why };
    }
    return { key: "other", why: [f.domClass ? `${f.domClass} hardware, ${plural(f.nModels, "model")}` : "configured, no rule matched"] };
  }

  // ---------- formatting ----------
  const nf = new Intl.NumberFormat("en-US");
  const fmt = (n) => nf.format(Math.round(n || 0));
  const pct = (n, d = 0) => `${((n || 0) * 100).toFixed(d)}%`;
  const flag = (iso) => (iso && /^[A-Z]{2}$/.test(iso) ? String.fromCodePoint(...[...iso].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)) : "🌐");
  let regionNames = null;
  try { regionNames = new Intl.DisplayNames(["en"], { type: "region" }); } catch (e) { regionNames = null; }
  const countryName = (iso) => { if (!iso) return "Unknown"; try { return (regionNames && regionNames.of(iso)) || iso; } catch (e) { return iso; } };
  const dfmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short" });
  const dtfmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  const date = (t) => dfmt.format(new Date(t));
  const datetime = (t) => dtfmt.format(new Date(t));
  let NOW = Date.now();
  function ago(t) {
    const s = (NOW - t) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  }
  function dur(ms) {
    const s = Math.max(0, Math.round((ms || 0) / 1000));
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  }
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Visitor-local time: Cloudflare timezone when present, else the country's main zone, else UTC.
  const COUNTRY_TZ = { US: "America/Chicago", CA: "America/Toronto", MX: "America/Mexico_City", BR: "America/Sao_Paulo", AR: "America/Argentina/Buenos_Aires",
    CL: "America/Santiago", CO: "America/Bogota", GB: "Europe/London", IE: "Europe/Dublin", FR: "Europe/Paris", DE: "Europe/Berlin", NL: "Europe/Amsterdam",
    BE: "Europe/Brussels", ES: "Europe/Madrid", PT: "Europe/Lisbon", IT: "Europe/Rome", CH: "Europe/Zurich", AT: "Europe/Vienna", PL: "Europe/Warsaw",
    CZ: "Europe/Prague", SE: "Europe/Stockholm", NO: "Europe/Oslo", DK: "Europe/Copenhagen", FI: "Europe/Helsinki", UA: "Europe/Kyiv", RO: "Europe/Bucharest",
    GR: "Europe/Athens", TR: "Europe/Istanbul", IL: "Asia/Jerusalem", AE: "Asia/Dubai", SA: "Asia/Riyadh", EG: "Africa/Cairo", NG: "Africa/Lagos",
    ZA: "Africa/Johannesburg", KE: "Africa/Nairobi", IN: "Asia/Kolkata", PK: "Asia/Karachi", BD: "Asia/Dhaka", CN: "Asia/Shanghai", HK: "Asia/Hong_Kong",
    TW: "Asia/Taipei", JP: "Asia/Tokyo", KR: "Asia/Seoul", SG: "Asia/Singapore", MY: "Asia/Kuala_Lumpur", TH: "Asia/Bangkok", VN: "Asia/Ho_Chi_Minh",
    ID: "Asia/Jakarta", PH: "Asia/Manila", AU: "Australia/Sydney", NZ: "Pacific/Auckland", RU: "Europe/Moscow" };
  const tzFmts = new Map();
  function tzOf(v) { return v.tz || COUNTRY_TZ[v.country] || "UTC"; }
  function localParts(t, v) {
    const tz = tzOf(v);
    let f = tzFmts.get(tz);
    if (!f) {
      try { f = new Intl.DateTimeFormat("en-US", { timeZone: tz, weekday: "short", hour: "numeric", minute: "2-digit", hourCycle: "h23" }); }
      catch (e) { f = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", weekday: "short", hour: "numeric", minute: "2-digit", hourCycle: "h23" }); }
      tzFmts.set(tz, f);
    }
    const parts = Object.fromEntries(f.formatToParts(new Date(t)).map((p) => [p.type, p.value]));
    const wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].indexOf(parts.weekday);
    return { wd, h: Number(parts.hour) % 24, hm: `${String(Number(parts.hour) % 24).padStart(2, "0")}:${parts.minute}`, tz };
  }

  // ---------- filter state ----------
  const store = {
    get(k, d) { try { const v = localStorage.getItem("adm:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("adm:" + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
  };
  const filter = { range: store.get("range", 30), internal: store.get("internal", false), agents: store.get("agents", true) };
  const A = { visitors: [], meta: {}, catalog: CAT };

  function visible() {
    return A.visitors.filter((v) => (filter.internal || !v.internal) && (filter.agents || v.persona !== "agent"));
  }
  const engaged = (v) => v.persona !== "bouncer";

  function filterBar(el, onChange, extra) {
    el.innerHTML = `
      <span class="fl-label">Range</span>
      <span class="seg" data-k="range">${[7, 30, 90].map((d) => `<button type="button" data-v="${d}" class="${filter.range === d ? "on" : ""}">${d}d</button>`).join("")}</span>
      <label class="chk"><input type="checkbox" data-k="agents" ${filter.agents ? "checked" : ""}> AI agents</label>
      <label class="chk"><input type="checkbox" data-k="internal" ${filter.internal ? "checked" : ""}> Internal traffic (you)</label>
      ${extra || ""}
      <span class="spacer"></span>
      <span class="muted" id="fl-count"></span>`;
    el.querySelectorAll(".seg[data-k=range] button").forEach((b) => b.addEventListener("click", () => {
      filter.range = +b.dataset.v; store.set("range", filter.range);
      el.querySelectorAll(".seg[data-k=range] button").forEach((x) => x.classList.toggle("on", x === b));
      load().then(onChange);
    }));
    el.querySelectorAll("input[type=checkbox][data-k]").forEach((c) => c.addEventListener("change", () => {
      filter[c.dataset.k] = c.checked; store.set(c.dataset.k, c.checked); onChange();
    }));
  }
  function setCount(n) { const c = document.getElementById("fl-count"); if (c) c.textContent = `${fmt(n)} visitors in view`; }

  // ---------- data ----------
  function prepare(payload) {
    CAT = payload.catalog || { gpus: {}, models: {} };
    A.catalog = CAT;
    A.meta = payload.meta || {};
    NOW = A.meta.now || Date.now();
    A.visitors = (payload.visitors || []).filter((v) => v.sessions && v.sessions.length);
    for (const v of A.visitors) {
      v.f = features(v);
      const c = classify(v, v.f);
      v.persona = c.key;
      v.why = c.why;
    }
  }
  function load() {
    const main = document.getElementById("content");
    return fetch(`/admin/data.json?days=${filter.range}`, { headers: { Accept: "application/json" }, credentials: "same-origin" })
      .then((r) => { if (r.status === 401) { location.reload(); throw new Error("login"); } return r.json(); })
      .then(prepare)
      .catch((err) => { if (main && err.message !== "login") main.innerHTML = `<div class="empty-state">Could not load visitor data: ${esc(err.message)}</div>`; throw err; });
  }
  function emptyState() {
    const st = A.meta.status || {};
    return `<div class="empty-state"><b>No visitor data in this range yet.</b><br>Tracking records new visits from now on.
      ${st.events ? `The store holds ${fmt(st.events)} events overall; try a longer range.` : ""}</div>`;
  }
  function start(render) {
    load().then(() => {
      const loading = document.getElementById("loading");
      if (loading) loading.remove();
      const content = document.getElementById("content");
      if (content) content.hidden = false;
      render();
    }).catch(() => {});
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
  document.addEventListener("mousemove", (ev) => {
    const t = ev.target.closest && ev.target.closest("[data-tip]");
    if (t) tip(t.getAttribute("data-tip"), ev); else tip(null);
  });

  // ---------- theme ----------
  (function initTheme() {
    const saved = store.get("theme", null);
    if (saved) document.documentElement.dataset.theme = saved;
    document.addEventListener("DOMContentLoaded", () => {
      const btn = document.getElementById("theme-btn");
      if (!btn) return;
      btn.addEventListener("click", () => {
        const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        const next = cur === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next; store.set("theme", next);
      });
    });
  })();

  // ---------- small chart helpers ----------
  function hbars(rows, opts = {}) {
    if (!rows.length) return `<div class="muted">No data yet.</div>`;
    const max = opts.max || Math.max(1, ...rows.map((r) => r.value));
    const fmtV = opts.fmt || fmt;
    return `<div style="display:grid;grid-template-columns:minmax(90px,${opts.labelW || "38%"}) 1fr auto;gap:5px 10px;align-items:center;font-size:12px">` +
      rows.map((r) => `
        <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(r.plain || "")}">${r.label}</div>
        <div class="bar-track" data-tip="${esc(r.tip || `<b>${esc(r.plain || "")}</b><div>${fmtV(r.value)}</div>`)}" style="height:14px;display:flex;align-items:center">
          <div class="bar" style="width:${Math.max(0.5, (r.value / max) * 100)}%;background:${r.color || "var(--accent)"};height:${opts.h || 10}px"></div>
        </div>
        <div class="num" style="text-align:right;color:var(--text-2)">${fmtV(r.value)}${r.sub ? ` <span class="muted">${r.sub}</span>` : ""}</div>`).join("") + `</div>`;
  }
  function countBy(arr, keyFn) {
    const m = new Map();
    for (const x of arr) { const k = keyFn(x); if (k == null) continue; m.set(k, (m.get(k) || 0) + 1); }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }
  function ticks(max, n) {
    if (max <= 0) return [0];
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((k) => k * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
    return out;
  }

  window.UI = { A, PERSONAS, PERSONA_ORDER, RULES, pColor, pchip, GPU, MODEL, gpuName, shortGpu, modelName, shortModel, sizeClass, SIZE_CLASSES, GPU_CLASS_LABEL,
    precLabel, fmt, pct, flag, countryName, date, datetime, ago, dur, esc, localParts, tzOf, filter, visible, engaged, filterBar, setCount, start, load, emptyState,
    tip, hbars, countBy, ticks, store, now: () => NOW };
})();
