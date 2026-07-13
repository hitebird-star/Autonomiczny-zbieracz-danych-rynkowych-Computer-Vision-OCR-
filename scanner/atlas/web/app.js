"use strict";
// Atlas rynku — frontend (canvas 2D + izometria 3D, pan/zoom, hover, ustawienia, SSE).

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

const view = { mode: "2d", zoom: 1, panX: 0, panY: 0, scale: 1, cx: 0, cy: 0 };
let data = { boundary: [], shops: [], live: [], bounds: null, stats: {} };
let hover = null;

const COLORS = { atlas: "#37d67a", game: "#d6a93a", live: "#3aa0ff" };

// ---- projekcja world(game) -> plaszczyzna widoku ----
function project(x, y) {
  if (view.mode === "iso") return { vx: x - y, vy: (x + y) * 0.5 };
  return { vx: x, vy: y };
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  fitToData(false);
  draw();
}

function allPoints() {
  const pts = data.shops.map(s => [s.x, s.y]);
  for (const p of data.boundary) pts.push(p);
  for (const p of data.live) pts.push([p.x, p.y]);
  return pts;
}

function fitToData(resetPan) {
  const pts = allPoints();
  if (!pts.length) return;
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const [x, y] of pts) {
    const p = project(x, y);
    minx = Math.min(minx, p.vx); maxx = Math.max(maxx, p.vx);
    miny = Math.min(miny, p.vy); maxy = Math.max(maxy, p.vy);
  }
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const spanX = Math.max(maxx - minx, 1e-6), spanY = Math.max(maxy - miny, 1e-6);
  view.scale = Math.min(W / spanX, H / spanY) * 0.86;
  view.cx = (minx + maxx) / 2;
  view.cy = (miny + maxy) / 2;
  if (resetPan) { view.panX = 0; view.panY = 0; view.zoom = 1; }
}

function toCanvas(x, y) {
  const p = project(x, y);
  const W = canvas.clientWidth, H = canvas.clientHeight;
  return {
    sx: (p.vx - view.cx) * view.scale * view.zoom + W / 2 + view.panX,
    sy: (p.vy - view.cy) * view.scale * view.zoom + H / 2 + view.panY,
  };
}

// ---- rysowanie ----
function draw() {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);

  // granica
  if (data.boundary.length) {
    ctx.beginPath();
    data.boundary.forEach(([x, y], i) => {
      const c = toCanvas(x, y);
      i ? ctx.lineTo(c.sx, c.sy) : ctx.moveTo(c.sx, c.sy);
    });
    ctx.closePath();
    ctx.fillStyle = "#4f8cff14";
    ctx.strokeStyle = "#4f8cff88";
    ctx.lineWidth = 1.5;
    ctx.fill(); ctx.stroke();
  }

  // sklepy
  const iso = view.mode === "iso";
  for (const s of data.shops) {
    const c = toCanvas(s.x, s.y);
    const col = COLORS[s.source] || "#aaa";
    if (iso) drawStall(c.sx, c.sy, col, s === hover);
    else drawDot(c.sx, c.sy, col, s === hover);
  }

  // punkty live
  ctx.fillStyle = COLORS.live;
  for (const p of data.live) {
    const c = toCanvas(p.x, p.y);
    ctx.beginPath(); ctx.arc(c.sx, c.sy, 3, 0, 7); ctx.fill();
  }
}

function drawDot(sx, sy, color, hot) {
  ctx.beginPath();
  ctx.arc(sx, sy, hot ? 6 : 4, 0, 7);
  ctx.fillStyle = color; ctx.fill();
  if (hot) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
}

function drawStall(sx, sy, color, hot) {
  const w = hot ? 8 : 6, h = w * 0.55, ht = w * 1.1;
  // ścianki (3D)
  ctx.fillStyle = shade(color, -30);
  ctx.beginPath();
  ctx.moveTo(sx - w, sy); ctx.lineTo(sx, sy + h); ctx.lineTo(sx, sy + h - ht); ctx.lineTo(sx - w, sy - ht); ctx.closePath(); ctx.fill();
  ctx.fillStyle = shade(color, -60);
  ctx.beginPath();
  ctx.moveTo(sx + w, sy); ctx.lineTo(sx, sy + h); ctx.lineTo(sx, sy + h - ht); ctx.lineTo(sx + w, sy - ht); ctx.closePath(); ctx.fill();
  // dach (romb)
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(sx, sy - ht); ctx.lineTo(sx + w, sy - ht + h); ctx.lineTo(sx, sy - ht + 2 * h); ctx.lineTo(sx - w, sy - ht + h); ctx.closePath(); ctx.fill();
  if (hot) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.2; ctx.stroke(); }
}

function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.max(0, Math.min(255, (n >> 16) + amt));
  const g = Math.max(0, Math.min(255, ((n >> 8) & 255) + amt));
  const b = Math.max(0, Math.min(255, (n & 255) + amt));
  return `rgb(${r},${g},${b})`;
}

// ---- interakcja ----
function findHover(mx, my) {
  let best = null, bestD = 14 * 14;
  for (const s of data.shops) {
    const c = toCanvas(s.x, s.y);
    const d = (c.sx - mx) ** 2 + (c.sy - my) ** 2;
    if (d < bestD) { bestD = d; best = s; }
  }
  return best;
}

canvas.addEventListener("mousemove", (e) => {
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  if (dragging) {
    view.panX += mx - lastX; view.panY += my - lastY;
    lastX = mx; lastY = my; draw(); return;
  }
  const h = findHover(mx, my);
  if (h !== hover) { hover = h; draw(); }
  if (h) {
    tooltip.hidden = false;
    tooltip.style.left = (mx + 14) + "px";
    tooltip.style.top = (my + 14) + "px";
    tooltip.innerHTML =
      `<b>${h.seller || "(brak sprzedawcy)"}</b><br>` +
      `poz: ${h.x.toFixed(1)}, ${h.y.toFixed(1)} <span class="muted">(${h.source})</span><br>` +
      `pewność: ${h.confidence} · fp: ${String(h.fingerprint).slice(0, 8)}`;
  } else tooltip.hidden = true;
});

let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener("mousedown", (e) => {
  const r = canvas.getBoundingClientRect();
  dragging = true; lastX = e.clientX - r.left; lastY = e.clientY - r.top;
});
window.addEventListener("mouseup", () => dragging = false);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  view.zoom = Math.max(0.2, Math.min(20, view.zoom * f));
  draw();
}, { passive: false });

// ---- statystyki + widok ----
function updateStats() {
  const s = data.stats || {};
  set("stat-total", `sklepy: ${s.total ?? "–"}`);
  set("stat-located", `metryczne: ${s.located ?? 0}`);
  set("stat-stand", `na postoju: ${s.on_stand ?? 0}`);
  set("stat-sellers", `sprzedawcy: ${s.sellers ?? 0}`);
  set("stat-calib", `kalibracja: ${data.has_calibration ? "jest" : "brak"}`);
}
const set = (id, txt) => document.getElementById(id).textContent = txt;

document.querySelectorAll(".viewbtn").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".viewbtn").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  view.mode = b.dataset.view;
  fitToData(true); draw();
}));
document.getElementById("btn-reload").addEventListener("click", async () => {
  await fetch("/api/reload", { method: "POST" }); loadAtlas();
});

// ---- ustawienia ----
const settingsEl = document.getElementById("settings");
document.getElementById("btn-settings").addEventListener("click", () => {
  settingsEl.hidden = !settingsEl.hidden;
  if (!settingsEl.hidden) loadSettings();
});
document.getElementById("btn-close-settings").addEventListener("click", () => settingsEl.hidden = true);

async function loadSettings() {
  const cfg = await (await fetch("/api/config")).json();
  const form = document.getElementById("settings-form");
  form.innerHTML = "";
  for (const [k, v] of Object.entries(cfg)) {
    const val = Array.isArray(v) ? v.join(", ") : v;
    const type = typeof v === "number" ? "number" : "text";
    form.insertAdjacentHTML("beforeend",
      `<div class="field"><label>${k}</label>` +
      `<input data-key="${k}" data-kind="${Array.isArray(v) ? "arr" : typeof v}" type="${type}" value="${val}" step="any"></div>`);
  }
}
document.getElementById("btn-save").addEventListener("click", async () => {
  const patch = {};
  document.querySelectorAll("#settings-form input").forEach(inp => {
    const k = inp.dataset.key, kind = inp.dataset.kind;
    if (kind === "number") patch[k] = parseFloat(inp.value);
    else if (kind === "arr") patch[k] = inp.value.split(",").map(s => s.trim()).filter(Boolean);
    else patch[k] = inp.value;
  });
  const res = await fetch("/api/config", { method: "POST", body: JSON.stringify(patch) });
  const msg = document.getElementById("settings-msg");
  msg.textContent = res.ok ? "Zapisano ✓" : "Błąd zapisu";
  loadAtlas();
});

// ---- dane ----
function setData(payload) {
  data = payload;
  updateStats();
  draw();
}
async function loadAtlas() {
  try {
    const p = await (await fetch("/api/atlas")).json();
    const first = data.shops.length === 0;
    setData(p);
    if (first) fitToData(true), draw();
  } catch (e) { console.error("atlas load", e); }
}

// SSE live
function connectSSE() {
  try {
    const es = new EventSource("/api/events");
    es.onmessage = (ev) => { try { setData(JSON.parse(ev.data)); } catch {} };
    es.onerror = () => { es.close(); setTimeout(connectSSE, 3000); };
  } catch (e) { console.warn("SSE off", e); }
}

window.addEventListener("resize", resize);
resize();
loadAtlas().then(() => { fitToData(true); draw(); connectSSE(); });
