// Interactive charts for the website: EDA counts, light timeline, event timeline (click = seek), risk curve.
const VIDEOS = ["C3897", "C3902", "C3905", "C3896"];
const LABELLED = { C3897: "daylight", C3902: "dusk", C3905: "dusk, jam at the end", C3896: "evening (not labelled)" };
const EV_COLORS = {
  jaywalking: "#22c55e", stopped_vehicle: "#eab308", failure_to_yield: "#14b8a6", congestion: "#3b82f6",
  stop_line: "#fb7185", solid_line_crossing: "#f59e0b", illegal_turn: "#6366f1", near_miss: "#f97316",
};
const CLS_COLORS = { car: "#3b82f6", person: "#f97316", bus: "#22c55e", truck: "#a855f7", motorcycle: "#ef4444", bicycle: "#0ea5e9" };
const NS = "http://www.w3.org/2000/svg";
const $ = (s) => document.querySelector(s);
const fmt = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function picker(id, onPick) {
  const box = $(id);
  VIDEOS.forEach((v, i) => {
    const b = document.createElement("button");
    b.textContent = `${v} · ${LABELLED[v]}`;
    b.onclick = () => { box.querySelectorAll("button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); onPick(v); };
    box.appendChild(b);
    if (i === 0) b.classList.add("on");
  });
  onPick(VIDEOS[0]);
}

function lineChart(target, xs, series, opts = {}) {
  const W = 1000, H = opts.h || 260, L = 44, B = 26, T = 10, R = 10;
  const ymax = opts.ymax ?? Math.max(1, ...Object.values(series).flat());
  const xmax = Math.max(...xs, 1);
  const box = $(target); box.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  const X = (x) => L + (W - L - R) * x / xmax, Y = (y) => T + (H - T - B) * (1 - y / ymax);
  for (let k = 0; k <= 4; k++) {
    const y = ymax * k / 4;
    el("line", { x1: L, x2: W - R, y1: Y(y), y2: Y(y), stroke: "#e2e8f0" }, svg);
    el("text", { x: L - 6, y: Y(y) + 4, "text-anchor": "end", "font-size": 11, fill: "#64748b" }, svg).textContent = +y.toFixed(2);
  }
  const step = xmax > 200 ? 60 : xmax > 60 ? 20 : 10;
  for (let x = 0; x <= xmax; x += step)
    el("text", { x: X(x), y: H - 6, "text-anchor": "middle", "font-size": 11, fill: "#64748b" }, svg).textContent = fmt(x).replace(/\.\d$/, "");
  if (opts.threshold != null)
    el("line", { x1: L, x2: W - R, y1: Y(opts.threshold), y2: Y(opts.threshold), stroke: "#ef4444", "stroke-dasharray": "5 4" }, svg);
  const legend = document.createElement("div"); legend.className = "legend";
  for (const [name, ys] of Object.entries(series)) {
    const d = ys.map((y, i) => `${i ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(y).toFixed(1)}`).join("");
    el("path", { d, fill: "none", stroke: opts.colors?.[name] || "#2563eb", "stroke-width": 1.8 }, svg);
    legend.innerHTML += `<span><i style="background:${opts.colors?.[name] || "#2563eb"}"></i>${name}</span>`;
  }
  if (!opts.noLegend) box.appendChild(legend);
  return { svg, X, Y };
}

// ---------- EDA ----------
fetch("data/eda.json").then((r) => r.json()).then((eda) => {
  const tb = $("#facts tbody");
  for (const v of VIDEOS) {
    const d = eda[v]; if (!d) continue;
    const t = d.tracks;
    tb.innerHTML += `<tr><td>${v}</td><td>${d.resolution}</td><td>${d.fps}</td><td>${fmt(d.duration)}</td>
      <td>${LABELLED[v].split(" (")[0].split(",")[0]} (mean brightness ${d.brightness})</td>
      <td>${t.car || 0} / ${t.person || 0} / ${t.bus || 0} / ${t.truck || 0}</td></tr>`;
  }
  picker("#edaPick", (v) => {
    const d = eda[v];
    const keep = Object.fromEntries(Object.entries(d.counts).filter(([k]) => ["car", "person", "bus", "truck", "motorcycle"].includes(k)));
    lineChart("#countsChart", d.bins, keep, { colors: CLS_COLORS });
    const box = $("#lightChart"); box.innerHTML = "";
    const svg = el("svg", { viewBox: "0 0 1000 40" }, box);
    const L = d.lights, dur = d.duration;
    for (let i = 0; i < L.length - 1; i++) {
      const c = { red: "#ef4444", green: "#22c55e", amber: "#f59e0b", yellow: "#f59e0b" }[L[i][1]] || "#cbd5e1";
      el("rect", { x: 1000 * L[i][0] / dur, y: 8, width: Math.max(1, 1000 * (L[i + 1][0] - L[i][0]) / dur) + 0.5, height: 24, fill: c }, svg);
    }
    $("#heatImg").src = `img/heat_${v}.jpg`;
    $("#trajImg").src = `img/traj_${v}.jpg`;
  });
});

// ---------- Results ----------
let labels = {};
fetch("data/dev_labels.json").then((r) => r.json()).then((l) => { labels = l; picker("#resPick", showResult); });

function showResult(v) {
  const player = $("#player");
  player.src = `videos/${v}.mp4`;
  fetch(`data/${v}.json`).then((r) => r.json()).then((d) => {
    drawTimeline(d, labels[v] || null, player);
    const rt = d.risk.map((r) => r[0]), rv = d.risk.map((r) => r[1]);
    const c = lineChart("#riskChart", rt, { risk: rv }, { ymax: 1, threshold: 0.5, h: 180, colors: { risk: "#ef4444" }, noLegend: true });
    const tb = $("#evTable tbody"); tb.innerHTML = "";
    d.events.forEach(([s, e, l]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${fmt(s)}</td><td>${fmt(e)}</td><td>${l}</td>`;
      tr.style.cursor = "pointer"; tr.onclick = () => { player.currentTime = s; player.play(); };
      tb.appendChild(tr);
    });
  });
}

function drawTimeline(d, gt, player, target = "#timeline") {
  const box = $(target); box.innerHTML = "";
  const classes = [...new Set([...d.events.map((e) => e[2]), ...(gt || []).map((e) => e[2])])].sort();
  const W = 1000, L = 150, R = 10, row = gt ? 30 : 22, H = 24 + row * classes.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  const X = (t) => L + (W - L - R) * t / d.duration;
  for (let t = 0; t <= d.duration; t += 30) {
    el("line", { x1: X(t), x2: X(t), y1: 16, y2: H, stroke: "#eef2f7" }, svg);
    el("text", { x: X(t), y: 12, "text-anchor": "middle", "font-size": 10, fill: "#64748b" }, svg).textContent = fmt(t).replace(/\.\d$/, "");
  }
  classes.forEach((c, i) => {
    const y = 20 + i * row;
    el("text", { x: 4, y: y + 14, "font-size": 12, fill: "#0f172a" }, svg).textContent = c;
    d.events.filter((e) => e[2] === c).forEach(([s, e]) => {
      const r = el("rect", { x: X(s), y: y + 2, width: Math.max(2, X(e) - X(s)), height: gt ? 13 : 16, rx: 3, fill: EV_COLORS[c] || "#94a3b8", class: "bar" }, svg);
      el("title", {}, r).textContent = `${c}: ${fmt(s)}–${fmt(e)} (click to play)`;
      r.addEventListener("click", () => { player.currentTime = s; player.play(); });
    });
    (gt || []).filter((e) => e[2] === c).forEach(([s, e]) => {
      const r = el("rect", { x: X(s), y: y + 17, width: Math.max(2, X(e) - X(s)), height: 9, rx: 2, fill: "none", stroke: EV_COLORS[c] || "#64748b", "stroke-width": 1.5, class: "bar" }, svg);
      el("title", {}, r).textContent = `our label — ${c}: ${fmt(s)}–${fmt(e)}`;
      r.addEventListener("click", () => { player.currentTime = s; player.play(); });
    });
  });
  const head = el("line", { x1: L, x2: L, y1: 16, y2: H, stroke: "#0f172a", "stroke-width": 1.5 }, svg);
  player.ontimeupdate = () => { head.setAttribute("x1", X(player.currentTime)); head.setAttribute("x2", X(player.currentTime)); };
  const legend = document.createElement("div"); legend.className = "legend";
  legend.innerHTML = gt ? "<span>filled bars = our system · outlined bars = our labels · click any bar to jump there</span>"
                        : "<span>click any bar to jump there (this video is not labelled)</span>";
  box.appendChild(legend);
}

// ---------- examples ----------
const EX = { jaywalking: "a pedestrian on the carriageway outside a crossing", stopped_vehicle: "cars parked at the far kerb for the whole video",
  failure_to_yield: "a car drives through the crossing while people walk on it", congestion: "the approach jammed and spilling into the junction",
  stop_line: "a car stopped past the stop line on red", solid_line_crossing: "a vehicle across a solid lane divider",
  illegal_turn: "a left turn from the wrong lane" };
const g = $("#examples");
for (const [k, txt] of Object.entries(EX))
  g.innerHTML += `<figure><img src="img/examples/${k}.jpg" alt="${k}" loading="lazy"><figcaption><b>${k}</b> — ${txt}</figcaption></figure>`;

// ---------- live demo ----------
const fileIn = $("#demoFile"), runBtn = $("#demoRun");
fileIn.onchange = () => { runBtn.disabled = !fileIn.files.length; };
function setStatus(frac, msg) {
  $("#demoStatus").hidden = false;
  $("#demoBar").style.width = `${Math.round(100 * frac)}%`;
  $("#demoMsg").textContent = msg;
}
runBtn.onclick = () => {
  const f = fileIn.files[0];
  if (!f) return;
  if (f.size > 800e6) { setStatus(0, "File is larger than 800 MB."); return; }
  runBtn.disabled = true; $("#demoOut").hidden = true;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "api/jobs");
  xhr.setRequestHeader("X-Filename", f.name);
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) setStatus(0.1 * e.loaded / e.total, `Uploading ${(e.loaded / 1e6).toFixed(0)} / ${(e.total / 1e6).toFixed(0)} MB`); };
  xhr.onload = () => {
    let r = {}; try { r = JSON.parse(xhr.responseText); } catch (e) {}
    if (xhr.status !== 200) { setStatus(0, r.error || "Upload failed."); runBtn.disabled = false; return; }
    poll(r.id);
  };
  xhr.onerror = () => { setStatus(0, "Upload failed — check your connection."); runBtn.disabled = false; };
  xhr.send(f);
};
function poll(id) {
  fetch(`api/jobs/${id}`).then((r) => r.json()).then((j) => {
    if (j.error) { setStatus(0, j.error); runBtn.disabled = false; return; }
    if (j.state === "queued") setStatus(0.1, `In the queue (${j.position} ahead of you) — one video is processed at a time.`);
    else if (j.state === "running") setStatus(0.1 + 0.9 * j.progress, j.message);
    if (j.state === "done") { setStatus(1, "Done."); showDemo(j); runBtn.disabled = false; return; }
    if (j.state === "error") { setStatus(0, j.message); runBtn.disabled = false; return; }
    setTimeout(() => poll(id), 2000);
  }).catch(() => setTimeout(() => poll(id), 4000));
}
function showDemo(j) {
  const d = j.result, player = $("#demoPlayer");
  $("#demoOut").hidden = false;
  player.src = j.video || "";
  $("#demoSummary").innerHTML = `<b>${d.events.length} events</b> in ${d.duration.toFixed(0)} s of video · ` +
    `${d.processed_frames} frames analysed in ${d.seconds.toFixed(0)} s.` + (d.notes.length ? `<br><small>${d.notes.join(" ")}</small>` : "");
  drawTimeline(d, null, player, "#demoTimeline");
  if (d.risk.length) lineChart("#demoRisk", d.risk.map((r) => r[0]), { risk: d.risk.map((r) => r[1]) },
    { ymax: 1, threshold: 0.5, h: 180, colors: { risk: "#ef4444" }, noLegend: true });
  const tb = $("#demoTable tbody"); tb.innerHTML = "";
  d.events.forEach(([s, e, l]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${fmt(s)}</td><td>${fmt(e)}</td><td>${l}</td>`;
    tr.style.cursor = "pointer"; tr.onclick = () => { player.currentTime = s; player.play(); };
    tb.appendChild(tr);
  });
  $("#demoJson").href = j.json;
}
