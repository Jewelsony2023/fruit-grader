// Fruit Quality Grader — front-end logic (no framework, easy to read).
const $ = (id) => document.getElementById(id);
const GRADE_COLOUR = { A: "var(--A)", B: "var(--B)", C: "var(--C)", Reject: "var(--Reject)" };
let INFO = null, currentFile = null, currentModel = "multitask", lastResult = null;

// ---------- tabs ----------
document.querySelectorAll(".tab[data-view]").forEach((t) => t.onclick = () => {
  document.querySelectorAll(".tab[data-view]").forEach((x) => x.classList.toggle("active", x === t));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + t.dataset.view));
  if (t.dataset.view === "performance") loadReport();
});

// ---------- model toggle ----------
document.querySelectorAll("#model-toggle button").forEach((b) => b.onclick = () => {
  if (b.disabled) return;
  currentModel = b.dataset.model;
  document.querySelectorAll("#model-toggle button").forEach((x) => x.classList.toggle("on", x === b));
  if (currentFile) run();
});

// ---------- file input + drag & drop ----------
const drop = $("drop");
$("file").onchange = (e) => e.target.files[0] && setFile(e.target.files[0]);
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

function setFile(file, autoRun = false) {
  if (!file.type.startsWith("image/")) { status("That is not an image file.", true); return; }
  currentFile = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  $("drop-hint").hidden = true;
  $("run").disabled = !INFO || !Object.keys(INFO.models).length;
  status("");
  if (autoRun) run();
}
$("run").onclick = run;

function status(msg, err = false, busy = false) {
  $("status").innerHTML = (busy ? '<span class="spinner"></span>' : "") + msg;
  $("status").style.color = err ? "var(--Reject)" : "";
}

// ---------- predict ----------
async function run() {
  if (!currentFile) return;
  $("run").disabled = true;
  status("Analysing… (20 MC-Dropout passes + Grad-CAM)", false, true);
  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("model", currentModel);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    lastResult = data;
    render(data);
    status("Done.");
  } catch (e) {
    status("Error: " + e.message, true);
  } finally {
    $("run").disabled = false;
  }
}

const pct = (v, d = 1) => (v == null ? "–" : v.toFixed(d) + "%");

function bar(label, value, max, colour, text, top = false) {
  const w = Math.max(0, Math.min(100, (value / max) * 100));
  return `<div class="bar ${top ? "top" : ""}"><span>${label}</span>
    <div class="track"><div class="fill" style="width:${w}%;background:${colour}"></div></div>
    <span class="val">${text}</span></div>`;
}

function render(r) {
  $("empty").hidden = true;
  $("out").hidden = false;

  // grade card
  $("grade").textContent = r.grade;
  $("grade-card").style.background = GRADE_COLOUR[r.grade];
  $("conf").textContent = `Confidence ${(r.confidence * 100).toFixed(1)}%  ·  ${r.model} (${r.backbone})`;

  // severity
  const area = r.defect_area_pct.total;
  $("sev-area").textContent = area.toFixed(1);
  $("sev-fill").style.left = `calc(${Math.min(area, 100)}% - 2px)`;
  $("sev-head").textContent = r.severity_std != null
    ? `Severity regression head: ${r.severity_score.toFixed(1)} ± ${r.severity_std.toFixed(1)} (0–100)`
    : "";

  // review flag
  const rv = $("review");
  rv.hidden = false;
  if (r.needs_review) {
    rv.className = "review";
    rv.innerHTML = "<b>⚠ Needs manual review:</b> " + r.review_reasons.join("; ") + ".";
  } else {
    rv.className = "review ok";
    rv.innerHTML = "<b>✓ Confident prediction.</b> Classifier, defect area and MC-Dropout all agree.";
  }
  $("note").hidden = !r.note;
  $("note").textContent = r.note || "";

  // images
  $("im-input").src = r.images.input;
  $("im-overlay").src = r.images.defect_overlay;
  $("im-mask").src = r.images.mask;
  $("im-cam").src = r.images.gradcam;

  // defect breakdown
  const c = INFO.colours;
  const d = r.defect_area_pct;
  let html = "";
  if ("bruise" in d) {
    html += bar("Bruise", d.bruise, 50, c.bruise, pct(d.bruise));
    html += bar("Rot", d.rot, 50, c.rot, pct(d.rot));
    html += bar("Mechanical", d.mechanical, 50, c.mechanical, pct(d.mechanical));
  }
  html += bar("Total", d.total, 100, "var(--text)", pct(d.total), true);
  if (r.rule_grade) html += `<p class="muted small">Area-based rule grade: <b>${r.rule_grade}</b></p>`;
  $("breakdown").innerHTML = html;
  $("legend").innerHTML = ["healthy", "bruise", "rot", "mechanical"]
    .map((k) => `<span><i style="background:${c[k]}"></i>${k}</span>`).join("");

  // probabilities
  $("probs").innerHTML = INFO.grades.map((g) =>
    bar(g, r.grade_probs[g], 1, GRADE_COLOUR[g], pct(r.grade_probs[g] * 100), g === r.grade)).join("");
  $("uncert").innerHTML = `
    <div>Confidence<b>${(r.confidence * 100).toFixed(1)}%</b></div>
    <div>MC std<b>${r.mc_std.toFixed(3)}</b></div>
    <div>Entropy<b>${r.entropy.toFixed(2)}</b></div>`;
}

// ---------- lightbox ----------
document.querySelectorAll(".imgs img").forEach((im) => im.onclick = () => {
  $("lightbox-img").src = im.src; $("lightbox").hidden = false;
});
$("lightbox").onclick = () => ($("lightbox").hidden = true);

// ---------- download ----------
$("download").onclick = () => {
  if (!lastResult) return;
  const { images, ...rest } = lastResult;
  const blob = new Blob([JSON.stringify(rest, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (lastResult.filename || "result").replace(/\.[^.]+$/, "") + "_grade.json";
  a.click();
};

// ---------- samples ----------
async function loadSamples() {
  const list = await (await fetch("/api/samples")).json();
  $("samples").innerHTML = "";
  list.forEach((s) => {
    const b = document.createElement("button");
    b.title = `True grade ${s.grade}, severity ${s.severity.toFixed(1)}% (${s.source})`;
    b.innerHTML = `<img src="${s.url}" loading="lazy"><span>${s.grade}</span>`;
    b.onclick = async () => {
      const blob = await (await fetch(s.url)).blob();
      setFile(new File([blob], s.url.split("/").pop(), { type: blob.type || "image/jpeg" }), true);
    };
    $("samples").appendChild(b);
  });
  if (!list.length) $("samples").innerHTML = '<p class="muted small">No test split found.</p>';
}

// ---------- performance tab ----------
async function loadReport() {
  const r = await (await fetch("/api/report")).json();
  if (!r.metrics) { $("perf").innerHTML = "<p>No report yet. Run <code>python -m fruitgrade.evaluate</code>.</p>"; return; }
  const m = r.metrics;
  const rows = [
    ["Grade accuracy", (x) => x.classification.accuracy * 100, "%", true],
    ["Grade macro-F1", (x) => x.classification.macro_f1, "", true],
    ["Mean defect IoU", (x) => x.segmentation.mean_defect_iou, "", true],
    ["Mean defect Dice", (x) => x.segmentation.mean_defect_dice, "", true],
    ["Any-defect IoU", (x) => x.segmentation.any_defect_iou, "", true],
    ["Any-defect Dice", (x) => x.segmentation.any_defect_dice, "", true],
    ["Severity MAE (% points)", (x) => x.severity_mae_from_mask, "", false],
  ];
  let h = r.title ? `<p class="review"><b>${r.title}</b></p>` : "";
  h += `<table><tr><th>Metric</th>${m.map((x) => `<th>${x.kind}</th>`).join("")}</tr>`;
  rows.forEach(([name, fn, unit, higher]) => {
    const vals = m.map(fn);
    const best = higher ? Math.max(...vals) : Math.min(...vals);
    h += `<tr><td>${name}</td>${vals.map((v) =>
      `<td class="${m.length > 1 && v === best ? "win" : ""}">${v.toFixed(unit === "%" ? 1 : 3)}${unit}</td>`).join("")}</tr>`;
  });
  h += `</table><p class="muted small">Test images: ${m[0].n}. Baseline IoU/MAE use a Grad-CAM proxy mask (it has no segmentation head).</p>`;
  h += `<div class="perf-imgs">${m.map((x) => `<div><h3>${x.kind}</h3><img src="/api/report-image/confusion_${x.kind}.png" onerror="this.remove()"></div>`).join("")}</div>`;
  $("perf").innerHTML = h;
}

// ---------- rules ----------
function renderRules() {
  const r = INFO.rules;
  $("rules").innerHTML = `<table>
    <tr><th>Grade</th><th>Condition (first match wins)</th></tr>
    <tr><td><b style="color:var(--Reject)">Reject</b></td><td>total ≥ ${r.reject.total_pct}% or rot ≥ ${r.reject.rot_pct}%</td></tr>
    <tr><td><b style="color:var(--C)">C</b></td><td>total ≥ ${r.c.total_pct}% or rot ≥ ${r.c.rot_pct}%</td></tr>
    <tr><td><b style="color:var(--B)">B</b></td><td>total ≥ ${r.b.total_pct}%</td></tr>
    <tr><td><b style="color:var(--A)">A</b></td><td>otherwise</td></tr></table>
    <p class="muted small">Manual review if confidence &lt; ${INFO.review.review_confidence * 100}% or MC std &gt; ${INFO.review.review_std}.</p>`;
}

// ---------- start ----------
(async () => {
  INFO = await (await fetch("/api/info")).json();
  document.querySelectorAll("#model-toggle button").forEach((b) => {
    b.disabled = !(b.dataset.model in INFO.models);
  });
  if (!("multitask" in INFO.models) && "baseline" in INFO.models) {
    currentModel = "baseline";
    document.querySelector('[data-model="baseline"]').click();
  }
  if (!Object.keys(INFO.models).length) status("No trained model found — train one first (see README).", true);
  renderRules();
  loadSamples();
})();
