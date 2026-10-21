// Presentation dashboard. Every number shown comes from /api/dashboard (app/dashboard.py),
// which reads it from files in the report folder. This file only formats and draws.
"use strict";

const GRADES = ["A", "B", "C", "Reject"];
const MODELS = ["multitask", "baseline"];
const NAME = { multitask: "Multitask", baseline: "Baseline" };
const COL = { multitask: "#1b7046", baseline: "#d97a22", head: "#86c09c", ideal: "#9aa39d" };
const GRADE_COL = { A: "#2e9d57", B: "#8ab43c", C: "#e5a12a", Reject: "#c8372d" };
let D = null, MASK_COLOURS = {};
const rendered = new Set();

// ---------- formatting ----------
const pct = (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d) + "%");
const f3 = (v) => (v == null ? "—" : v.toFixed(3));
const f2 = (v) => (v == null ? "—" : v.toFixed(2));
const signed = (v, fmt) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v));
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- Chart.js setup: no animation, big fonts, values drawn on the chart (no hover needed) ----------
Chart.defaults.font.family = '"Segoe UI", system-ui, Arial, sans-serif';
Chart.defaults.font.size = window.innerHeight >= 1000 ? 16 : 14;
Chart.defaults.color = "#1b241f";
Chart.defaults.animation = false;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.boxHeight = 10;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.scale.grid.color = "#ecebe4";

Chart.register({
  id: "valueLabels",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.fmt) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.font = `600 ${opts.size || Chart.defaults.font.size - 1}px ${Chart.defaults.font.family}`;
    ctx.textAlign = "center";
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden || meta.type !== "bar") return;
      meta.data.forEach((el, i) => {
        const v = ds.data[i];
        if (v == null) return;
        const txt = opts.fmt(v, di, i);
        if (txt == null || txt === "") return;
        if (opts.inside) {
          if (Math.abs(el.base - el.y) < 18) return;
          ctx.fillStyle = "#fff";
          ctx.textBaseline = "middle";
          ctx.fillText(txt, el.x, (el.y + el.base) / 2);
        } else {
          ctx.fillStyle = "#1b241f";
          ctx.textBaseline = "bottom";
          ctx.fillText(txt, el.x, el.y - 3);
        }
      });
    });
    ctx.restore();
  },
});
Chart.register({
  id: "vlines",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.lines) return;
    const { ctx, chartArea: a, scales } = chart;
    ctx.save();
    opts.lines.forEach((l, k) => {
      const x = scales.x.getPixelForValue(l.index);
      ctx.strokeStyle = l.color;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(x, a.top); ctx.lineTo(x, a.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `600 ${Chart.defaults.font.size - 2}px ${Chart.defaults.font.family}`;
      ctx.textAlign = x > a.left + (a.right - a.left) * 0.7 ? "right" : "left";
      ctx.textBaseline = "top";
      const tx = x + (ctx.textAlign === "left" ? 4 : -4), ty = a.top + 2 + k * (Chart.defaults.font.size + 2);
      const tw = ctx.measureText(l.text).width;
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      ctx.fillRect(ctx.textAlign === "left" ? tx - 2 : tx - tw - 2, ty - 1, tw + 4, Chart.defaults.font.size);
      ctx.fillStyle = l.textColor || l.color;
      ctx.fillText(l.text, tx, ty);
    });
    ctx.restore();
  },
});

function chart(id, cfg) {
  const el = document.getElementById(id);
  const old = Chart.getChart(el);
  if (old) old.destroy();
  return new Chart(el, cfg);
}
const yPct = (max = 1) => ({ min: 0, max, ticks: { stepSize: 0.2, callback: (v) => (v > 1.0001 ? "" : Math.round(v * 100) + "%") } });

// ---------- tabs ----------
const TABS = ["overview", "confusion", "pergrade", "segmentation", "severity", "confidence", "training", "data", "errors"];
const RENDER = {};

function show(tab) {
  if (!TABS.includes(tab)) tab = "overview";
  document.querySelectorAll(".tab-view").forEach((s) => s.classList.toggle("active", s.id === "tab-" + tab));
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  if (D && !rendered.has(tab)) {
    rendered.add(tab);
    RENDER[tab]();
  }
  if (location.hash !== "#" + tab) history.replaceState(null, "", "#" + tab);
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea")) return;
  const cur = TABS.indexOf(location.hash.slice(1));
  if (e.key === "ArrowRight") show(TABS[Math.min(TABS.length - 1, cur + 1)]);
  else if (e.key === "ArrowLeft") show(TABS[Math.max(0, cur - 1)]);
  else if (/^[1-9]$/.test(e.key)) show(TABS[+e.key - 1]);
});
window.addEventListener("hashchange", () => show(location.hash.slice(1)));

function segButtons(sel, onChange) {
  const box = $(sel);
  box.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      box.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset.v);
    }));
  return () => box.querySelector("button.on").dataset.v;
}

// ---------- 1 overview ----------
const KPIS = [
  { key: "accuracy", name: "Grade accuracy", get: (m) => m.classification.accuracy, fmt: pct, dfmt: (v) => (v * 100).toFixed(1) + " pts" },
  { key: "macro_f1", name: "Grade macro-F1", get: (m) => m.classification.macro_f1, fmt: f3, dfmt: f3 },
  { key: "any_defect_dice", name: "Any-defect Dice", get: (m) => m.segmentation.any_defect_dice, fmt: f3, dfmt: f3 },
  { key: "severity_mae_from_mask", name: "Severity MAE, mask (lower = better)", get: (m) => m.severity_mae_from_mask, fmt: f2, dfmt: f2 },
];
const PAIRED_NAMES = {
  accuracy: "Grade accuracy", macro_f1: "Grade macro-F1", any_defect_dice: "Any-defect Dice",
  severity_mae_from_mask: "Severity MAE, mask (% pts)",
  severity_mae_head_vs_baseline_mask: "Severity MAE, multitask head vs baseline (% pts)",
};

function verdictClass(p) {
  if (!p.excludes_zero) return "";
  return p.verdict.startsWith("multitask") ? "good" : "bad";
}

RENDER.overview = () => {
  const M = D.metrics, P = D.paired.metrics;
  $("#kpis").innerHTML = KPIS.map((k) => {
    const rows = MODELS.map((mo) => {
      const ci = (M[mo].ci95 || {})[k.key];
      return `<div class="m ${mo === "multitask" ? "mt" : "bl"}"><span class="who"><i class="sw ${mo === "multitask" ? "mt" : "bl"}"></i>${NAME[mo]}</span>
        <b>${k.fmt(k.get(M[mo]))}</b><span class="ci">${ci ? `${k.fmt(ci[0])} – ${k.fmt(ci[1])}` : ""}</span></div>`;
    }).join("");
    let extra = "";
    if (k.key === "severity_mae_from_mask" && M.multitask.severity_mae_regression_head != null) {
      const ci = M.multitask.ci95.severity_mae_regression_head;
      extra = `<div class="note">Multitask regression head: <b>${f2(M.multitask.severity_mae_regression_head)}</b> (${f2(ci[0])} – ${f2(ci[1])})</div>`;
    }
    const p = P[k.key];
    const v = p ? `<div class="verdict ${verdictClass(p)}">Paired Δ ${signed(p.diff, k.dfmt)} [${signed(p.ci95[0], k.dfmt)}, ${signed(p.ci95[1], k.dfmt)}]<br><b>${p.excludes_zero ? esc(p.verdict) : "no clear difference (CI includes 0)"}</b></div>` : "";
    return `<div class="kpi"><h3>${k.name} <small>· 95% CI</small></h3>${rows}${extra}${v}</div>`;
  }).join("");

  const fmtFor = (key) => (key === "accuracy" ? (v) => (v * 100).toFixed(1) + " pts" : key.startsWith("severity") ? f2 : f3);
  $("#paired").innerHTML = `<tr><th>Metric</th><th>Δ</th><th>95% CI of Δ</th><th>Verdict</th></tr>` +
    Object.entries(P).map(([key, p]) => {
      const f = fmtFor(key);
      return `<tr><td>${PAIRED_NAMES[key] || key}</td><td class="nw">${signed(p.diff, f)}</td><td class="nw">[${signed(p.ci95[0], f)}, ${signed(p.ci95[1], f)}]</td>
        <td class="nw ${p.excludes_zero ? (verdictClass(p) === "good" ? "good" : "bad") : "neutral"}">${p.excludes_zero ? esc(p.verdict) : "no clear difference"}</td></tr>`;
    }).join("");
  $("#paired-note").textContent = `Δ = multitask − baseline. ${D.paired.resamples} resamples of the ${D.paired.n} ${D.paired.split} images, both models scored on the same resampled images (seed ${D.paired.seed}). "Better" only when the 95% CI excludes 0. For MAE, negative Δ = multitask better.`;

  const I = D.info.models;
  const row = (label, f) => `<tr><td>${label}</td>${MODELS.map((m) => `<td>${f(I[m]) ?? "—"}</td>`).join("")}</tr>`;
  $("#model-info").innerHTML = `<tr><th></th>${MODELS.map((m) => `<th><i class="sw ${m === "multitask" ? "mt" : "bl"}"></i>${NAME[m]}</th>`).join("")}</tr>` +
    row("Backbone", (i) => `${i.backbone}${i.pretrained ? " (ImageNet)" : ""}`) +
    row("Parameters", (i) => (i.parameters / 1e6).toFixed(2) + " M") +
    row("Image size / batch", (i) => `${i.image_size} px / ${i.batch_size ?? "—"}`) +
    row("Epochs run (first frozen)", (i) => `${i.epochs_run} (${i.frozen_backbone_epochs})`) +
    row("Best epoch (val score)", (i) => i.best_epoch) +
    row("Train time", (i) => (i.train_minutes != null ? i.train_minutes + " min" : "—")) +
    row("GPU", (i) => esc(i.gpu_this_machine || i.train_device || "—")) +
    row("MC Dropout passes", (i) => `${i.mc_passes} (seed ${i.mc_seed})`);
};

// ---------- 2 confusion ----------
let cmSel = null;
RENDER.confusion = () => {
  const mode = segButtons("#cm-mode", drawCMs);
  function drawCMs() {
    const m = mode();
    $("#cms").innerHTML = MODELS.map((mo) => {
      const cm = D.metrics[mo].classification.confusion_matrix;
      const body = cm.map((r, i) => {
        const tot = r.reduce((a, b) => a + b, 0);
        return `<tr><th>${GRADES[i]}</th>` + r.map((v, j) => {
          const frac = tot ? v / tot : 0, a = 0.06 + 0.86 * frac;
          const sel = cmSel && cmSel.mo === mo && cmSel.i === i && cmSel.j === j;
          return `<td data-mo="${mo}" data-i="${i}" data-j="${j}" class="${i === j ? "diag" : ""} ${sel ? "sel" : ""}"
            style="background:rgba(27,112,70,${a.toFixed(3)});color:${a > 0.5 ? "#fff" : "#1b241f"}">${m === "count" ? v : pct(frac, 0)}</td>`;
        }).join("") + `<th class="ax">${tot}</th></tr>`;
      }).join("");
      const acc = D.metrics[mo].classification.accuracy;
      return `<div class="card cm"><h2><i class="sw ${mo === "multitask" ? "mt" : "bl"}"></i>${NAME[mo]}</h2>
        <table><tr><th class="ax">true ↓ / pred →</th>${GRADES.map((g) => `<th>${g}</th>`).join("")}<th class="ax">n</th></tr>${body}</table>
        <div class="acc">Accuracy ${pct(acc)} (${cm.reduce((s, r, i) => s + r[i], 0)} of ${D.metrics[mo].n} on the diagonal)</div></div>`;
    }).join("");
    document.querySelectorAll(".cm td").forEach((td) => td.addEventListener("click", () => {
      cmSel = { mo: td.dataset.mo, i: +td.dataset.i, j: +td.dataset.j };
      drawCMs();
    }));
    drawCmGallery();
  }
  drawCMs();
};
function drawCmGallery() {
  if (!cmSel) {
    $("#cm-gallery").innerHTML = `<div class="empty-g">Click any cell above to show those images here.</div>`;
    return;
  }
  const { mo, i, j } = cmSel;
  const ids = D.confusion[mo][`${i},${j}`] || [];
  $("#cm-gallery").innerHTML = gallery(mo, ids, `${NAME[mo]}: true ${GRADES[i]} → predicted ${GRADES[j]} (${ids.length} image${ids.length === 1 ? "" : "s"})`);
}

// ---------- galleries ----------
function gallery(mo, ids, title) {
  const names = ["Input", "True mask (pseudo)", mo === "multitask" ? "Predicted mask" : "Proxy mask (Grad-CAM > 0.5)",
    mo === "multitask" ? "Grad-CAM, predicted grade" : "Grad-CAM, Reject class"];
  const legend = Object.entries(MASK_COLOURS).filter(([k]) => k !== "background")
    .map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("");
  const cards = ids.map((id) => card(mo, id)).join("") || `<div class="empty-g">No images in this cell.</div>`;
  return `<div class="gallery-head"><h2>${esc(title)}</h2><div class="legend-row">${legend}</div></div>
    <div class="gallery"><div class="panel-names">${names.map((n) => `<span>${n}</span>`).join("")}</div>
    ${ids.length > 1 ? `<div class="panel-names">${names.map((n) => `<span>${n}</span>`).join("")}</div>` : ""}${cards}</div>`;
}
function card(mo, id) {
  const r = D.predictions[mo][id];
  const head = r.pred_severity_head != null ? ` / head ${f2(r.pred_severity_head)}` : "";
  return `<div class="g-card"><img loading="lazy" src="/api/dashboard/thumb/${mo}/${encodeURIComponent(id)}" alt="${esc(id)}">
    <div class="cap"><span><b>${esc(id)}</b> · ${esc(r.fruit)}, labelled ${esc(r.fresh_rotten)}</span>
    <span>true <b>${r.true_grade}</b> → predicted <b class="${r.correct ? "" : "wrong"}">${r.pred_grade}</b> (conf ${pct(r.confidence, 0)}, std ${f2(r.mc_std)})</span>
    <span>severity true ${f2(r.true_severity)} / mask ${f2(r.pred_severity_mask)}${head} %</span>
    ${r.flagged ? `<span class="flag">⚑ review: ${esc(r.flag_reasons)}</span>` : ""}</div></div>`;
}

// ---------- 3 per grade ----------
RENDER.pergrade = () => {
  const PC = (mo) => D.metrics[mo].classification.per_class;
  $("#pg-table").innerHTML =
    `<tr><th>Grade</th><th>Support</th>${MODELS.map((m) => `<th colspan="3"><i class="sw ${m === "multitask" ? "mt" : "bl"}"></i>${NAME[m]}</th>`).join("")}</tr>` +
    `<tr><th></th><th></th>${MODELS.map(() => "<th>Prec.</th><th>Recall</th><th>F1</th>").join("")}</tr>` +
    GRADES.map((g) => `<tr><td><b>${g}</b></td><td>${PC("multitask")[g].support}</td>` +
      MODELS.map((m) => { const c = PC(m)[g]; return `<td>${f3(c.precision)}</td><td>${f3(c.recall)}</td><td><b>${f3(c.f1)}</b></td>`; }).join("") + "</tr>").join("") +
    `<tr><td><b>Macro avg</b></td><td>${D.metrics.multitask.n}</td>` + MODELS.map((m) => `<td></td><td></td><td><b>${f3(D.metrics[m].classification.macro_f1)}</b></td>`).join("") + "</tr>";
  const draw = (metric) => chart("pg-chart", {
    type: "bar",
    data: {
      labels: GRADES.map((g) => `${g} (n=${PC("multitask")[g].support})`),
      datasets: MODELS.map((m) => ({ label: NAME[m], data: GRADES.map((g) => PC(m)[g][metric]), backgroundColor: COL[m], borderRadius: 4, borderSkipped: "start" })),
    },
    options: { scales: { y: { min: 0, max: 1.1, ticks: { stepSize: 0.2, callback: (v) => (v > 1 ? "" : v.toFixed(1)) } } },
      plugins: { valueLabels: { fmt: (v) => v.toFixed(2) } } },
  });
  segButtons("#pg-metric", draw);
  draw("f1");
};

// ---------- 4 segmentation ----------
RENDER.segmentation = () => {
  const cls = ["bruise", "rot", "mechanical", "healthy"];
  const n = D.seg_images.multitask;
  const labels = [...cls.map((c) => (n[c] != null ? [c, `${n[c]} test images`] : [c, "fruit skin"])), ["any defect", "binary"]];
  for (const [id, key, any] of [["seg-iou", "iou", "any_defect_iou"], ["seg-dice", "dice", "any_defect_dice"]]) {
    chart(id, {
      type: "bar",
      data: {
        labels,
        datasets: MODELS.map((m) => {
          const s = D.metrics[m].segmentation;
          return { label: NAME[m], data: [...cls.map((c) => s[key][c]), s[any]], backgroundColor: COL[m], borderRadius: 4, borderSkipped: "start" };
        }),
      },
      options: { scales: { y: { min: 0, max: 1.1, ticks: { stepSize: 0.2, callback: (v) => (v > 1 ? "" : v.toFixed(1)) } } },
        plugins: { valueLabels: { fmt: (v) => v.toFixed(2) } } },
    });
  }
  const px = D.metrics.multitask.segmentation.gt_pixels;
  $("#seg-note").innerHTML = `Pixels pooled over all ${D.metrics.multitask.n} test images. True defect pixels: bruise ${px.bruise.toLocaleString()}, rot ${px.rot.toLocaleString()}, mechanical ${px.mechanical.toLocaleString()}. ` +
    `<b>Baseline has no segmentation head:</b> its mask is the true fruit outline with Grad-CAM (Reject) &gt; 0.5 marked as rot, so its bruise and mechanical scores are 0 by construction and its healthy score uses the true outline.`;
};

// ---------- 5 severity ----------
RENDER.severity = () => {
  const S = D.severity, M = D.metrics;
  const maxV = Math.max(...MODELS.flatMap((m) => S[m].points.flatMap((p) => [p[0], p[1], p[2] ?? 0])));
  const top = Math.min(100, Math.ceil(maxV / 10) * 10);
  const ideal = { type: "line", label: "y = x (perfect)", data: [{ x: 0, y: 0 }, { x: top, y: top }], borderColor: COL.ideal, borderDash: [6, 4], borderWidth: 2, pointRadius: 0 };
  const axes = { x: { type: "linear", min: 0, max: top, title: { display: true, text: "true severity (% of fruit)" } },
    y: { min: 0, max: top, title: { display: true, text: "predicted (%)" } } };
  chart("sev-sc-mt", { type: "scatter", data: { datasets: [
    { label: `mask (MAE ${f2(M.multitask.severity_mae_from_mask)})`, data: S.multitask.points.map((p) => ({ x: p[0], y: p[1] })), backgroundColor: COL.multitask + "b0", pointRadius: 4 },
    { label: `head (MAE ${f2(M.multitask.severity_mae_regression_head)})`, data: S.multitask.points.map((p) => ({ x: p[0], y: p[2] })), backgroundColor: "#ffffff00", borderColor: "#0d3b24", pointStyle: "triangle", pointRadius: 5, borderWidth: 1.5 },
    ideal] }, options: { scales: axes } });
  chart("sev-sc-bl", { type: "scatter", data: { datasets: [
    { label: `Grad-CAM proxy mask (MAE ${f2(M.baseline.severity_mae_from_mask)})`, data: S.baseline.points.map((p) => ({ x: p[0], y: p[1] })), backgroundColor: COL.baseline + "b0", pointStyle: "rectRot", pointRadius: 5 },
    ideal] }, options: { scales: axes } });

  const series = [["multitask", "mask", "Multitask mask", COL.multitask], ["multitask", "head", "Multitask head", COL.head], ["baseline", "mask", "Baseline proxy mask", COL.baseline]]
    .filter(([m, s]) => S[m][`hist_${s}`]);
  const g = S.multitask.mae_by_grade_mask;
  chart("sev-grade", { type: "bar", data: {
    labels: GRADES.map((x) => [x, `n=${g[x] ? g[x].n : 0}`]),
    datasets: series.map(([m, s, label, c]) => ({ label, data: GRADES.map((x) => (S[m][`mae_by_grade_${s}`][x] || {}).mae ?? null), backgroundColor: c, borderRadius: 4, borderSkipped: "start" })) },
    options: { scales: { y: { min: 0, grace: "12%", title: { display: true, text: "MAE (% pts)" } }, x: { ticks: { maxRotation: 0 } } }, plugins: { valueLabels: { fmt: (v) => v.toFixed(1), size: 12 } } } });

  const e = S.multitask.hist_mask.edges;
  const labels = ["< " + e[0], ...e.slice(0, -1).map((v, i) => `${v}…${e[i + 1]}`), "≥ " + e[e.length - 1]];
  chart("sev-hist", { type: "bar", data: { labels,
    datasets: series.map(([m, s, label, c]) => ({ label, data: S[m][`hist_${s}`].counts, backgroundColor: c, borderRadius: 3, borderSkipped: "start" })) },
    options: { scales: { y: { min: 0, title: { display: true, text: "images" } }, x: { ticks: { autoSkip: false, maxRotation: 0, font: { size: 12 } } } } } });
};

// ---------- 6 confidence ----------
RENDER.confidence = () => {
  const C = D.confidence;
  const bins = C.multitask.reliability.bins.map((b, i) => i).filter((i) => MODELS.some((m) => C[m].reliability.bins[i].n > 0));
  const lab = bins.map((i) => { const b = C.multitask.reliability.bins[i]; return `${b.lo.toFixed(1)}–${b.hi.toFixed(1)}`; });
  $("#rel-title").textContent = `Reliability: confidence vs actual accuracy (ECE multitask ${f3(C.multitask.reliability.ece)}, baseline ${f3(C.baseline.reliability.ece)})`;
  chart("rel", { type: "bar", data: { labels: lab, datasets: [
    ...MODELS.map((m) => ({ label: NAME[m], data: bins.map((i) => C[m].reliability.bins[i].accuracy), backgroundColor: COL[m], borderRadius: 4, borderSkipped: "start" })),
    { type: "line", label: "perfect calibration", data: bins.map((i) => (C.multitask.reliability.bins[i].lo + C.multitask.reliability.bins[i].hi) / 2), borderColor: COL.ideal, borderDash: [6, 4], pointRadius: 0, borderWidth: 2 },
  ] }, options: { scales: { y: yPct(1.12), x: { title: { display: true, text: "MC Dropout confidence (mean probability of predicted grade)" } } },
    plugins: { valueLabels: { size: 12, fmt: (_v, di, i) => (di < 2 ? "n=" + C[MODELS[di]].reliability.bins[bins[i]].n : null) } } } });

  const T = C.multitask.threshold;
  chart("thr", { type: "line", data: { labels: T.map((t) => t.t.toFixed(2)), datasets: MODELS.flatMap((m) => [
    { label: `${NAME[m]} accuracy`, data: C[m].threshold.map((t) => t.accuracy), borderColor: COL[m], backgroundColor: COL[m], borderWidth: 2.5, pointRadius: 3, pointStyle: m === "baseline" ? "triangle" : "circle" },
    { label: `${NAME[m]} accepted`, data: C[m].threshold.map((t) => t.coverage), borderColor: COL[m], backgroundColor: COL[m], borderDash: [7, 5], borderWidth: 2, pointRadius: 0 },
  ]) }, options: { scales: { y: yPct(1), x: { title: { display: true, text: "confidence threshold" } } } } });

  $("#flags").innerHTML = `<tr><th>Model · flag rule</th><th>Flagged for review</th><th>Wrong grades caught</th><th>Accuracy of the rest</th></tr>` +
    MODELS.flatMap((m) => ["flagged_uncertainty", "flagged"].map((k) => {
      const f = C[m].flags[k];
      return `<tr><td><i class="sw ${m === "multitask" ? "mt" : "bl"}"></i><b>${NAME[m]}</b> · ${esc(f.rule)}</td><td>${pct(f.flag_rate)} (${f.flagged}/${f.n})</td>
        <td>${pct(f.errors_caught_rate)} (${f.errors_caught}/${f.errors})</td><td>${pct(f.accuracy_unflagged)}</td></tr>`;
    })).join("");
};

// ---------- 7 training ----------
RENDER.training = () => {
  const H = D.history, I = D.info.models;
  const n = Math.max(...MODELS.map((m) => (H[m] ? H[m].epoch.length : 0)));
  const labels = Array.from({ length: n }, (_, i) => String(i + 1));
  const lines = MODELS.filter((m) => I[m].best_epoch).map((m) => ({ index: I[m].best_epoch - 1, color: COL[m], textColor: m === "baseline" ? "#8a4a0f" : COL[m], text: `${NAME[m]} best: epoch ${I[m].best_epoch}` }));
  const line = (m, col, label, opts = {}) => (H[m] && H[m][col] && H[m][col].some((v) => v != null))
    ? [{ label, data: H[m][col], borderColor: opts.color || COL[m], backgroundColor: opts.color || COL[m], borderWidth: 2.5, pointRadius: 2,
         borderDash: opts.dash || (m === "baseline" ? [7, 4] : []), pointStyle: m === "baseline" ? "triangle" : "circle" }] : [];
  const opt = (y, onlyMt) => ({ scales: { y, x: { title: { display: true, text: "epoch" } } },
    plugins: { vlines: { lines: onlyMt ? lines.filter((l) => l.color === COL.multitask) : lines } } });
  chart("tr-loss", { type: "line", data: { labels, datasets: [
    ...line("multitask", "train_total", "Multitask total (cls + seg + 2×sev)"), ...line("multitask", "train_cls", "Multitask classification", { color: COL.head }),
    ...line("baseline", "train_cls", "Baseline classification")] }, options: opt({ min: 0 }) });
  chart("tr-acc", { type: "line", data: { labels, datasets: [
    ...line("multitask", "val_acc", "Multitask accuracy"), ...line("multitask", "val_f1", "Multitask macro-F1", { color: COL.head }),
    ...line("baseline", "val_acc", "Baseline accuracy"), ...line("baseline", "val_f1", "Baseline macro-F1", { color: "#f0b27a" })] }, options: opt({ min: 0.4, max: 1, ticks: { stepSize: 0.1 } }) });
  chart("tr-dice", { type: "line", data: { labels, datasets: line("multitask", "val_defect_dice", "Multitask mean-defect Dice") }, options: opt({ min: 0 }, true) });
  chart("tr-mae", { type: "line", data: { labels, datasets: line("multitask", "val_sev_mae", "Multitask head MAE (% pts)") }, options: opt({ min: 0 }, true) });
  $("#tr-note").textContent = `Validation split, one pass per epoch (no MC Dropout). Best epoch = highest val score (macro-F1 + mean-defect IoU − head MAE/100 for multitask; macro-F1 for baseline). ` +
    `First ${I.multitask.frozen_backbone_epochs} epochs: backbone frozen. Train time: multitask ${I.multitask.train_minutes} min, baseline ${I.baseline.train_minutes} min.`;
};

// ---------- 8 data ----------
RENDER.data = () => {
  const d = D.data;
  const lf = d.label_filter;
  $("#facts").innerHTML = [
    [`${d.files_total.toLocaleString()} = ${d.originals.toLocaleString()} × ${d.versions_per_original}`, "Kaggle files = original photos × versions per photo: " +
      Object.entries(d.augmentation_type_counts).map(([k, v]) => `${k.replace(/_+$/, "").replace(/_/g, " ")} ×${v / d.originals}`).join(", ")],
    [`${d.kaggle_split_leak_originals.toLocaleString()} of ${d.originals.toLocaleString()}`, "originals with copies on BOTH sides of Kaggle's train/test split: we never use that split"],
    [`${d.originals_used.toLocaleString()} originals`, `used (augmented copies dropped) · ${d.merged_pairs} of ${d.candidate_pairs} near-duplicate pairs merged, ${d.label_conflicts} label conflicts`],
    [`${lf.excluded} rows`, "excluded from train/val by the label-consistency filter (test untouched)"],
  ].map(([b, s]) => `<div class="fact"><b>${b}</b><span>${s}</span></div>`).join("");
  const sp = ["train", "val", "test"];
  const classes = Object.keys(d.split_by_class.train).sort();
  $("#split-table").innerHTML = `<tr><th>Class</th>${sp.map((s) => `<th>${s}</th>`).join("")}</tr>` +
    classes.map((c) => `<tr><td>${c.replace("_", " ")}</td>${sp.map((s) => `<td>${d.split_by_class[s][c] ?? 0}</td>`).join("")}</tr>`).join("") +
    `<tr><td><b>Images</b></td>${sp.map((s) => `<td><b>${d.split_images[s]}</b></td>`).join("")}</tr>` +
    `<tr><td><b>Groups</b></td>${sp.map((s) => `<td>${d.split_groups[s]}</td>`).join("")}</tr>`;
  const G = d.grade_by_split;
  chart("grade-split", { type: "bar", data: { labels: sp,
    datasets: GRADES.map((g) => ({ label: g, data: sp.map((s) => G[s][g] || 0), backgroundColor: GRADE_COL[g], borderColor: "#fff", borderWidth: { right: 2 } })) },
    options: { indexAxis: "y", scales: { x: { stacked: true, min: 0 }, y: { stacked: true } }, plugins: { legend: { position: "top" } } },
    plugins: [{ id: "stackLabels", afterDatasetsDraw(ch) {
      const ctx = ch.ctx; ctx.save(); ctx.font = `600 13px ${Chart.defaults.font.family}`; ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ch.data.datasets.forEach((ds, di) => ch.getDatasetMeta(di).data.forEach((el, i) => { if (Math.abs(el.x - el.base) > 28) ctx.fillText(ds.data[i], (el.x + el.base) / 2, el.y); }));
      ctx.restore(); } }] });
  $("#grade-table").innerHTML = `<tr><th>Split</th>${GRADES.map((g) => `<th>${g}</th>`).join("")}<th>Total</th></tr>` +
    sp.map((s) => `<tr><td>${s}</td>${GRADES.map((g) => `<td>${G[s][g] || 0}</td>`).join("")}<td><b>${GRADES.reduce((a, g) => a + (G[s][g] || 0), 0)}</b></td></tr>`).join("");
  const t = d.tuning_sample;
  $("#data-note").textContent = `Split by group, stratified by class, seed ${d.seed}; ${d.groups.toLocaleString()} groups.` +
    (t ? ` Caveat: the ${t.images}-image pseudo-mask tuning sample was drawn from all splits (${Object.entries(t.by_split).map(([k, v]) => `${k} ${v}`).join(", ")}).` : "");
};

// ---------- 9 errors ----------
RENDER.errors = () => {
  const mo = segButtons("#err-model", draw), kind = segButtons("#err-kind", draw);
  function draw() {
    const m = mo(), k = kind();
    const ids = D.errors[m][k];
    const title = k === "confident_wrong" ? `${NAME[m]}: the ${ids.length} wrong grades with the highest confidence`
      : `${NAME[m]}: the ${ids.length} largest severity errors (|mask − true|)`;
    $("#err-gallery").innerHTML = gallery(m, ids, title) +
      `<p class="note">"True" grade = the pseudo-mask grade, not a human label. Known pseudo-mask limits: white mould is often missed and deep-red apple skin can read as rot, so some of these "errors" are label errors. The CVAT-verified test set will settle this.</p>`;
  }
  draw();
};

// ---------- start ----------
async function start() {
  try {
    const [d, info] = await Promise.all([fetch("/api/dashboard"), fetch("/api/info")]);
    if (!d.ok) throw new Error((await d.json()).detail || d.statusText);
    D = await d.json();
    if (info.ok) MASK_COLOURS = (await info.json()).colours || {};
  } catch (e) {
    $("#load-error").hidden = false;
    $("#load-error").textContent = "Could not load dashboard data: " + e.message;
    return;
  }
  if (D.warnings.length) {
    $("#load-error").hidden = false;
    $("#load-error").textContent = "Warning: " + D.warnings.join("; ");
  }
  $("#testset").textContent = `${D.info.n_images} ${D.info.split} images · report folder ${D.report_dir}`;
  document.querySelectorAll(".src").forEach((p) => {
    p.innerHTML = "Source: " + p.dataset.src.split(",").filter((k) => D.sources[k]).map((k) => `<code>${D.sources[k]}</code>`).join(", ");
  });
  show(location.hash.slice(1) || "overview");
}
start();
