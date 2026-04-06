const API_BASE = "";

let sessionId = localStorage.getItem("well_ml_session") || "";
let wellColumns = [];
let charts = { oil: null, water: null, choke: null };

function headersJson() {
  return {
    "Content-Type": "application/json",
    "X-Session-Id": sessionId,
  };
}

function showToast(msg, isErr) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.toggle("hidden", false);
  el.classList.toggle("err", !!isErr);
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 4500);
}

/**
 * Tampilkan loading bar di kartu / modal dan nonaktifkan kontrol sementara.
 * @param {{ cardId?: string|null, root?: HTMLElement|null, label: string, buttons?: HTMLElement[], fn: () => Promise<void> }} opts
 */
async function withStepLoading(opts) {
  const { cardId, root, label, buttons = [], fn } = opts;
  const el = root || (cardId ? document.getElementById(cardId) : null);
  const btnList = buttons.filter(Boolean);
  const prevDisabled = btnList.map((b) => b.disabled);
  if (el) {
    const t = el.querySelector(".process-loading-text");
    if (t) t.textContent = label;
    el.classList.add("is-busy");
    el.setAttribute("aria-busy", "true");
  }
  btnList.forEach((b) => {
    b.disabled = true;
  });
  try {
    await fn();
  } finally {
    if (el) {
      el.classList.remove("is-busy");
      el.removeAttribute("aria-busy");
    }
    btnList.forEach((b, i) => {
      b.disabled = prevDisabled[i];
    });
  }
}

async function api(path, opts = {}) {
  const h = opts.headers || {};
  if (!h["X-Session-Id"] && sessionId) h["X-Session-Id"] = sessionId;
  const res = await fetch(API_BASE + path, { ...opts, headers: h });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    const err = data.error || res.statusText || "Request gagal";
    throw new Error(err);
  }
  return data;
}

async function ensureSession() {
  if (sessionId) {
    try {
      await api("/api/state", { headers: { "X-Session-Id": sessionId } });
      return;
    } catch {
      sessionId = "";
    }
  }
  const d = await api("/api/session", { method: "POST" });
  sessionId = d.session_id;
  localStorage.setItem("well_ml_session", sessionId);
  document.getElementById("sessionLabel").textContent = "Sesi: " + sessionId;
}

function applyTestDateBounds(minD, maxD) {
  const a = document.getElementById("optDateStart");
  const b = document.getElementById("optDateEnd");
  const hint = document.getElementById("optDateHint");
  if (!a || !b || !hint) return;
  a.value = "";
  b.value = "";
  if (minD && maxD) {
    a.min = minD;
    a.max = maxD;
    b.min = minD;
    b.max = maxD;
    a.disabled = false;
    b.disabled = false;
    hint.textContent = `Opsional — DATEPRD di test set antara ${minD} dan ${maxD}.`;
  } else {
    ["min", "max"].forEach((attr) => {
      a.removeAttribute(attr);
      b.removeAttribute(attr);
    });
    a.disabled = true;
    b.disabled = true;
    hint.textContent =
      "Rentang tanggal tidak tersedia (kolom DATEPRD tidak ada atau tidak valid di test set).";
  }
}

async function syncTestDateBoundsFromServer() {
  try {
    const s = await api("/api/state", { headers: { "X-Session-Id": sessionId } });
    if (s.has_split && s.test_date_min && s.test_date_max) {
      applyTestDateBounds(s.test_date_min, s.test_date_max);
    } else {
      applyTestDateBounds(null, null);
    }
  } catch {
    applyTestDateBounds(null, null);
  }
}

function applyTestDateBounds(minD, maxD) {
  const a = document.getElementById("optDateStart");
  const b = document.getElementById("optDateEnd");
  const hint = document.getElementById("optDateHint");
  if (!a || !b || !hint) return;
  a.value = "";
  b.value = "";
  if (minD && maxD) {
    a.min = minD;
    a.max = maxD;
    b.min = minD;
    b.max = maxD;
    a.disabled = false;
    b.disabled = false;
    hint.textContent = `Opsional — DATEPRD di test set: ${minD} sampai ${maxD}.`;
  } else {
    a.removeAttribute("min");
    a.removeAttribute("max");
    b.removeAttribute("min");
    b.removeAttribute("max");
    a.disabled = true;
    b.disabled = true;
    hint.textContent =
      "Rentang tanggal tidak tersedia (DATEPRD tidak ada atau tidak valid di test set).";
  }
}

async function syncTestDateBoundsFromServer() {
  try {
    const s = await api("/api/state", { headers: { "X-Session-Id": sessionId } });
    if (s.has_split && s.test_date_min && s.test_date_max) {
      applyTestDateBounds(s.test_date_min, s.test_date_max);
    } else {
      applyTestDateBounds(null, null);
    }
  } catch {
    applyTestDateBounds(null, null);
  }
}

function fmtCvBlock(title, cv) {
  if (!cv) return "(belum ada)";
  const scores = cv.scores.map((x) => Number(x).toFixed(4)).join(", ");
  return (
    `=== ${title} ===\n` +
    `Skor R² tiap lipatan : ${scores}\n` +
    `Rata-rata R²         : ${cv.mean.toFixed(4)}\n` +
    `Standar deviasi      : ${cv.std.toFixed(4)} (makin kecil makin stabil)`
  );
}

function chokeToPct(v) {
  if (v == null || Number.isNaN(v)) return v;
  const n = Number(v);
  if (n <= 1.5) return n * 100;
  return n;
}

function buildOrUpdateChart(key, canvasId, labels, datasets) {
  const ctx = document.getElementById(canvasId);
  if (charts[key]) charts[key].destroy();
  charts[key] = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#c8d4e6" } } },
      scales: {
        x: {
          ticks: { color: "#8b9bb4", maxRotation: 45 },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          ticks: { color: "#8b9bb4" },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
      },
    },
  });
}

function renderCharts(series) {
  const labels = series.map((r) => (r.DATEPRD || "").slice(0, 10));
  const cAct = series.map((r) => chokeToPct(r.Choke_Aktual));
  const cRec = series.map((r) => chokeToPct(r.Choke_Rekomendasi));

  buildOrUpdateChart("oil", "chartOil", labels, [
    {
      label: "Oil aktual",
      data: series.map((r) => r.Oil_Actual),
      borderColor: "#94a3b8",
      tension: 0.15,
    },
    {
      label: "Oil prediksi (choke optimal)",
      data: series.map((r) => r.Oil_Pred_OptimalChoke),
      borderColor: "#34d399",
      tension: 0.15,
    },
  ]);

  buildOrUpdateChart("water", "chartWater", labels, [
    {
      label: "Water aktual",
      data: series.map((r) => r.Water_Actual),
      borderColor: "#94a3b8",
      tension: 0.15,
    },
    {
      label: "Water prediksi (choke optimal)",
      data: series.map((r) => r.Water_Pred_OptimalChoke),
      borderColor: "#60a5fa",
      tension: 0.15,
    },
  ]);

  buildOrUpdateChart("choke", "chartChoke", labels, [
    {
      label: "Choke aktual (%)",
      data: cAct,
      borderColor: "#fb923c",
      tension: 0.15,
    },
    {
      label: "Choke rekomendasi (%)",
      data: cRec,
      borderColor: "#2dd4bf",
      tension: 0.15,
    },
  ]);
}

function wellTableHtml(wells) {
  if (!wells || !wells.length) return "<p>Tidak ada sumur.</p>";
  let h =
    "<table class='well-table'><thead><tr><th>Nama sumur</th><th>Jumlah data</th></tr></thead><tbody>";
  wells.forEach((w) => {
    h += `<tr><td>${escapeHtml(w.well)}</td><td>${w.count}</td></tr>`;
  });
  h += "</tbody></table>";
  return h;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function fillWellSelect(wells) {
  const sel = document.getElementById("wellSelect");
  sel.innerHTML = '<option value="">— pilih sumur —</option>';
  (wells || []).forEach((w) => {
    const o = document.createElement("option");
    o.value = w.well;
    o.textContent = `${w.well} (${w.count})`;
    sel.appendChild(o);
  });
  sel.disabled = !wells || wells.length === 0;
}

function openFeatureModal() {
  const box = document.getElementById("featureChecks");
  const to = document.getElementById("targetOil");
  const tw = document.getElementById("targetWater");
  box.innerHTML = "";
  to.innerHTML = "";
  tw.innerHTML = "";
  wellColumns.forEach((c) => {
    const lab = document.createElement("label");
    const inp = document.createElement("input");
    inp.type = "checkbox";
    inp.dataset.col = c;
    lab.appendChild(inp);
    lab.appendChild(document.createTextNode(" " + c));
    box.appendChild(lab);
    const o1 = document.createElement("option");
    o1.value = c;
    o1.textContent = c;
    to.appendChild(o1);
    const o2 = document.createElement("option");
    o2.value = c;
    o2.textContent = c;
    tw.appendChild(o2);
  });
  const defOil = wellColumns.indexOf("BORE_OIL_VOL");
  const defWat = wellColumns.indexOf("BORE_WAT_VOL");
  if (defOil >= 0) to.selectedIndex = defOil;
  if (defWat >= 0) tw.selectedIndex = defWat;
  document.getElementById("modal").classList.remove("hidden");
}

function closeFeatureModal() {
  document.getElementById("modal").classList.add("hidden");
}

document.getElementById("btnUpload").addEventListener("click", async () => {
  const f = document.getElementById("fileInput").files[0];
  if (!f) {
    showToast("Pilih file .xlsx", true);
    return;
  }
  const fileInput = document.getElementById("fileInput");
  try {
    await withStepLoading({
      cardId: "step-upload",
      label: "Mengunggah & membaca Excel…",
      buttons: [document.getElementById("btnUpload"), fileInput],
      fn: async () => {
        await ensureSession();
        const fd = new FormData();
        fd.append("file", f);
        const res = await fetch(API_BASE + "/api/upload", {
          method: "POST",
          headers: { "X-Session-Id": sessionId },
          body: fd,
        });
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          throw new Error(data.error || "Upload gagal");
        }
        document.getElementById("uploadSummary").innerHTML =
          `<strong>${data.n_rows}</strong> baris, <strong>${data.n_wells}</strong> sumur.` +
          wellTableHtml(data.wells);
        fillWellSelect(data.wells);
        document.getElementById("btnPreprocess").disabled = false;
        document.getElementById("btnSelectWell").disabled = false;
      },
    });
    showToast("Upload berhasil");
  } catch (e) {
    showToast(e.message, true);
  }
});

document.getElementById("btnPreprocess").addEventListener("click", async () => {
  try {
    await withStepLoading({
      cardId: "step-preprocess",
      label: "Memfilter data (oil > 0, production, OP)…",
      buttons: [document.getElementById("btnPreprocess")],
      fn: async () => {
        await ensureSession();
        const d = await api("/api/preprocess", {
          method: "POST",
          headers: headersJson(),
          body: "{}",
        });
        let skip = "";
        if (d.skipped_rules_missing_columns && d.skipped_rules_missing_columns.length) {
          skip =
            "<br><span class='muted'>Aturan dilewati (kolom hilang): " +
            d.skipped_rules_missing_columns.join(", ") +
            "</span>";
        }
        document.getElementById("preSummary").innerHTML =
          `Setelah praproses: <strong>${d.n_rows}</strong> baris, <strong>${d.n_wells}</strong> sumur.` +
          wellTableHtml(d.wells) +
          skip;
        fillWellSelect(d.wells);
      },
    });
    showToast("Praproses selesai");
  } catch (e) {
    showToast(e.message, true);
  }
});

document.getElementById("btnSelectWell").addEventListener("click", async () => {
  const well = document.getElementById("wellSelect").value;
  if (!well) {
    showToast("Pilih nama sumur", true);
    return;
  }
  const wellSelect = document.getElementById("wellSelect");
  try {
    await withStepLoading({
      cardId: "step-well",
      label: "Memuat data sumur terpilih…",
      buttons: [document.getElementById("btnSelectWell"), wellSelect],
      fn: async () => {
        await ensureSession();
        const d = await api("/api/select-well", {
          method: "POST",
          headers: headersJson(),
          body: JSON.stringify({ well_name: well }),
        });
        wellColumns = d.columns;
        document.getElementById("wellSummary").textContent =
          `${d.well_name}: ${d.n_rows} baris, ${d.columns.length} kolom.`;
        document.getElementById("btnOpenFeatures").disabled = false;
        document.getElementById("btnSplit").disabled = true;
        document.getElementById("featureSummary").textContent = "";
      },
    });
    showToast("Sumur dipilih");
  } catch (e) {
    showToast(e.message, true);
  }
});

document.getElementById("btnOpenFeatures").addEventListener("click", () => {
  if (!wellColumns.length) return;
  openFeatureModal();
});

document.getElementById("btnModalCancel").addEventListener("click", closeFeatureModal);

document.getElementById("btnModalSave").addEventListener("click", async () => {
  const checks = document.querySelectorAll("#featureChecks input[type=checkbox]:checked");
  const feature_columns = Array.from(checks).map((el) => el.dataset.col);
  const target_oil = document.getElementById("targetOil").value;
  const target_water = document.getElementById("targetWater").value;
  if (feature_columns.length === 0) {
    showToast("Pilih minimal satu fitur", true);
    return;
  }
  const modalInner = document.getElementById("modalInner");
  try {
    await withStepLoading({
      root: modalInner,
      label: "Menyimpan kolom fitur & target…",
      buttons: [
        document.getElementById("btnModalSave"),
        document.getElementById("btnModalCancel"),
      ],
      fn: async () => {
        await ensureSession();
        await api("/api/features", {
          method: "POST",
          headers: headersJson(),
          body: JSON.stringify({ feature_columns, target_oil, target_water }),
        });
        document.getElementById("featureSummary").textContent =
          `${feature_columns.length} fitur · oil: ${target_oil} · water: ${target_water}`;
        document.getElementById("btnSplit").disabled = false;
        document.getElementById("algoSelect").disabled = false;
        closeFeatureModal();
      },
    });
    showToast("Fitur disimpan");
  } catch (e) {
    showToast(e.message, true);
  }
});

document.getElementById("btnSplit").addEventListener("click", async () => {
  const test_size = parseFloat(document.getElementById("testSize").value);
  const random_state = parseInt(document.getElementById("randomState").value, 10);
  let splitResp = null;
  try {
    await withStepLoading({
      cardId: "step-split",
      label: "Membagi train / test…",
      buttons: [
        document.getElementById("btnSplit"),
        document.getElementById("testSize"),
        document.getElementById("randomState"),
      ],
      fn: async () => {
        await ensureSession();
        const d = await api("/api/split", {
          method: "POST",
          headers: headersJson(),
          body: JSON.stringify({ test_size, random_state }),
        });
        splitResp = d;
        document.getElementById("splitSummary").textContent =
          `Train: ${d.train_rows} · Test: ${d.test_rows}`;
        document.getElementById("btnTrain").disabled = false;
        document.getElementById("btnCv").disabled = true;
        document.getElementById("btnOptimize").disabled = true;
        document.getElementById("cvOil").textContent = "";
        document.getElementById("cvWater").textContent = "";
      },
    });
    if (splitResp) {
      applyTestDateBounds(splitResp.test_date_min, splitResp.test_date_max);
    }
    showToast("Split selesai");
  } catch (e) {
    showToast(e.message, true);
  }
});

document.getElementById("btnTrain").addEventListener("click", async () => {
  const algorithm = document.getElementById("algoSelect").value;
  if (!algorithm) {
    showToast("Pilih algoritma terlebih dahulu", true);
    return;
  }
  try {
    await withStepLoading({
      cardId: "step-train",
      label: "Melatih model oil & water + cross-validation…",
      buttons: [
        document.getElementById("btnTrain"),
        document.getElementById("algoSelect"),
      ],
      fn: async () => {
        await ensureSession();
        const d = await api("/api/train", {
          method: "POST",
          headers: headersJson(),
          body: JSON.stringify({ algorithm }),
        });
        document.getElementById("cvFolds").value = String(d.cv_folds_used || 5);
        document.getElementById("cvOil").textContent = fmtCvBlock(
          `CV Oil (${d.cv_folds_used}-fold)`,
          d.cross_validation_oil
        );
        document.getElementById("cvWater").textContent = fmtCvBlock(
          `CV Water (${d.cv_folds_used}-fold)`,
          d.cross_validation_water
        );
        document.getElementById("trainSummary").textContent =
          (d.message || "Training selesai.") +
          (d.algorithm ? ` · Algoritma: ${d.algorithm}` : "");
        document.getElementById("btnCv").disabled = false;
        document.getElementById("btnOptimize").disabled = false;
      },
    });
    showToast("Training selesai");
  } catch (e) {
    showToast(e.message, true);
  }
});

document.getElementById("btnCv").addEventListener("click", async () => {
  const cv = parseInt(document.getElementById("cvFolds").value, 10);
  try {
    await withStepLoading({
      cardId: "step-cv",
      label: `Menjalankan cross-validation (${cv} lipatan)…`,
      buttons: [document.getElementById("btnCv"), document.getElementById("cvFolds")],
      fn: async () => {
        await ensureSession();
        const d = await api("/api/cross-validate", {
          method: "POST",
          headers: headersJson(),
          body: JSON.stringify({ cv }),
        });
        document.getElementById("cvOil").textContent = fmtCvBlock(
          `CV Oil (${cv}-fold)`,
          d.cross_validation_oil
        );
        document.getElementById("cvWater").textContent = fmtCvBlock(
          `CV Water (${cv}-fold)`,
          d.cross_validation_water
        );
      },
    });
    showToast("Cross-validation diperbarui");
  } catch (e) {
    showToast(e.message, true);
  }
});

document.getElementById("btnOptimize").addEventListener("click", async () => {
  const n_days = parseInt(document.getElementById("nDays").value, 10);
  const method = document.getElementById("optMethod").value;
  const start_date = document.getElementById("optDateStart").value || "";
  const end_date = document.getElementById("optDateEnd").value || "";
  const datePart =
    start_date && end_date ? `, rentang ${start_date}–${end_date}` : "";
  try {
    await withStepLoading({
      cardId: "step-optimize",
      label: `Optimasi choke (maks. ${n_days} baris${datePart}, ${method})…`,
      buttons: [
        document.getElementById("btnOptimize"),
        document.getElementById("nDays"),
        document.getElementById("optMethod"),
        document.getElementById("optDateStart"),
        document.getElementById("optDateEnd"),
      ],
      fn: async () => {
        await ensureSession();
        const d = await api("/api/optimize", {
          method: "POST",
          headers: headersJson(),
          body: JSON.stringify({ n_days, method, start_date, end_date }),
        });
        let msg = `Optimasi ${d.method}: ${d.n_rows} baris dipakai.`;
        if (d.subset) {
          if (d.subset.mode === "date_range" && d.subset.date_filter) {
            msg += ` Filter DATEPRD ${d.subset.date_filter.start} … ${d.subset.date_filter.end}: ${d.subset.matched_in_range} baris dalam rentang → dipakai ${d.subset.n_rows}.`;
          } else if (d.subset.mode === "head") {
            msg += ` Urutan baris test awal (${d.subset.n_rows} baris).`;
          }
        }
        document.getElementById("optSummary").textContent = msg;
        renderCharts(d.rows);
      },
    });
    showToast("Optimasi selesai");
  } catch (e) {
    showToast(e.message, true);
  }
});

(async function init() {
  try {
    await ensureSession();
    document.getElementById("sessionLabel").textContent = "Sesi: " + sessionId;
    await syncTestDateBoundsFromServer();
  } catch (e) {
    showToast(e.message, true);
  }
})();
