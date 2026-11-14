"use strict";

/* ------------------------------------------------------------------ i18n */
const I18N = {
  en: {
    app_title: "Chest X-ray Pneumonia Assistant",
    app_sub: "AI decision support · research prototype",
    ruo: "Research use only · not for diagnosis",
    theme_toggle: "Toggle dark mode",
    github: "Source code",
    load_study: "Load study",
    drop_title: "Drop a chest X-ray here",
    drop_or: "or",
    browse: "browse files",
    drop_hint: "PNG, JPEG or DICOM · frontal view (PA/AP)",
    samples: "Sample studies",
    samples_hint: "Held-out test images with a known reference label",
    pediatric: "Pediatric",
    adult: "Adult",
    src_kermany: "Kermany test set",
    src_rsna: "RSNA test split",
    ref_pneumonia: "Reference: pneumonia",
    ref_normal: "Reference: normal",
    info: "Study information",
    info_file: "File",
    info_size: "Matrix",
    info_loaded: "Analysed",
    mode_original: "Original",
    mode_overlay: "Heatmap",
    mode_split: "Side by side",
    opacity: "Overlay",
    brightness: "Brightness",
    contrast: "Contrast",
    invert: "Invert",
    reset: "Reset",
    legend_low: "Lower",
    legend_high: "Higher contribution",
    legend_note: "Grad-CAM: regions that most increased the pneumonia score",
    tag_original: "Original",
    tag_overlay: "Grad-CAM",
    empty_title: "No study loaded",
    empty_text: "Upload a frontal chest X-ray or choose a sample study to start the analysis.",
    analysing: "Analysing…",
    findings: "AI findings",
    awaiting: "Awaiting a study",
    awaiting_text: "Results appear here after analysis.",
    pos_title: "Findings suggestive of pneumonia",
    pos_sub: "Probability is above the decision threshold.",
    neg_title: "No pneumonia detected",
    neg_sub: "Probability is below the decision threshold.",
    borderline: "Close to the decision threshold – interpret with caution.",
    prob_caption: "Calibrated probability of pneumonia",
    threshold: "Decision threshold",
    threshold_short: "Threshold",
    evidence: "Strongest evidence",
    reference: "Reference label",
    agrees: "agrees",
    disagrees: "disagrees",
    pneumonia: "Pneumonia",
    normal: "Normal",
    time: "Analysis time",
    side_right: "Right lung",
    side_left: "Left lung",
    zone_upper: "upper zone",
    zone_middle: "middle zone",
    zone_lower: "lower zone",
    evidence_note: "Patient's side, radiological convention (image left = patient's right). Approximate location of the strongest Grad-CAM activation.",
    copy: "Copy report",
    copied: "Copied",
    print: "Print",
    error_title: "Analysis failed",
    error_generic: "The server could not analyse this file.",
    perf: "Validated performance",
    perf_note: "Held-out test sets; threshold fixed before testing. Sensitivity is lower in adults, in patients aged 60+ and on PA views.",
    population: "Population",
    sens: "Sensitivity",
    spec: "Specificity",
    ped_long: "Pediatric (Kermany)",
    adult_long: "Adult (RSNA)",
    model: "Model",
    model_desc: "ConvNeXt-Tiny with ImageNet-22k pre-training, fine-tuned on 24,928 chest X-rays (Kermany pediatric + RSNA adult). Explanations: Grad-CAM.",
    model_version: "Version",
    disclaimer: "Not a medical device. This research prototype must not be used for diagnosis, triage or treatment decisions. Always rely on the assessment of a qualified clinician.",
    report_header: "AI-assisted chest X-ray analysis",
    report_result: "Result",
    report_prob: "Calibrated probability",
    report_model: "Model",
    report_perf: "Validated performance",
    footer: "Built by Emir Mutlu · MIT License",
    model_card: "Model card",
    privacy_note: "Runs entirely in your browser: images are never uploaded.",
    drop_hint_browser: "PNG or JPEG · frontal view (PA/AP)",
    model_loading: "Loading the AI model",
    model_ready: "AI model ready",
    model_failed: "The AI model could not be loaded. Please reload the page.",
    err_dicom: "DICOM files need the server version. Please use PNG or JPEG here.",
    err_decode: "Could not decode the image (PNG or JPEG expected).",
  },
  tr: {
    app_title: "Göğüs Röntgeni Pnömoni Asistanı",
    app_sub: "Yapay zekâ karar desteği · araştırma prototipi",
    ruo: "Yalnızca araştırma amaçlı · tanı için kullanılamaz",
    theme_toggle: "Koyu modu aç/kapat",
    github: "Kaynak kod",
    load_study: "Çalışma yükle",
    drop_title: "Göğüs röntgenini buraya sürükleyin",
    drop_or: "veya",
    browse: "dosya seçin",
    drop_hint: "PNG, JPEG veya DICOM · ön görünüm (PA/AP)",
    samples: "Örnek çalışmalar",
    samples_hint: "Referans etiketi bilinen bağımsız test görüntüleri",
    pediatric: "Pediatrik",
    adult: "Erişkin",
    src_kermany: "Kermany test seti",
    src_rsna: "RSNA test bölümü",
    ref_pneumonia: "Referans: pnömoni",
    ref_normal: "Referans: normal",
    info: "Çalışma bilgileri",
    info_file: "Dosya",
    info_size: "Matris",
    info_loaded: "Analiz",
    mode_original: "Orijinal",
    mode_overlay: "Isı haritası",
    mode_split: "Yan yana",
    opacity: "Katman",
    brightness: "Parlaklık",
    contrast: "Kontrast",
    invert: "Negatif",
    reset: "Sıfırla",
    legend_low: "Düşük",
    legend_high: "Yüksek katkı",
    legend_note: "Grad-CAM: pnömoni skorunu en çok artıran bölgeler",
    tag_original: "Orijinal",
    tag_overlay: "Grad-CAM",
    empty_title: "Çalışma yüklenmedi",
    empty_text: "Analizi başlatmak için bir ön göğüs röntgeni yükleyin veya örnek bir çalışma seçin.",
    analysing: "Analiz ediliyor…",
    findings: "Yapay zekâ bulguları",
    awaiting: "Çalışma bekleniyor",
    awaiting_text: "Sonuçlar analizden sonra burada görünür.",
    pos_title: "Pnömoni ile uyumlu bulgular",
    pos_sub: "Olasılık karar eşiğinin üzerinde.",
    neg_title: "Pnömoni saptanmadı",
    neg_sub: "Olasılık karar eşiğinin altında.",
    borderline: "Karar eşiğine yakın – dikkatle yorumlayın.",
    prob_caption: "Kalibre edilmiş pnömoni olasılığı",
    threshold: "Karar eşiği",
    threshold_short: "Eşik",
    evidence: "En güçlü bulgu bölgesi",
    reference: "Referans etiket",
    agrees: "uyumlu",
    disagrees: "uyumsuz",
    pneumonia: "Pnömoni",
    normal: "Normal",
    time: "Analiz süresi",
    side_right: "Sağ akciğer",
    side_left: "Sol akciğer",
    zone_upper: "üst zon",
    zone_middle: "orta zon",
    zone_lower: "alt zon",
    evidence_note: "Hastanın tarafı, radyolojik konvansiyon (görüntünün solu = hastanın sağı). En güçlü Grad-CAM aktivasyonunun yaklaşık konumu.",
    copy: "Raporu kopyala",
    copied: "Kopyalandı",
    print: "Yazdır",
    error_title: "Analiz başarısız",
    error_generic: "Sunucu bu dosyayı analiz edemedi.",
    perf: "Doğrulanmış performans",
    perf_note: "Bağımsız test setleri; eşik testten önce sabitlendi. Duyarlılık erişkinlerde, 60 yaş üstünde ve PA çekimlerde daha düşüktür.",
    population: "Popülasyon",
    sens: "Duyarlılık",
    spec: "Özgüllük",
    ped_long: "Pediatrik (Kermany)",
    adult_long: "Erişkin (RSNA)",
    model: "Model",
    model_desc: "ImageNet-22k ön eğitimli ConvNeXt-Tiny, 24.928 göğüs röntgeniyle ince ayar (Kermany pediatrik + RSNA erişkin). Açıklama: Grad-CAM.",
    model_version: "Sürüm",
    disclaimer: "Tıbbi cihaz değildir. Bu araştırma prototipi tanı, triyaj veya tedavi kararlarında kullanılamaz. Her zaman yetkin bir hekimin değerlendirmesine başvurun.",
    report_header: "Yapay zekâ destekli göğüs röntgeni analizi",
    report_result: "Sonuç",
    report_prob: "Kalibre edilmiş olasılık",
    report_model: "Model",
    report_perf: "Doğrulanmış performans",
    footer: "Geliştiren: Emir Mutlu · MIT Lisansı",
    model_card: "Model kartı",
    privacy_note: "Tamamen tarayıcınızda çalışır: görüntüler hiçbir sunucuya yüklenmez.",
    drop_hint_browser: "PNG veya JPEG · ön görünüm (PA/AP)",
    model_loading: "Yapay zekâ modeli yükleniyor",
    model_ready: "Yapay zekâ modeli hazır",
    model_failed: "Yapay zekâ modeli yüklenemedi. Lütfen sayfayı yenileyin.",
    err_dicom: "DICOM dosyaları yalnızca sunucu sürümünde desteklenir. Lütfen PNG veya JPEG kullanın.",
    err_decode: "Görüntü okunamadı (PNG veya JPEG bekleniyor).",
  },
};

/* ---------------------------------------------------------------- engines */
// Server engine: the FastAPI backend in app/web.py (PyTorch, Grad-CAM, DICOM support).
const ServerEngine = {
  name: "server",
  async init() {},
  modelInfo: () => fetch("api/model").then((r) => (r.ok ? r.json() : null)),
  examples: () => fetch("api/examples").then((r) => r.json()),
  exampleUrl: (file) => `examples/${encodeURIComponent(file)}`,
  async analyse(file) {
    const form = new FormData();
    form.append("file", file, file.name);
    const res = await fetch("api/analyse", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || "");
    body.image.src = "data:image/png;base64," + body.image.png;
    body.heatmap.src = "data:image/png;base64," + body.heatmap.png;
    return body;
  },
};
// Browser engine (static hosting): ONNX Runtime Web, see engine-browser.js.
const CONFIG = window.CXR_CONFIG || {};
const engine = CONFIG.engine === "browser" && window.BrowserEngine ? window.BrowserEngine : ServerEngine;

const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
};

const $ = (sel) => document.querySelector(sel);
const params = new URLSearchParams(location.search);
const state = {
  lang: (I18N[params.get("lang")] && params.get("lang")) || store.get("cxr-lang") ||
    ((navigator.language || "en").toLowerCase().startsWith("tr") ? "tr" : "en"),
  mode: "overlay",
  opacity: 70,
  brightness: 100,
  contrast: 100,
  invert: false,
  result: null,
  study: null,
  model: null,
  examples: [],
  busy: false,
  engineStatus: engine.name === "browser" ? { phase: "loading", progress: 0 } : { phase: "ready" },
  engineReady: Promise.resolve(),
};

const t = (key) => (I18N[state.lang] && I18N[state.lang][key]) ?? I18N.en[key] ?? key;
const locale = () => (state.lang === "tr" ? "tr-TR" : "en-GB");
const pct = (p, digits = 1) => new Intl.NumberFormat(locale(), { style: "percent", minimumFractionDigits: digits, maximumFractionDigits: digits }).format(p);
const num = (x, digits = 3) => new Intl.NumberFormat(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(x);
function probText(p) {
  if (p >= 0.999) return "> " + pct(0.999);
  if (p <= 0.001) return "< " + pct(0.001);
  return pct(p);
}

/* ------------------------------------------------------------- rendering */
function applyI18n() {
  document.documentElement.lang = state.lang;
  document.title = t("app_title");
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === state.lang)));
  if (engine.name === "browser") $(".drop-hint").textContent = t("drop_hint_browser");
  renderEngine();
  renderSamples();
  renderModel();
  renderResult();
}

function renderEngine() {
  const browser = engine.name === "browser";
  $("#privacy-note").hidden = !browser;
  $("#api-link").hidden = browser;
  const status = $("#engine-status");
  const s = state.engineStatus;
  status.hidden = !browser || (s.phase === "ready" && Boolean(state.result));
  if (!browser) return;
  status.className = "engine-status " + s.phase;
  if (s.phase === "loading") {
    const mb = s.total ? ` · ${(s.got / 1e6).toFixed(0)} / ${(s.total / 1e6).toFixed(0)} MB` : "";
    status.textContent = `${t("model_loading")} ${pct(s.progress || 0, 0)}${mb}`;
  } else if (s.phase === "ready") {
    status.textContent = `✓ ${t("model_ready")}`;
  } else {
    status.textContent = t("model_failed");
  }
}

function renderSamples() {
  const list = $("#samples");
  list.replaceChildren();
  for (const ex of state.examples) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample";
    btn.setAttribute("aria-current", String(state.study?.sample?.file === ex.file));
    btn.addEventListener("click", () => loadSample(ex));
    const img = document.createElement("img");
    img.src = engine.exampleUrl(ex.file);
    img.alt = "";
    img.loading = "lazy";
    const text = document.createElement("span");
    text.className = "sample-text";
    const title = document.createElement("span");
    title.className = "sample-title";
    title.textContent = `${t(ex.population)} · ${ex.view}`;
    const src = document.createElement("span");
    src.className = "sample-src";
    src.textContent = t(`src_${ex.source}`);
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = t(`ref_${ex.reference}`);
    text.append(title, src, chip);
    btn.append(img, text);
    li.append(btn);
    list.append(li);
  }
}

function renderModel() {
  const m = state.model;
  if (!m) return;
  $("#model-thr").textContent = pct(m.threshold);
  $("#model-version").textContent = m.version;
  const body = $("#perf-body");
  body.replaceChildren();
  const rows = [["kermany", "ped_long"], ["rsna", "adult_long"]];
  for (const [key, label] of rows) {
    const met = m.test_metrics?.[key];
    if (!met) continue;
    const tr = document.createElement("tr");
    for (const v of [t(label), pct(met.sensitivity), pct(met.specificity), num(met.auroc)]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(td);
    }
    body.append(tr);
  }
}

function regionText(region) {
  return `${t("side_" + region.side)}, ${t("zone_" + region.zone)}`;
}

function renderResult() {
  const r = state.result;
  $("#placeholder").hidden = Boolean(r) || Boolean(state.error);
  $("#result-card").hidden = !r;
  $("#error-card").hidden = !state.error;
  if (state.error) $("#error-text").textContent = state.error;
  if (!r) return;

  const status = $("#status");
  status.className = "status " + (r.positive ? "positive" : "negative");
  status.querySelector("use").setAttribute("href", r.positive ? "#i-alert" : "#i-check");
  $("#status-title").textContent = t(r.positive ? "pos_title" : "neg_title");
  $("#status-sub").textContent = t(r.positive ? "pos_sub" : "neg_sub");
  $("#borderline").hidden = Math.abs(r.probability - r.threshold) >= 0.1;

  $("#prob-value").textContent = probText(r.probability);
  const meter = $("#meter");
  meter.className = "meter" + (r.positive ? " positive" : "");
  meter.setAttribute("aria-label", `${t("prob_caption")}: ${probText(r.probability)}; ${t("threshold")}: ${pct(r.threshold)}`);
  $("#meter-fill").style.width = `${(r.probability * 100).toFixed(2)}%`;
  $("#meter-tick").style.left = `${(r.threshold * 100).toFixed(2)}%`;
  const thr = $("#meter-thr");
  thr.style.left = `${(r.threshold * 100).toFixed(2)}%`;
  thr.textContent = `${t("threshold_short")} ${pct(r.threshold)}`;

  $("#evidence").textContent = regionText(r.region);
  const sample = state.study?.sample;
  $("#ref-row").hidden = !sample;
  if (sample) {
    const truthPositive = sample.reference === "pneumonia";
    const agrees = truthPositive === r.positive;
    const dd = $("#reference");
    dd.replaceChildren();
    const label = document.createElement("span");
    label.textContent = `${t(sample.reference)} · `;
    const verdict = document.createElement("span");
    verdict.className = agrees ? "agree" : "disagree";
    verdict.textContent = (agrees ? "✓ " : "✗ ") + t(agrees ? "agrees" : "disagrees");
    dd.append(label, verdict);
  }
  const device = r.device ? ` · ${r.device}` : "";
  $("#elapsed").textContent = `${Math.round(r.inference_ms)} ms${device}`;
  $("#info-file").textContent = state.study?.name || "–";
  $("#info-size").textContent = `${r.image.width} × ${r.image.height}`;
  $("#info-time").textContent = state.study?.at?.toLocaleTimeString(locale(), { hour: "2-digit", minute: "2-digit" }) || "–";
  $("#print-meta").textContent = `${state.study?.name || ""} · ${state.study?.at?.toLocaleString(locale()) || ""}`;
  renderViewer();
}

function renderViewer() {
  const r = state.result;
  $("#empty").hidden = Boolean(r);
  $("#frames").hidden = !r;
  $("#legend").hidden = !r || state.mode === "original";
  $(".toolbar").classList.toggle("disabled", !r);
  document.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode)));
  $("#opacity").disabled = state.mode === "original";
  if (!r) return;

  const baseSrc = r.image.src;
  const heatSrc = r.heatmap.src;
  const filter = `brightness(${state.brightness}%) contrast(${state.contrast}%)${state.invert ? " invert(1)" : ""}`;
  const a = $("#frame-a");
  const b = $("#frame-b");
  const split = state.mode === "split";
  b.hidden = !split;

  const setup = (frame, withOverlay, tag) => {
    const base = frame.querySelector(".base");
    const over = frame.querySelector(".overlay");
    if (base.src !== baseSrc) base.src = baseSrc;
    if (over.src !== heatSrc) over.src = heatSrc;
    base.style.filter = filter;
    over.hidden = !withOverlay;
    over.style.opacity = String(state.opacity / 100);
    frame.classList.toggle("show-peak", withOverlay);
    const peak = frame.querySelector(".peak");
    peak.style.left = `${r.region.x * 100}%`;
    peak.style.top = `${r.region.y * 100}%`;
    frame.querySelector(".frame-tag").textContent = split ? t(tag) : "";
  };
  setup(a, state.mode === "overlay", "tag_original");
  if (split) {
    setup(a, false, "tag_original");
    setup(b, true, "tag_overlay");
  }
  fitFrames();
}

/* Fit each frame inside the stage while keeping the image aspect ratio. */
function fitFrames() {
  const r = state.result;
  if (!r) return;
  const frames = $("#frames");
  const n = state.mode === "split" ? 2 : 1;
  const gap = 14;
  const availW = (frames.clientWidth - gap * (n - 1)) / n;
  const availH = frames.clientHeight || $("#stage").clientHeight - 28;
  const ar = r.image.width / r.image.height;
  let w = availW;
  let h = w / ar;
  if (h > availH) { h = availH; w = h * ar; }
  for (const id of ["#frame-a", "#frame-b"]) {
    const f = $(id);
    f.style.width = `${Math.max(0, Math.floor(w))}px`;
    f.style.height = `${Math.max(0, Math.floor(h))}px`;
  }
}

/* --------------------------------------------------------------- actions */
async function analyse(file, sample = null) {
  if (state.busy) return;
  state.busy = true;
  state.error = null;
  $("#busy").hidden = false;
  try {
    await state.engineReady;
    const result = await engine.analyse(file);
    for (const old of [state.result?.image?.src, state.result?.heatmap?.src]) {
      if (old && old.startsWith("blob:")) URL.revokeObjectURL(old);
    }
    state.result = result;
    state.study = { name: file.name, sample, at: new Date() };
  } catch (err) {
    state.result = null;
    state.study = null;
    state.error = (err.code && t("err_" + err.code)) || err.message || t("error_generic");
  } finally {
    state.busy = false;
    $("#busy").hidden = true;
    renderEngine();
    renderSamples();
    renderResult();
    renderViewer();
  }
}

async function loadSample(ex) {
  const res = await fetch(engine.exampleUrl(ex.file));
  const blob = await res.blob();
  await analyse(new File([blob], ex.file, { type: blob.type }), ex);
}

function reportText() {
  const r = state.result;
  if (!r) return "";
  const m = state.model;
  const lines = [
    `${t("report_header")} — ${t("ruo")}`,
    `${t("info_file")}: ${state.study?.name || "-"} (${r.image.width} × ${r.image.height})`,
    `${t("info_loaded")}: ${state.study?.at?.toLocaleString(locale()) || "-"}`,
    "",
    `${t("report_result")}: ${t(r.positive ? "pos_title" : "neg_title")}`,
    `${t("report_prob")}: ${probText(r.probability)} (${t("threshold").toLowerCase()} ${pct(r.threshold)})`,
    `${t("evidence")}: ${regionText(r.region)}`,
  ];
  if (Math.abs(r.probability - r.threshold) < 0.1) lines.push(t("borderline"));
  if (m) {
    lines.push("", `${t("report_model")}: ConvNeXt-Tiny (${(m.train_sources || []).join(" + ")}), v${m.version}`);
    const k = m.test_metrics?.kermany;
    const s = m.test_metrics?.rsna;
    if (k && s) {
      lines.push(
        `${t("report_perf")}: ${t("ped_long")} ${t("sens").toLowerCase()} ${pct(k.sensitivity)}, ${t("spec").toLowerCase()} ${pct(k.specificity)}; ` +
          `${t("adult_long")} ${t("sens").toLowerCase()} ${pct(s.sensitivity)}, ${t("spec").toLowerCase()} ${pct(s.specificity)}`,
      );
    }
  }
  lines.push("", t("disclaimer"));
  return lines.join("\n");
}

async function copyReport() {
  const text = reportText();
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  const label = $("#copy-label");
  label.textContent = t("copied");
  setTimeout(() => { label.textContent = t("copy"); }, 1600);
}

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  store.set("cxr-theme", root.dataset.theme);
}

/* ------------------------------------------------------------------ init */
function bind() {
  document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => {
    state.lang = b.dataset.lang;
    store.set("cxr-lang", state.lang);
    applyI18n();
  }));
  $("#theme-toggle").addEventListener("click", toggleTheme);
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; renderViewer(); }));
  for (const id of ["opacity", "brightness", "contrast"]) {
    $("#" + id).addEventListener("input", (e) => { state[id] = Number(e.target.value); renderViewer(); });
  }
  $("#invert").addEventListener("change", (e) => { state.invert = e.target.checked; renderViewer(); });
  $("#reset-view").addEventListener("click", () => {
    Object.assign(state, { mode: "overlay", opacity: 70, brightness: 100, contrast: 100, invert: false });
    $("#opacity").value = 70; $("#brightness").value = 100; $("#contrast").value = 100; $("#invert").checked = false;
    renderViewer();
  });
  const input = $("#file-input");
  input.addEventListener("change", () => { if (input.files[0]) analyse(input.files[0]); input.value = ""; });
  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove("drag")));
  dz.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) analyse(f); });
  document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) analyse(f); });
  $("#copy-report").addEventListener("click", copyReport);
  $("#print-report").addEventListener("click", () => window.print());
  new ResizeObserver(fitFrames).observe($("#stage"));
}

async function init() {
  bind();
  applyI18n();
  renderViewer();
  state.engineReady = engine
    .init(CONFIG, (progress, got, total) => {
      state.engineStatus = { phase: "loading", progress, got, total };
      renderEngine();
    })
    .then(() => {
      state.engineStatus = { phase: "ready" };
      renderEngine();
    })
    .catch((err) => {
      console.error(err);
      state.engineStatus = { phase: "failed" };
      renderEngine();
      throw Object.assign(new Error(t("model_failed")), { code: "" });
    });
  state.engineReady.catch(() => {});
  const [model, examples] = await Promise.all([
    engine.modelInfo(CONFIG).catch(() => null),
    engine.examples().catch(() => []),
  ]);
  state.model = model;
  state.examples = examples;
  applyI18n();
  const auto = params.get("example");
  if (auto) {
    const ex = examples.find((e) => e.file === auto) || examples[Number(auto)];
    if (ex) loadSample(ex);
  }
}

init();
