// Loaded from config.json injected by CDK at deploy time
let API_URL = null;

// ── Boot ─────────────────────────────────────────────────────────────────────

(async () => {
  await loadConfig();
  await loadSamples();
  wireUpload();
})();

async function loadConfig() {
  try {
    const res = await fetch("/config.json");
    if (!res.ok) throw new Error("not found");
    const cfg = await res.json();
    API_URL = cfg.apiUrl.replace(/\/$/, "");
  } catch {
    // Running locally without a deployed API — samples still work visually,
    // but classification will show an error when clicked.
    console.warn("config.json not found — running in local preview mode.");
  }
}

// Loads samples/manifest.json — { hotdog: [...], not_hotdog: [...] }
// Renders two labeled rows of clickable thumbnails.
async function loadSamples() {
  try {
    const res = await fetch("/samples/manifest.json");
    if (!res.ok) return;
    const manifest = await res.json();

    const hasHotdog = manifest.hotdog?.length > 0;
    const hasNotHotdog = manifest.not_hotdog?.length > 0;
    if (!hasHotdog && !hasNotHotdog) return;

    if (hasHotdog) {
      const grid = document.getElementById("samplesHotdog");
      manifest.hotdog.forEach(name => grid.appendChild(makeSampleThumb(name)));
    }
    if (hasNotHotdog) {
      const grid = document.getElementById("samplesNotHotdog");
      manifest.not_hotdog.forEach(name => grid.appendChild(makeSampleThumb(name)));
    }

    document.getElementById("samplesCard").classList.remove("hidden");
  } catch {
    // no samples — that's fine
  }
}

function makeSampleThumb(name) {
  const img = document.createElement("img");
  img.src = `/samples/${name}`;
  img.alt = name;
  img.className = [
    "w-full aspect-square object-cover rounded-xl cursor-pointer",
    "border-2 border-transparent hover:border-yellow-400",
    "transition-all hover:scale-105",
  ].join(" ");
  img.addEventListener("click", () => classifyUrl(`/samples/${name}`, img.src));
  return img;
}

// ── Upload wiring ─────────────────────────────────────────────────────────────

function wireUpload() {
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("fileInput");

  zone.addEventListener("click", () => input.click());

  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("border-yellow-500");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("border-yellow-500"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("border-yellow-500");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) handleFile(input.files[0]);
  });
}

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select an image file (JPG, PNG, WebP).");
    return;
  }
  const objectUrl = URL.createObjectURL(file);
  resizeToBase64(objectUrl, 512).then(b64 => classify(b64, objectUrl));
}

async function classifyUrl(src, displaySrc) {
  const b64 = await resizeToBase64(src, 512);
  classify(b64, displaySrc);
}

// ── Image resize (client-side, keeps payload under 6 MB) ─────────────────────

function resizeToBase64(src, maxPx) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const scale = Math.min(1, maxPx / Math.max(img.width, img.height));
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      // strip the "data:image/jpeg;base64," prefix
      resolve(canvas.toDataURL("image/jpeg", 0.9).split(",")[1]);
    };
    img.onerror = reject;
    img.src = src;
  });
}

// ── API call ──────────────────────────────────────────────────────────────────

async function classify(base64, previewSrc) {
  if (!API_URL) {
    showError("API URL not loaded. Check config.json.");
    return;
  }

  setLoading(true);
  hideResult();
  hideError();

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    showResult(data, previewSrc);
  } catch (err) {
    showError(`Prediction failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function showResult(data, previewSrc) {
  const isHotdog = data.prediction === "hotdog";
  const pct = Math.round(data.confidence * 100);

  document.getElementById("previewImg").src = previewSrc;
  document.getElementById("predLabel").textContent = isHotdog ? "🌭 Hotdog!" : "❌ Not Hotdog";
  document.getElementById("predLabel").className =
    `text-2xl font-bold mb-3 ${isHotdog ? "text-yellow-400" : "text-red-400"}`;

  document.getElementById("confPct").textContent = `${pct}%`;
  const fill = document.getElementById("confFill");
  fill.className = `h-2 rounded-full transition-all duration-700 ${isHotdog ? "bg-yellow-400" : "bg-red-500"}`;
  // Trigger CSS transition
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fill.style.width = `${pct}%`;
  }));

  document.getElementById("p0").textContent = `${Math.round(data.hotdog * 100)}%`;
  document.getElementById("p1").textContent = `${Math.round(data.not_hotdog * 100)}%`;

  document.getElementById("result").classList.remove("hidden");
  document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideResult() {
  document.getElementById("result").classList.add("hidden");
  document.getElementById("confFill").style.width = "0%";
}

function showError(msg) {
  document.getElementById("errorMsg").textContent = msg;
  document.getElementById("error").classList.remove("hidden");
}

function hideError() {
  document.getElementById("error").classList.add("hidden");
}

function setLoading(on) {
  document.getElementById("loading").classList.toggle("hidden", !on);
}
