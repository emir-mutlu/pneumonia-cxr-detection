"use strict";
/*
 * In-browser inference with ONNX Runtime Web: the X-ray never leaves the device.
 *
 * The ONNX graph (`pneumo export-browser`) outputs the calibrated probability and a Grad-CAM
 * map computed in closed form, so no backward pass is needed. Pre-processing mirrors
 * pneumonia_cxr.data.imaging (Pillow): ITU-R 601 luma, then anti-aliased bilinear resampling
 * to 256 px and 224 px with the same fixed-point arithmetic as Pillow.
 */
(function () {
  const INFERNO =
    "00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4";
  const LUT = new Uint8Array(768);
  for (let i = 0; i < 768; i++) LUT[i] = parseInt(INFERNO.substr(i * 2, 2), 16);

  const PRECISION_BITS = 22; // Pillow: 32 - 8 - 2
  const ONE = 1 << PRECISION_BITS;

  /* Pillow's precompute_coeffs + normalize_coeffs_8bpc for the bilinear (triangle) filter. */
  function coefficients(inSize, outSize) {
    const scale = inSize / outSize;
    const filterscale = Math.max(scale, 1);
    const support = filterscale;
    const out = [];
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const ss = 1 / filterscale;
      const xmin = Math.max(0, Math.trunc(center - support + 0.5));
      const xmax = Math.min(inSize, Math.trunc(center + support + 0.5)) - xmin;
      const w = new Float64Array(xmax);
      let total = 0;
      for (let x = 0; x < xmax; x++) {
        const t = Math.abs((x + xmin - center + 0.5) * ss);
        w[x] = t < 1 ? 1 - t : 0;
        total += w[x];
      }
      const k = new Int32Array(xmax);
      for (let x = 0; x < xmax; x++) {
        const v = total ? w[x] / total : 0;
        k[x] = v < 0 ? Math.trunc(v * ONE - 0.5) : Math.trunc(v * ONE + 0.5);
      }
      out.push([xmin, k]);
    }
    return out;
  }
  const clip8 = (v) => (v <= 0 ? 0 : v >= 255 ? 255 : v);

  function horizontal(src, w, h, outW) {
    const co = coefficients(w, outW);
    const dst = new Uint8Array(outW * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < outW; x++) {
        const [xmin, k] = co[x];
        let s = 1 << (PRECISION_BITS - 1);
        for (let i = 0; i < k.length; i++) s += src[row + xmin + i] * k[i];
        dst[y * outW + x] = clip8(Math.floor(s / ONE));
      }
    }
    return dst;
  }

  function vertical(src, w, h, outH) {
    const co = coefficients(h, outH);
    const dst = new Uint8Array(w * outH);
    for (let y = 0; y < outH; y++) {
      const [ymin, k] = co[y];
      for (let x = 0; x < w; x++) {
        let s = 1 << (PRECISION_BITS - 1);
        for (let i = 0; i < k.length; i++) s += src[(ymin + i) * w + x] * k[i];
        dst[y * w + x] = clip8(Math.floor(s / ONE));
      }
    }
    return dst;
  }

  /* Horizontal pass first, then vertical, with an 8-bit intermediate, as Pillow does. */
  function resize(src, w, h, outW, outH) {
    let buf = src;
    if (outW !== w) buf = horizontal(buf, w, h, outW);
    if (outH !== h) buf = vertical(buf, outW, h, outH);
    return buf;
  }

  async function decode(file) {
    const bitmap = await createImageBitmap(file, {
      imageOrientation: "none",
      colorSpaceConversion: "none",
      premultiplyAlpha: "none",
    });
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const n = bitmap.width * bitmap.height;
    const gray = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      gray[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >> 16;
    }
    return { gray, width: bitmap.width, height: bitmap.height, bitmap };
  }

  const canvasUrl = (canvas) =>
    new Promise((resolve) => canvas.toBlob((b) => resolve(URL.createObjectURL(b)), "image/png"));

  async function displayUrl(img, maxSide = 1024) {
    const s = Math.min(1, maxSide / Math.max(img.width, img.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(img.width * s));
    canvas.height = Math.max(1, Math.round(img.height * s));
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.filter = "grayscale(1)";
    ctx.drawImage(img.bitmap, 0, 0, canvas.width, canvas.height);
    return canvasUrl(canvas);
  }

  /* Inferno heatmap whose transparency follows the activation (same curve as the server). */
  async function heatmapUrl(cam, size) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d");
    const data = ctx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const h = Math.min(1, Math.max(0, cam[i]));
      const c = Math.round(h * 255) * 3;
      data.data[i * 4] = LUT[c];
      data.data[i * 4 + 1] = LUT[c + 1];
      data.data[i * 4 + 2] = LUT[c + 2];
      data.data[i * 4 + 3] = Math.round(Math.pow(Math.min(1, Math.max(0, (h - 0.15) / 0.85)), 0.8) * 255);
    }
    ctx.putImageData(data, 0, 0);
    return canvasUrl(canvas);
  }

  function strongestRegion(cam, size) {
    let best = 0;
    for (let i = 1; i < cam.length; i++) if (cam[i] > cam[best]) best = i;
    const y = Math.floor(best / size);
    const x = best % size;
    return {
      side: x < size / 2 ? "right" : "left",
      zone: y < size / 3 ? "upper" : y < (2 * size) / 3 ? "middle" : "lower",
      x: x / size,
      y: y / size,
    };
  }

  async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Model download failed (HTTP ${res.status})`);
    const total = Number(res.headers.get("content-length")) || 0;
    if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (onProgress) onProgress(Math.min(1, got / total), got, total);
    }
    const buf = new Uint8Array(got);
    let offset = 0;
    for (const c of chunks) {
      buf.set(c, offset);
      offset += c.length;
    }
    return buf;
  }

  let session = null;
  let meta = null;

  window.BrowserEngine = {
    name: "browser",
    async init(config, onProgress) {
      const ort = window.ort;
      if (!ort) throw new Error("ONNX Runtime Web could not be loaded");
      ort.env.wasm.wasmPaths = config.wasmPaths;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      meta = meta || (await fetch(config.metaUrl).then((r) => r.json()));
      const bytes = await fetchWithProgress(config.modelUrl, onProgress);
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      const n = meta.img_size;
      await session.run({ image: new ort.Tensor("float32", new Float32Array(n * n).fill(0.5), [1, 1, n, n]) });
    },
    async modelInfo(config) {
      meta = meta || (await fetch(config.metaUrl).then((r) => r.json()));
      return {
        arch: meta.arch,
        train_sources: meta.train_sources,
        img_size: meta.img_size,
        threshold: meta.threshold,
        version: meta.package_version,
        test_metrics: meta.test_metrics,
      };
    },
    examples: () => fetch("examples/examples.json").then((r) => r.json()),
    exampleUrl: (file) => `examples/${encodeURIComponent(file)}`,
    async analyse(file) {
      if (/\.dcm$/i.test(file.name)) {
        throw Object.assign(new Error("DICOM files need the server version."), { code: "dicom" });
      }
      let img;
      try {
        img = await decode(file);
      } catch (e) {
        throw Object.assign(new Error("Could not decode the image."), { code: "decode" });
      }
      const size = meta.img_size;
      const cache = meta.cache_size || 256;
      const t0 = performance.now();
      const small = resize(resize(img.gray, img.width, img.height, cache, cache), cache, cache, size, size);
      const input = new Float32Array(size * size);
      for (let i = 0; i < input.length; i++) input[i] = small[i] / 255;
      const out = await session.run({ image: new window.ort.Tensor("float32", input, [1, 1, size, size]) });
      const probability = out.probability.data[0];
      const cam = out.cam.data;
      const elapsed = performance.now() - t0;
      const [imageSrc, heatSrc] = await Promise.all([displayUrl(img), heatmapUrl(cam, size)]);
      img.bitmap.close();
      return {
        filename: file.name,
        probability,
        threshold: meta.threshold,
        positive: probability >= meta.threshold,
        label: probability >= meta.threshold ? "PNEUMONIA" : "NORMAL",
        region: strongestRegion(cam, size),
        image: { width: img.width, height: img.height, src: imageSrc },
        heatmap: { src: heatSrc },
        inference_ms: elapsed,
        device: "WASM",
      };
    },
  };
})();
