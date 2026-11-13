// Dashboard Light Identifier, browser version.
// Everything runs on the visitor's own device: the photo is never uploaded anywhere.
// The steps are the same as the Python version (src/find_lights.py + src/pipeline.py):
//   photo -> find bright colorful spots -> crop each one -> model says which light it is, or "unsure".

const DETECT_SIDE = 1024;   // find the lights on a copy no bigger than this
const MIN_AREA = 25;        // ignore specks whose bounding box is smaller than this many pixels
const MARGIN = 0.25;        // empty border added around each light, as a fraction of its size
const GAP_FRAC = 0.01;      // pieces of one icon closer than this fraction of the image get merged
const MAX_LIGHTS = 40;      // more spots than this means the photo is not a clean shot of a dashboard
const MAX_PIXELS = 100_000_000;
const POSSIBLE_FROM = 0.4;  // below the identify threshold but at least this sure: show a low-confidence guess
const NOTE = "General information only, not a diagnosis. Your owner's manual is the authority for your car. " +
             "If a warning light is on and you are unsure, have the car checked by a mechanic.";

let session, classes, config, meanings;                 // loaded once, when the page opens

// ---- finding the lights (a port of find_lights.py) ----------------------------------------------------
function litMask(data, width, height) {
  // "lit" = colorful (saturation > 0.5) and bright (value > 0.45). On 0-255 numbers that is S >= 128 and V >= 115.
  const mask = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const saturation = max === 0 ? 0 : Math.floor((255 * (max - min)) / max);
    if (saturation >= 128 && max >= 115) mask[i] = 1;
  }
  return mask;
}

function components(mask, width, height) {
  // Bounding box of every group of touching lit pixels (up/down/left/right neighbors), like scipy's label().
  const seen = new Uint8Array(width * height), stack = new Int32Array(width * height), boxes = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    let top = 0, x0 = width, y0 = height, x1 = -1, y1 = -1;
    stack[top++] = start; seen[start] = 1;
    while (top) {
      const p = stack[--top], x = p % width, y = (p / width) | 0;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (x > 0 && mask[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack[top++] = p - 1; }
      if (x < width - 1 && mask[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack[top++] = p + 1; }
      if (y > 0 && mask[p - width] && !seen[p - width]) { seen[p - width] = 1; stack[top++] = p - width; }
      if (y < height - 1 && mask[p + width] && !seen[p + width]) { seen[p + width] = 1; stack[top++] = p + width; }
    }
    if ((x1 - x0 + 1) * (y1 - y0 + 1) >= MIN_AREA) boxes.push([x0, y0, x1 + 1, y1 + 1]);
  }
  return boxes;
}

function mergeBoxes(boxes, gap) {
  // Join boxes that overlap or sit within `gap` pixels of each other (the parts of one icon).
  boxes = boxes.map((b) => b.slice());
  let changed = true;
  while (changed) {
    changed = false;
    search: for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        if (a[0] - gap <= b[2] && b[0] - gap <= a[2] && a[1] - gap <= b[3] && b[1] - gap <= a[3]) {
          boxes[i] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
          boxes.splice(j, 1); changed = true; break search;
        }
      }
    }
  }
  return boxes;
}

function proposeLights(imageData) {
  const { data, width, height } = imageData;
  const merged = mergeBoxes(components(litMask(data, width, height), width, height), Math.floor(GAP_FRAC * Math.min(width, height)));
  return merged.map(([x0, y0, x1, y1]) => {                       // square boxes with a margin, so crops are never stretched
    const side = Math.floor(Math.max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN));
    const cx = Math.floor((x0 + x1) / 2), cy = Math.floor((y0 + y1) / 2);
    const sx = cx - Math.floor(side / 2), sy = cy - Math.floor(side / 2);
    return [sx, sy, sx + side, sy + side];
  });
}

// ---- the whole job (a port of pipeline.py) ------------------------------------------------------------
function makeCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  return { canvas, ctx };
}

async function analyze(file) {
  let bitmap;
  try { bitmap = await createImageBitmap(file); }                  // also applies the photo's rotation information
  catch (err) { throw new Error("This browser cannot open that file. Try a JPEG or PNG photo."); }
  if (bitmap.width * bitmap.height > MAX_PIXELS) throw new Error("That image is too large. Please use a smaller photo.");

  const k = Math.min(1, DETECT_SIDE / Math.max(bitmap.width, bitmap.height));
  const small = makeCanvas(Math.max(1, Math.round(bitmap.width * k)), Math.max(1, Math.round(bitmap.height * k)));
  small.ctx.drawImage(bitmap, 0, 0, small.canvas.width, small.canvas.height);          // the small copy used for detection
  const found = proposeLights(small.ctx.getImageData(0, 0, small.canvas.width, small.canvas.height));
  found.sort((a, b) => Math.floor(a[1] / 40) - Math.floor(b[1] / 40) || a[0] - b[0]);  // rows about 40 px tall, left to right
  if (found.length > MAX_LIGHTS) {
    throw new Error(`Found ${found.length} bright spots, more than a dashboard normally shows. Try a straighter, closer photo of just the instrument cluster.`);
  }
  const scale = bitmap.width / small.canvas.width;
  const boxes = found.map((b) => b.map((c) => Math.round(c * scale)));                  // back to the original photo's pixels

  const size = config.img, perCrop = 3 * size * size, batch = new Float32Array(boxes.length * perCrop);
  const crop = makeCanvas(size, size);
  boxes.forEach(([x0, y0, x1, y1], n) => {                          // cut each crop from the ORIGINAL photo (more detail)
    crop.ctx.clearRect(0, 0, size, size);
    crop.ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, size, size);
    const pixels = crop.ctx.getImageData(0, 0, size, size).data;
    for (let i = 0; i < size * size; i++) {                         // the same preparation as training: 0..1, then (x - 0.5) / 0.5
      for (let c = 0; c < 3; c++) batch[n * perCrop + c * size * size + i] = (pixels[i * 4 + c] / 255 - 0.5) / 0.5;
    }
  });

  const lights = [];
  if (boxes.length) {
    const output = await session.run({ input: new ort.Tensor("float32", batch, [boxes.length, 3, size, size]) });
    const logits = output.logits.data, nClasses = classes.length;
    boxes.forEach((box, n) => {
      const row = Array.from(logits.slice(n * nClasses, (n + 1) * nClasses));
      const top = Math.max(...row), exps = row.map((v) => Math.exp(v - top)), total = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((v) => v / total), best = probs.indexOf(Math.max(...probs));
      const guess = classes[best], confidence = probs[best];
      const sure = confidence >= config.threshold && guess !== "none";    // identified only if confident AND a real light
      const maybe = !sure && guess !== "none" && confidence >= (config.possible_threshold ?? POSSIBLE_FROM);
      lights.push({ id: n + 1, box, status: sure ? "identified" : maybe ? "possible" : "unsure", label: sure ? guess : "unsure",
                    best_guess: guess, confidence, meaning: sure ? meanings[guess] || null : null });
    });
  }
  return { lights, image: annotate(bitmap, lights) };
}

function annotate(bitmap, lights) {
  // A picture of the photo (at most 1024 px wide) with a numbered box on every light: green = identified, orange = unsure.
  const k = Math.min(1, DETECT_SIDE / Math.max(bitmap.width, bitmap.height));
  const view = makeCanvas(Math.round(bitmap.width * k), Math.round(bitmap.height * k));
  view.ctx.drawImage(bitmap, 0, 0, view.canvas.width, view.canvas.height);
  view.ctx.lineWidth = 3; view.ctx.font = "14px sans-serif";
  for (const l of lights) {
    view.ctx.strokeStyle = view.ctx.fillStyle = { identified: "#3cc85a", possible: "#5aa9ff", unsure: "#ffaa00" }[l.status];
    const [x0, y0, x1, y1] = l.box.map((c) => c * k);
    view.ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    view.ctx.fillText(String(l.id), x0 + 4, y0 + 15);
  }
  return view.canvas.toDataURL("image/jpeg", 0.85);
}

// ---- the page --------------------------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;       // textContent, never innerHTML: nothing is ever run as code
  return e;
}

function show(result) {
  $("photo").src = result.image;
  const box = $("lights"); box.replaceChildren();
  if (!result.lights.length) {
    const c = el("div", "card"); c.append(el("p", "", "I could not find any lit lights. Try a closer, sharper photo with the lights on."));
    box.append(c);
  }
  for (const l of result.lights) {
    const c = el("div", "card light");
    if (l.status === "identified" && l.meaning) {
      const h = el("h3", l.meaning.severity, `${l.id}. ${l.meaning.name}`);
      h.append(el("span", "chip " + l.meaning.severity, l.meaning.severity));
      c.append(h, el("p", "", l.meaning.summary), el("p", "", "What to do: " + l.meaning.what_to_do),
               el("p", "muted", `Confidence ${Math.round(l.confidence * 100)}%`));
    } else if (l.status === "possible") {
      const h = el("h3", "possible", `${l.id}. Possible match`);
      h.append(el("span", "chip possible", "low confidence"));
      const name = (meanings[l.best_guess] || {}).name || l.best_guess;
      c.append(h, el("p", "", `This might be: ${name} (${Math.round(l.confidence * 100)}% sure). That is not enough to rely on, so no advice is shown. Check your owner's manual.`));
    } else {
      c.append(el("h3", "unsure", `${l.id}. Unsure`),
               el("p", "", "A bright spot is here, but I am not confident what it is. Check your owner's manual."),
               el("p", "muted", `Closest guess: ${l.best_guess === "none" ? "not a known light" : l.best_guess} (${Math.round(l.confidence * 100)}%)`));
    }
    box.append(c);
  }
  $("note").textContent = NOTE;
  $("result").hidden = false;
}

async function start() {
  $("status").textContent = "Loading the model...";
  try {
    [classes, config, meanings] = await Promise.all(["model/classes.json", "model/config.json", "meanings.json"].map((u) => fetch(u).then((r) => r.json())));
    ort.env.wasm.numThreads = 1;                      // plain WebAssembly: works on every host, no special server settings
    session = await ort.InferenceSession.create("model/model.onnx", { executionProviders: ["wasm"] });
    $("status").textContent = "";
    $("go").disabled = !$("file").files.length;
    window.__ready = true;
  } catch (err) {
    $("status").textContent = ""; $("error").textContent = "Could not load the model: " + err.message;
  }
}

$("file").addEventListener("change", () => { $("go").disabled = !$("file").files.length || !session; });
$("go").addEventListener("click", async () => {
  const file = $("file").files[0];
  if (!file) return;
  $("go").disabled = true; $("error").textContent = ""; $("status").textContent = "Analyzing..."; $("result").hidden = true;
  try {
    const result = await analyze(file);
    window.__lastResult = result;                     // kept so automated tests can read the answers
    show(result); $("status").textContent = "";
  } catch (err) {
    $("status").textContent = ""; $("error").textContent = err.message;
  } finally {
    $("go").disabled = false;
  }
});
start();