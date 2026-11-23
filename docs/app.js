// Dashboard Light Identifier, browser version.
// Everything runs on the visitor's own device: the photo is never uploaded anywhere.
// The steps are the same as the Python version (src/find_lights.py + src/pipeline.py):
//   photo -> find bright colorful spots -> crop each one -> model says which light it is, or "unsure".
// If the finder misses a light, the visitor can draw a box around it and only that area is classified.

const DETECT_SIDE = 1024;   // find the lights on a copy no bigger than this
const MIN_AREA = 25;        // ignore specks whose bounding box is smaller than this many pixels
const MARGIN = 0.25;        // empty border added around each light, as a fraction of its size
const GAP_FRAC = 0.005;     // pieces of one icon closer than this fraction of the image's short side get merged
const SPARSE_FRAC = 0.10;   // a spot bigger than this fraction of the short side ...
const SPARSE_FILL = 0.15;   // ... that fills less than 15% of its box is a needle or scale arc, not a light
const MAX_LIGHTS = 40;      // more spots than this means the photo is not a clean shot of a dashboard
const MAX_PIXELS = 100_000_000;
const POSSIBLE_FROM = 0.4;  // below the identify threshold but at least this sure: show a low-confidence guess
const COLLAPSE_FROM = 0.8;  // the model is at least this sure a spot is NOT a warning light: list it under "other spots"
const NOTE = "General information only, not a diagnosis. Your owner's manual is the authority for your car. " +
             "If a warning light is on and you are unsure, have the car checked by a mechanic.";

let session, classes, config, meanings;                 // loaded once, when the page opens
let current = null;                                     // the photo on screen: { bitmap, lights, userBoxes }

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
  // Every group of touching lit pixels (up/down/left/right neighbors), like scipy's label():
  // its bounding box and how many pixels it has.
  const seen = new Uint8Array(width * height), stack = new Int32Array(width * height), found = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    let top = 0, count = 0, x0 = width, y0 = height, x1 = -1, y1 = -1;
    stack[top++] = start; seen[start] = 1;
    while (top) {
      const p = stack[--top], x = p % width, y = (p / width) | 0;
      count++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (x > 0 && mask[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack[top++] = p - 1; }
      if (x < width - 1 && mask[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack[top++] = p + 1; }
      if (y > 0 && mask[p - width] && !seen[p - width]) { seen[p - width] = 1; stack[top++] = p - width; }
      if (y < height - 1 && mask[p + width] && !seen[p + width]) { seen[p + width] = 1; stack[top++] = p + width; }
    }
    found.push({ box: [x0, y0, x1 + 1, y1 + 1], count });
  }
  return found;
}

function keepLikelyLights(found, width, height) {
  // Drop specks, and big sparse shapes (needles, scale arcs): the same two rules as find_lights.py.
  const short = Math.min(width, height);
  return found.filter(({ box: [x0, y0, x1, y1], count }) => {
    const w = x1 - x0, h = y1 - y0;
    if (w * h < MIN_AREA) return false;
    if (Math.max(w, h) > SPARSE_FRAC * short && count / (w * h) < SPARSE_FILL) return false;
    return true;
  }).map((c) => c.box);
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

function squareAround([x0, y0, x1, y1]) {
  // A square box with a margin around a light, so crops are never stretched.
  const side = Math.floor(Math.max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN));
  const cx = Math.floor((x0 + x1) / 2), cy = Math.floor((y0 + y1) / 2);
  const sx = cx - Math.floor(side / 2), sy = cy - Math.floor(side / 2);
  return [sx, sy, sx + side, sy + side];
}

function proposeLights(imageData) {
  const { data, width, height } = imageData;
  const kept = keepLikelyLights(components(litMask(data, width, height), width, height), width, height);
  return mergeBoxes(kept, Math.floor(GAP_FRAC * Math.min(width, height))).map(squareAround);
}

// ---- classifying crops (a port of pipeline.py) --------------------------------------------------------
function makeCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  return { canvas, ctx };
}

function wholePhotoBox(width, height) {
  // "Close-up" mode: the whole photo is the light. Pad it to a square so it is not stretched.
  const side = Math.max(width, height);
  return [Math.floor((width - side) / 2), Math.floor((height - side) / 2), Math.floor((width - side) / 2) + side, Math.floor((height - side) / 2) + side];
}

function judge(probs) {
  // Turn the model's probabilities for one crop into what the page shows.
  const none = classes.indexOf("none");
  const best = probs.indexOf(Math.max(...probs)), guess = classes[best], confidence = probs[best];
  const lights = classes.map((name, i) => ({ name, p: probs[i] })).filter((c) => c.name !== "none").sort((a, b) => b.p - a.p);
  let status;
  if (guess !== "none" && confidence >= config.threshold) status = "identified";              // confident AND a real light
  else if (guess !== "none" && confidence >= (config.possible_threshold ?? POSSIBLE_FROM)) status = "possible";
  else if (guess === "none" && probs[none] >= (config.collapse_none_from ?? COLLAPSE_FROM)) status = "other";
  else status = "unsure";
  return { status, best_guess: guess, confidence, none_prob: none >= 0 ? probs[none] : 0, top3: lights.slice(0, 3),
           meaning: status === "identified" ? meanings[guess] || null : null };
}

function renderKnownLights() {
  // The list of lights the model can name, built from the same files the model uses, so it always matches the model.
  const groups = [["warning", "Serious: stop safely or get it checked soon"], ["caution", "Have it checked, or act on it"], ["info", "Information only"]];
  const list = $("knownlist"); list.replaceChildren();
  for (const [severity, title] of groups) {
    const names = classes.filter((c) => c !== "none" && (meanings[c] || {}).severity === severity).map((c) => meanings[c].name);
    if (!names.length) continue;
    const ul = el("ul");
    for (const n of names) ul.append(el("li", "", n));
    list.append(el("h4", severity, title), ul);
  }
  $("known").querySelector("summary").textContent = `Which lights can it recognize? (${classes.length - 1})`;
}

async function classify(bitmap, boxes) {
  // Cut each box from the ORIGINAL photo (more detail), prepare it like the training images, and run the model.
  if (!boxes.length) return [];
  const size = config.img, perCrop = 3 * size * size, batch = new Float32Array(boxes.length * perCrop);
  const crop = makeCanvas(size, size);
  boxes.forEach(([x0, y0, x1, y1], n) => {
    crop.ctx.clearRect(0, 0, size, size);
    crop.ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, size, size);
    const pixels = crop.ctx.getImageData(0, 0, size, size).data;
    for (let i = 0; i < size * size; i++) {                         // the same preparation as training: 0..1, then (x - 0.5) / 0.5
      for (let c = 0; c < 3; c++) batch[n * perCrop + c * size * size + i] = (pixels[i * 4 + c] / 255 - 0.5) / 0.5;
    }
  });
  const output = await session.run({ input: new ort.Tensor("float32", batch, [boxes.length, 3, size, size]) });
  const logits = output.logits.data, nClasses = classes.length, none = classes.indexOf("none");
  return boxes.map((box, n) => {
    const row = Array.from(logits.slice(n * nClasses, (n + 1) * nClasses));
    if (none >= 0) row[none] -= config.none_penalty ?? 0;          // the 'none' penalty chosen in compare_models.py (section 4)
    const top = Math.max(...row), exps = row.map((v) => Math.exp(v - top)), total = exps.reduce((a, b) => a + b, 0);
    return { box, ...judge(exps.map((v) => v / total)) };
  });
}

async function analyze(file, single) {
  let bitmap;
  try { bitmap = await createImageBitmap(file); }                  // also applies the photo's rotation information
  catch (err) { throw new Error("This browser cannot open that file. Try a JPEG or PNG photo."); }
  if (bitmap.width * bitmap.height > MAX_PIXELS) throw new Error("That image is too large. Please use a smaller photo.");

  const k = Math.min(1, DETECT_SIDE / Math.max(bitmap.width, bitmap.height));
  const small = makeCanvas(Math.max(1, Math.round(bitmap.width * k)), Math.max(1, Math.round(bitmap.height * k)));
  small.ctx.drawImage(bitmap, 0, 0, small.canvas.width, small.canvas.height);          // the small copy used for detection
  const found = single ? [wholePhotoBox(small.canvas.width, small.canvas.height)]       // close-up: skip the finder
                       : proposeLights(small.ctx.getImageData(0, 0, small.canvas.width, small.canvas.height));
  found.sort((a, b) => Math.floor(a[1] / 40) - Math.floor(b[1] / 40) || a[0] - b[0]);  // rows about 40 px tall, left to right
  if (found.length > MAX_LIGHTS) {
    throw new Error(`Found ${found.length} bright spots, more than a dashboard normally shows. Try a straighter, closer photo of just the instrument cluster.`);
  }
  const scale = bitmap.width / small.canvas.width;
  const boxes = found.map((b) => b.map((c) => Math.round(c * scale)));                  // back to the original photo's pixels
  const lights = (await classify(bitmap, boxes)).map((l, n) => ({ id: String(n + 1), ...l }));
  return { bitmap, lights, userBoxes: [] };
}

// ---- drawing the results ------------------------------------------------------------------------------
const STYLE = {                                  // how each kind of box is drawn on the photo
  identified: { color: "#3cc85a", width: 3, dash: [] },
  possible:   { color: "#5aa9ff", width: 3, dash: [] },
  unsure:     { color: "#ffaa00", width: 2, dash: [] },
  other:      { color: "rgba(200,200,200,0.7)", width: 1.5, dash: [6, 5] },
  user:       { color: "#d97cff", width: 3, dash: [] },
};

function annotate(view) {
  // The photo (at most 1024 px wide) with a box on every spot; "other" spots are faint dashed outlines without a number.
  const { bitmap, lights, userBoxes } = view;
  const k = Math.min(1, DETECT_SIDE / Math.max(bitmap.width, bitmap.height));
  const out = makeCanvas(Math.round(bitmap.width * k), Math.round(bitmap.height * k));
  out.ctx.drawImage(bitmap, 0, 0, out.canvas.width, out.canvas.height);
  out.ctx.font = "bold 14px sans-serif";
  for (const l of [...lights, ...userBoxes]) {
    const s = STYLE[l.user ? "user" : l.status];
    out.ctx.strokeStyle = out.ctx.fillStyle = s.color; out.ctx.lineWidth = s.width; out.ctx.setLineDash(s.dash);
    const [x0, y0, x1, y1] = (l.drawn || l.box).map((c) => c * k);
    out.ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (l.status !== "other" || l.user) out.ctx.fillText(l.id, x0 + 4, y0 + 15);
  }
  return out.canvas.toDataURL("image/jpeg", 0.85);
}

const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;       // textContent, never innerHTML: nothing is ever run as code
  return e;
}

const pct = (p) => `${Math.round(p * 100)}%`;
const lightName = (name) => (meanings[name] || {}).name || name;
function candidates(l) {
  // The model's best light types for this spot, but only those with at least a 10% chance: lower ones are noise.
  const likely = l.top3.filter((c) => c.p >= 0.10);
  return likely.length ? "Could be: " + likely.map((c) => `${lightName(c.name)} (${pct(c.p)})`).join(", ")
                       : "No warning-light type fits it well.";
}

function card(l) {
  // One result card. Only "identified" lights get advice; everything else says how unsure it is.
  const c = el("div", "card light");
  const label = l.user ? `Your box ${l.id}:` : `${l.id}.`;
  if (l.status === "identified" && l.meaning) {
    const h = el("h3", l.meaning.severity, `${label} ${l.meaning.name}`);
    h.append(el("span", "chip " + l.meaning.severity, l.meaning.severity));
    c.append(h, el("p", "", l.meaning.summary), el("p", "", "What to do: " + l.meaning.what_to_do),
             el("p", "muted", `Confidence ${pct(l.confidence)}`));
  } else if (l.status === "possible") {
    const h = el("h3", "possible", `${label} Possibly: ${lightName(l.best_guess)}`);
    h.append(el("span", "chip possible", "low confidence"));
    c.append(h, el("p", "", `${pct(l.confidence)} sure. That is not enough to rely on, so no advice is shown. Check your owner's manual.`),
             el("p", "muted", candidates(l)));
  } else if (l.status === "other") {
    c.append(el("h3", "muted-title", `${label} Probably not a warning light`),
             el("p", "muted", `${pct(l.none_prob)} sure this is something else (a needle, text, glare or a light it does not know). ${candidates(l)}`));
  } else if (l.best_guess === "none") {
    c.append(el("h3", "unsure", `${label} Unsure, probably not a warning light`),
             el("p", "", `This bright spot is probably something else (a needle, text or glare), but I am only ${pct(l.none_prob)} sure. If it is a warning light, check your owner's manual.`),
             el("p", "muted", candidates(l)));
  } else {
    c.append(el("h3", "unsure", `${label} Unsure`),
             el("p", "", "Something is lit here, but I am not confident what it is. Check your owner's manual."),
             el("p", "muted", candidates(l)));
  }
  return c;
}

function show(view) {
  $("photo").src = annotate(view);
  const box = $("lights"); box.replaceChildren();
  const main = view.lights.filter((l) => l.status !== "other"), others = view.lights.filter((l) => l.status === "other");
  const order = { identified: 0, possible: 1, unsure: 2 };
  if (!view.lights.length) box.append(Object.assign(el("div", "card"), { textContent: "I could not find any lit lights. If you can see one, draw a box around it below." }));
  else if (!main.length) box.append(Object.assign(el("div", "card"), { textContent: "Nothing here looks like a warning light. If one is lit, draw a box around it below." }));
  for (const l of [...view.userBoxes].reverse()) box.append(card(l));                   // your own boxes first, newest on top
  for (const l of [...main].sort((a, b) => order[a.status] - order[b.status] || Number(a.id) - Number(b.id))) box.append(card(l));
  if (others.length) {
    const d = el("details", "card");
    d.append(el("summary", "", `Other bright spots (${others.length}): probably not warning lights`));
    for (const l of others) d.append(card(l));
    box.append(d);
  }
  $("note").textContent = NOTE;
  $("result").hidden = false;
}

// ---- drawing your own box -----------------------------------------------------------------------------
let drawing = false, dragStart = null;

function setDrawMode(on) {
  drawing = on;
  $("frame").classList.toggle("drawing", on);
  $("draw").textContent = on ? "Cancel drawing" : "Draw a box around a light";
  $("drawhint").hidden = !on;
}

function pointerToPhoto(event) {
  // Where the pointer is, in the ORIGINAL photo's pixels (the picture on screen is scaled to fit).
  const r = $("photo").getBoundingClientRect(), f = current.bitmap.width / r.width;
  return [Math.min(Math.max(event.clientX - r.left, 0), r.width) * f, Math.min(Math.max(event.clientY - r.top, 0), r.height) * f];
}

function showRubberBand(a, b) {
  const r = $("photo").getBoundingClientRect(), f = r.width / current.bitmap.width, band = $("band");
  band.hidden = false;
  band.style.left = `${Math.min(a[0], b[0]) * f}px`; band.style.top = `${Math.min(a[1], b[1]) * f}px`;
  band.style.width = `${Math.abs(b[0] - a[0]) * f}px`; band.style.height = `${Math.abs(b[1] - a[1]) * f}px`;
}

$("frame").addEventListener("pointerdown", (e) => {
  if (!drawing || !current) return;
  e.preventDefault(); $("frame").setPointerCapture(e.pointerId); dragStart = pointerToPhoto(e);
});
$("frame").addEventListener("pointermove", (e) => { if (drawing && dragStart) showRubberBand(dragStart, pointerToPhoto(e)); });
$("frame").addEventListener("pointerup", async (e) => {
  if (!drawing || !dragStart) return;
  const end = pointerToPhoto(e), a = dragStart; dragStart = null; $("band").hidden = true;
  const drawn = [Math.min(a[0], end[0]), Math.min(a[1], end[1]), Math.max(a[0], end[0]), Math.max(a[1], end[1])].map(Math.round);
  const minSide = current.bitmap.width / 100;                                   // ignore a plain tap
  if (drawn[2] - drawn[0] < minSide || drawn[3] - drawn[1] < minSide) return;
  setDrawMode(false);
  const [l] = await classify(current.bitmap, [squareAround(drawn)]);
  current.userBoxes.push({ ...l, id: String.fromCharCode(65 + current.userBoxes.length), user: true, drawn });
  window.__lastResult = current;
  show(current);
  $("lights").firstElementChild.scrollIntoView({ behavior: "smooth", block: "nearest" });   // bring the new answer into view
});
$("draw").addEventListener("click", () => setDrawMode(!drawing));

// ---- the page --------------------------------------------------------------------------------------------
async function start() {
  $("status").textContent = "Loading the model...";
  try {
    [classes, config, meanings] = await Promise.all(["model/classes.json", "model/config.json", "meanings.json"].map((u) => fetch(u).then((r) => r.json())));
    ort.env.wasm.numThreads = 1;                      // plain WebAssembly: works on every host, no special server settings
    renderKnownLights();
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
  setDrawMode(false);
  try {
    const single = document.querySelector('input[name="mode"]:checked').value === "single";
    current = await analyze(file, single);
    window.__lastResult = current;                    // kept so automated tests can read the answers
    show(current); $("status").textContent = "";
  } catch (err) {
    $("status").textContent = ""; $("error").textContent = err.message;
  } finally {
    $("go").disabled = false;
  }
});
start();