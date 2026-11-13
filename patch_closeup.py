"""One-time patch: adds a "close-up of one light" checkbox to the web demo (docs/app.js and docs/index.html).

Usage (from the repo folder):  python patch_closeup.py
With the box ticked, the light finder is skipped and the whole photo (padded to a square) is classified as one light.
Each edit must match exactly once, otherwise nothing is written."""
from pathlib import Path

JS = [
    ('async function analyze(file) {\n',
     'function wholePhotoBox(width, height) {\n'
     '  // "Close-up" mode: the whole photo is the light. Pad it to a square so it is not stretched.\n'
     '  const side = Math.max(width, height);\n'
     '  return [Math.floor((width - side) / 2), Math.floor((height - side) / 2), Math.floor((width - side) / 2) + side, Math.floor((height - side) / 2) + side];\n'
     '}\n\n'
     'async function analyze(file, single) {\n'),
    ('  const found = proposeLights(small.ctx.getImageData(0, 0, small.canvas.width, small.canvas.height));\n',
     '  const found = single ? [wholePhotoBox(small.canvas.width, small.canvas.height)]       // close-up: skip the finder\n'
     '                       : proposeLights(small.ctx.getImageData(0, 0, small.canvas.width, small.canvas.height));\n'),
    ('    const result = await analyze(file);\n',
     '    const result = await analyze(file, $("single").checked);\n'),
]
HTML = [
    ('    <button id="go" disabled>Identify lights</button>\n',
     '    <button id="go" disabled>Identify lights</button>\n'
     '    <p class="muted"><label><input type="checkbox" id="single"> This photo is a close-up of one light (skips the light finder)</label></p>\n'),
]

def apply(path, edits):
    text = Path(path).read_text()
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f"STOPPED, nothing written to {path}: text found {text.count(old)} times (expected 1):\n{old[:90]}")
        text = text.replace(old, new)
    return text

new_js, new_html = apply("docs/app.js", JS), apply("docs/index.html", HTML)      # check everything first, then write
Path("docs/app.js").write_text(new_js); Path("docs/index.html").write_text(new_html)
print("OK: patched docs/app.js and docs/index.html")