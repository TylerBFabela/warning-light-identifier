"""One-time patch: adds a middle tier, "possible", to the web demo (docs/app.js and docs/index.html).

Usage (from the repo folder):  python patch_possible.py
A light is "identified" at high confidence, "possible" at middling confidence (shown as a low-confidence guess with
NO advice), and "unsure" otherwise. Each edit must match exactly once, otherwise nothing is written."""
from pathlib import Path

JS = [
    ('const MAX_PIXELS = 100_000_000;\n',
     'const MAX_PIXELS = 100_000_000;\nconst POSSIBLE_FROM = 0.4;  // below the identify threshold but at least this sure: show a low-confidence guess\n'),
    ('      const sure = confidence >= config.threshold && guess !== "none";    // identified only if confident AND a real light\n'
     '      lights.push({ id: n + 1, box, status: sure ? "identified" : "unsure", label: sure ? guess : "unsure",\n',
     '      const sure = confidence >= config.threshold && guess !== "none";    // identified only if confident AND a real light\n'
     '      const maybe = !sure && guess !== "none" && confidence >= (config.possible_threshold ?? POSSIBLE_FROM);\n'
     '      lights.push({ id: n + 1, box, status: sure ? "identified" : maybe ? "possible" : "unsure", label: sure ? guess : "unsure",\n'),
    ('    view.ctx.strokeStyle = view.ctx.fillStyle = l.status === "identified" ? "#3cc85a" : "#ffaa00";\n',
     '    view.ctx.strokeStyle = view.ctx.fillStyle = { identified: "#3cc85a", possible: "#5aa9ff", unsure: "#ffaa00" }[l.status];\n'),
    ('    } else {\n      c.append(el("h3", "unsure", `${l.id}. Unsure`),\n',
     '    } else if (l.status === "possible") {\n'
     '      const h = el("h3", "possible", `${l.id}. Possible match`);\n'
     '      h.append(el("span", "chip possible", "low confidence"));\n'
     '      const name = (meanings[l.best_guess] || {}).name || l.best_guess;\n'
     '      c.append(h, el("p", "", `This might be: ${name} (${Math.round(l.confidence * 100)}% sure). That is not enough to rely on, so no advice is shown. Check your owner\'s manual.`));\n'
     '    } else {\n      c.append(el("h3", "unsure", `${l.id}. Unsure`),\n'),
]
HTML = [
    ('--info:#2457d6; --caution:#b36b00; --warning:#c0262d; --unsure:#7a6a00; }\n',
     '--info:#2457d6; --caution:#b36b00; --warning:#c0262d; --unsure:#7a6a00; --possible:#2f6fb5; }\n'),
    ('--info:#6f9bff; --caution:#f0b04a; --warning:#ff6b73; --unsure:#d8c85a; } }\n',
     '--info:#6f9bff; --caution:#f0b04a; --warning:#ff6b73; --unsure:#d8c85a; --possible:#7fb4ff; } }\n'),
    ('.unsure { color:var(--unsure); }\n', '.unsure { color:var(--unsure); } .possible { color:var(--possible); }\n'),
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