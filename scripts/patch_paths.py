from pathlib import Path

p = Path("wsp.py")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()
for i, l in enumerate(lines):
    if l.startswith("ROOT = Path("):
        lines[i] = "ROOT = Path(__file__).resolve().parent"
    elif 'DIR_EDI = ROOT / "ediciones"' in l:
        lines[i] = 'DIR_EDI = ROOT / "edits" if (ROOT / "edits").exists() else ROOT / "ediciones"'
    elif 'DIR_SRV = ROOT / "servidos"' in l:
        lines[i] = 'DIR_SRV = ROOT / "served" if (ROOT / "served").exists() else ROOT / "servidos"'
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[OK] wsp.py paths updated")
