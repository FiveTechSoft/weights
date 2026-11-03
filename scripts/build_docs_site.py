"""
build_docs_site.py
Generates clean, styled HTML pages for docs/*.md so the GitHub Pages site is 100% self-contained.
"""

import re
from pathlib import Path

STYLE = """
<style>
  :root {
    --bg: #090D16;
    --card-bg: #0F172A;
    --card-border: #1E293B;
    --accent-cyan: #38BDF8;
    --accent-purple: #A855F7;
    --accent-emerald: #10B981;
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
    --code-bg: #0B1120;
  }
  body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    background-color: var(--bg);
    color: var(--text-main);
    line-height: 1.6;
    padding: 0;
    margin: 0;
  }
  .nav {
    background: #0D1322;
    border-bottom: 1px solid var(--card-border);
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .nav a { color: var(--accent-cyan); text-decoration: none; font-weight: 600; }
  .container {
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1.5rem 4rem 1.5rem;
  }
  h1, h2, h3, h4 { color: #FFFFFF; margin-top: 1.75rem; margin-bottom: 0.75rem; }
  h1 { font-size: 2.2rem; border-bottom: 1px solid var(--card-border); padding-bottom: 0.5rem; }
  h2 { font-size: 1.5rem; color: var(--accent-cyan); }
  h3 { font-size: 1.2rem; }
  p, li { color: #CBD5E1; margin-bottom: 0.85rem; }
  ul, ol { padding-left: 1.5rem; margin-bottom: 1rem; }
  li { margin-bottom: 0.35rem; }
  a { color: var(--accent-cyan); text-decoration: none; }
  a:hover { text-decoration: underline; }
  pre {
    background: var(--code-bg);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--card-border);
    overflow-x: auto;
    font-family: 'Consolas', monospace;
    font-size: 0.9rem;
    color: #E2E8F0;
  }
  code {
    background: rgba(56, 189, 248, 0.1);
    color: var(--accent-cyan);
    padding: 0.15rem 0.35rem;
    border-radius: 0.25rem;
    font-family: 'Consolas', monospace;
    font-size: 0.9em;
  }
  pre code {
    background: none;
    color: inherit;
    padding: 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5rem 0;
  }
  th, td {
    border: 1px solid var(--card-border);
    padding: 0.65rem 0.85rem;
    text-align: left;
  }
  th { background: #131D33; color: var(--accent-cyan); }
  blockquote {
    border-left: 4px solid var(--accent-purple);
    padding: 0.5rem 1rem;
    margin: 1rem 0;
    background: rgba(168, 85, 247, 0.08);
    color: #E2E8F0;
    font-style: italic;
  }
  hr {
    border: none;
    border-top: 1px solid var(--card-border);
    margin: 2rem 0;
  }
</style>
"""

def md_to_html(md_text, title):
    # Very simple markdown converter for headings, code, bold, lists
    html = md_text
    # Escape basic html chars in code blocks first
    code_blocks = []
    def save_cb(m):
        code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"
    html = re.sub(r"```[\s\S]*?```", save_cb, html)
    
    # Inline code
    inline_codes = []
    def save_ic(m):
        inline_codes.append(f"<code>{m.group(1)}</code>")
        return f"__INLINE_CODE_{len(inline_codes)-1}__"
    html = re.sub(r"`([^`]+)`", save_ic, html)

    # Headers
    html = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", html, flags=re.M)
    html = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", html, flags=re.M)
    html = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", html, flags=re.M)

    # Bold and italics
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)

    # Blockquotes
    html = re.sub(r"^> (.*?)$", r"<blockquote>\1</blockquote>", html, flags=re.M)
    # Horizontal rules
    html = re.sub(r"^---$", r"<hr>", html, flags=re.M)

    # Restore code blocks
    for i, cb in enumerate(code_blocks):
        lines = cb.splitlines()
        lang = lines[0].replace("```", "").strip()
        code_content = "\n".join(lines[1:-1])
        code_content = code_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = html.replace(f"__CODE_BLOCK_{i}__", f"<pre><code>{code_content}</code></pre>")

    # Restore inline code
    for i, ic in enumerate(inline_codes):
        html = html.replace(f"__INLINE_CODE_{i}__", ic)

    # Paragraphs (lines separated by blank lines)
    paragraphs = html.split("\n\n")
    processed = []
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        if p.startswith("<h") or p.startswith("<pre") or p.startswith("<blockquote") or p.startswith("<hr") or p.startswith("<table"):
            processed.append(p)
        elif p.startswith("- ") or p.startswith("* "):
            items = [f"<li>{line[2:]}</li>" for line in p.splitlines() if line.startswith("- ") or line.startswith("* ")]
            processed.append(f"<ul>{''.join(items)}</ul>")
        elif re.match(r"^\d+\. ", p):
            items = [f"<li>{re.sub(r'^\d+\.\s*', '', line)}</li>" for line in p.splitlines() if re.match(r"^\d+\. ", line)]
            processed.append(f"<ol>{''.join(items)}</ol>")
        else:
            processed.append(f"<p>{p.replace(chr(10), '<br>')}</p>")
            
    body = "\n\n".join(processed)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} · FiveTechSoft / weights</title>
  {STYLE}
</head>
<body>
  <div class="nav">
    <a href="index.html">← Back to WSP-Lang Overview</a>
    <a href="https://github.com/FiveTechSoft/weights" target="_blank">GitHub Repository ↗</a>
  </div>
  <div class="container">
    {body}
  </div>
</body>
</html>"""

def main():
    docs_dir = Path("docs")
    for md_name in ["getting_started.md", "behavioral_steering_and_dreaming.md"]:
        md_file = docs_dir / md_name
        if md_file.exists():
            html_name = md_name.replace(".md", ".html")
            html_content = md_to_html(md_file.read_text(encoding="utf-8"), md_name.replace(".md", ""))
            (docs_dir / html_name).write_text(html_content, encoding="utf-8")
            print(f"[OK] Generated {docs_dir / html_name}")

if __name__ == "__main__":
    main()
