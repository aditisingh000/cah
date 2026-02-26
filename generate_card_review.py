"""
Generate an HTML page that displays all extracted card images in a grid for quick review.
Run this script, then open card_review.html in a browser (or use a local server).
"""

import os
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent
    extracted = base / "extracted_cards"
    if not extracted.exists():
        print("extracted_cards folder not found")
        return

    categories = {}
    for subdir in sorted(extracted.iterdir()):
        if subdir.is_dir():
            cards = sorted(subdir.glob("*.png"), key=lambda p: p.name)
            categories[subdir.name] = [
                str(Path("extracted_cards") / subdir.name / p.name) for p in cards
            ]

    sections_html = []
    for cat_name, paths in categories.items():
        title = cat_name.replace("_", " ").title()
        cards_html = []
        for path in paths:
            name = Path(path).stem
            cards_html.append(
                f'<div class="card-cell" title="{name}">'
                f'<img src="{path}" alt="{name}" loading="lazy" />'
                f'<span class="card-label">{name}</span></div>'
            )
        sections_html.append(
            f'<section id="{cat_name}" class="category">'
            f'<h2>{title} ({len(paths)} cards)</h2>'
            f'<div class="card-grid">{"".join(cards_html)}</div></section>'
        )

    nav = "".join(
        f'<a href="#{cat_name}">{cat_name.replace("_", " ").title()}</a>'
        for cat_name in categories
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CAH Cards Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1rem; background: #1a1a1a; color: #eee; }}
    h1 {{ margin-top: 0; }}
    .nav {{ margin-bottom: 1.5rem; display: flex; gap: 1rem; flex-wrap: wrap; }}
    .nav a {{ color: #7dd; text-decoration: none; padding: 0.4rem 0.8rem; background: #333; border-radius: 6px; }}
    .nav a:hover {{ background: #444; }}
    .category {{ margin-bottom: 3rem; }}
    .category h2 {{ color: #9cf; margin-bottom: 1rem; font-size: 1.25rem; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }}
    .card-cell {{ background: #2a2a2a; border-radius: 8px; padding: 6px; text-align: center; }}
    .card-cell img {{ max-width: 100%; height: auto; max-height: 180px; object-fit: contain; display: block; margin: 0 auto; border-radius: 4px; }}
    .card-label {{ font-size: 0.7rem; color: #888; display: block; margin-top: 4px; }}
    .card-cell:hover {{ background: #353535; }}
    .card-cell img:hover {{ cursor: pointer; }}
    /* Lightbox */
    .lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 9999; align-items: center; justify-content: center; padding: 2rem; }}
    .lightbox.show {{ display: flex; }}
    .lightbox img {{ max-width: 95%; max-height: 95%; object-fit: contain; }}
    .lightbox-close {{ position: absolute; top: 1rem; right: 1rem; color: #fff; font-size: 2rem; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>CAH Extracted Cards Review</h1>
  <nav class="nav">{nav}</nav>
  {"".join(sections_html)}
  <div id="lightbox" class="lightbox" onclick="closeLightbox(event)">
    <span class="lightbox-close">&times;</span>
    <img id="lightbox-img" src="" alt="">
  </div>
  <script>
    document.querySelectorAll('.card-cell img').forEach(img => {{
      img.addEventListener('click', () => {{
        document.getElementById('lightbox-img').src = img.src;
        document.getElementById('lightbox').classList.add('show');
      }});
    }});
    function closeLightbox(e) {{ if (e.target.id === 'lightbox' || e.target.classList.contains('lightbox-close')) document.getElementById('lightbox').classList.remove('show'); }}
  </script>
</body>
</html>
"""

    out = base / "card_review.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    print("To view: run  python -m http.server 8765  then open  http://localhost:8765/card_review.html")
    print("(Or open card_review.html directly; some browsers may block images when opened as file://)")

if __name__ == "__main__":
    main()
