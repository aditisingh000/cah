# CAH – Cards Against Humanity (card assets & game)

Card assets extracted from CAH PDFs and tooling for a **browser-based, mobile-friendly multiplayer** game where friends join a room by code and play together.

---

## What’s done

### Card extraction (“card cutting”) pipeline

- **`extract_cards.py`**  
  - Reads CAH PDFs (Family Edition and Original) and extracts individual card images.
  - Uses a fixed grid (columns × rows) and optional margins per PDF.
  - Detects white vs black cards (e.g. by background) and saves them into:
    - `extracted_cards/white_cards_family/`
    - `extracted_cards/black_cards_family/`
    - `extracted_cards/white_cards_original/`
    - `extracted_cards/black_cards_original/`
  - Output: one PNG per card (e.g. `card_0001.png`) with consistent naming.

- **`re_extract_problematic.py`**  
  - Re-extracts only specific card numbers that were cut wrong, using the same grid logic from `extract_cards.py`.
  - Useful for fixing mis-cut cards without re-running the full extraction.

- **`generate_card_review.py`**  
  - Scans `extracted_cards/` and generates **`card_review.html`**.
  - Shows all cards in a grid by category; click to open a lightbox.
  - Run a local server (e.g. `python -m http.server 8765`) and open `card_review.html` to review extractions.

- **Dependencies**  
  - Listed in `requirements.txt`: PyMuPDF, Pillow, numpy, opencv-python.

**How to run extraction (after placing PDFs in project):**

```bash
pip install -r requirements.txt
python extract_cards.py
python generate_card_review.py
# Then open http://localhost:8765/card_review.html (with a server running)
```

---

## What’s to be done (gameplay implementation)

The goal is a **browser-based mobile game** where:

- One person **creates a room** and gets a **short code**.
- Friends **join the same room** by entering that code.
- Everyone plays **Cards Against Humanity** in real time (deal, play white cards, Card Czar picks winner, scoring, next round).

Planned work is captured as **GitHub issues** so you can track and prioritize. A full list with titles and descriptions is in:

- **`docs/GITHUB_ISSUES.md`**

Summary of the main areas:

| Area | Description |
|------|-------------|
| **Setup** | Frontend project (e.g. Vite + framework or vanilla), dev server, loading card assets. |
| **Data** | Card manifest (JSON) from `extracted_cards/` so the game knows deck contents and image paths. |
| **Rooms** | Create room, join by code, list players in the lobby. |
| **Real-time** | WebSockets (or equivalent) so all clients stay in sync (game state, actions, events). |
| **Game loop** | Rounds, Card Czar rotation, play phase → reveal → Czar pick → scoring → next round (or game over). |
| **UI** | Mobile-first lobby, hand view (select white cards), black card display, reveal + Czar pick screens. |
| **Content** | Use black card images (and optionally add text/OCR) so blanks and prompts are clear. |
| **Decks** | Let host choose Family, Original, or both decks. |
| **Robustness** | Reconnection, error messages (room not found, full, disconnected). |
| **Polish** | Animations, optional sound, better UX. |
| **Deploy** | Host backend + frontend so friends can play over the internet (HTTPS, env config). |

Suggested order: do setup and card manifest first, then rooms + real-time, then game loop and UI, then content/decks/robustness/polish and deployment.

---

## Future expansion ideas

- **Other card games**  
  - Reuse the same “room by code” and real-time sync to add more games, e.g.:
    - Apples to Apples–style (judge picks best answer).
    - Simple party/trivia or custom “white card + black card” variants.
  - Design the game engine and message protocol so new game types can be added without rewriting the whole app.

- **Adding to “the app”**  
  - If you later build a native or packaged app (e.g. React Native, Capacitor, PWA):
    - Use the same backend and WebSocket API; the app becomes another client.
    - Share one codebase for “join by code” and game state; only the shell (native vs browser) changes.
  - Consider PWA first (installable, offline-capable lobby/instructions) before investing in a full native app.

- **Content and moderation**  
  - Optional: custom/community decks (with clear “official” vs “custom” and moderation).
  - Optional: text-only mode (no card images) for accessibility or low bandwidth.

- **Analytics and tuning**  
  - Anonymous stats (round length, drop-off) to improve UX and balance (e.g. hand size, points to win).

---

## Repo structure (current)

```
cah/
├── extract_cards.py          # Main PDF → card images extraction
├── re_extract_problematic.py # Re-extract specific cards
├── generate_card_review.py   # Build card_review.html from extracted_cards/
├── card_review.html          # Generated review page (open via local server)
├── requirements.txt          # Python deps for extraction
├── extracted_cards/          # Output: white_cards_*, black_cards_* (PNGs)
├── docs/
│   └── GITHUB_ISSUES.md      # Full issue list for gameplay implementation
├── README.md                 # This file
└── LICENSE                   # CC BY-NC-SA 2.0
```

---

## License

Card content and project are under the **Creative Commons Attribution-NonCommercial-ShareAlike 2.0 Generic (CC BY-NC-SA 2.0)** license. See `LICENSE` for details.
