"""
Script to re-extract only the problematic cards that were cut incorrectly.
"""

from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from extract_cards import (
    pdf_to_images,
    detect_card_boxes_opencv_grid,
    detect_card_boxes_with_grid_from_separators,
    fix_card_crop,
    is_white_card
)

def re_extract_specific_cards(pdf_path: str, output_base_dir: str, card_type: str,
                              grid_cols: int, grid_rows: int,
                              target_cards: dict,  # {'white': [list of card numbers], 'black': [list]}
                              margin_left: int = None, margin_right: int = None,
                              margin_top: int = None, margin_bottom: int = None):
    """
    Re-extract specific cards by card number.
    
    Args:
        target_cards: Dict with 'white' and 'black' keys, each containing list of card numbers to extract
    """
    print(f"Re-extracting specific cards from {pdf_path}...")
    
    # Create output directories
    white_dir = Path(output_base_dir) / f"white_cards_{card_type}"
    black_dir = Path(output_base_dir) / f"black_cards_{card_type}"
    white_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert PDF to images
    pages = pdf_to_images(pdf_path)
    
    # Track which cards we've extracted so far
    white_counter = 0
    black_counter = 0
    
    # Cards per page (4x5 = 20)
    cards_per_page = grid_cols * grid_rows
    
    for page_num, page_image in enumerate(pages):
        # Skip first two pages
        if page_num < 2:
            continue
        
        print(f"  Processing page {page_num + 1}/{len(pages)}...")
        
        # Detect card boxes on this page
        card_boxes = detect_card_boxes_opencv_grid(page_image, grid_cols, grid_rows)
        
        if not card_boxes:
            card_boxes = detect_card_boxes_with_grid_from_separators(
                page_image, grid_cols, grid_rows
            )
        
        if not card_boxes:
            print(f"    Warning: Could not detect cards on page {page_num + 1}")
            # Skip this page, but still increment counters
            # Estimate: assume half white, half black (rough estimate)
            white_counter += cards_per_page // 2
            black_counter += cards_per_page // 2
            continue
        
        # Process each card on this page
        for box_num, (x, y, w, h) in enumerate(card_boxes):
            page_width, page_height = page_image.size
            
            # Clamp coordinates
            x = max(0, min(x, page_width - 1))
            y = max(0, min(y, page_height - 1))
            w = min(w, page_width - x)
            h = min(h, page_height - y)
            
            # Filter by aspect ratio
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = h / w
            if not (1.2 <= aspect_ratio <= 1.9):
                continue
            
            # Inset box
            border_inset = 4
            if w > 2 * border_inset and h > 2 * border_inset:
                x += border_inset
                y += border_inset
                w -= 2 * border_inset
                h -= 2 * border_inset
            
            # Crop the card
            card_image = page_image.crop((x, y, x + w, y + h))
            
            # Skip if too small
            if card_image.size[0] < 200 or card_image.size[1] < 300:
                continue
            
            # Check content
            gray_card = np.array(card_image.convert('L'))
            if np.var(gray_card) < 100:
                continue
            
            # Fix crop if needed
            x, y, w, h = fix_card_crop(card_image, page_image, x, y, w, h)
            card_image = page_image.crop((x, y, x + w, y + h))
            
            # Determine if white or black
            if is_white_card(card_image):
                white_counter += 1
                # Only save if this card number is in our target list
                if white_counter in target_cards.get('white', []):
                    output_path = white_dir / f"card_{white_counter:04d}.png"
                    card_image.save(output_path, 'PNG')
                    print(f"    ✓ Re-extracted white card {white_counter:04d}")
            else:
                black_counter += 1
                # Only save if this card number is in our target list
                if black_counter in target_cards.get('black', []):
                    output_path = black_dir / f"card_{black_counter:04d}.png"
                    card_image.save(output_path, 'PNG')
                    print(f"    ✓ Re-extracted black card {black_counter:04d}")
            
            # Debug: show progress for cards near our targets
            if white_counter in range(525, 535) or black_counter in range(80, 100):
                card_type = "white" if is_white_card(card_image) else "black"
                counter = white_counter if card_type == "white" else black_counter
                print(f"    Debug: Page {page_num+1}, Box {box_num}, {card_type} card #{counter}")

def main():
    base_dir = Path(__file__).parent
    
    family_pdf = base_dir / "CAH_FamilyGame-1.1-SmallCards.pdf"
    output_dir = base_dir / "extracted_cards"
    
    # Define which cards to re-extract
    # Family black cards 0085-0095 (cut too high)
    # Family white card 0530 (cut too low)
    
    target_cards = {
        'black': list(range(85, 96)),  # 85-95 inclusive
        'white': [530]
    }
    
    print("Re-extracting problematic cards...")
    print(f"Black cards to re-extract: {target_cards['black']}")
    print(f"White cards to re-extract: {target_cards['white']}")
    
    if family_pdf.exists():
        re_extract_specific_cards(
            str(family_pdf),
            str(output_dir),
            "family",
            grid_cols=4,
            grid_rows=5,
            target_cards=target_cards
        )
        print("\nRe-extraction complete!")
    else:
        print(f"Error: {family_pdf} not found")

if __name__ == "__main__":
    main()
