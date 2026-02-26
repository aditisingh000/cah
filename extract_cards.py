"""
Script to extract cards from CAH PDFs and sort them into categories:
- White cards family edition
- White cards original
- Black cards family edition
- Black cards original
"""

import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from typing import List, Tuple
import shutil
import cv2

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert PDF pages to PIL Images."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image at 300 DPI for good quality
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def find_separator_lines(gray: np.ndarray, axis: int, min_brightness: int = 200) -> List[int]:
    """
    Find thin white separator lines by looking for bright horizontal or vertical lines.
    
    Args:
        gray: Grayscale image array
        axis: 0 for horizontal lines (y coordinates), 1 for vertical lines (x coordinates)
        min_brightness: Minimum brightness to consider a line (white lines are bright)
    
    Returns:
        List of line coordinates
    """
    height, width = gray.shape
    lines = []
    line_thickness = 5  # Check a window around each potential line
    
    if axis == 0:  # Horizontal lines (y coordinates) - separate rows
        # For each row, check if it's a bright separator line
        for y in range(line_thickness, height - line_thickness):
            # Get the row and surrounding rows
            row_region = gray[y-line_thickness:y+line_thickness, :]
            
            # Calculate statistics
            avg_brightness = np.mean(row_region)
            std_brightness = np.std(row_region)
            min_brightness_in_region = np.min(row_region)
            
            # A separator line should be:
            # 1. Very bright on average (> min_brightness)
            # 2. Relatively consistent (low std, < 40)
            # 3. Even the darkest parts should be bright (min > 180)
            if (avg_brightness > min_brightness and 
                std_brightness < 40 and 
                min_brightness_in_region > 180):
                lines.append(y)
    else:  # Vertical lines (x coordinates) - separate columns
        # For each column, check if it's a bright separator line
        for x in range(line_thickness, width - line_thickness):
            # Get the column and surrounding columns
            col_region = gray[:, x-line_thickness:x+line_thickness]
            
            # Calculate statistics
            avg_brightness = np.mean(col_region)
            std_brightness = np.std(col_region)
            min_brightness_in_region = np.min(col_region)
            
            # Same criteria as horizontal lines
            if (avg_brightness > min_brightness and 
                std_brightness < 40 and 
                min_brightness_in_region > 180):
                lines.append(x)
    
    # Merge nearby lines - separator lines should be at least 50 pixels apart
    # (cards are much larger than that)
    merged = []
    for line in sorted(lines):
        if not merged:
            merged.append(line)
        else:
            # Check if this line is far enough from existing lines
            min_distance = min(abs(line - m) for m in merged)
            if min_distance > 50:  # Cards are at least 200-300 pixels, so separators should be 50+ apart
                merged.append(line)
    
    return merged

def detect_card_boxes(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detect individual card boxes by finding white separator lines.
    Returns list of (x, y, width, height) bounding boxes.
    """
    # Convert to numpy array
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2).astype(np.uint8)
    
    height, width = gray.shape
    
    # Find white separator lines
    vertical_lines = find_separator_lines(gray, 1)  # x coordinates (vertical lines separate columns)
    horizontal_lines = find_separator_lines(gray, 0)  # y coordinates (horizontal lines separate rows)
    
    cards = []
    
    # If we found separator lines, use them to define card boundaries
    if len(vertical_lines) >= 1 or len(horizontal_lines) >= 1:
        # Add page edges as boundaries
        all_vertical = [0] + sorted(vertical_lines) + [width]
        all_horizontal = [0] + sorted(horizontal_lines) + [height]
        
        # Extract cards between the lines
        for i in range(len(all_horizontal) - 1):
            for j in range(len(all_vertical) - 1):
                x = all_vertical[j]
                y = all_horizontal[i]
                w = all_vertical[j + 1] - x
                h = all_horizontal[i + 1] - y
                
                # Add small padding to avoid including the separator line itself
                # But be careful not to cut into the card
                padding = 5
                x += padding
                y += padding
                w -= 2 * padding
                h -= 2 * padding
                
                # Only add if it's a reasonable card size
                if w > 200 and h > 300:
                    cards.append((x, y, w, h))
    else:
        # Fallback: if no separator lines found, try alternative detection
        print("    Warning: No separator lines detected, using fallback method")
        
        # Try to find lines with lower brightness threshold
        vertical_lines = find_separator_lines(gray, 1, min_brightness=180)
        horizontal_lines = find_separator_lines(gray, 0, min_brightness=180)
        
        if len(vertical_lines) >= 1 or len(horizontal_lines) >= 1:
            all_vertical = [0] + sorted(vertical_lines) + [width]
            all_horizontal = [0] + sorted(horizontal_lines) + [height]
            
            for i in range(len(all_horizontal) - 1):
                for j in range(len(all_vertical) - 1):
                    x = all_vertical[j]
                    y = all_horizontal[i]
                    w = all_vertical[j + 1] - x
                    h = all_horizontal[i + 1] - y
                    
                    padding = 5
                    x += padding
                    y += padding
                    w -= 2 * padding
                    h -= 2 * padding
                    
                    if w > 200 and h > 300:
                        cards.append((x, y, w, h))
        else:
            # Last resort: use intelligent grid detection
            aspect_ratio = width / height
            if aspect_ratio > 1.3:
                cols, rows = 3, 3
            elif aspect_ratio < 0.8:
                cols, rows = 2, 4
            else:
                cols, rows = 3, 3
            
            card_width = width // cols
            card_height = height // rows
            margin = 30
            
            for row in range(rows):
                for col in range(cols):
                    x = col * card_width + margin
                    y = row * card_height + margin
                    w = card_width - 2 * margin
                    h = card_height - 2 * margin
                    
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w > 200 and h > 300:
                        cards.append((x, y, w, h))
    
    # Sort cards by position (top to bottom, left to right)
    cards.sort(key=lambda box: (box[1] // (height // 10), box[0]))
    
    return cards

def detect_card_boxes_opencv_grid(
    image: Image.Image, grid_cols: int, grid_rows: int
) -> List[Tuple[int, int, int, int]]:
    """
    Detect card boxes using OpenCV:
    - Runs edge detection + Hough line transform to find long straight lines.
    - Uses those to infer a regular grid with grid_cols x grid_rows cells.

    Returns list of (x, y, w, h) boxes. Empty list if detection clearly fails.
    """
    img_array = np.array(image)

    # Convert PIL RGB -> OpenCV BGR/gray
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Hough transform to detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=200,
        minLineLength=min(width, height) * 0.5,
        maxLineGap=20,
    )

    if lines is None:
        return []

    vertical_positions = []
    horizontal_positions = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx < 10 and dy > height * 0.5:
            # Almost vertical, long enough
            x_avg = (x1 + x2) // 2
            vertical_positions.append(x_avg)
        elif dy < 10 and dx > width * 0.5:
            # Almost horizontal, long enough
            y_avg = (y1 + y2) // 2
            horizontal_positions.append(y_avg)

    def cluster_positions(positions: List[int], min_separation: int = 30) -> List[int]:
        if not positions:
            return []
        positions = sorted(positions)
        clusters = [[positions[0]]]
        for p in positions[1:]:
            if abs(p - clusters[-1][-1]) <= min_separation:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        # Use mean of each cluster
        return [int(np.mean(c)) for c in clusters]

    vertical_lines = cluster_positions(vertical_positions)
    horizontal_lines = cluster_positions(horizontal_positions)

    # If not enough lines, bail out
    if len(vertical_lines) < grid_cols + 1 or len(horizontal_lines) < grid_rows + 1:
        return []

    vertical_lines = sorted(vertical_lines)
    horizontal_lines = sorted(horizontal_lines)

    # Helper to pick a fixed number of roughly evenly spaced boundaries
    def pick_bounds(lines: List[int], needed: int, min_val: int, max_val: int) -> List[int]:
        lines = [p for p in lines if min_val <= p <= max_val]
        lines = sorted(set(lines))
        if len(lines) < needed:
            # Add edges if we don't have enough
            lines = [min_val] + lines + [max_val]
        # Now resample to exactly 'needed' positions
        indices = np.linspace(0, len(lines) - 1, needed, dtype=int)
        return [lines[i] for i in indices]

    # For N columns/rows we need N+1 boundaries
    v_bounds = pick_bounds(vertical_lines, grid_cols + 1, 0, width)
    h_bounds = pick_bounds(horizontal_lines, grid_rows + 1, 0, height)

    boxes: List[Tuple[int, int, int, int]] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x0 = v_bounds[col]
            x1 = v_bounds[col + 1]
            y0 = h_bounds[row]
            y1 = h_bounds[row + 1]

            x = max(0, min(x0, width - 1))
            y = max(0, min(y0, height - 1))
            w = max(0, min(x1, width) - x)
            h = max(0, min(y1, height) - y)

            if w > 200 and h > 300:
                boxes.append((x, y, w, h))

    # Sanity check: expect exactly grid_cols * grid_rows boxes
    if len(boxes) != grid_cols * grid_rows:
        # If we're close, still return; otherwise, signal failure
        if len(boxes) < grid_cols * grid_rows * 0.7:
            return []

    # Sort in reading order
    boxes.sort(key=lambda box: (box[1] // (height // 10), box[0]))
    return boxes

def detect_card_boxes_with_grid_from_separators(
    image: Image.Image, grid_cols: int, grid_rows: int
) -> List[Tuple[int, int, int, int]]:
    """
    Detect card boxes using the actual detected white separator lines, constrained
    to a known grid size (grid_cols x grid_rows).

    Returns an empty list if we can't confidently match the expected grid.
    """
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2).astype(np.uint8)

    height, width = gray.shape

    # Try to find separator lines with the standard threshold
    vertical_lines = find_separator_lines(gray, 1)
    horizontal_lines = find_separator_lines(gray, 0)

    # If not enough lines, try again with a slightly lower brightness threshold
    if len(vertical_lines) < grid_cols + 1 or len(horizontal_lines) < grid_rows + 1:
        vertical_lines = find_separator_lines(gray, 1, min_brightness=180)
        horizontal_lines = find_separator_lines(gray, 0, min_brightness=180)

    if len(vertical_lines) < grid_cols + 1 or len(horizontal_lines) < grid_rows + 1:
        # Not enough reliable lines detected for the expected grid
        return []

    vertical_lines = sorted(vertical_lines)
    horizontal_lines = sorted(horizontal_lines)

    # Helper to pick a fixed number of roughly evenly spaced lines
    def pick_lines(lines: List[int], needed: int) -> List[int]:
        if len(lines) == needed:
            return lines
        indices = np.linspace(0, len(lines) - 1, needed, dtype=int)
        return [lines[i] for i in indices]

    # For N columns we need N+1 vertical boundaries (left/right of each column)
    v_bounds = pick_lines(vertical_lines, grid_cols + 1)
    h_bounds = pick_lines(horizontal_lines, grid_rows + 1)

    boxes: List[Tuple[int, int, int, int]] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x0 = v_bounds[col]
            x1 = v_bounds[col + 1]
            y0 = h_bounds[row]
            y1 = h_bounds[row + 1]

            x = max(0, min(x0, width - 1))
            y = max(0, min(y0, height - 1))
            w = max(0, min(x1, width) - x)
            h = max(0, min(y1, height) - y)

            if w > 200 and h > 300:
                boxes.append((x, y, w, h))

    return boxes

def detect_margin(region: np.ndarray, axis: int) -> int:
    """Detect margin by finding where content (non-white/non-black) starts."""
    # Simple approach: find first significant variation from edge
    if axis == 0:  # Vertical margin (top)
        projection = np.std(region, axis=1)
        threshold = np.max(projection) * 0.2
        for i, val in enumerate(projection):
            if val > threshold:
                return max(0, i - 5)  # Add small buffer
    else:  # Horizontal margin (left)
        projection = np.std(region, axis=0)
        threshold = np.max(projection) * 0.2
        for i, val in enumerate(projection):
            if val > threshold:
                return max(0, i - 5)  # Add small buffer
    return 0

def is_white_card(image: Image.Image) -> bool:
    """Determine if a card is white (light background) or black (dark background)."""
    # Convert to grayscale
    gray = np.array(image.convert('L'))
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # White cards have high brightness (> 200), black cards have low brightness (< 100)
    return avg_brightness > 150

def fix_card_crop(image: Image.Image, page_image: Image.Image, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Check if card has white line at top (cut too high) or black line at top (cut too low)
    and adjust the crop box accordingly.
    
    Returns adjusted (x, y, w, h)
    """
    # Check top edge of the card (first 15 pixels)
    gray = np.array(image.convert('L'))
    if gray.shape[0] < 15:
        return (x, y, w, h)
    
    top_edge = gray[:15, :]
    top_brightness = np.mean(top_edge)
    top_std = np.std(top_edge)
    
    # Check if top edge is a uniform white line (bright and low variance)
    # This indicates we cut too high and included the separator line
    if top_brightness > 230 and top_std < 20:
        # Move y down to skip the white line
        adjust = 8
        new_y = min(y + adjust, page_image.size[1] - h)
        new_h = h - adjust if new_y == y + adjust else h
        return (x, new_y, w, new_h)
    
    # Check if top edge has a dark line on what should be a white card
    # (indicates we cut too low and included part of the previous card)
    card_brightness = np.mean(gray[15:50, :]) if gray.shape[0] > 50 else np.mean(gray[15:, :])
    if card_brightness > 150:  # This is a white card
        # Check if top has a dark horizontal line
        top_row = gray[0, :]
        if np.mean(top_row) < 80 and np.std(top_row) < 30:
            # Move y up to skip the dark line
            adjust = 8
            new_y = max(0, y - adjust)
            new_h = h + adjust if new_y == y - adjust else h
            return (x, new_y, w, new_h)
    
    return (x, y, w, h)

def extract_cards_from_pdf(pdf_path: str, output_base_dir: str, card_type: str, 
                           grid_cols: int = None, grid_rows: int = None, 
                           margin: int = None,
                           margin_left: int = None, margin_right: int = None,
                           margin_top: int = None, margin_bottom: int = None):
    """
    Extract cards from a PDF and save them to appropriate directories.
    
    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base directory for output
        card_type: 'family' or 'original'
        grid_cols: Optional number of columns (if None, auto-detect)
        grid_rows: Optional number of rows (if None, auto-detect)
        margin: Optional uniform margin in pixels (if None, auto-detect)
        margin_left: Optional left margin in pixels
        margin_right: Optional right margin in pixels
        margin_top: Optional top margin in pixels
        margin_bottom: Optional bottom margin in pixels
    """
    print(f"Processing {pdf_path}...")
    
    # Create output directories
    white_dir = Path(output_base_dir) / f"white_cards_{card_type}"
    black_dir = Path(output_base_dir) / f"black_cards_{card_type}"
    white_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert PDF to images
    pages = pdf_to_images(pdf_path)
    
    card_counter = {'white': 0, 'black': 0}
    
    for page_num, page_image in enumerate(pages):
        # Skip the first two pages (covers / instructions)
        if page_num < 2:
            print(f"  Skipping page {page_num + 1} (cover/intro)")
            continue

        print(f"  Processing page {page_num + 1}/{len(pages)}...")
        
        # If manual grid specified, prefer using OpenCV line detection to build the grid
        if grid_cols and grid_rows:
            # First try OpenCV-based grid detection (most robust)
            card_boxes = detect_card_boxes_opencv_grid(page_image, grid_cols, grid_rows)

            if not card_boxes:
                # Fallback 1: use separator-based grid if OpenCV fails
                card_boxes = detect_card_boxes_with_grid_from_separators(
                    page_image, grid_cols, grid_rows
                )

            if not card_boxes:
                # Fallback 2: use margin-based grid if both fail
                page_width, page_height = page_image.size

                # Calculate margins
                if margin is not None:
                    # Use uniform margin if specified
                    margin_left_val = margin_right_val = margin_top_val = margin_bottom_val = margin
                else:
                    # Use individual margins if specified, otherwise defaults
                    margin_left_val = margin_left if margin_left is not None else 30
                    margin_right_val = margin_right if margin_right is not None else 30
                    margin_top_val = margin_top if margin_top is not None else 0
                    margin_bottom_val = margin_bottom if margin_bottom is not None else 30

                # Calculate usable area (page minus margins)
                usable_width = page_width - margin_left_val - margin_right_val
                usable_height = page_height - margin_top_val - margin_bottom_val

                # Calculate card dimensions
                card_width = usable_width // grid_cols
                card_height = usable_height // grid_rows

                card_boxes = []
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        x = margin_left_val + col * card_width
                        y = margin_top_val + row * card_height
                        w = card_width
                        h = card_height

                        # Ensure we don't go out of bounds
                        w = min(w, page_width - x)
                        h = min(h, page_height - y)

                        if w > 200 and h > 300:
                            card_boxes.append((x, y, w, h))
        else:
            # Auto-detect card boxes
            card_boxes = detect_card_boxes(page_image)
        
        print(f"    Found {len(card_boxes)} potential cards")
        
        for box_num, (x, y, w, h) in enumerate(card_boxes):
            # Ensure coordinates are within image bounds
            page_width, page_height = page_image.size

            # First clamp the original box
            x = max(0, min(x, page_width - 1))
            y = max(0, min(y, page_height - 1))
            w = min(w, page_width - x)
            h = min(h, page_height - y)

            # Filter out obviously bad boxes by aspect ratio
            # CAH cards are portrait, height/width roughly between 1.0 and 2.0
            # (allowing some flexibility for detection variations)
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = h / w
            if not (1.0 <= aspect_ratio <= 2.0):
                continue

            # Slightly inset the box so we never include anything outside the white outline.
            # We trim a few pixels from each side to remove any remaining white grid line.
            border_inset = 4
            if w > 2 * border_inset and h > 2 * border_inset:
                x += border_inset
                y += border_inset
                w -= 2 * border_inset
                h -= 2 * border_inset

            # Crop the card
            card_image = page_image.crop((x, y, x + w, y + h))
            
            # Fix crop if there's a white line at top (cut too high) or black line at top (cut too low)
            x, y, w, h = fix_card_crop(card_image, page_image, x, y, w, h)
            
            # Ensure coordinates are still within bounds after fixing
            x = max(0, min(x, page_width - 1))
            y = max(0, min(y, page_height - 1))
            w = min(w, page_width - x)
            h = min(h, page_height - y)
            
            # Re-crop with fixed coordinates
            card_image = page_image.crop((x, y, x + w, y + h))
            
            # Skip if card is too small (likely empty space or detection error)
            if card_image.size[0] < 200 or card_image.size[1] < 300:
                continue
            
            # Check if the card has meaningful content (not just blank space)
            gray_card = np.array(card_image.convert('L'))
            # Check variance - blank cards have low variance
            variance = np.var(gray_card)
            if variance < 100:
                # Debug: print first few failures
                if box_num < 3:
                    print(f"      Box {box_num} filtered: variance too low {variance:.1f}")
                continue
            
            # Determine if white or black
            if is_white_card(card_image):
                card_counter['white'] += 1
                output_path = white_dir / f"card_{card_counter['white']:04d}.png"
                card_image.save(output_path, 'PNG')
            else:
                card_counter['black'] += 1
                output_path = black_dir / f"card_{card_counter['black']:04d}.png"
                card_image.save(output_path, 'PNG')
    
    print(f"  Extracted {card_counter['white']} white cards and {card_counter['black']} black cards")
    return card_counter

def main():
    """Main function to process both PDFs."""
    base_dir = Path(__file__).parent
    
    # Define PDF files
    family_pdf = base_dir / "CAH_FamilyGame-1.1-SmallCards.pdf"
    original_pdf = base_dir / "CAH_PrintPlay2022-RegularInk-FINAL-outlined.pdf"
    
    # Output directory
    output_dir = base_dir / "extracted_cards"
    
    # ============================================================================
    # MANUAL GRID CONFIGURATION
    # ============================================================================
    # Set these to manually specify the grid layout for each PDF
    # Leave as None to use auto-detection (which may not work well)
    # 
    # Common layouts: 
    #   - 3 columns x 3 rows = 9 cards per page
    #   - 2 columns x 4 rows = 8 cards per page
    #   - 2 columns x 3 rows = 6 cards per page
    #
    # margin: Padding in pixels to leave around each card (at 300 DPI)
    #         Increase if cards are being cut off, decrease if too much white space
    # ============================================================================
    
    # Family Edition settings
    FAMILY_GRID_COLS = 4  # 4 columns
    FAMILY_GRID_ROWS = 5  # 5 rows
    FAMILY_MARGIN_LEFT = 75   # Left margin in pixels
    FAMILY_MARGIN_RIGHT = 75   # Right margin in pixels
    FAMILY_MARGIN_BOTTOM = 250  # Bottom margin in pixels
    FAMILY_MARGIN_TOP = 75    # Top margin in pixels
    
    # Original Edition settings
    ORIGINAL_GRID_COLS = 4  # 4 columns
    ORIGINAL_GRID_ROWS = 5  # 5 rows
    ORIGINAL_MARGIN_LEFT = 75   # Left margin in pixels
    ORIGINAL_MARGIN_RIGHT = 75   # Right margin in pixels
    ORIGINAL_MARGIN_BOTTOM = 250  # Bottom margin in pixels
    ORIGINAL_MARGIN_TOP = 75    # Top margin in pixels
    
    # ============================================================================
    
    # Process family edition
    if family_pdf.exists():
        print(f"\n{'='*60}")
        print("Processing Family Edition PDF")
        print(f"{'='*60}")
        if FAMILY_GRID_COLS and FAMILY_GRID_ROWS:
            print(f"Using manual grid: {FAMILY_GRID_COLS} columns x {FAMILY_GRID_ROWS} rows")
            extract_cards_from_pdf(str(family_pdf), str(output_dir), "family", 
                                   grid_cols=FAMILY_GRID_COLS, 
                                   grid_rows=FAMILY_GRID_ROWS,
                                   margin_left=FAMILY_MARGIN_LEFT,
                                   margin_right=FAMILY_MARGIN_RIGHT,
                                   margin_top=FAMILY_MARGIN_TOP,
                                   margin_bottom=FAMILY_MARGIN_BOTTOM)
        else:
            print("Using auto-detection (may not work well)")
            extract_cards_from_pdf(str(family_pdf), str(output_dir), "family")
    else:
        print(f"Warning: {family_pdf} not found")
    
    # Process original edition
    if original_pdf.exists():
        print(f"\n{'='*60}")
        print("Processing Original Edition PDF")
        print(f"{'='*60}")
        if ORIGINAL_GRID_COLS and ORIGINAL_GRID_ROWS:
            print(f"Using manual grid: {ORIGINAL_GRID_COLS} columns x {ORIGINAL_GRID_ROWS} rows")
            extract_cards_from_pdf(str(original_pdf), str(output_dir), "original",
                                   grid_cols=ORIGINAL_GRID_COLS, 
                                   grid_rows=ORIGINAL_GRID_ROWS,
                                   margin_left=ORIGINAL_MARGIN_LEFT,
                                   margin_right=ORIGINAL_MARGIN_RIGHT,
                                   margin_top=ORIGINAL_MARGIN_TOP,
                                   margin_bottom=ORIGINAL_MARGIN_BOTTOM)
        else:
            print("Using auto-detection (may not work well)")
            extract_cards_from_pdf(str(original_pdf), str(output_dir), "original")
    else:
        print(f"Warning: {original_pdf} not found")
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"Cards saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
