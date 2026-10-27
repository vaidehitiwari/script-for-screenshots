import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def normalize_brightness_and_lighting(
    gray: np.ndarray,
    target_brightness: int = 240,
    bg_kernel_size: int = 51
) -> Tuple[np.ndarray, np.ndarray]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bg_kernel_size, bg_kernel_size))
    dilated_bg = cv2.dilate(gray, kernel)
    median_bg = cv2.medianBlur(dilated_bg, 21)

    bg_float = np.maximum(median_bg.astype(np.float32), 1.0)
    gray_float = gray.astype(np.float32)

    norm_float = (gray_float / bg_float) * float(target_brightness)
    normalized_gray = np.clip(norm_float, 0, 255).astype(np.uint8)

    binary_mask = cv2.adaptiveThreshold(
        normalized_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=12
    )

    return normalized_gray, binary_mask


def is_page_blank(binary_mask: np.ndarray, min_ink_ratio: float = 0.002) -> Tuple[bool, float]:
    total_pixels = binary_mask.size
    ink_pixels = int(np.count_nonzero(binary_mask))
    ink_ratio = ink_pixels / total_pixels
    return (ink_ratio < min_ink_ratio), ink_ratio


def remove_margin_and_ruling_lines(
    binary_mask: np.ndarray,
    suppress_ruled: bool = True
) -> np.ndarray:
    h, w = binary_mask.shape
    cleaned = binary_mask.copy()

    vert_kernel_len = max(35, h // 25)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    detected_vert = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vert_kernel)
    cleaned = cv2.subtract(cleaned, detected_vert)

    my = max(4, int(h * 0.015))
    mx = max(4, int(w * 0.015))
    cleaned[:my, :] = 0
    cleaned[-my:, :] = 0
    cleaned[:, :mx] = 0
    cleaned[:, -mx:] = 0

    if suppress_ruled:
        horiz_kernel_len = max(25, w // 20)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
        detected_horiz = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horiz_kernel)
        cleaned = cv2.subtract(cleaned, detected_horiz)

    return cleaned


def has_real_handwriting(
    norm_gray: np.ndarray,
    cleaned_mask: np.ndarray,
    y1: int,
    y2: int,
    w: int,
    h: int,
    min_ink_pixels: int = 350
) -> bool:
    if y1 < int(h * 0.03) and (y2 - y1) < 40:
        return False
    if y2 > int(h * 0.97) and (y2 - y1) < 40:
        return False

    x_left = int(w * 0.12)
    x_right = int(w * 0.88)
    body_slice = cleaned_mask[y1:y2, x_left:x_right]

    if body_slice.size == 0:
        return False

    ink_pixels = int(np.count_nonzero(body_slice))
    if ink_pixels < min_ink_pixels:
        return False

    cols_with_ink = np.where(np.any(body_slice > 0, axis=0))[0]
    if len(cols_with_ink) == 0:
        return False

    span = cols_with_ink[-1] - cols_with_ink[0]
    if span < int(w * 0.20):
        return False

    return True


def detect_and_group_lines(
    norm_gray: np.ndarray,
    cleaned_mask: np.ndarray,
    lines_per_crop: int = 3,
    min_line_distance: int = 22,
    min_peak_prominence: float = 12.0,
    min_peak_height: float = 25.0,
    pad_y: int = 8
) -> List[Dict[str, Any]]:
    h, w = cleaned_mask.shape

    raw_proj = np.count_nonzero(cleaned_mask, axis=1)
    smoothed_proj = gaussian_filter1d(raw_proj.astype(np.float32), sigma=7.0)

    peaks, _ = find_peaks(
        smoothed_proj,
        distance=min_line_distance,
        prominence=min_peak_prominence,
        height=min_peak_height
    )

    if len(peaks) == 0:
        return []

    valleys = []
    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i + 1]
        valley_y = p1 + int(np.argmin(smoothed_proj[p1:p2]))
        valleys.append(valley_y)

    max_half_h = max(28, int(h * 0.04))
    raw_lines = []

    for i in range(len(peaks)):
        p = peaks[i]

        if i == 0:
            up_idx = p
            while up_idx > 0 and smoothed_proj[up_idx] > (min_peak_height * 0.35) and (p - up_idx) < max_half_h:
                up_idx -= 1
            y1 = max(0, up_idx - pad_y)
        else:
            v_prev = valleys[i - 1]
            y1 = max(0, max(v_prev - (pad_y // 2), p - max_half_h))

        if i == len(peaks) - 1:
            down_idx = p
            while down_idx < h and smoothed_proj[down_idx] > (min_peak_height * 0.35) and (down_idx - p) < max_half_h:
                down_idx += 1
            y2 = min(h, down_idx + pad_y)
        else:
            v_next = valleys[i]
            y2 = min(h, min(v_next + (pad_y // 2), p + max_half_h))

        if y2 > y1 + 10:
            if has_real_handwriting(norm_gray, cleaned_mask, y1, y2, w, h):
                raw_lines.append((y1, y2, p))

    if not raw_lines:
        return []

    crops_info = []

    if lines_per_crop <= 1:
        for idx, (y1, y2, p) in enumerate(raw_lines):
            crops_info.append({
                "y1": y1,
                "y2": y2,
                "num_lines": 1,
                "line_indices": [idx + 1]
            })
    else:
        chunks = []
        for idx in range(0, len(raw_lines), lines_per_crop):
            chunk = raw_lines[idx:idx + lines_per_crop]
            if len(chunk) == 1 and len(chunks) > 0:
                chunks[-1].extend(chunk)
            else:
                chunks.append(chunk)

        current_line_idx = 1
        for chunk in chunks:
            chunk_y1 = chunk[0][0]
            chunk_y2 = chunk[-1][1]
            n_lines = len(chunk)
            line_ids = list(range(current_line_idx, current_line_idx + n_lines))
            current_line_idx += n_lines

            crops_info.append({
                "y1": chunk_y1,
                "y2": chunk_y2,
                "num_lines": n_lines,
                "line_indices": line_ids
            })

    return crops_info


def process_single_image(
    image_path: Path,
    output_crops_dir: Path,
    debug_dir: Optional[Path] = None,
    lines_per_crop: int = 3,
    target_brightness: int = 240,
    pad_y: int = 8,
    min_page_ink_ratio: float = 0.002,
    suppress_ruled: bool = True,
    work_width: int = 2480
) -> Dict[str, Any]:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"status": "error", "error": f"Failed to load image: {image_path}"}

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Detection thresholds are in pixels and tuned for an A4 page at 300 DPI
    # (~2480 px wide). Resize every page to that width so a single handwritten
    # line isn't split into several "lines" on large scans.
    if work_width and gray.shape[1] != work_width:
        scale = work_width / gray.shape[1]
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (work_width, round(gray.shape[0] * scale)), interpolation=interp)

    h, w = gray.shape[:2]

    norm_gray, binary_mask = normalize_brightness_and_lighting(
        gray, target_brightness=target_brightness
    )

    blank, ink_ratio = is_page_blank(binary_mask, min_ink_ratio=min_page_ink_ratio)
    if blank:
        return {
            "status": "skipped",
            "reason": f"Blank page / empty screenshot (Ink ratio: {ink_ratio*100:.3f}% < {min_page_ink_ratio*100:.3f}%)"
        }

    cleaned_mask = remove_margin_and_ruling_lines(binary_mask, suppress_ruled=suppress_ruled)

    crop_chunks = detect_and_group_lines(
        norm_gray,
        cleaned_mask,
        lines_per_crop=lines_per_crop,
        pad_y=pad_y
    )

    if not crop_chunks:
        return {
            "status": "skipped",
            "reason": "No valid handwriting text lines detected above threshold."
        }

    stem = image_path.stem
    crops_saved = 0
    debug_overlay = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR) if debug_dir else None

    for idx, chunk in enumerate(crop_chunks):
        crop_num = idx + 1
        y1 = chunk["y1"]
        y2 = chunk["y2"]
        n_lines = chunk["num_lines"]

        x1 = 0
        x2 = w

        crop_img = norm_gray[y1:y2, x1:x2]
        crop_filename = f"{stem}_crop_{crop_num:03d}.png"
        crop_path = output_crops_dir / crop_filename

        cv2.imwrite(str(crop_path), crop_img)
        crops_saved += 1

        if debug_overlay is not None:
            cv2.rectangle(debug_overlay, (x1 + 4, y1), (x2 - 4, y2), (0, 200, 0), 3)
            label = f"Crop #{crop_num} ({n_lines} lines)"
            cv2.putText(
                debug_overlay, label,
                (25, max(28, y1 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

    if debug_overlay is not None:
        debug_path = debug_dir / f"{stem}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_overlay)

    return {
        "status": "success",
        "crops_count": crops_saved,
        "ink_ratio": ink_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="Crop 3-4 lines of handwritten text per image with dynamic brightness and blank rejection.")
    parser.add_argument("--input", "-i", required=True, help="Path to a single image file OR a folder containing page images.")
    parser.add_argument("--output_dir", "-o", default="data/line_crops", help="Folder to save cropped line images.")
    parser.add_argument("--debug_dir", "-d", default=None, help="Optional directory to save debug overlay images.")
    parser.add_argument("--lines_per_crop", "-n", type=int, default=3, help="Number of text lines to include in each crop (default: 3).")
    parser.add_argument("--target_brightness", type=int, default=240, help="Target paper brightness (0-255, default: 240).")
    parser.add_argument("--pad_y", type=int, default=8, help="Vertical padding around text boundaries (default: 8px).")
    parser.add_argument("--min_ink_ratio", type=float, default=0.002, help="Minimum ink density ratio to avoid blank screenshots (default: 0.002 = 0.2%%).")
    parser.add_argument("--work_width", type=int, default=2480, help="Resize pages to this width before processing so line detection works on any scan size (default: 2480 = A4 @ 300 DPI, 0 = keep original).")
    parser.add_argument("--no_ruled_suppression", action="store_true", help="Disable notebook ruling line suppression.")

    args = parser.parse_args()

    input_p = Path(args.input)
    if not input_p.exists():
        sys.exit(f"[ERROR] Input path does not exist: {input_p}")

    crops_dir = Path(args.output_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    if input_p.is_file():
        image_files = [input_p]
    else:
        image_files = sorted([f for f in input_p.glob("*.*") if f.suffix.lower() in valid_exts])

    if not image_files:
        sys.exit(f"[WARNING] No image files found at: {input_p}")

    print("\n" + "=" * 60)
    print("HANDWRITTEN TEXT CROPER (3-4 LINES PER CROP)")
    print(f"   Lines per crop:    {args.lines_per_crop}")
    print(f"   Target brightness: {args.target_brightness}/255")
    print(f"   Crop span:         Full page width")
    print(f"   Blank rejection:   Active")
    print(f"   Images to process: {len(image_files)}")
    print("=" * 60 + "\n")

    total_crops = 0
    skipped_count = 0

    for img_path in tqdm(image_files, desc="Processing images", unit="img"):
        res = process_single_image(
            image_path=img_path,
            output_crops_dir=crops_dir,
            debug_dir=debug_dir,
            lines_per_crop=args.lines_per_crop,
            target_brightness=args.target_brightness,
            pad_y=args.pad_y,
            min_page_ink_ratio=args.min_ink_ratio,
            suppress_ruled=not args.no_ruled_suppression,
            work_width=args.work_width
        )

        if res["status"] == "success":
            total_crops += res["crops_count"]
        elif res["status"] == "skipped":
            skipped_count += 1
            tqdm.write(f"[SKIPPED] {img_path.name}: {res['reason']}")
        else:
            tqdm.write(f"[ERROR] {img_path.name}: {res.get('error', 'Unknown')}")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print(f"   Total multi-line crops: {total_crops}")
    print(f"   Blank pages skipped:    {skipped_count}")
    print(f"   Crops saved to:         {crops_dir.resolve()}")
    if debug_dir:
        print(f"   Debug overlay:          {debug_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
