import sys
import shutil
import argparse
from pathlib import Path
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_pipeline(
    input_path: str,
    output_root: str = "dataset",
    batch_name: str = None,
    dpi: int = 300,
    lines_per_crop: int = 3,
    target_brightness: int = 240,
    work_width: int = 2480,
    debug: bool = False,
    cleanup_pages: bool = False
) -> None:
    input_p = Path(input_path)
    if not input_p.exists():
        sys.exit(f"[ERROR] Input path does not exist: {input_p}")

    if batch_name:
        batch_id = batch_name
    elif input_p.is_file():
        batch_id = input_p.stem
    else:
        batch_id = input_p.name

    root_p = Path(output_root) / batch_id
    page_images_dir = root_p / "page_images"
    line_crops_dir = root_p / "line_crops"
    debug_dir = root_p / "debug_overlays" if debug else None

    page_images_dir.mkdir(parents=True, exist_ok=True)
    line_crops_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"STARTING HANDWRITING PIPELINE FOR BATCH: '{batch_id}'")
    print(f"Input:      {input_p.resolve()}")
    print(f"Output dir: {root_p.resolve()}")
    print("=" * 60)

    print("\n[STEP 1/2] Converting PDFs to high-resolution page images...")
    cmd_step1 = [
        sys.executable,
        "1_pdf_to_images.py",
        "--input", str(input_p),
        "--output_dir", str(page_images_dir),
        "--dpi", str(dpi)
    ]
    res1 = subprocess.run(cmd_step1)
    if res1.returncode != 0:
        sys.exit("[ERROR] Failed during PDF conversion step. Aborting.")

    print("\n[STEP 2/2] Segmenting and cropping text lines...")
    cmd_step2 = [
        sys.executable,
        "2_crop_lines.py",
        "--input", str(page_images_dir),
        "--output_dir", str(line_crops_dir),
        "--lines_per_crop", str(lines_per_crop),
        "--target_brightness", str(target_brightness),
        "--work_width", str(work_width)
    ]
    if debug_dir:
        cmd_step2.extend(["--debug_dir", str(debug_dir)])

    res2 = subprocess.run(cmd_step2)
    if res2.returncode != 0:
        sys.exit("[ERROR] Failed during line cropping step. Aborting.")

    if cleanup_pages:
        print("\nCleaning up intermediate page images to save disk space...")
        shutil.rmtree(page_images_dir, ignore_errors=True)
        print("Intermediate page images removed.")

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY FOR: {batch_id}")
    print(f"   Line crops:   {line_crops_dir.resolve()}")
    if debug_dir:
        print(f"   Debug visual: {debug_dir.resolve()}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-end handwriting dataset ingestion pipeline.")
    parser.add_argument("--input", "-i", required=True, help="Path to uploaded .zip file or folder of PDFs.")
    parser.add_argument("--output_dir", "-o", default="dataset", help="Root folder to save dataset output.")
    parser.add_argument("--batch_name", "-b", default=None, help="Optional custom name for this batch/user.")
    parser.add_argument("--dpi", type=int, default=300, help="Rendering resolution (default: 300 DPI).")
    parser.add_argument("--lines_per_crop", "-n", type=int, default=3, help="Number of text lines per crop (default: 3).")
    parser.add_argument("--target_brightness", type=int, default=240, help="Normalized target paper brightness (0-255).")
    parser.add_argument("--work_width", type=int, default=2480, help="Resize pages to this width before cropping (default: 2480, 0 = keep original).")
    parser.add_argument("--debug", action="store_true", help="Generate debug visualization images with bounding boxes.")
    parser.add_argument("--cleanup_pages", action="store_true", help="Delete full-page images after cropping to save space.")

    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        output_root=args.output_dir,
        batch_name=args.batch_name,
        dpi=args.dpi,
        lines_per_crop=args.lines_per_crop,
        target_brightness=args.target_brightness,
        work_width=args.work_width,
        debug=args.debug,
        cleanup_pages=args.cleanup_pages
    )


if __name__ == "__main__":
    main()
