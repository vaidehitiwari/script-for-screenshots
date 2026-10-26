import os
import sys
import zipfile
import argparse
import re
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RENDER_BACKEND = None
try:
    import pymupdf as fitz
    RENDER_BACKEND = "pymupdf"
except ImportError:
    try:
        import fitz
        RENDER_BACKEND = "pymupdf"
    except ImportError:
        try:
            import pypdfium2 as pdfium
            RENDER_BACKEND = "pdfium"
        except ImportError:
            RENDER_BACKEND = None


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| \t\n\r]+', '_', name).strip('_')


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> List[Path]:
    extracted_pdfs = []
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_dir_resolved = extract_dir.resolve()

    print(f"Extracting archive: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if "__MACOSX" in name or os.path.basename(name).startswith("._") or name.endswith(".DS_Store"):
                continue

            target_path = (extract_dir / name).resolve()
            if not str(target_path).startswith(str(extract_dir_resolved)):
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

            if target_path.suffix.lower() == ".pdf":
                extracted_pdfs.append(target_path)

    return extracted_pdfs


def discover_pdfs(input_path: Path, temp_extract_dir: Path) -> List[Tuple[Path, str]]:
    pdf_entries = []

    if input_path.is_file():
        if input_path.suffix.lower() == ".zip":
            unpacked_dir = temp_extract_dir / input_path.stem
            pdfs = safe_extract_zip(input_path, unpacked_dir)
            batch_id = sanitize_filename(input_path.stem)
            for p in pdfs:
                pdf_entries.append((p, batch_id))
        elif input_path.suffix.lower() == ".pdf":
            pdf_entries.append((input_path, "upload"))
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}. Expected .zip or .pdf.")
    elif input_path.is_dir():
        zip_files = list(input_path.glob("*.zip")) + list(input_path.glob("**/*.zip"))
        for z in zip_files:
            unpacked_dir = temp_extract_dir / z.stem
            pdfs = safe_extract_zip(z, unpacked_dir)
            batch_id = sanitize_filename(z.stem)
            for p in pdfs:
                pdf_entries.append((p, batch_id))

        direct_pdfs = [p for p in input_path.rglob("*.pdf") if not str(p.resolve()).startswith(str(temp_extract_dir.resolve()))]
        for p in direct_pdfs:
            batch_id = sanitize_filename(p.parent.name) if p.parent != input_path else "batch"
            pdf_entries.append((p, batch_id))
    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    return pdf_entries


def render_pdf_pymupdf(pdf_path: Path, output_dir: Path, file_prefix: str, dpi: int = 300, img_format: str = "jpg", quality: int = 95, skip_existing: bool = True) -> int:
    doc = fitz.open(pdf_path)
    rendered_count = 0
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_idx in range(len(doc)):
        out_filename = f"{file_prefix}_p{page_idx + 1:04d}.{img_format}"
        out_path = output_dir / out_filename

        if skip_existing and out_path.exists():
            rendered_count += 1
            continue

        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if img_format.lower() in ("jpg", "jpeg"):
            pix.save(str(out_path), "jpeg", jpg_quality=quality)
        else:
            pix.save(str(out_path))

        rendered_count += 1

    doc.close()
    return rendered_count


def render_pdf_pdfium(pdf_path: Path, output_dir: Path, file_prefix: str, dpi: int = 300, img_format: str = "jpg", quality: int = 95, skip_existing: bool = True) -> int:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    rendered_count = 0
    scale = dpi / 72.0

    for page_idx in range(len(pdf)):
        out_filename = f"{file_prefix}_p{page_idx + 1:04d}.{img_format}"
        out_path = output_dir / out_filename

        if skip_existing and out_path.exists():
            rendered_count += 1
            continue

        page = pdf.get_page(page_idx)
        pil_image = page.render(scale=scale).to_pil()

        if img_format.lower() in ("jpg", "jpeg"):
            pil_image.save(out_path, "JPEG", quality=quality)
        else:
            pil_image.save(out_path, "PNG")

        rendered_count += 1

    pdf.close()
    return rendered_count


def process_pdfs(
    input_path: str,
    output_dir: str,
    dpi: int = 300,
    img_format: str = "jpg",
    quality: int = 95,
    skip_existing: bool = True,
    temp_dir: str = "_extracted_temp"
) -> None:
    if RENDER_BACKEND is None:
        sys.exit("[ERROR] Neither PyMuPDF nor pypdfium2 is installed. Run: pip install pymupdf")

    input_p = Path(input_path)
    output_p = Path(output_dir)
    temp_p = Path(temp_dir)

    output_p.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for PDFs in: {input_p}...")
    pdf_entries = discover_pdfs(input_p, temp_p)

    if not pdf_entries:
        print(f"[WARNING] No PDF files found in {input_p}.")
        return

    print(f"Found {len(pdf_entries)} PDF file(s) across uploads.")
    print(f"Backend: {RENDER_BACKEND.upper()} | Resolution: {dpi} DPI | Format: {img_format.upper()}")

    success_count = 0
    total_pages = 0
    failed_files = []

    for pdf_path, batch_id in tqdm(pdf_entries, desc="Converting PDFs", unit="pdf"):
        safe_name = sanitize_filename(pdf_path.stem)
        file_prefix = f"{batch_id}_{safe_name}"

        try:
            if RENDER_BACKEND == "pymupdf":
                pages = render_pdf_pymupdf(
                    pdf_path, output_p, file_prefix,
                    dpi=dpi, img_format=img_format, quality=quality,
                    skip_existing=skip_existing
                )
            else:
                pages = render_pdf_pdfium(
                    pdf_path, output_p, file_prefix,
                    dpi=dpi, img_format=img_format, quality=quality,
                    skip_existing=skip_existing
                )
            total_pages += pages
            success_count += 1
        except Exception as e:
            failed_files.append((str(pdf_path), str(e)))

    print("\n" + "=" * 50)
    print(f"Conversion complete")
    print(f"   Successfully processed: {success_count}/{len(pdf_entries)} PDFs")
    print(f"   Total pages rendered:   {total_pages}")
    print(f"   Page images saved in:   {output_p.resolve()}")

    if failed_files:
        print(f"\nEncountered errors in {len(failed_files)} file(s):")
        for f, err in failed_files:
            print(f"   - {f}: {err}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Convert assignment PDFs or ZIPs of PDFs to high-res page images.")
    parser.add_argument("--input", "-i", required=True, help="Path to input .zip file, folder of .zip files, or folder of PDFs.")
    parser.add_argument("--output_dir", "-o", default="data/page_images", help="Directory where rendered page images will be stored.")
    parser.add_argument("--dpi", type=int, default=300, help="Rendering resolution in DPI (default: 300).")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Image format (default: jpg for space efficiency).")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality if format is jpg (default: 95).")
    parser.add_argument("--no_skip", action="store_true", help="Overwrite existing rendered images rather than skipping.")
    parser.add_argument("--temp_dir", default="_temp_extracted", help="Temporary directory for unzipping files.")

    args = parser.parse_args()
    process_pdfs(
        input_path=args.input,
        output_dir=args.output_dir,
        dpi=args.dpi,
        img_format=args.format,
        quality=args.quality,
        skip_existing=not args.no_skip,
        temp_dir=args.temp_dir
    )


if __name__ == "__main__":
    main()
