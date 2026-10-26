# Handwriting Dataset Ingestion & Line Cropper

A pipeline to parse handwritten assignment PDFs (or ZIP archives) and crop text into clean, multi-line image strips across the full page width for handwriting dataset curation.

---

## 1. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Usage

You can run the entire pipeline in **one command**:

### Process a single PDF:
```bash
python process_upload.py --input "assignment.pdf"
```

### Process a folder of PDFs (or a .zip of PDFs):
```bash
python process_upload.py --input "path/to/folder_of_pdfs"
```

### Output:
All outputs are saved automatically in the `dataset/` directory:
* `dataset/<name>/line_crops/` — The final cropped handwriting images (3–4 lines per image).

---

## 3. Useful Flags

* `--lines_per_crop 3` — Number of text lines per crop (default: `3`, automatically makes 3–4 lines).
* `--debug` — Generates visual inspection images (`dataset/<name>/debug_overlays/`) showing green bounding boxes over the pages.
* `--cleanup_pages` — Automatically deletes full-page intermediate images after cropping to save disk space.
* `--dpi 300` — Rendering resolution (default: `300` DPI).

Example with flags:
```bash
python process_upload.py --input "assignment.pdf" --lines_per_crop 3 --debug
```

---

## 4. Running Scripts Individually (Optional)

If you prefer to run the steps manually:

**Step 1: Convert PDFs to Page Images**
```bash
python 1_pdf_to_images.py --input "assignment.pdf" --output_dir "data/page_images" --dpi 300
```

**Step 2: Crop Lines from Images**
```bash
python 2_crop_lines.py --input "data/page_images" --output_dir "data/line_crops" --lines_per_crop 3
```
