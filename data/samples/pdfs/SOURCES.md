# Downloaded Arabic PDFs (Internet Archive)

| File | Source | Notes |
|------|--------|-------|
| `arabic_bayan.pdf` | https://archive.org/details/ArabicBayan_201810 | Scanned lithograph (not used by digital-only agent) |
| `arabic_bayan_text.pdf` | same item (`Arabic Bayan_text.pdf`) | Archive OCR text-layer PDF for comparison |

Re-download:

```bash
mkdir -p data/samples/pdfs
curl -L -o data/samples/pdfs/arabic_bayan.pdf \
  "https://archive.org/download/ArabicBayan_201810/Arabic%20Bayan.pdf"
curl -L -o data/samples/pdfs/arabic_bayan_text.pdf \
  "https://archive.org/download/ArabicBayan_201810/Arabic%20Bayan_text.pdf"
```
