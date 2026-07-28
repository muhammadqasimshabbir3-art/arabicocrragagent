"""Qari-OCR backend (NAMAA Arabic VLM) with automatic model download."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from config.settings import get_settings
from loaders.base import LoadedDocument, LoadedPage
from ocr.base import OCRDocument, OCREngine, OCRError, OCRPage
from preprocess.arabic_normalize import clean_ocr_text, dedupe_repeated_lines
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _model_dir() -> Path:
    settings = get_settings()
    if settings.ocr_model_path:
        return Path(settings.ocr_model_path).expanduser().resolve()
    safe_name = settings.ocr_model_id.replace("/", "__")
    return settings.models_dir / safe_name


def _model_ready(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").is_file() or (path / "model.safetensors").is_file()
    )


@lru_cache(maxsize=1)
def _load_qari():
    """Lazy-load the Qari OCR VLM (downloads to models/ on first use)."""
    settings = get_settings()
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:
        raise OCRError(
            "Qari OCR requires `transformers`, `torch`, and related packages. "
            "Install project dependencies and try again."
        ) from exc

    path = _model_dir()
    model_id = settings.ocr_model_id
    source = str(path) if _model_ready(path) else model_id

    logger.info("Loading OCR model from %s", source)
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)

    device = settings.ocr_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    except Exception:
        # Older Qwen2-VL checkpoints may need the dedicated class.
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

    if device == "cpu":
        model = model.to("cpu")

    if not _model_ready(path) and source == model_id:
        path.mkdir(parents=True, exist_ok=True)
        try:
            processor.save_pretrained(str(path))
            model.save_pretrained(str(path))
            logger.info("Saved OCR model to %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist OCR model locally: %s", exc)

    return processor, model, device


def _page_to_pil(page: LoadedPage, document: LoadedDocument):
    from PIL import Image

    if page.image_bytes:
        return Image.open(BytesIO(page.image_bytes)).convert("RGB")

    # Render PDF page via pypdfium2 / pymupdf when available.
    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise OCRError(
            "Scanned PDF OCR requires `pymupdf` to render pages. "
            "Install project dependencies, or upload page images."
        ) from exc

    pdf = fitz.open(stream=document.raw_bytes, filetype="pdf")
    try:
        index = max(0, page.page_number - 1)
        if index >= len(pdf):
            raise OCRError(f"Page {page.page_number} is out of range.")
        pix = pdf[index].get_pixmap(matrix=fitz.Matrix(2, 2))
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        pdf.close()


def _ocr_image(image, max_tokens: int) -> str:
    processor, model, device = _load_qari()
    import torch

    prompt = (
        "Extract all Arabic and English text from this document image. "
        "Preserve reading order, headings, paragraphs, lists, and tables. "
        "Return clean text (use simple HTML for tables/headings when helpful)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    if device == "cuda":
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    # Trim prompt tokens when present.
    generated = output_ids
    if "input_ids" in inputs:
        trimmed = []
        for in_ids, out_ids in zip(inputs["input_ids"], output_ids, strict=False):
            trimmed.append(out_ids[len(in_ids) :])
        generated = trimmed

    decoded = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (decoded[0] if decoded else "").strip()


class QariOCREngine(OCREngine):
    """Arabic-specialized Qari VLM OCR with automatic Hugging Face download."""

    name = "qari"

    def extract(self, document: LoadedDocument) -> OCRDocument:
        settings = get_settings()
        pages: list[OCRPage] = []
        for page in document.pages:
            try:
                image = _page_to_pil(page, document)
                raw = _ocr_image(image, settings.ocr_max_tokens)
            except OCRError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise OCRError(f"Qari OCR failed on page {page.page_number}: {exc}") from exc

            text = dedupe_repeated_lines(clean_ocr_text(raw))
            html = raw if "<" in raw and ">" in raw else ""
            pages.append(
                OCRPage(
                    page_number=page.page_number,
                    text=text,
                    html=html,
                    metadata={"engine": self.name, "model": settings.ocr_model_id},
                )
            )

        if not any(page.text.strip() for page in pages):
            raise OCRError("OCR completed but no text was extracted.")

        return OCRDocument(
            filename=document.filename,
            fingerprint=document.fingerprint,
            pages=pages,
            engine=self.name,
            metadata={"model": settings.ocr_model_id, "page_count": len(pages)},
        )
