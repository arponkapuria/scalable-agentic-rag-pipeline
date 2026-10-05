"""
Unified parser + chunker for PDF/DOCX/HTML/MD via Docling — one format-agnostic path for every supported file type, wired into pipeline.py.

Layout detection and picture classification run via ONNXRuntime — no torch needed for either. Table structure (TableFormer) has no first-party ONNX path in mainline Docling, so it runs via torch — kept on anyway since table accuracy matters for this project's literature-survey corpus. Pinned to TableStructureOptions V1, not V2 — V2 has an open upstream bug duplicating rows on multi-page tables.

Image captioning deliberately does NOT use Docling's built-in picture-description option — that would call the vision API directly from inside Docling's own HTTP client, bypassing this app's shared rate limiter/circuit breaker on mistral_client that every other LLM/vision call goes through. Docling only classifies and crops figures here; captioning is a separate step through mistral_client.
"""
import asyncio
import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import PictureItem
from docling_core.types.doc.document import PictureClassificationData
from transformers import AutoTokenizer

try:
    from docling.datamodel.object_detection_engine_options import (
        OnnxRuntimeObjectDetectionEngineOptions,
    )
    from docling.datamodel.picture_classification_options import (
        DocumentPictureClassifierOptions,
    )
    _ONNX_LAYOUT_AVAILABLE = True
except ImportError:
    # Older/newer Docling may have moved these — fall back to Docling's own default
    # (torch-backed) engine rather than failing ingestion entirely.
    _ONNX_LAYOUT_AVAILABLE = False

from services.api.app.config import settings
from services.api.app.clients.llm.mistral_client import mistral_client
from pipelines.ingestion.debug_dump import dump_debug_artifact

logger = logging.getLogger(__name__)

_converter: Optional[DocumentConverter] = None  # built once, reused — model/tokenizer loading isn't free
_chunker: Optional[HybridChunker] = None


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = settings.PDF_ENABLE_OCR
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
        mode=(
            TableFormerMode.ACCURATE
            if settings.TABLE_STRUCTURE_MODE == "accurate"
            else TableFormerMode.FAST
        ),
    )
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

    if _ONNX_LAYOUT_AVAILABLE:
        layout_options = LayoutObjectDetectionOptions.from_preset("layout_heron_default")
        layout_options.engine_options = OnnxRuntimeObjectDetectionEngineOptions()
        pipeline_options.layout_options = layout_options

        picture_classifier_options = DocumentPictureClassifierOptions.from_preset(
            "document_figure_classifier_v2"
        )
        pipeline_options.picture_classification_options = picture_classifier_options
        pipeline_options.do_picture_classification = True
    else:
        logger.warning(
            "ONNXRuntime layout/picture-classifier options unavailable in this "
            "Docling version — falling back to defaults (likely torch-backed)."
        )

    pipeline_options.do_picture_description = False  # own captioning pass below, not Docling's
    pipeline_options.generate_picture_images = True  # keeps cropped bitmaps available via PictureItem.get_image()

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
        # DOCX/HTML/MD use Docling's own default (non-PDF) pipelines — no enrichment models needed.
    )


class _MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Overrides HybridChunker's table serialization to compact Markdown pipe tables, since these chunks are for embedding/LLM consumption, not human display."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


def _build_chunker() -> HybridChunker:
    # Tokenizer matches the real embedder (FASTEMBED_MODEL), so the token ceiling here is the
    # one that actually matters at embed time.
    hf_tokenizer = AutoTokenizer.from_pretrained(settings.FASTEMBED_MODEL)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=settings.CHUNK_MAX_TOKENS)
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        # Docling's default table serializer (TripletTableSerializer) produces a wall of
        # low-signal text for real academic tables — MarkdownTableSerializer is far more compact
        # for the same information.
        serializer_provider=_MarkdownTableSerializerProvider(),
        # repeat_table_header defaults True, stated explicitly since the serializer is
        # overridden — a table split across chunks should keep its header in every fragment.
        repeat_table_header=True,
    )


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = _build_converter()
    return _converter


def _get_chunker() -> HybridChunker:
    global _chunker
    if _chunker is None:
        _chunker = _build_chunker()
    return _chunker


def _convert_and_chunk_sync(file_bytes: bytes, filename: str) -> Tuple[Any, HybridChunker, list]:
    """Runs in a worker thread — both conversion and chunking are CPU-bound/blocking."""
    stream = DocumentStream(name=filename, stream=io.BytesIO(file_bytes))
    doc = _get_converter().convert(stream).document
    chunker = _get_chunker()
    raw_chunks = list(chunker.chunk(dl_doc=doc))
    return doc, chunker, raw_chunks


def _top_picture_class(picture: PictureItem) -> Optional[str]:
    for ann in picture.annotations:
        if isinstance(ann, PictureClassificationData) and ann.predicted_classes:
            return ann.predicted_classes[0].class_name.lower()
    return None


async def _describe_image(png_bytes: bytes, filename: str, index: int, corpus_id: Optional[str]) -> Optional[str]:
    """Captions one figure via the shared, rate-limited mistral_client.

    Args:
        png_bytes: The cropped figure image.
        filename: The source document's filename, for logging.
        index: The figure's position in the document, for logging.
        corpus_id: For log-line scoping.

    Returns:
        The caption text, or None if captioning failed (soft-fails per image, never fails the doc).
    """
    log_prefix = f"[ingest:{corpus_id}] " if corpus_id else ""
    try:
        b64_image = base64.b64encode(png_bytes).decode("utf-8")
        content = await mistral_client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this figure/chart/diagram from a research paper in 1-2 sentences, focused on what data or concept it conveys.",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    ],
                }
            ],
            temperature=0.0,
        )
        return content.strip()
    except Exception as e:
        logger.warning(f"{log_prefix}Image captioning failed for {filename} figure {index}: {e}")
        return None


async def _caption_pictures(doc: Any, filename: str, corpus_id: Optional[str]) -> Dict[str, str]:
    """Captions every non-skipped figure in a document, sequentially and rate-paced.

    Args:
        doc: The parsed Docling document.
        filename: The source document's filename, for logging.
        corpus_id: For log-line scoping and the debug artifact.

    Returns:
        A dict mapping each captioned picture's self_ref to its caption text.
    """
    log_prefix = f"[ingest:{corpus_id}] " if corpus_id else ""
    pictures = list(doc.pictures)
    if not pictures:
        logger.info(f"{log_prefix}No pictures detected in {filename}.")
        return {}

    skip_classes = {c.strip().lower() for c in settings.PICTURE_SKIP_CLASSES.split(",") if c.strip()}
    captions: Dict[str, str] = {}
    skipped = 0
    failed = 0
    last_call_started = 0.0

    for i, picture in enumerate(pictures, start=1):
        top_class = _top_picture_class(picture)
        if top_class and top_class in skip_classes:
            skipped += 1
            continue
        try:
            image = picture.get_image(doc)
            if image is None:
                failed += 1
                continue
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            # Paced, not just sequential — the provider's per-second cap is invisible to a
            # per-minute rate limiter, so back-to-back calls would still draw 429s.
            wait = settings.CAPTION_MIN_INTERVAL_SECONDS - (time.monotonic() - last_call_started)
            if wait > 0:
                await asyncio.sleep(wait)
            last_call_started = time.monotonic()
            caption = await _describe_image(buf.getvalue(), filename, i, corpus_id)
            if caption:
                logger.info(f"{log_prefix}Captioned figure {i}/{len(pictures)}: {caption[:100]}")
                captions[picture.self_ref] = caption
            else:
                failed += 1
        except Exception as e:
            logger.warning(f"{log_prefix}Captioning failed for figure {i}/{len(pictures)} in {filename}: {e}")
            failed += 1

    logger.info(
        f"{log_prefix}Captioning for {filename}: {len(captions)}/{len(pictures)} succeeded, "
        f"{skipped} skipped by class, {failed} failed."
    )
    if corpus_id:
        dump_debug_artifact(
            corpus_id,
            "docling_pictures",
            [
                {
                    "self_ref": p.self_ref,
                    "top_class": _top_picture_class(p),
                    "captioned": p.self_ref in captions,
                }
                for p in pictures
            ],
        )
    return captions


async def parse_and_chunk(file_bytes: bytes, filename: str, corpus_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parses and chunks a document, with figures captioned inline.

    Args:
        file_bytes: The raw document bytes.
        filename: The document's filename (used for format detection).
        corpus_id: For log-line scoping and debug artifacts.

    Returns:
        A list of {"text", "metadata"} chunk dicts.
    """
    log_prefix = f"[ingest:{corpus_id}] " if corpus_id else ""
    ext = filename.lower().rsplit(".", 1)[-1]

    t0 = time.monotonic()
    doc, chunker, raw_chunks = await asyncio.to_thread(_convert_and_chunk_sync, file_bytes, filename)
    convert_elapsed = time.monotonic() - t0
    logger.info(
        f"{log_prefix}stage=parse_chunk docling convert+chunk={convert_elapsed:.1f}s chunks={len(raw_chunks)}"
    )

    captions: Dict[str, str] = {}
    if settings.PDF_DESCRIBE_IMAGES:
        captions = await _caption_pictures(doc, filename, corpus_id)

    chunks: List[Dict[str, Any]] = []
    for i, chunk in enumerate(raw_chunks):
        text = chunker.contextualize(chunk)

        headings: List[str] = []
        pages: List[int] = []
        has_table = False
        chunk_captions: List[str] = []
        try:
            doc_items = chunk.meta.doc_items or []
            headings = list(chunk.meta.headings or [])
            for item in doc_items:
                for prov in getattr(item, "prov", []) or []:
                    if getattr(prov, "page_no", None) is not None:
                        pages.append(prov.page_no)
                if item.__class__.__name__ == "TableItem":
                    has_table = True
                if isinstance(item, PictureItem) and item.self_ref in captions:
                    chunk_captions.append(captions[item.self_ref])
        except Exception as e:
            # Defensive: don't let an unexpected docling_core shape kill ingestion.
            logger.warning(f"{log_prefix}Chunk metadata extraction fell back to defaults: {e}")

        if chunk_captions:
            text += "\n\n" + "\n".join(f"[Figure caption: {c}]" for c in chunk_captions)

        pages = sorted(set(pages))
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "filename": filename,
                    "type": ext,
                    "section": " > ".join(headings) if headings else "Document",
                    "page": pages[0] if pages else 0,
                    "pages": pages,
                    "chunk_index": i,
                    "has_table": has_table,
                    "image_count": len(captions),
                },
            }
        )

    return chunks