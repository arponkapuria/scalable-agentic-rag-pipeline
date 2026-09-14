"""
Unified parser + chunker for PDF/DOCX/HTML/MD via Docling — the active
loader for the in-process (default, INGESTION_BACKEND=in_process)
pipeline. One format-agnostic path replaces the old PDF-via-OpenDataLoader
/ DOCX+HTML-via-homegrown-parsers split.

pipelines/ingestion/main.py (Ray path, frozen), loaders/opendataloader_pdf.py,
loaders/docx.py, loaders/html.py, chunking/section_splitter.py, and
chunking/splitter.py are all left untouched — INGESTION_BACKEND=ray still
uses that original chain if ever reactivated. This module is wired into
pipeline.py only.

Resourcing (why Docling is viable here now, see PROGRESS.md): layout
detection (Heron/RT-DETR) and picture classification run via ONNXRuntime,
configured below — no torch needed for either. Table structure
(TableFormer) has no first-party ONNX path in mainline Docling (confirmed
via Docling's own technical report: "For inference, our implementation
relies on PyTorch") — kept enabled anyway since tables matter for a
literature-survey corpus and accuracy is the top priority; the earlier
Docling disk exhaustion was Docling's overall model/dependency footprint,
not specifically a CUDA torch build (this repo's M1/Oracle-ARM targets
never had a CUDA wheel in play regardless of library). Pinned to
TableStructureOptions (V1, not V2) — V2 has an open bug (docling#3553,
Jun 2026) duplicating rows on multi-page tables, exactly the shape of a
references/bibliography table.

Image captioning deliberately does NOT use Docling's built-in
do_picture_description/PictureDescriptionApiOptions — that would call
Gemini directly from inside Docling's own HTTP client, bypassing this
app's shared BackendRateLimiter/CircuitBreaker on gemini_client that every
other LLM/vision call goes through. Instead: Docling only classifies +
crops figures (do_picture_classification=True, do_picture_description=
False), and captioning is a separate step through the existing
gemini_client, exactly like the code it replaces.

NOT live-verified end-to-end in this environment (no network for a
multi-hundred-MB install here) — several exact import paths below
(OnnxRuntimeObjectDetectionEngineOptions' module, DocMeta field names)
were confirmed via Docling's current docs/examples but not run. If an
import fails or chunk.meta fields differ, run:
    python -c "from docling_core.transforms.chunker.hierarchical_chunker import DocChunk; help(DocChunk)"
to check the installed version's actual shape — same "verify against a
real run" caveat opendataloader_pdf.py carried for the same reason.
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
    # Older/newer Docling may have moved these — fall back to Docling's
    # own default engine (torch) rather than failing ingestion entirely.
    # See module docstring's verification note.
    _ONNX_LAYOUT_AVAILABLE = False

from services.api.app.config import settings
from services.api.app.clients.llm.gemini_client import gemini_client
from pipelines.ingestion.debug_dump import dump_debug_artifact

logger = logging.getLogger(__name__)

# Built once, reused for the process's lifetime — model/tokenizer loading
# is not free (same lazy-singleton pattern as models/embeddings/fastembed_client.py).
_converter: Optional[DocumentConverter] = None
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

    # Own captioning pass below, not Docling's — see module docstring.
    pipeline_options.do_picture_description = False
    pipeline_options.generate_picture_images = True  # keeps cropped bitmaps available via PictureItem.get_image()

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
        # DOCX/HTML/MD use Docling's own default (non-PDF) pipelines —
        # no enrichment models needed for those formats.
    )


class _MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Docling's own documented pattern for overriding table serialization
    in HybridChunker (see docling docs' "Advanced chunking &
    serialization" page) — compact_tables=True drops extra Markdown
    padding, no reason to keep it since these chunks are for
    embedding/LLM consumption, not human display."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


def _build_chunker() -> HybridChunker:
    # Tokenizer matches the real embedder (FASTEMBED_MODEL) so the token
    # ceiling here is the one that actually matters at embed time — the
    # code this replaces only approximated tokens via character count.
    hf_tokenizer = AutoTokenizer.from_pretrained(settings.FASTEMBED_MODEL)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=settings.CHUNK_MAX_TOKENS)
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        # Confirmed via live testing (not the docs default): Docling's
        # default table serializer is TripletTableSerializer — one
        # "RowLabel, ColLabel = value" line per cell, including empty
        # cells as bare ".". For a real academic table (Table 3 in a
        # transformer-architecture paper) this produced a wall of
        # low-signal repeated text, at a real cost: this is very likely
        # what was pushing graph extraction into max_tokens truncation on
        # table-heavy chunks specifically. Switched to MarkdownTableSerializer
        # (docling's own documented alternative) for a compact pipe-table
        # instead — same information, far fewer tokens.
        serializer_provider=_MarkdownTableSerializerProvider(),
        # repeat_table_header defaults True already, but stated explicitly
        # since we're overriding the serializer — a table split across
        # chunks should keep its header in every fragment or entity
        # extraction loses column meaning for later fragments. Docling has
        # an open bug (docling#2975) where this doesn't always propagate
        # correctly with MarkdownTableSerializer specifically — not fixed
        # on our side, just something to watch for in chunks.json if a
        # split table's later chunks look header-less.
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
    """Runs in a worker thread (see parse_and_chunk) — both conversion
    (layout/table inference) and chunking are CPU-bound/blocking."""
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
    """Same shared, rate-limited gemini_client as the code this replaces
    (opendataloader_pdf.py) — soft-fails per image, never fails the doc."""
    log_prefix = f"[ingest:{corpus_id}] " if corpus_id else ""
    try:
        b64_image = base64.b64encode(png_bytes).decode("utf-8")
        content = await gemini_client.chat_completion(
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
    """Returns {picture.self_ref: caption}. Skips pictures the classifier
    scores into PICTURE_SKIP_CLASSES before spending a Gemini call —
    direct answer to "don't burn API calls on logos." Sequential, not
    concurrent, same reasoning as before: Gemini free tier is ~15 RPM."""
    log_prefix = f"[ingest:{corpus_id}] " if corpus_id else ""
    pictures = list(doc.pictures)
    if not pictures:
        logger.info(f"{log_prefix}No pictures detected in {filename}.")
        return {}

    skip_classes = {c.strip().lower() for c in settings.PICTURE_SKIP_CLASSES.split(",") if c.strip()}
    captions: Dict[str, str] = {}
    skipped = 0
    failed = 0

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
            caption = await _describe_image(buf.getvalue(), filename, i, corpus_id)
            if caption:
                logger.info(f"{log_prefix}Gemini captioned figure {i}/{len(pictures)}: {caption[:100]}")
                captions[picture.self_ref] = caption
            else:
                failed += 1
        except Exception as e:
            logger.warning(f"{log_prefix}Captioning failed for figure {i}/{len(pictures)} in {filename}: {e}")
            failed += 1

    logger.info(
        f"{log_prefix}Gemini captioning for {filename}: {len(captions)}/{len(pictures)} succeeded, "
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
    """
    Returns chunk dicts in the {"text", "metadata"} shape pipeline.py
    already expects (enrich_metadata/embed/index_qdrant downstream) —
    format-agnostic: PDF/DOCX/HTML/MD all go through this one path.
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
            # Defensive: don't let an unexpected docling_core shape kill
            # ingestion — fall back to bare text, log for investigation
            # (see module docstring's verification note).
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
