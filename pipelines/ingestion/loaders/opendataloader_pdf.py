"""
PDF-only for now (Phase 3/4 scope) — DOCX/HTML stay on the old flat-text
loaders until a later phase.

Replaces Docling: Docling pulls in torch (via docling-ibm-models/rapidocr)
as a hard dependency, and even the CPU-only build plus everything else
pushed a demo-scale Docker build past available disk on a constrained dev
machine. OpenDataLoader PDF is rule-based/deterministic (no ML model, no
torch, no GPU) — Java process instead, ranks #1 in a 200-PDF benchmark
covering reading order, table, and heading extraction (0.907 overall) and
handles multi-column academic layouts via its XY-Cut++ reading-order
algorithm.

Outputs Markdown directly (headings preserved as # / ## / ### lines,
tables as markdown tables) — section_splitter.py now splits on markdown
headers directly instead of consuming a structured element list, per the
simpler "convert to markdown, then chunk that" approach.

NOT live-verified end-to-end in this environment (network/disk
constraints) — in particular the JSON output's exact per-element schema
(field names for type/page/bbox) is inferred from public docs, not
confirmed against a real run. Verify during Phase 3/4 testing; the image
detection step below is written defensively (multiple key-name guesses)
specifically because of this.
"""
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz  # `import fitz` is deprecated per PyMuPDF's own
                         # warning; used only to crop the actual image
                         # bytes at a page+bbox OpenDataLoader's JSON
                         # identifies as a figure/image (its Markdown
                         # output flags *that* an image exists, not the
                         # raw image bytes).
import opendataloader_pdf

from services.api.app.config import settings
from services.api.app.clients.llm.gemini_client import gemini_client
from pipelines.ingestion.debug_dump import dump_debug_artifact

logger = logging.getLogger(__name__)


async def parse_pdf_bytes(file_bytes: bytes, filename: str, corpus_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (markdown_text, metadata). markdown_text is what
    section_splitter.py's split_markdown_by_sections() consumes directly —
    no separate structured-elements step needed, OpenDataLoader's Markdown
    output already preserves heading hierarchy and table structure.

    Async now (captioning does a real network call via Gemini) — the
    JVM/PyMuPDF work below is still synchronous/blocking; callers running
    inside an event loop (in-process ingestion) should await this from a
    context where a blocking call is acceptable, or wrap the sync portion
    in asyncio.to_thread if called from a latency-sensitive path.

    `corpus_id`: optional, used only to prefix log lines consistently
    with other ingestion stage logs and to name debug-dump artifacts.
    """
    log_prefix = f"[ingest:{corpus_id}] " if corpus_id else ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, filename)
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        out_dir = os.path.join(tmp_dir, "out")
        # Spawns a JVM process — slow per-call, but this loader is only
        # invoked once per ingestion job (one PDF per job), not in a
        # tight loop, so the fixed JVM-startup cost doesn't compound.
        opendataloader_pdf.convert(
            input_path=[pdf_path],
            output_dir=out_dir,
            format="markdown,json",
        )

        base_name = os.path.splitext(filename)[0]
        md_path = os.path.join(out_dir, f"{base_name}.md")
        json_path = os.path.join(out_dir, f"{base_name}.json")

        if not os.path.exists(md_path):
            raise FileNotFoundError(f"OpenDataLoader did not produce expected output: {md_path}")

        with open(md_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        image_count = 0
        if os.path.exists(json_path) and settings.PDF_DESCRIBE_IMAGES:
            markdown_text, image_count = await _inline_image_captions(
                markdown_text, json_path, pdf_path, filename, corpus_id
            )
        elif not os.path.exists(json_path):
            logger.warning(f"{log_prefix}OpenDataLoader produced no JSON output for {filename} — image captioning skipped entirely.")
        elif not settings.PDF_DESCRIBE_IMAGES:
            logger.info(f"{log_prefix}PDF_DESCRIBE_IMAGES=false — image captioning skipped.")

        metadata = {
            "filename": filename,
            "type": "pdf",
            "has_tables": "|" in markdown_text,  # markdown tables use pipe syntax
            "image_count": image_count,
        }

        return markdown_text, metadata


async def _inline_image_captions(
    markdown_text: str, json_path: str, pdf_path: str, filename: str, corpus_id: Optional[str] = None
) -> Tuple[str, int]:
    """
    Reads OpenDataLoader's JSON output for image/figure elements, crops
    the actual pixels via PyMuPDF at the given page+bbox, captions via one
    Gemini vision call per figure, and appends captions to the markdown
    text (image bytes themselves aren't in the JSON, just their detected
    location — schema/key names below are best-effort, see module
    docstring). Figures are captioned sequentially, not concurrently —
    Gemini's free tier is ~15 RPM, so fanning out N concurrent calls for
    an N-figure paper would just trade rate-limit skips for 429s.
    """
    import json

    log_prefix = f"[ingest:{corpus_id}] " if corpus_id else ""

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc_json = json.load(f)
    except Exception as e:
        logger.warning(f"{log_prefix}Could not read OpenDataLoader JSON for {filename}: {e}")
        return markdown_text, 0

    all_elements = _flatten(doc_json.get("elements", doc_json.get("pages", [])))
    image_elements = [
        el for el in all_elements
        if isinstance(el, dict) and str(el.get("type", el.get("category", ""))).lower() in ("image", "figure")
    ]

    if not image_elements:
        # This used to be silent — the exact bug that made "did Gemini
        # even run?" unanswerable from logs alone. Log the type/category
        # values actually seen so a schema mismatch (the module docstring's
        # known risk — key names were never live-verified) is diagnosable
        # directly from this line instead of guessing.
        seen_types = sorted({
            str(el.get("type", el.get("category", "<none>")))
            for el in all_elements if isinstance(el, dict)
        })
        logger.info(
            f"{log_prefix}No image/figure elements detected in {filename} "
            f"({len(all_elements)} total elements scanned; types seen: {seen_types[:15]})"
        )
        return markdown_text, 0

    logger.info(f"{log_prefix}Found {len(image_elements)} image/figure element(s) in {filename} — captioning via Gemini.")

    if corpus_id:
        dump_debug_artifact(corpus_id, "detected_images", image_elements)

    captions = []
    caption_failures = 0
    try:
        doc = fitz.open(pdf_path)
        for i, el in enumerate(image_elements, start=1):
            page_no = el.get("page", el.get("page_number", 1)) - 1
            bbox = el.get("bbox", el.get("bounding_box"))
            if page_no < 0 or page_no >= len(doc) or not bbox:
                logger.warning(f"{log_prefix}Skipping figure {i}/{len(image_elements)}: invalid page/bbox ({page_no=}, {bbox=})")
                caption_failures += 1
                continue

            page = doc[page_no]
            rect = fitz.Rect(*bbox)
            pix = page.get_pixmap(clip=rect, dpi=150)
            png_bytes = pix.tobytes("png")

            caption = await _describe_image(png_bytes, filename, i, corpus_id)
            if caption:
                logger.info(f"{log_prefix}Gemini captioned figure {i}/{len(image_elements)}: {caption[:100]}")
                captions.append(f"\n\n[Figure {i}]: {caption}\n")
            else:
                caption_failures += 1
        doc.close()
    except Exception as e:
        logger.warning(f"{log_prefix}Image extraction failed for {filename}: {e}")

    logger.info(
        f"{log_prefix}Gemini captioning done for {filename}: "
        f"{len(captions)}/{len(image_elements)} succeeded, {caption_failures} failed/skipped."
    )

    if captions:
        markdown_text += "".join(captions)

    return markdown_text, len(captions)


def _flatten(nested) -> List[dict]:
    """OpenDataLoader's JSON structure (page-nested vs flat element list)
    isn't confirmed — this walks either shape looking for dicts."""
    result = []
    if isinstance(nested, dict):
        result.append(nested)
        for v in nested.values():
            result.extend(_flatten(v))
    elif isinstance(nested, list):
        for item in nested:
            result.extend(_flatten(item))
    return result


async def _describe_image(png_bytes: bytes, filename: str, index: int, corpus_id: Optional[str] = None) -> Optional[str]:
    """One Gemini vision call per figure, via the shared gemini_client
    (rate-limited/circuit-breaker-protected like every other LLM client
    here — see clients/llm/gemini_client.py). Soft-failing: a caption
    failure (budget exhausted, bad response, network error) drops that
    one image, not the whole document."""
    import base64

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
