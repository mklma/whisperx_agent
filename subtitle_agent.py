"""
WhisperX Subtitle Agent
=======================

MP4 -> Faster-Whisper/WhisperX transcription -> word alignment
    -> local Ollama translation to Simplified Chinese -> smart 2-line SRT

Designed for:
    Windows 11
    NVIDIA RTX 3070 Ti Laptop 8 GB
    Python 3.11
    CUDA GPU

Usage:
    python subtitle_agent.py "C:\\videos\\input.mp4"

Optional:
    python subtitle_agent.py "input.mp4" --source-language ja
    python subtitle_agent.py "input.mp4" --source-language en
    python subtitle_agent.py "input.mp4" --ollama-model qwen3.5:9b
    python subtitle_agent.py "input.mp4" --compute-type int8_float16
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
import torch
import whisperx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "large-v3"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"

# RTX 3070 Ti 8 GB: start conservatively.
DEFAULT_BATCH_SIZE = 2
DEFAULT_COMPUTE_TYPE = "float16"

# Subtitle readability
MAX_CHARS_PER_LINE = 22
MAX_TOTAL_CHARS = 44
MIN_DURATION = 0.9
MAX_DURATION = 7.0
MIN_GAP = 0.05

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
# OLLAMA_BASE_URL = "http://localhost:11434/v1"  ## for chat.completions API with messages

@dataclass
class Subtitle:
    index: int
    start: float
    end: float
    source: str
    text: str


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure ffmpeg.exe "
            "is available on PATH."
        )


def srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))

    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    millis = ms % 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# WhisperX transcription + alignment
# ---------------------------------------------------------------------------

def transcribe_and_align(
    video_path: str,
    source_language: Optional[str],
    model_name: str,
    batch_size: int,
    compute_type: str,
):
    """
    WhisperX uses Faster-Whisper internally for the ASR stage.

    Returns an aligned WhisperX result containing:
        segments
        word-level timestamps
        detected language
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("WARNING: CUDA is not available. This will run on CPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

    print(f"\nLoading WhisperX/Faster-Whisper model: {model_name}")
    print(f"device={device}, compute_type={compute_type}, batch_size={batch_size}")

    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=source_language,
    )

    audio = whisperx.load_audio(video_path)

    print("\n[1/3] Transcribing...")
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=source_language,
        task="transcribe",
    )

    detected_language = result["language"]
    print(f"Detected language: {detected_language}")
    print(f"Initial segments: {len(result['segments'])}")

    # Release ASR model before loading alignment model where possible.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[2/3] Loading forced-alignment model...")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )

    print("[3/3] Aligning words...")
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    del align_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return aligned, detected_language


# ---------------------------------------------------------------------------
# Convert aligned words into sensible subtitle units
# ---------------------------------------------------------------------------

def get_words(segment: dict) -> list[dict]:
    words = []
    for w in segment.get("words", []):
        word = clean_text(str(w.get("word", "")))
        start = w.get("start")
        end = w.get("end")

        if not word or start is None or end is None:
            continue

        words.append(
            {
                "word": word,
                "start": float(start),
                "end": float(end),
            }
        )
    return words


def split_long_segment(segment: dict) -> list[Subtitle]:
    """
    Use aligned word boundaries to break very long ASR segments.
    This happens BEFORE translation, preserving timing anchors.
    """
    words = get_words(segment)

    if not words:
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start + 1))
        text = clean_text(segment.get("text", ""))
        if not text:
            return []
        return [Subtitle(0, start, end, text, "")]

    chunks = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return

        chunks.append(
            Subtitle(
                index=0,
                start=current[0]["start"],
                end=current[-1]["end"],
                source=clean_text(" ".join(w["word"] for w in current)),
                text="",
            )
        )
        current = []

    for word in words:
        current.append(word)

        current_text = " ".join(w["word"] for w in current)
        duration = current[-1]["end"] - current[0]["start"]

        # Keep individual ASR units reasonably sized.
        if len(current_text) >= 80 or duration >= 7.0:
            flush()

    flush()

    return chunks

# remove spaces between two CJK characters  ja/zh/ko
def fix_cjk_spaces(text: str) -> str:
    return re.sub(r"(?<=[\u3000-\u9fff\uff00-\uffef])\s+(?=[\u3000-\u9fff\uff00-\uffef])", "", text)


def build_source_segments(aligned_result: dict) -> list[Subtitle]:
    segments = []

    for segment in aligned_result["segments"]:
        pieces = split_long_segment(segment)
        segments.extend(pieces)

    for segment in segments:
        segment.source = fix_cjk_spaces(segment.source)

    # Merge tiny adjacent pieces where sensible.
    merged = []
    for item in segments:
        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]
        gap = item.start - previous.end

        if (
            previous.end - previous.start < MIN_DURATION
            and gap < 0.7
            and len(previous.source) + len(item.source) + 1 <= 100
        ):
            previous.source = clean_text(previous.source + " " + item.source)
            previous.end = item.end
        else:
            merged.append(item)

    for i, item in enumerate(merged, 1):
        item.index = i

    return merged


# ---------------------------------------------------------------------------
# Local Ollama translation
# ---------------------------------------------------------------------------

TRANSLATION_SYSTEM = """You are a professional subtitle translator.

Translate the supplied transcript into natural Simplified Chinese.

Rules:
1. Translate meaning, not word-for-word structure.
2. Keep names, product names, model numbers, brands and technical terms accurate.
3. Do not add explanations.
4. Do not omit meaningful content.
5. Return ONLY the Chinese translation.
6. Do not use quotation marks around the answer unless they are part of the original meaning.
7. Keep the translation reasonably concise for subtitles.
"""


def ollama_translate(
    text: str,
    model: str,
    source_language: str,
    timeout: int = 180,
    _retry_truncated: bool = True,
) -> str:
    prompt = (
        f"Source language: {source_language}\n\n"
        f"Transcript:\n{text}\n\n"
        "Translate this into Simplified Chinese."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
        },
    }

    try:       ## How to use ollama library to translate the text is shown in the test.ipynb file
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:500]

        # Repetition-loop abort: retry once with only the first 10 characters by using recursive call.
        if "token repeat limit" in body and _retry_truncated:
            short_text = text[:10]
            print(f"\n  Repeat-limit error; retrying with first 10 chars: {short_text!r}")
            return ollama_translate(
                short_text,
                model,
                source_language,
                timeout=timeout,
                _retry_truncated=False,   # prevent infinite recursion
            )
        raise RuntimeError(
            f"Ollama HTTP {response.status_code}: {body}\n"
            f"Input text: {text!r}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc!r}") from exc

    data = response.json()
    translated = data.get("message", {}).get("content", "").strip()

    if not translated:
        raise RuntimeError(f"Ollama returned no translation: {data}")

    # Remove common model chatter if present.
    translated = re.sub(r"^\s*(translation|translated text)\s*:\s*", "", translated, flags=re.I)
    return clean_text(translated)


def translate_segments(
    segments: list[Subtitle],
    ollama_model: str,
    source_language: str,
) -> list[Subtitle]:
    print(f"\nTranslating {len(segments)} subtitle units with Ollama: {ollama_model}")

    for i, segment in enumerate(segments, 1):
        print(f"  Translating {i}/{len(segments)}", end="\r")

        try:
            segment.text = ollama_translate(
                segment.source,
                ollama_model,
                source_language,
            )
        except RuntimeError as exc:
            print(f"\n  Skipped #{i} ({segment.source[:10]!r}): {exc}")
            segment.text = ""

    print()
    return segments


# ---------------------------------------------------------------------------
# Intelligent Chinese subtitle line splitting
# ---------------------------------------------------------------------------

PUNCTUATION = "，。！？；：、,.!?;:"

def chinese_length(text: str) -> int:
    # Ignore spaces for Chinese readability.
    return len(re.sub(r"\s+", "", text))


def find_best_split(text: str) -> Optional[int]:
    """
    Find a visually natural split point close to the middle.
    Preference:
        punctuation > whitespace > nearest character boundary
    """
    text = text.strip()
    n = len(text)

    if n < 2:
        return None

    target = n / 2

    candidates = []

    for i, char in enumerate(text, 1):
        distance = abs(i - target)

        # Strong preference for Chinese punctuation.
        if char in PUNCTUATION:
            candidates.append((0, distance, i))
        elif char.isspace():
            candidates.append((1, distance, i))

    if candidates:
        candidates.sort()
        return candidates[0][2]

    # Chinese has no word spaces, so a character boundary is acceptable.
    return max(1, min(n - 1, round(target)))


def split_chinese_two_lines(
    text: str,
    max_line_chars: int = MAX_CHARS_PER_LINE,
) -> str:
    """
    Produce at most two lines.

    The function tries:
        1. Keep short subtitles on one line.
        2. Split around punctuation.
        3. Otherwise split near the visual midpoint.
        4. Avoid very unbalanced lines.
    """
    text = clean_text(text)

    if chinese_length(text) <= max_line_chars:
        return text

    split_at = find_best_split(text)

    if split_at is None:
        return text

    left = text[:split_at].strip()
    right = text[split_at:].strip()

    # If punctuation was used as the split point, keep it with line 1.
    if right and right[0] in PUNCTUATION:
        left += right[0]
        right = right[1:].strip()

    if not left or not right:
        return text

    # If one side is still far too long, use midpoint.
    if chinese_length(left) > max_line_chars * 1.25:
        midpoint = max_line_chars
        left = text[:midpoint].strip()
        right = text[midpoint:].strip()

    if chinese_length(right) > max_line_chars * 1.25:
        midpoint = len(text) - max_line_chars
        left = text[:midpoint].strip()
        right = text[midpoint:].strip()

    return f"{left}\n{right}"


def format_subtitles(segments: list[Subtitle]) -> list[Subtitle]:
    """
    Add line breaks and adjust very short subtitle durations.
    """
    output = []

    for segment in segments:
        text = clean_text(segment.text)
        if not text:
            continue

        duration = segment.end - segment.start

        # Minimum readable duration.
        if duration < MIN_DURATION:
            new_end = segment.start + MIN_DURATION

            # Don't overlap the next subtitle; final adjustment happens below.
            segment.end = new_end

        # Avoid excessively long display time.
        if segment.end - segment.start > MAX_DURATION:
            segment.end = segment.start + MAX_DURATION

        segment.text = split_chinese_two_lines(text)
        output.append(segment)

    # Prevent overlaps.
    for i in range(len(output) - 1):
        if output[i].end > output[i + 1].start - MIN_GAP:
            output[i].end = max(
                output[i].start + 0.2,
                output[i + 1].start - MIN_GAP,
            )

    # Re-index.
    for i, item in enumerate(output, 1):
        item.index = i

    return output


# ---------------------------------------------------------------------------
# SRT writer
# ---------------------------------------------------------------------------

def write_srt(segments: list[Subtitle], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig") as f:
        for segment in segments:
            f.write(f"{segment.index}\n")
            f.write(
                f"{srt_time(segment.start)} --> "
                f"{srt_time(segment.end)}\n"
            )
            f.write(segment.text.strip() + "\n\n")


def write_debug_json(
    segments: list[Subtitle],
    output_path: Path,
) -> None:
    data = [asdict(s) for s in segments]

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    source_language: Optional[str],
    whisper_model: str,
    ollama_model: str,
    batch_size: int,
    compute_type: str,
):
    video = Path(video_path)

    if not video.exists():
        raise FileNotFoundError(f"Input video not found: {video}")

    if video.suffix.lower() not in {".mp4", ".mkv", ".mov", ".avi", ".m4v"}:
        print("Warning: this file extension is not a common video format.")

    check_ffmpeg()

    output_dir = video.parent / f"{video.stem}_subtitles"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WhisperX Subtitle Agent")
    print("=" * 70)
    print(f"Input : {video}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    start_time = time.time()

    # Stage 1: ASR + alignment
    aligned_result, detected_language = transcribe_and_align(
        str(video),
        source_language=source_language,
        model_name=whisper_model,
        batch_size=batch_size,
        compute_type=compute_type,
    )

    source_segments = build_source_segments(aligned_result)

    # Save raw aligned transcript before translation.
    raw_json = output_dir / f"{video.stem}_aligned.json"
    write_debug_json(source_segments, raw_json)

    # Stage 2: Translation
    translated = translate_segments(
        source_segments,
        ollama_model=ollama_model,
        source_language=detected_language,
    )

    # Stage 3: Subtitle formatting
    final_segments = format_subtitles(translated)

    srt_path = output_dir / f"{video.stem}_zh-CN.srt"
    write_srt(final_segments, srt_path)

    # Also save a human-readable TXT.
    txt_path = output_dir / f"{video.stem}_zh-CN.txt"
    with txt_path.open("w", encoding="utf-8-sig") as f:
        for segment in final_segments:
            f.write(segment.text.replace("\n", " ") + "\n")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"SRT       : {srt_path}")
    print(f"Aligned   : {raw_json}")
    print(f"Text      : {txt_path}")
    print(f"Subtitles : {len(final_segments)}")
    print(f"Time      : {elapsed / 60:.1f} minutes")
    print("=" * 70)

    return srt_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local MP4 -> WhisperX -> Chinese SRT subtitle agent"
    )

    parser.add_argument(
        "video",
        help="Path to input MP4/video",
    )

    parser.add_argument(
        "--source-language",
        default=None,
        help="Source language code, e.g. en, ja, ko. "
             "If omitted, WhisperX detects it.",
    )

    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_MODEL,
        help="Whisper model. Default: %(default)s",
    )

    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama translation model. Default: {DEFAULT_OLLAMA_MODEL}",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="WhisperX batch size. Use 2 for an 8 GB GPU.",
    )

    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        choices=["float16", "int8_float16", "int8", "float32"],
        help="GPU precision. float16 is preferred; int8_float16 uses less VRAM.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        process_video(
            video_path=args.video,
            source_language=args.source_language,
            whisper_model=args.whisper_model,
            ollama_model=args.ollama_model,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
        )
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(130)
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        sys.exit(1)
