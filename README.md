# WhisperX Subtitle Agent — RTX 3070 Ti 8 GB

This project creates Simplified Chinese subtitles from a local video.

Pipeline:

MP4
  -> WhisperX / Faster-Whisper ASR
  -> WhisperX forced word alignment
  -> local Ollama translation
  -> intelligent Chinese 2-line formatting
  -> UTF-8 SRT

## 1. Create a clean Conda environment

Open Anaconda PowerShell:

```powershell
conda create -n whisperx python=3.11 -y
conda activate whisperx
```

## 2. Install FFmpeg

Make sure this works:

```powershell
ffmpeg -version
```

If it does not, install FFmpeg and add its `bin` directory to Windows PATH.

## 3. Install WhisperX

Current WhisperX documentation recommends:

```powershell
pip install whisperx
```

For GPU acceleration, use the CUDA setup required by the current WhisperX release.

Then install the remaining requirements:

```powershell
pip install -r requirements.txt
```

## 4. Check the NVIDIA GPU

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

You want something similar to:

```text
True
NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

## 5. Install Ollama

Make sure Ollama is running.

Then choose a local translation model. For example:

```powershell
ollama pull qwen3:8b
```

Check:

```powershell
ollama list
```

If your installed model has another name, use:

```powershell
python subtitle_agent.py "C:\Videos\test.mp4" --ollama-model YOUR_MODEL
```

## 6. Run

For an English video:

```powershell
python subtitle_agent.py "C:\Videos\test.mp4" --source-language en
```

For Japanese:

```powershell
python subtitle_agent.py "C:\Videos\test.mp4" --source-language ja
```

If you omit `--source-language`, WhisperX will detect it:

```powershell
python subtitle_agent.py "C:\Videos\test.mp4"
```

## 7. Output

For:

```text
C:\Videos\test.mp4
```

the agent creates:

```text
C:\Videos\test_subtitles\
    test_zh-CN.srt
    test_zh-CN.txt
    test_aligned.json
```

The SRT looks like:

```text
1
00:00:00,520 --> 00:00:03,100
欢迎来到我们的展厅。

2
00:00:03,100 --> 00:00:06,500
今天我们来看一下
最新的电磁炉产品。
```

## RTX 3070 Ti 8 GB settings

Start with:

```powershell
python subtitle_agent.py "C:\Videos\test.mp4" --batch-size 2 --compute-type float16
```

If CUDA out-of-memory occurs:

```powershell
python subtitle_agent.py "C:\Videos\test.mp4" --batch-size 1 --compute-type int8_float16
```

Do not start with a large batch size on an 8 GB GPU.

## Important architecture detail

WhisperX currently uses Faster-Whisper as its ASR backend. Therefore you do NOT need to separately run a Faster-Whisper transcription and then feed that transcription into WhisperX.

The practical pipeline is:

WhisperX
  -> Faster-Whisper ASR
  -> forced alignment
  -> word timestamps

This avoids doing the same transcription twice.

## Translation

The default translator is Ollama because this keeps the video/transcript local.

The translation model is configurable:

```powershell
python subtitle_agent.py "test.mp4" --ollama-model qwen3:8b
```

If your Ollama model is different, substitute its exact name.

## Why translation happens after alignment

The timestamps belong to the original speech. Translation should not modify them.

Example:

Original:

```text
This induction cooktop gives you a lot more power.
```

Timing:

```text
12.43 -> 16.82
```

Chinese:

```text
这款电磁炉可以提供更强劲的火力。
```

The Chinese subtitle keeps:

```text
12.43 -> 16.82
```

This makes the subtitle timing stable.

## Future extensions

This architecture can later add:

- Gradio UI
- microphone/audio input
- speaker diarization
- bilingual SRT
- English SRT
- Chinese/English dual subtitles
- subtitle preview
- FFmpeg hard-subtitle rendering
- automatic video export
- subtitle editing
- product-name glossary for e&s appliance terminology
