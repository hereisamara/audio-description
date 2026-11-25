# Audio Description Studio

Generate, inspect, and edit audio descriptions for video on your own machine.
The Studio runs locally with Docker Compose. Import an MP4, start a job, and
review the suggested narration alongside the original dialogue. Export cues
as WebVTT or save a reviewed MP4.

![Studio review with an Apollo 11 video and the cue editor](docs/images/studio-review.jpg)

## Quick start

Install [Docker with Compose](https://docs.docker.com/compose/install/) and the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) for
the initial checkpoint download.

```bash
git clone https://github.com/hereisamara/audio-description.git
cd audio-description

hf download openai/whisper-tiny.en --local-dir models/whisper-tiny.en
hf download facebook/mms-tts-eng --local-dir models/mms-tts-eng-local
hf download Qwen/Qwen3-VL-2B-Instruct-GGUF \
  Qwen3VL-2B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
  --local-dir models/qwen3-vl-2b-instruct-gguf

docker compose up --build -d
```

Open [Studio](http://127.0.0.1:7860). Choose **Import video**, pick an MP4
with an audio stream, and select **Start job**. Keep **Auto** for the default
profile. Choose **Inspect dialogue and gaps** as soon as Whisper finishes.
Open completed jobs under **Recent jobs** and use the right panel to edit cues
or export the result. Use **Import run** to review results from another host.

The initial build and downloads need internet access. Inference uses the
staged models offline. Compose stores uploads, job state, and results in
the local `job-data` volume. `docker compose stop` stops the services and
keeps the volume.

## How the containers interact

The browser talks to `studio` over HTTP. All three containers mount
`job-data` at `/data`. SQLite in that volume holds the job queue and
dialogue records. The original video, analysis WAV, and finished artifacts
are files on the same volume. Both workers load checkpoints from the
read-only `/models` mount.

```mermaid
sequenceDiagram
    actor Browser
    participant Studio as studio
    participant DB as SQLite
    participant Files as Files on job-data
    participant Whisper as whisper
    participant AD as ad-maker

    Browser->>Studio: POST /api/media with MP4 bytes
    Studio->>Files: Save source.mp4
    Studio->>DB: Register media_id and source path
    Studio-->>Browser: Media ID and audio track metadata

    Browser->>Studio: POST /api/media/{media_id}/generate with device preference
    Studio->>DB: Enqueue the Whisper stage with media_id and device preference
    Studio-->>Browser: HTTP 202 with job_id and queued state

    Whisper->>DB: Claim a Whisper job and look up its source
    DB-->>Whisper: Job ID, source path, and selected device
    Files-->>Whisper: Original MP4
    Whisper->>Files: Save 16 kHz mono analysis-audio.wav
    Whisper->>DB: Save timestamped dialogue and gaps, then queue AD

    AD->>DB: Claim an AD job and load its Whisper result
    DB-->>AD: Source path, analysis WAV path, dialogue, gaps, and device
    Files-->>AD: Original MP4 and analysis WAV
    AD->>Files: Save described MP4, WebVTT, narration plan, and cue WAVs
    AD->>DB: Mark the job finished and record the result directory

    Browser->>Studio: GET /api/jobs/{job_id}
    Studio->>DB: Read job state and result directory
    Studio-->>Browser: Stage and worker devices, plus file URLs once finished
    Browser->>Studio: GET /result/{job_id}/... with an optional Range header
    Studio->>Files: Read the requested file or byte range
    Studio-->>Browser: File bytes, with HTTP 206 for a requested range
```

- **Studio** saves the uploaded MP4 once. The browser polls the job endpoint
  during processing and can request the transcript and gaps at
  `GET /api/jobs/{job_id}/dialogue` as soon as Whisper publishes them.
- **Whisper** extracts the analysis audio and transcribes it. Saving the
  dialogue and queuing AD happens in one transaction, so AD Maker picks up
  the job with its input ready.
- **AD Maker** combines the original frames with the saved dialogue and gaps.
  It plans cues, runs the available cue editor and MMS-TTS, then mixes the
  narration into a selectable track. It publishes the result directory
  after writing the files. Studio serves MP4s and cue WAVs by byte range
  when the browser requests partial content.

Click a VLM marker to inspect the saved prompt, selected frames, and reply.
The timeline also shows shots, scenes, dialogue, and placed cues. Review the
draft cues before publishing, especially around missed or overlapping dialogue.

![Studio evidence panel with a selected VLM frame, prompt, and raw response](docs/images/studio-evidence.jpg)

The screenshots use illustrative cue and timing data with public-domain
Apollo 11 footage digitized by Kipp Teague. [NASA Apollo 11 Video Library](https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/video11.html)

## Optional models

**Cue editor.** Stage Qwen3-1.7B to have AD Maker rewrite drafts into short,
grounded narration using the dialogue and available time. Compose detects
the checkpoint automatically.

```bash
hf download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q4_K_M.gguf --local-dir models/cpu-llm
docker compose up -d --force-recreate ad-maker
```

**NVIDIA CUDA.** On an x86-64 Linux host with [NVIDIA GPU support for Docker](https://docs.docker.com/compose/how-tos/gpu-support/),
Auto can use Whisper Small English and Qwen3-VL-8B-Instruct. The default
Compose file uses the Qwen3-VL-2B CPU profile.

```bash
hf download openai/whisper-small.en --local-dir models/whisper-small.en-transformers
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
hf download facebook/mms-tts-eng --local-dir models/mms-tts-eng

docker compose -f compose.yaml -f compose.cuda.yaml up --build -d
```

For endpoints, JSON types, and request examples, see the [API reference](docs/api.md).
For standalone commands and replacing a worker, see the
[worker integration guide](docs/worker-integration.md).

The [Lanta POC branch](https://github.com/hereisamara/audio-description/tree/lanta-poc)
contains the Slurm launchers and research benchmarks.
