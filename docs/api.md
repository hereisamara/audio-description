# Studio API and container contracts

In the default Compose stack, clients reach `studio` at
`http://127.0.0.1:7860` from the host or `http://studio:7860` from
another container on the Compose network. Studio accepts these host
names and checks the `Origin` header on POST requests when present.
The API is intended for local use and has no authentication. If you set
`STUDIO_PORT` in Compose, use that port in the host URL.

| Compose service | Interface | How to use it |
| --- | --- | --- |
| `studio` | HTTP with JSON, MP4 uploads, and file downloads | Upload media, submit jobs, read dialogue, and stream results. |
| `whisper` | SQLite job queue and files on `/data` | Submit through Studio. Read transcript and gaps through Studio, or implement the Whisper contract when replacing the worker. |
| `ad-maker` | SQLite job queue and files on `/data` | The Whisper result queues this stage. Read the result manifest through Studio, or implement the AD Maker contract when replacing the worker. |

## Submit a job

Run these examples with a local `video.mp4` containing an audio stream,
plus `curl` and `jq` on PATH.

```bash
base=http://127.0.0.1:7860
curl -fsS "$base/api/capabilities" | jq '{worker, profile, workers}'
```

Check `worker: true` before submission. `GET /api/capabilities` returns
a JSON object. In Compose, `split_workers` is `true`, `worker` (boolean)
indicates that both worker heartbeats are current and FFmpeg is available,
`cuda` (boolean) indicates at least one CUDA worker, and `profile` is a
string. `workers` maps the active roles `whisper` and `ad` to objects
with `host` (string) and `cuda` (boolean).

### Upload

`POST /api/media` accepts raw MP4 bytes with `Content-Type: video/mp4`
and a content length from 1 byte to 512 MiB. The video must contain an
audio stream. Use the resulting ID for operations on the uploaded media.

```bash
upload=$(curl -fsS -H 'Content-Type: video/mp4' \
  --data-binary @video.mp4 "$base/api/media")
media_id=$(printf '%s\n' "$upload" | jq -er '.id')
printf '%s\n' "$upload" | jq .
```

HTTP 200 returns an object with:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string, 32 lowercase hexadecimal characters | Media ID. A job has its own ID. |
| `tracks` | array of objects | Audio tracks with `index` (zero-based integer), `label` (string), `language` (string or null), and `ad` (boolean). |

To reuse video already on the server, send JSON such as
`{"url":"/result/<job_id>/original.mp4"}` to the same route. Use a
server-relative route returned by this Studio instance. A finished
job's `video-ad.mp4` can also be registered this way.

### Queue generation

`POST /api/media/{media_id}/generate` takes this JSON:

| Field | Type | Default |
| --- | --- | --- |
| `device` | `"auto"`, `"cpu"`, or `"cuda"` | `"auto"` |
| `fallback` | boolean | `false`. With `"cuda"`, set `true` to allow a CPU-only worker to process its stage. |
| `title` | string | Source filename. Studio trims the title and keeps up to 140 characters. |

```bash
job=$(curl -fsS -H 'Content-Type: application/json' \
  --data '{"device":"auto","title":"API walkthrough"}' \
  "$base/api/media/$media_id/generate")
job_id=$(printf '%s\n' "$job" | jq -er '.id')
```

HTTP 202 returns a job object. Poll `GET /api/jobs/{job_id}` to read
its current state. `GET /api/jobs` returns `{"jobs":[...]}` with up to
50 jobs, newest first.

| Job field | Type | Meaning |
| --- | --- | --- |
| `id`, `title`, `host`, `profile` | string | Job ID, display title, Studio host, and model profile. |
| `stage` | `"whisper"`, `"ad"`, or `"complete"` | Current processing stage. |
| `state` | `"queued"`, `"running"`, `"finished"`, or `"failed"` | A job can return to queued when Whisper hands it off to AD Maker. |
| `requested` | `"auto"`, `"cpu"`, or `"cuda"` | Requested device. |
| `device` | `"pending"`, `"cpu"`, or `"cuda"` | Selected AD device when assigned, otherwise the Whisper device or pending. |
| `devices` | object | Role keys become available on claim. Each value has `device` and `host` strings. |
| `fallback`, `dialogue_ready` | boolean | A CUDA request is using CPU for at least one stage, and the Whisper result is ready, respectively. |
| `error` | string or null | Failure reason when available. |
| `original` | file object | Source video route. A file object contains `name` and `url` strings. |
| `files`, `log` | optional file object array, optional string | The finished job's artifact manifest and the recent log excerpt. |

```bash
curl -fsS "$base/api/jobs/$job_id" | jq '{stage, state, devices, dialogue_ready, error, files}'
```

## Read dialogue and results

### Whisper dialogue

`GET /api/jobs/{job_id}/dialogue` returns HTTP 200 once Whisper writes the
record. HTTP 409 means the dialogue is still unavailable. Timestamps are
seconds relative to the source video.

| Field | Type |
| --- | --- |
| `schema_version` | integer, currently 1 |
| `media` | object with `duration` and `fps` (numbers), `width` and `height` (integers), and `has_audio` (boolean) |
| `transcript` | object with `report` (diagnostic object) and arrays `segments`, `words`, `raw_segments`, and `raw_words`. Each item has `start` and `end` (numbers) and `text` (string). |
| `dialogue_validation` | diagnostic object |
| `speech_and_gaps` | object with `speech` (interval array), `gaps` (interval array with integer `word_budget`), and `gap_audit` (diagnostic object array). Intervals use numeric `start` and `end`. |

```bash
curl -fsS "$base/api/jobs/$job_id/dialogue" | jq '.speech_and_gaps.gaps'
```

### AD Maker artifacts

After `state: "finished"`, follow the URLs in `files`:

| Artifact | Data |
| --- | --- |
| `video-ad.mp4` | The rendered MP4, with an AD audio track when cues were placed. |
| `descriptions.vtt` | Timed AD captions in WebVTT format. |
| `narration-plan.json` | Versioned JSON with `schema_version` (integer, currently 1), `language` (string, currently `"en"`), and `cues` (array). |
| Review files such as `ad-proposals.json` and `runtime-summary.json` | Saved visual evidence, prompts, decisions, and per-stage timings. Read the manifest to see which files are available. |

Each narration cue has `start` and `end` (numbers in source-video
seconds), `text` and `audio` (strings), `overlaps_dialogue` (boolean),
and `dialogue_overlap_seconds` (number). Resolve `audio` relative to
`/result/{job_id}/`. A job with zero placed cues can return `"cues": []`.

```bash
curl -fsS "$base/result/$job_id/narration-plan.json" | jq '.cues'
curl -fsS -D - -o /dev/null -H 'Range: bytes=0-63' \
  "$base/result/$job_id/video-ad.mp4"
```

File requests return 200 for the full file, 206 with `Content-Range` for
a valid single byte range, or 416 for an invalid range. Missing jobs or
files return 404. Invalid POST bodies, unavailable workers, and rejected
device requests can return 400. A rejected POST origin returns 403.
JSON errors use `{"error":"message"}`.
Some route and host errors use an HTML response, so check the status and
content type before parsing JSON.

## Optional review endpoints

These endpoints accept JSON and return HTTP 200 with a JSON object on
success. A cue row is `{"start":3.2,"end":5.6,"text":"A frog rests on a leaf."}`
with numeric seconds and string text.

| Route | Request | Response |
| --- | --- | --- |
| `POST /api/media/{media_id}/render` | `audio` (audio track index), optional `captions` object with cue arrays under `ad`, `adThai`, and `dialogue`. | `{"url":"..."}` for the exported MP4. |
| `POST /api/media/{media_id}/narration` | `audio` (original soundtrack index), `language` (`"en"` or `"th"`), and `cues` (array). Requires an available local voice reported by `voice` or `thai_voice` in capabilities. | `{"url":"...","voice":"..."}` for the generated audio mix. |
| `POST /api/translate-ad` | `cues` (1 to 100 English AD rows). Requires `thai_translation: true` in capabilities. | `{"cues":[...]}` with translated cue text and matching timings. |

For rendering, `revoice: true` can synthesize the English cues over the
original audio. When a mix is already generated for the same media ID,
pass `narration_language` and optionally `alternate_narration_language`
(`"en"` or `"th"`) to export one or both mixes.

## Work directly with a worker container

The workers claim jobs through [JobStore](../offline_ad_poc/job_store.py)
at `/data/studio-jobs.sqlite3` and exchange files through the same writable
`/data` mount. Worker roles
are `whisper` and `ad` in the store, with `ad-maker` as the AD Compose
service name.

| Operation | Input types | Output |
| --- | --- | --- |
| `report_worker(role, host, profile, cuda, preparer)` | Three strings and two booleans | Updates the role heartbeat and capability record. |
| `claim(role, cuda, host, profiles=None)` | Role string, CUDA boolean, host string, optional map from `cpu` and `cuda` to profile names for the AD worker | An attempt object or `None`. The object includes `id`, `media_id`, `attempt_dir`, `selected_device`, and `lease_token` as strings. |
| `source_path(media_id)` | Media ID string | Source `Path` or `None`. |
| `dialogue(job_id)` | Job ID string | `(work_dir: Path, payload: dict)` or `None`. The analysis WAV is in that work directory. |
| `renew(claim)`, `finish_dialogue(claim, payload)`, `finish_ad(claim)` | Claim object. Whisper also supplies the version 1 dialogue object. | Boolean. A false result means another claim owns the attempt or it is no longer active. |
| `fail(claim, error)` | Claim object and error string | Returns `None` and records the failure for the active lease. |

Write files in the assigned attempt directory and renew the lease while
processing. Whisper writes `work/analysis-audio.wav` in 16 kHz mono
signed 16-bit PCM. It finishes by publishing the JSON payload and
queuing AD in one transaction. AD Maker reads the very same media
and analysis audio, writes the review files, then calls `finish_ad`
to publish the result directory.

For sample payloads, standalone worker commands, and replacement
container instructions, see the [worker integration guide](worker-integration.md).
