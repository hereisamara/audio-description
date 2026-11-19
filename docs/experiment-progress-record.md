# Audio Description POC: Experiment and Progress Record

Last updated: 2026-11-19

This document records what was attempted, why it was attempted, what evidence
was produced, how it helped, why the preceding approach was insufficient, and
what decision follows. It deliberately separates completed experiments from
planned work.

Status labels:

- **Completed**: implemented and exercised successfully.
- **Partially validated**: implemented, but the available results are not yet
  sufficient for a final choice.
- **Proposed**: designed or discussed, but not yet implemented and tested.

## Quick reconnect: minimal progress list

1. **Completed — Offline real-video POC:** built a Conda-only, no-API pipeline
   that accepts a real video and exports an MP4 with an Audio Description track.
2. **Completed — Repeatable Lanta environment:** made environment creation
   update an existing Conda environment instead of failing when its prefix
   already exists.
3. **Completed — CUDA execution:** pinned the CUDA PyTorch variant and added a
   job preflight after CUDA was unavailable on a non-GPU/login context.
4. **Completed — First full semantic run:** Whisper + shot/scene analysis +
   Qwen2.5-VL + MMS-TTS completed on a 178.584-second real scene.
5. **Completed — Method and runtime logging:** every stage now records method,
   fallback, CPU/GPU use, time, memory, and counts.
6. **Partially validated — VLM resolution study:** isolated one-frame tests
   exposed cold-start and output-validity problems; the benchmark was changed
   to warm up and use repeated median measurements.
7. **Partially validated — Whisper gap correction:** invalid/stretched word
   timestamps are filtered, but the baseline still has measurable false
   positive and missed-dialogue time.
8. **Completed — Shot-boundary evidence:** 14 shots and 13 selected cuts were
   checked with before/after frames; 24 lower-confidence candidates were saved.
9. **Problem confirmed — Scene segmentation:** all 14 shots were merged into
   one scene, so the current scene result is too coarse to guide AD placement.
10. **Partially validated — Four-model ASR comparison:** the sequential offline
    benchmark exists; the recorded Whisper-small smoke result is available, but
    a complete comparable four-model result has not yet been recorded here.
11. **Completed — Dialogue review video:** original and predicted dialogue
    intervals can be rendered on the video for human inspection.
12. **Partially validated — Timestamped visual knowledge database:** SQLite
    indexing and gap retrieval are implemented; the next Lanta run must verify
    the resulting evidence and overhead.
13. **Partially validated — Event-centered AD placement:** event grounding and
    scheduling are implemented and unit tested; semantic quality still requires
    a Qwen/Lanta run.

## Current conclusion

The complete offline media path works. The largest verified runtime cost is
ASR. The largest quality problems are now upstream timing and context problems,
not video export: dialogue intervals are imperfect, gradual transitions may
remain soft candidates, global scene grouping is too coarse, and the VLM does
not return the timestamp of the event it describes. The next full-pipeline
improvement should therefore focus on timestamped evidence and event-aware
placement before increasing model size or image resolution.

## Detailed record

### 1. Offline real-video proof of concept

**Status:** Completed

**What was tried**

- A real input video with its original audio, rather than synthetic test media.
- A fully offline pipeline using local checkpoints and no external API calls.
- CPU and CUDA execution selected through a command-line device option.
- FFmpeg/FFprobe for media handling, Transformers Whisper for ASR, Qwen2.5-VL
  for visual AD decisions, Transformers MMS-TTS for narration, and FFmpeg for
  the final mix.
- Generated media and multi-gigabyte checkpoints were excluded from Git and
  staged directly on Lanta.

**Why**

The POC needed to remain close to the eventual use case and be reproducible on
Lanta without depending on internet services.

**How it helped**

It established that the entire path from real video to a selectable AD audio
track could run as one offline batch job.

**Why this did not solve quality by itself**

End-to-end completion verifies integration, not the accuracy of ASR gaps,
scene boundaries, descriptions, or narration placement. Those components
needed isolated evaluation.

**Evidence**

- Initial implementation: Git commit `c505159`.
- CPU/GPU Conda support: Git commit `2f74598`.
- Lanta Slurm submission: Git commit `756da63`.

### 2. Repeatable Conda environment on Lanta

**Status:** Completed

**What was tried**

The first environment command failed in two ways:

1. `conda` was unavailable before loading `Mamba/23.11.0-0`.
2. `conda env create` failed because `offline-ad-gpu` already existed.

The setup script was changed to create a missing environment or update an
existing environment with pruning.

**Why**

Cluster jobs and collaborators need a setup command that can be safely rerun
after pulling dependency changes.

**How it helped**

The same command now converges an existing environment instead of requiring
manual removal or a second undocumented update command.

**Why the previous approach failed**

`conda env create` is only an initial-creation operation and treats an existing
prefix as an error.

**Evidence**

- Repeatable setup: Git commit `aa95edc`.
- Current script: `scripts/create_conda_env.sh`.

### 3. CUDA availability and GPU job preflight

**Status:** Completed

**What was tried**

The environment initially reported PyTorch `2.5.1` with
`torch.cuda.is_available() == False`, and a pipeline run requested `--device
cuda` and failed. The GPU environment was pinned to `pytorch-cuda=12.1`, and
the Slurm script was extended to record the allocation, visible devices,
PyTorch build, CUDA build, availability, and device count before running the
pipeline.

**Why**

CUDA can be unavailable because the command is running outside a GPU
allocation, because Slurm did not expose a device, or because a CPU PyTorch
variant was installed. These cases need to be distinguished.

**How it helped**

The successful job reported one visible NVIDIA A100-SXM4-40GB device and
`CUDA available: True`. Failures now occur before model loading with enough
information to diagnose the environment.

**Why the previous approach failed**

Loading the Conda module does not allocate a GPU, and an environment named
"gpu" does not guarantee that its PyTorch build includes CUDA support.

**Evidence**

- CUDA dependency and preflight: Git commit `ce60e30`.
- Job entry point: `scripts/submit_lanta_gpu.sh`.

### 4. Qwen image-processing compatibility

**Status:** Completed

**What was tried**

Torchvision was first added for Qwen visual processing, then removed in favor
of the Transformers PIL/image-processor path.

**Why**

The visual backend had to load local Qwen2.5-VL frames without introducing an
unnecessary compiled dependency that could conflict with the cluster PyTorch
build.

**How it helped**

The current Qwen path uses locally opened PIL images and the model's
`AutoImageProcessor`, reducing environment coupling.

**Why the previous approach was insufficient**

Adding Torchvision treated a preprocessing symptom as another environment
dependency. The model could be driven directly through the Transformers image
processor instead.

**Evidence**

- Torchvision attempt: Git commit `d853554`.
- Image-only processing: Git commit `25aed37`.

### 5. First complete semantic pipeline result

**Status:** Completed, with quality problems identified

**What was tried**

The first saved complete run used:

- input duration: 178.584 seconds;
- Whisper Small English Transformers checkpoint;
- OpenCV frame-difference shot detector fallback;
- lightweight multimodal scene grouping;
- Qwen2.5-VL-3B-Instruct;
- MMS-TTS English;
- CUDA for model stages and CPU for media/feature stages.

**Result**

- 14 shots;
- 1 scene;
- 1 candidate speech gap;
- 1 processed gap;
- 1 accepted AD event;
- no measured dialogue overlap;
- TTS fit rate of 1.0;
- 137.617 seconds total processing time in the first saved report;
- real-time factor 0.7706.

The generated description was:

> A woman walks into a room with a chandelier and a desk.

The candidate gap was 25.750–38.750 seconds. The 4.096-second narration was
placed at 30.202–34.298 seconds.

**How it helped**

This run proved that all local models and media stages could complete together
and produced concrete artifacts for component-level diagnosis.

**Why the result was not yet acceptable**

- A single candidate gap meant most of the film had no opportunity for AD.
- One scene containing all 14 shots provided almost no narrative segmentation.
- The selected evidence frames mixed title/transition material and the later
  action.
- The narration was centered in the complete gap rather than attached to the
  timestamp of the described action.
- One plausible sentence is not evidence of consistent factual grounding.

**Evidence**

- Saved report: `lanta-results/real-scene-01-work/evaluation-report.json`
  in the parent project results directory.
- Decision: `lanta-results/real-scene-01-work/ad-decisions.json` in the parent
  project results directory.

### 6. AD excerpt and placement investigation

**Status:** Problem confirmed

**What was tried**

The single generated AD event was isolated and rendered with voice-over and
captions for review. Its four evidence frames were at approximately 17.75,
24.73, 31.72, and 38.70 seconds.

**Why**

Reviewing only the full three-minute output made it difficult to see exactly
which visual evidence caused the sentence and whether the narration was placed
near the relevant action.

**How it helped**

The excerpt connected one model decision to its gap, evidence frames, TTS
duration, and final cue interval.

**Why the current placement approach is insufficient**

The VLM returns `AD` or `WAIT`, a description, a reason, and importance, but no
event timestamp. The TTS scheduler therefore has no event location and centers
the synthesized audio in the full speech gap. In a long gap, this can place a
correct sentence too early or too late.

**Decision**

The next placement algorithm should localize the described event first and
select an audio-safe slot close to that event. It should not treat the center
of the gap as the default ground truth.

### 7. Method, fallback, timing, and device profiling

**Status:** Completed

**What was tried**

Logging was added incrementally for:

- selected method and fallback reason;
- per-stage elapsed and CPU time;
- average CPU cores;
- process peak resident memory;
- CPU/GPU classification;
- synchronized PyTorch CUDA peak allocation and reservation;
- JSONL events that survive a later pipeline failure;
- a final structured runtime summary.

**Why**

Before optimizing resolution or changing models, the actual bottlenecks and
fallback methods needed to be observable.

**Profiled result (134.552-second run)**

| Stage | Time | Device | Method | Observed peak GPU allocation |
|---|---:|---|---|---:|
| ASR | 88.797 s | CUDA | Transformers Whisper | 2020.539 MB |
| Qwen initialization | 13.407 s | CUDA | Qwen2.5-VL | 7236.447 MB |
| Qwen AD decision | 6.975 s | CUDA | Qwen2.5-VL | 7527.621 MB |
| TTS | 8.830 s | CUDA | Transformers | 7406.967 MB |
| Video mixing | 4.049 s | CPU | FFmpeg filter graph | Not applicable |
| Shot detection | 2.073 s | CPU | OpenCV frame difference | Not applicable |
| Scene segmentation | 0.001 s | CPU | Weighted visual/audio/text distance | Not applicable |

ASR consumed approximately 66% of the total measured run, making it the first
runtime optimization target.

**How it helped**

It separated model initialization from inference, made the OpenCV shot fallback
explicit, and prevented optimization effort from being guided by assumption.

**Why the previous logs were insufficient**

A final total time could not identify which model or CPU stage caused delay,
and `nvidia-smi` snapshots did not attribute memory or time to pipeline stages.

**Evidence**

- Method/fallback logging: `df0cd30`.
- Stage timing: `9d41f58`.
- CPU/GPU resources: `543f112`.
- Profiling documentation: `881a882`.

### 8. VLM input-resolution experiments

**Status:** Partially validated

**What was tried**

Qwen was isolated from Whisper, TTS, and video rendering and tested on one
fixed frame at 31.72 seconds with long edges of 160, 320, 480, and 640 pixels.

The first recorded sweep produced:

| Long edge | Actual image | Time | Vision tokens | Incremental GPU peak | Output validity |
|---:|---:|---:|---:|---:|---|
| 160 | 160x120 | 16.008 s | 24 | 31.7 MB | `AD`, empty description |
| 320 | 320x240 | 1.942 s | 99 | 34.7 MB | `AD`, empty description |
| 480 | 480x360 | 2.446 s | 221 | 50.6 MB | Valid description |
| 640 | 640x480 | 1.696 s | 391 | 81.8 MB | `AD`, empty description |

At 480 pixels the model described the Educational Pictures title/logo. The
other three outputs were invalid because `AD` had no description.

**Why**

The goal was to find the smallest useful evidence resolution without repeatedly
running the entire pipeline or overloading the GPU.

**How it helped**

It showed that visual-token and incremental-memory cost increase with
resolution, while output quality was not monotonically related to resolution.
It also exposed that the 160-pixel timing contained first-use overhead.

**Why the first comparison could not select a resolution**

- Each resolution had only one measured output.
- The first run included cold-start/input-shape overhead.
- Three runs returned structurally invalid model output.
- One frame cannot measure action understanding across a scene.

**Correction made**

The benchmark now performs an excluded warm-up, uses three sequential measured
inferences per resolution, reports the median, and explicitly flags empty `AD`
descriptions and inconsistent repeated output.

**Remaining work**

Run the stabilized benchmark and then test the smallest reliable resolutions
on multiple event types, not only the title frame.

**Evidence**

- Configurable evidence resolution: `c06fa40`.
- Vision-token logging: `6c17ddd`.
- One-frame benchmark: `c544be8`.
- Stabilized measurements: `b2e4736`.

### 9. Whisper timestamp validation and dialogue-gap quality

**Status:** Partially validated

**Problem observed**

Whisper sometimes assigned words to intervals that were nearly zero length or
implausibly long. Treating every raw word timestamp as speech allowed one bad
word span to cover a real silence and remove a potential AD gap.

**What was tried**

- Preserve the unmodified ASR segments and words for auditing.
- Reject invalid or excessively stretched word intervals before generating
  speech masks.
- Rebuild segments from validated words.
- Record rejection counts and thresholds.
- Keep a `--dialogue-filter none` path to reproduce the legacy behavior.

**Why**

The AD opportunity detector depends on temporal speech coverage, not only the
transcript text. A readable transcript can still have unusable time spans.

**How it helped**

One validated full-pipeline run reduced ASR wall time from the earlier 88.797
seconds to 61.690 seconds, and the pipeline now makes the filtering step
explicit. More importantly, invalid word spans no longer silently become
dialogue intervals.

**Recorded Whisper-small smoke comparison**

- overlap with reference dialogue: 70.794 seconds;
- false-positive speech: 13.260 seconds;
- missed dialogue: 16.630 seconds;
- intro false-positive speech: 0 seconds;
- temporal precision: 0.842244;
- temporal recall: 0.809778;
- temporal F1: 0.825692.

**Why this is not a complete fix**

Timestamp filtering removes structurally implausible spans; it cannot correct
plausible-looking hallucinated speech or recover words that the model missed.
The remaining false-positive and missed-dialogue durations can still create or
remove AD gaps incorrectly.

**Evidence**

- Timestamp validation: Git commit `79824e8`.
- Dialogue reference: `references/real-scene-01-dialogue.json`.

### 10. Offline ASR model comparison

**Status:** Partially validated

**What was tried**

A sequential, one-model-at-a-time benchmark was added for:

- Whisper Small English;
- Whisper Large-v3 Turbo;
- Distil-Whisper Large-v3;
- Wav2Vec2 Large 960h LV60 Self.

It compares temporal precision, recall, F1, false-positive speech, missed
dialogue, intro false positives, WER, runtime, real-time factor, and peak GPU
memory. Ranking prioritizes temporal F1 before WER because safe AD gaps depend
on time coverage.

**Why**

Adjusting only timestamp post-processing may not overcome the baseline model's
recognition and hallucination errors. A fair model comparison is needed before
replacing Whisper Small in the main pipeline.

**How it helped**

All candidates use the same audio, human timestamp reference, metric code, and
offline execution constraints. Loading them sequentially limits GPU pressure.

**Why no model change is recorded yet**

The available record contains the Whisper-small smoke result, but not a complete
comparable result set for all four models. Therefore no alternative model is
yet documented as the winner.

**Evidence**

- Benchmark implementation: Git commit `b713de9`.
- Script: `scripts/submit_lanta_asr_benchmark.sh`.

### 11. Dialogue timeline and review-video visualization

**Status:** Completed

**What was tried**

Human dialogue and validated Whisper intervals were rendered as timed overlays
on the source video, with separate subtitles and cue metadata.

**Why**

Aggregate F1 can hide where timing errors occur. A visual review connects each
false positive or missed interval to the actual film audio and image.

**How it helped**

The project can now inspect dialogue errors one interval at a time rather than
relying only on a continuous timeline or scalar score.

**Why metrics alone were insufficient**

The same number of error seconds can have different consequences: a small
boundary drift may be harmless, while a false word across a quiet action can
eliminate the only usable AD slot.

**Evidence**

- Renderer: Git commit `5f7ca67`.
- Script: `scripts/render_dialogue_comparison.py`.
- Local ignored result: `outputs/dialogue-comparison-review/`.

### 12. Shot detection and boundary validation

**Status:** Completed for selected-boundary evidence; recall remains open

**What was tried**

The configured PySceneDetect path was unavailable in the minimal environment,
so the logged fallback used OpenCV frame differences. A separate validation
stage then:

- verified continuous, non-overlapping coverage of the video;
- decoded frames immediately before and after each selected boundary;
- measured pixel and histogram differences;
- wrote a before/after image for every selected boundary;
- collected lower-confidence frame changes as soft candidates.

**Result**

- 14 shot intervals;
- 13 selected boundaries;
- 13 decoded boundary evidence pairs;
- continuous valid timeline;
- complete evidence for the selected boundaries;
- 24 soft transition candidates.

Selected boundary timestamps in seconds:

`1.435, 49.783, 81.114, 89.556, 93.727, 101.568, 104.204, 114.815,
119.052, 123.757, 148.548, 151.451, 166.733`

**Why**

The initial result "14 shots" did not reveal whether boundaries corresponded
to visible cuts or whether gradual transitions were missed.

**How it helped**

Every boundary actually used by the pipeline can now be inspected. Soft
candidates make transition misses visible without automatically creating many
false cuts from motion, gestures, film damage, or animated titles.

**Why validation does not prove perfect shot detection**

`validation_passed=true` means the chosen shot intervals form a valid timeline
and have evidence. It does not establish recall. Gradual title transitions at
approximately 8.709, 12.880, 23.657, and 29.496 seconds remained soft
candidates rather than selected cuts.

**Evidence**

- Boundary validation: Git commit `6b81f6e`.
- Runtime method: `opencv_frame_difference`, CPU.

### 13. Multimodal scene segmentation

**Status:** Problem confirmed

**What was tried**

Adjacent shots were compared using weighted visual histogram, audio, and
transcript/topic distance. The resulting shot-level scores were thresholded to
form scenes.

**Result**

All 14 shots from 0.000–178.584 seconds were merged into Scene 1. Scene
boundary confidence was 0.0. The measured scene-segmentation stage took about
0.001 seconds on CPU.

**Why**

Shots are camera/editing units, while scenes should group shots that share a
location, time, and narrative event. The scene layer was intended to provide
larger narrative context to Qwen.

**How it helped**

The failed result is still informative: it proves that the current feature and
threshold combination contributes almost no segmentation information for this
film.

**Why the current approach did not work**

- Lightweight color histograms are sensitive to appearance but weak at
  semantic location/action understanding.
- Audio continuity and noisy ASR can make adjacent shots appear related.
- A global threshold can miss transitions whose evidence is distributed over
  several frames.
- The current AD placement does not use detected boundaries as scheduling
  anchors, so even correct shots do not position narration.

**Decision**

Do not make AD placement depend on a correct global scene ID. Use confirmed
cuts, soft candidates, frame embeddings, object/action changes, and audio
events as local evidence around each gap.

### 14. Timestamp-indexed visual knowledge database

**Status:** Implemented and unit tested; awaiting full Lanta validation

**Design implemented**

Create a local SQLite evidence store with:

- video metadata;
- frames with requested/decoded timestamps, shot IDs, scene IDs, and paths;
- dialogue intervals and text;
- confirmed boundaries and soft candidates;
- AD/WAIT decisions with the evidence timestamps used;
- final cue placement and timing method.

Images remain on disk; SQLite stores their references and structured knowledge.

**Implementation update (2026-11-19)**

- Added `offline_ad_poc/knowledge_db.py` using Python's built-in SQLite.
- Added `evidence-knowledge.sqlite3` to every work directory.
- Indexed validated transcript segments, shots, scenes, confirmed cuts, soft
  transition candidates, selected frame timestamps, AD decisions, event times,
  cue times, and placement methods.
- The VLM gap context is now retrieved from the database after the selected
  evidence frames are recorded.
- Per-frame semantic observations (objects, people, actions, location, and OCR)
  and consolidated narrative facts are not implemented yet; they remain the
  next database extension after timestamp placement is validated.

**Why**

The current pipeline resamples frames independently for every gap and retains
only previous AD strings. It cannot reliably answer what changed, when an
action occurred, or whether the same fact was already described.

**Expected benefit**

- reproducible evidence for every AD decision;
- chronological retrieval around a candidate gap;
- narrative continuity across gaps;
- less repeated description;
- easier comparison with human AD and dialogue ground truth.

**Risk and limitation**

Persisted model observations can also persist mistakes. Raw observations must
remain distinguishable from consolidated facts, and every fact should retain
its source frame timestamps. A database cannot correct an incorrect ASR gap by
itself.

### 15. Event-centered AD placement inside large gaps

**Status:** Implemented and unit tested; awaiting full Lanta validation

**Previous behavior being replaced**

For a gap, the pipeline samples four frames between up to eight seconds before
the gap and the gap end. Qwen produces one untimed description. After TTS, the
narration is centered in the full speech gap.

**Design implemented**

1. Coarsely sample a large gap at low resolution.
2. Combine frame similarity, motion, confirmed cuts, soft candidates, object
   changes, and audio changes into event candidates.
3. Densely sample only around the strongest candidate timestamps.
4. Ask Qwen to return event start/end and evidence timestamps with each
   description.
5. Fit TTS into safe candidate slots and minimize distance from the event,
   dialogue overlap, important-sound overlap, shot-crossing, and excessive
   narration delay.
6. Permit multiple non-redundant descriptions in a long gap when enough safe
   time exists.

**Implementation update (2026-11-19)**

- Large gaps are sampled on a low-cost time grid enriched with confirmed cuts
  and soft candidates.
- CPU pixel/histogram change plus boundary bonuses select up to six frames.
- Every selected frame carries timestamp, shot, scene, salience, and role.
- Qwen must return `event_start`, `event_end`, and supporting timestamps.
- Returned timestamps are snapped to frames that were actually decoded.
- TTS is placed near `event_start`; a cue is rejected when it can fit only by
  describing a late event too early.
- Scene IDs are prompt context only and cannot block local event selection.
- Multiple AD events inside one gap remain future work; the current
  implementation still returns at most one event per candidate gap.

**Why this is better than the nearest single frame**

A single nearest frame may show only the middle or end of an action. Placement
should use an event interval and begin after the action becomes understandable,
while still finishing before dialogue resumes.

**How this handles missed scene boundaries**

The scheduler uses local micro-events rather than treating scene IDs as gates.
Correct shot/scene results improve its evidence, but a globally merged scene
does not prevent local action detection or placement.

## Next experiment order

1. Run the updated event-aware pipeline on Lanta and inspect selected evidence,
   grounded event timestamps, cue placement, SQLite records, runtime, and GPU
   memory.
2. Compare its AD timing with the previous centered 30.202–34.298-second cue
   and with human AD timing.
3. Run the complete four-model ASR benchmark and select by temporal F1, then
   WER/runtime as tie-breakers.
4. Rerun the event-aware pipeline with the selected ASR and review gap intervals
   against the human dialogue timeline.
5. Evaluate placement error and description quality before adding multiple AD
   events per gap or replacing global scene segmentation.

## Recording template for future experiments

Copy this block for every new run:

```text
Experiment:
Date and Git commit:
Status: completed / partially validated / failed
Question being tested:
Input video and interval:
Models and checkpoint versions:
Parameters changed:
Parameters held constant:
Device and Slurm job ID:
Measured runtime and memory:
Quality metrics:
Human-review result:
What improved:
What failed or remained uncertain:
Why:
Artifacts and paths:
Decision / next experiment:
```
