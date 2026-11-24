"use strict";

const el = id => document.getElementById(id);
const video = el("video");
const narrationAudio = el("narration-audio");
const tracks = {ad: [], adThai: [], dialogue: [], reference: []};
const labels = {ad: "English AD", adThai: "Thai AD", dialogue: "Dialogue", reference: "Human AD"};
const modelProfiles = {
  auto: "Auto (CPU Qwen3-VL-2B, CUDA Qwen3-VL-8B)",
  qwen3vl: "Qwen3-VL-2B",
  qwen3vl8b: "Qwen3-VL-8B",
  qwen25: "Qwen2.5-VL-3B",
  smolvlm: "SmolVLM"
};
const narrationLanguages = {en: ["ad", "English"], th: ["adThai", "Thai"]};
const artifactLabels = {
  "ad-decisions.json": "AD decisions",
  "ad-proposals.json": "Visual evidence and proposals",
  "ad-preparation.json": "AD preparation and dialogue context",
  "narration-plan.json": "Narration cues and WAV files",
  "speech-and-gaps.json": "Speech and gaps",
  "transcript.json": "ASR transcript",
  "shots.json": "Shots",
  "scenes.json": "Scenes",
  "shot-boundary-validation.json": "Transitions and validation",
  "runtime-summary.json": "Stage time and device",
  "dialogue-runtime-summary.json": "Whisper stage time and device",
  "evaluation-report.json": "Evaluation report",
  "media.json": "Media information",
  "source.json": "Source and attribution"
};
const contextLabels = {
  draft: "Visual draft",
  max_words: "Maximum words",
  available_seconds: "Safe narration seconds",
  incomplete: "Incomplete draft",
  dialogue: "Preceding same-shot dialogue",
  previous_dialogue: "Earlier dialogue",
  previous_ad: "Previous prepared AD",
  current_scene: "Current scene",
  scene_previous_ads: "Earlier prepared AD in this scene",
  identifying_dialogue: "Identifying phrases",
  allowed_completions: "Allowed completions"
};
let artifacts = {};
let activeTrack = "ad";
let current = null;
let source = null;
let originalSource = null;
let originalMedia = null;
let objectURL = null;
let media = null;
let capabilities = {};
let running = false;
let selectedJob = null;
let sequence = 0;
let generatingNarration = false;
let translating = false;
let generatedNarration = {};
let playingNarration = null;
let previewMuted = false;

function notify(message, error = false) {
  el("notice").textContent = message;
  el("notice").hidden = !message;
  el("notice").dataset.error = error;
}

function errorMessage(error) { notify(error.message, true); }

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.headers.get("content-type")?.includes("application/json")) {
    throw new Error("This action needs the local demo server.");
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function post(url, data) {
  return request(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });
}

function clock(value, full = false) {
  const ms = Math.max(0, Math.round(value * 1000));
  const pad = (number, width = 2) => String(number).padStart(width, "0");
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor(ms / 60000) % 60;
  return (full || hours ? pad(hours) + ":" : "") + pad(minutes) + ":" +
    pad(Math.floor(ms / 1000) % 60) + "." + pad(ms % 1000, 3);
}

function seconds(value) {
  return value.replace(",", ".").split(":").reduce((sum, part) => sum * 60 + Number(part), 0);
}

function cue(start, end, text) { return {id: ++sequence, start, end, text: text.trim()}; }

function fields(rows) { return rows.map(({start, end, text}) => ({start, end, text})); }

function importRows(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.filter(row =>
    row && Number.isFinite(row.start) && Number.isFinite(row.end) &&
    row.start >= 0 && row.end > row.start && typeof row.text === "string" && row.text.trim()
  ).map(row => cue(row.start, row.end, row.text));
}

function linkAdPreparation() {
  const decisions = artifacts["ad-decisions.json"];
  if (!Array.isArray(decisions)) return;
  const key = (start, end) => clock(start, true) + "/" + clock(end, true);
  const byTime = new Map();
  decisions.forEach((item, index) => {
    if (item?.decision === "AD" && Number.isFinite(item.cue_start) && Number.isFinite(item.cue_end)) {
      byTime.set(key(item.cue_start, item.cue_end), index + 1);
    }
  });
  for (const item of tracks.ad) {
    item.decisionIndex ??= byTime.get(key(item.start, item.end));
    const decision = decisions[item.decisionIndex - 1];
    if (decision?.overlaps_dialogue) item.dialogueOverlap = decision.dialogue_overlap_seconds;
  }
}

function updateCueDialogueOverlap() {
  const speech = artifacts["speech-and-gaps.json"]?.speech;
  if (!Array.isArray(speech)) return;
  for (const item of tracks.ad) {
    item.dialogueOverlap = speech.reduce((total, row) => {
      if (!Number.isFinite(row?.start) || !Number.isFinite(row?.end)) return total;
      return total + Math.max(0, Math.min(item.end, row.end) - Math.max(item.start, row.start));
    }, 0);
  }
}

function parseSubtitles(text) {
  const pattern = /((?:\d+:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+((?:\d+:)?\d{2}:\d{2}[.,]\d{3})[^\n]*\n([\s\S]*?)(?=\n\s*\n|$)/g;
  return importRows([...text.replaceAll("\r", "").matchAll(pattern)].map(match => ({
    start: seconds(match[1]), end: seconds(match[2]), text: match[3]
  })));
}

function basename(file) { return (file.webkitRelativePath || file.name).split("/").at(-1); }

function importedVideo(files) {
  return files.find(file => basename(file) === "video-ad.mp4") ||
    files.find(file => /-ad\.mp4$/i.test(basename(file))) ||
    files.find(file => /\.mp4$/i.test(basename(file)));
}

function setPreview(asset, at = 0) {
  stopNarrationPlayback();
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = asset instanceof Blob ? URL.createObjectURL(asset) : null;
  video.src = objectURL || asset.url;
  video.hidden = false;
  el("empty-video").hidden = true;
  if (at > 0) video.addEventListener("loadedmetadata", () => { video.currentTime = at; }, {once: true});
}

function setSource(file, title = file.name, original = null) {
  media = null;
  source = file;
  originalSource = original === file ? null : original;
  originalMedia = null;
  generatedNarration = {};
  el("title").textContent = title.replace(/\.mp4$/i, "");
  setPreview(file);
  el("audio-track").replaceChildren(new Option("Inspect audio tracks"));
  updateControls();
}

async function openFiles(files, reset = true) {
  if (reset) {
    artifacts = {};
    el("timeline-detail").hidden = true;
    for (const kind of Object.keys(tracks)) tracks[kind] = [];
    generatedNarration = {};
    stopNarrationPlayback();
    fillEditor(null);
  }
  let edits;
  let dialogueImported = false;
  let adImported = false;
  for (const file of files) {
    const name = basename(file);
    if (name.endsWith(".json") && file.size <= 16 * 1024 * 1024) {
      const data = JSON.parse(await file.text());
      if (name === "ad-edits.json") edits = data;
      else if (name in artifactLabels) artifacts[name] = data;
    } else if (/\.(srt|vtt)$/i.test(name) && file.size <= 4 * 1024 * 1024) {
      const kind = /reference|human/i.test(name) ? "reference" : /dialogue|speech/i.test(name) ? "dialogue" :
        /^(?:ad|descriptions)[._-](?:th|thai)\./i.test(name) ? "adThai" : "ad";
      tracks[kind] = parseSubtitles(await file.text());
      dialogueImported ||= kind === "dialogue";
      adImported ||= kind === "ad";
    }
  }
  if (!dialogueImported && (reset || !tracks.dialogue.length)) {
    tracks.dialogue = importRows(artifacts["transcript.json"]?.segments);
  }
  if (!adImported && !tracks.ad.length && Array.isArray(artifacts["ad-decisions.json"])) {
    tracks.ad = importRows(artifacts["ad-decisions.json"].filter(item => item.decision === "AD").map(item => ({
      start: item.cue_start, end: item.cue_end, text: item.description
    })));
  }
  if (edits) for (const kind of Object.keys(tracks)) {
    if (Array.isArray(edits[kind])) tracks[kind] = importRows(edits[kind]);
  }
  linkAdPreparation();
  const file = importedVideo(files);
  if (file) {
    const folder = file.webkitRelativePath?.split("/")[0];
    setSource(file, folder || file.name, files.find(item => basename(item) === "original.mp4"));
  } else if (reset) {
    source = null;
    media = null;
    originalSource = null;
    originalMedia = null;
    video.removeAttribute("src");
    video.load();
    video.hidden = true;
    el("empty-video").hidden = false;
    el("title").textContent = "Imported cues";
  }
  if (tracks.ad.length) el("caption-mode").value = tracks.adThai.length ? "bilingual" : "ad";
  else if (tracks.adThai.length) el("caption-mode").value = "adThai";
  else if (tracks.dialogue.length) el("caption-mode").value = "dialogue";
  refresh();
  if (file?.url && capabilities.ffmpeg) {
    await inspectAudio().catch(errorMessage);
  }
}

async function openManifest(items) {
  const files = await Promise.all(items.map(async item => {
    if (item.name.endsWith(".mp4")) return item;
    const response = await fetch(item.url);
    if (!response.ok) throw new Error("Could not load " + item.name);
    const blob = await response.blob();
    return new File([blob], item.name, {type: blob.type});
  }));
  await openFiles(files);
}

function updateControls() {
  const selected = selectedNarration();
  const language = el("narration-language").value;
  const narrationTrack = narrationLanguages[language][0];
  const voice = language === "th" ? capabilities.thai_voice : capabilities.voice;
  el("inspect-audio").disabled = !source || !capabilities.ffmpeg;
  el("audio-track").disabled = !media;
  el("play-audio").disabled = !media;
  el("export-video").disabled = generatingNarration || !source || !capabilities.ffmpeg;
  el("translate-ad").disabled = translating || !capabilities.thai_translation || !tracks.ad.length;
  el("translate-ad").title = capabilities.thai_translation ? "" : "Requires a local Qwen3 translation model";
  el("generate-audio").disabled = generatingNarration || !source || !capabilities.ffmpeg || !voice ||
    !tracks[narrationTrack].length;
  el("generate-audio").title = voice ? "" : "Install a local " + narrationLanguages[language][1] + " voice";
  el("download-audio").disabled = !selected;
  el("revoice").disabled = !!selected || !source || !capabilities.voice || !tracks.ad.length ||
    !(originalSource || (media?.tracks.some(item => item.ad) && media?.tracks.some(item => !item.ad)));
  if (el("revoice").disabled) el("revoice").checked = false;
  el("export-detail").textContent = selected ? "Exports the selected " + narrationLanguages[selected.language][1] + " AD audio." :
    el("revoice").checked ?
    "Uses " + capabilities.voice + " on the original soundtrack. Each cue must fit its edited time range." :
    "MP4 export adds editable subtitles and keeps the selected audio mix.";
  el("generate").disabled = running || !source || !capabilities.worker;
  el("open-job").disabled = running || !el("jobs-list").value;
  el("add-cue").disabled = !source && !Object.values(tracks).some(rows => rows.length);
  el("export-vtt").disabled = !tracks[activeTrack].length;
  el("export-json").disabled = !Object.values(tracks).some(rows => rows.length);
}

function renderCueInput() {
  const panel = el("cue-input");
  panel.hidden = activeTrack !== "ad" || !current;
  if (panel.hidden) return;
  const record = Array.isArray(artifacts["ad-preparation.json"]) ?
    artifacts["ad-preparation.json"].find(row => row?.decision_index === current.decisionIndex) : null;
  const context = record?.context;
  const sent = record?.llm_input;
  const fields = el("cue-input-fields");
  fields.replaceChildren();
  const hasContext = context && typeof context === "object" && !Array.isArray(context);
  const hasPrompt = typeof sent?.system_prompt === "string" && sent.system_prompt.length > 0;
  el("cue-input-prompt").hidden = !hasPrompt;
  el("cue-input-system").textContent = hasPrompt ? sent.system_prompt : "";
  el("cue-input-context").hidden = !hasContext;
  el("cue-input-json").hidden = typeof sent?.user_prompt !== "string";
  el("cue-input-user").textContent = sent?.user_prompt ?? "";
  const info = el("cue-input-note");
  const generation = sent?.generation;
  const limits = generation ? " · Reasoning budget " + generation.reasoning_tokens +
    " tokens · Final JSON cap " + generation.final_tokens + " tokens" : "";
  info.textContent = !record ? "No LLM input was saved for this cue." :
    hasPrompt ? "Decision " + record.decision_index + limits :
    record.llm_input === null ? "No model request was recorded for this decision." :
    "System prompt unavailable in this run.";
  if (hasContext) for (const [key, value] of Object.entries(context)) {
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = contextLabels[key] ?? key;
    detail.textContent = Array.isArray(value) && value.every(item => typeof item === "string") ?
      value.join("\n") || "—" :
      typeof value === "object" && value !== null ? JSON.stringify(value, null, 2) : String(value ?? "") || "—";
    fields.append(term, detail);
  }
}

function fillEditor(item) {
  current = item;
  el("editor").hidden = !item;
  renderCueInput();
  if (!item) return;
  el("start").value = item.start.toFixed(3);
  el("end").value = item.end.toFixed(3);
  el("cue-text").value = item.text;
  el("delete-cue").disabled = !tracks[activeTrack].includes(item);
}

function selectCue(kind, item) {
  activeTrack = kind;
  fillEditor(item);
  if (source) video.currentTime = item.start;
  renderCues();
}

function renderCues() {
  document.querySelectorAll("[data-track]").forEach(button =>
    button.setAttribute("aria-pressed", String(button.dataset.track === activeTrack)));
  const list = el("cue-list");
  list.replaceChildren();
  for (const item of tracks[activeTrack]) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    const timing = document.createElement("time");
    const words = document.createElement("span");
    timing.textContent = clock(item.start);
    words.textContent = item.text;
    if (item.dialogueOverlap > 0) {
      const flag = document.createElement("small");
      flag.className = "overlap-flag";
      flag.textContent = "Dialogue overlap · " + item.dialogueOverlap.toFixed(2) + " s";
      words.append(flag);
    }
    button.setAttribute("aria-current", String(current === item));
    button.append(timing, words);
    button.onclick = () => selectCue(activeTrack, item);
    li.append(button);
    list.append(li);
  }
  if (!list.children.length) {
    const empty = document.createElement("li");
    empty.textContent = "No " + labels[activeTrack].toLowerCase() + " cues in this run.";
    empty.className = "editor-empty";
    list.append(empty);
  }
  updateControls();
}

function paintTime() {
  const now = video.currentTime || 0;
  const selected = el("caption-mode").value;
  const kinds = selected === "both" ? ["ad", "dialogue"] : selected === "bilingual" ? ["ad", "adThai"] : [selected];
  const text = kinds.flatMap(kind => (tracks[kind] || [])
    .filter(item => now >= item.start && now < item.end)
    .map(item => (selected === "both" && kind === "ad" ? "AD · " : "") + item.text)).join("\n");
  el("caption").textContent = text;
  el("caption").hidden = !text || !source;
  el("time").textContent = clock(now) + (Number.isFinite(video.duration) ? " / " + clock(video.duration) : "");
  const timeline = el("timeline");
  const duration = Number(timeline.dataset.duration);
  timeline.style.setProperty("--playhead", duration > 0 ? Math.min(100, now / duration * 100) + "%" : "0%");
}

function timelineIntervals(name, kind) {
  const rows = artifacts[name];
  if (!Array.isArray(rows)) return [];
  return rows.filter(row => row && Number.isFinite(row.start) && Number.isFinite(row.end) &&
    row.start >= 0 && row.end > row.start).map((row, index) => {
    const id = row[kind + "_id"] ?? index + 1;
    const name = (kind === "scene" ? "Scene " : "Shot ") + id;
    const title = name + " · " + clock(row.start) + "–" + clock(row.end);
    const memberShots = Array.isArray(row.shot_ids) ? "Shots " + row.shot_ids.join(", ") : "";
    return {start: row.start, end: row.end, label: name, title, detail: memberShots};
  });
}

const adReasonLabels = {
  gap_too_short: "The break between speech intervals is below the minimum.",
  gap_too_short_after_padding: "The break falls below the minimum after reserving space around speech.",
  gap_scan_limit: "The run reached its limit on gaps to scan.",
  no_clear_frame_inside_window: "No clear frame was available inside the window.",
  no_in_window_evidence: "No frame inside the window could support a cue.",
  no_vlm_output: "No VLM output was recorded.",
  empty_vlm_output: "The VLM returned no text.",
  empty_or_incomplete_caption: "The VLM caption was incomplete or unusable.",
  invalid_vlm_output: "The VLM response failed validation.",
  vlm_generation_failure: "VLM generation failed.",
  no_visual_event: "No visual event was identified.",
  uncertain_visual_evidence: "The visual evidence was uncertain.",
  low_visual_importance: "The event was judged too minor to describe.",
  already_communicated_by_dialogue: "The dialogue already conveyed this information.",
  repeated_caption: "The caption repeated an earlier description.",
  duplicate_visual_event: "Another window already covered the same event.",
  replaced_by_earlier_evidence: "A follow-up found earlier usable evidence for the same event.",
  timing_followup_not_selected: "The earlier follow-up did not find a usable placement.",
  single_event_plan: "The single-event plan selected a different window in this gap.",
  window_not_selected: "The pipeline did not select this proposal.",
  no_usable_description: "No usable description remained after applying the word budget.",
  ad_preparer_wait: "The AD preparer recommended skipping the cue.",
  ad_preparer_no_reply: "The AD preparer returned no description.",
  invalid_cue_response: "The AD preparer returned an invalid reply.",
  ad_preparer_execution_failed: "The AD preparer failed to run.",
  ad_preparer_word_limit: "The rewrite exceeded the word budget.",
  ad_preparer_sentence_limit: "The rewrite used more than one sentence.",
  ad_preparer_ungrounded_rewrite: "The rewrite did not pass the evidence check.",
  no_safe_post_event_slot: "Too little time remained after the visual anchor to attempt a cue.",
  incomplete_visual_description: "The AD preparer could not produce a complete description.",
  no_complete_description_within_word_budget: "No complete sentence fit the narration word budget."
};

const adOutcomeStages = {
  visual_review: "Visual review", event_selection: "Event selection",
  ad_preparation: "AD preparation", tts_and_placement: "Narration placement"
};

function savedWindowOutput(group, window, decisions) {
  const label = ["Window " + window.window_index, clock(window.start) + "–" + clock(window.end),
    window.scan === "timing_followup" ? "Timing follow-up" : ""].filter(Boolean).join(" · ");
  const decision = decisions.find(item => item?.gap_index === group.gap_index &&
    item.selected_window === window.window_index);
  const rawText = value => typeof value === "string" ? value : "";
  const responses = Array.isArray(window.attempts) && window.attempts.length ?
    window.attempts.map(attempt => rawText(attempt?.raw_output)) : [rawText(window.raw_caption ?? window.raw_output)];
  const outcome = window.outcome || (decision?.decision === "AD" && decision.tts_path ?
    {status: "ad_added", cue_start: decision.cue_start, cue_end: decision.cue_end} :
    {status: "no_ad", reason: decision?.reason || (window.decision === "AD" ?
      "window_not_selected" : window.reason || "no_vlm_output")});
  const prep = artifacts["ad-preparation.json"];
  const decisionIndex = outcome.decision_index ?? (decision ? decisions.indexOf(decision) + 1 : null);
  const preparation = Array.isArray(prep) ? prep.find(record => record?.decision_index === decisionIndex) : null;
  const attempts = Array.isArray(window.attempts) ? window.attempts : [];
  return {label, outcome, responses, preparation,
    prompts: attempts.map(attempt => attempt?.prompt),
    recentDialogue: attempts[0]?.recent_dialogue,
    validationErrors: attempts.map(
    attempt => attempt?.validation_error || attempt?.generation_error
  )};
}

function timelineGaps(groups, decisions, fallback = false) {
  const speech = artifacts["speech-and-gaps.json"];
  const audited = Array.isArray(speech?.gap_audit);
  const rows = audited ? speech.gap_audit : speech?.gaps;
  if (!Array.isArray(rows)) return [];
  const byGap = new Map(groups.map(group => [group.gap_index, group]));
  return rows.flatMap((row, index) => {
    if (!row || !Number.isFinite(row.start) || !Number.isFinite(row.end) || row.end <= row.start) return [];
    if ((row.kind === "scene_fallback") !== fallback) return [];
    const gapIndex = audited ? row.gap_index : index + 1;
    const group = byGap.get(gapIndex);
    const outputs = (group?.windows || []).filter(Boolean).map(window => savedWindowOutput(group, window, decisions));
    const savedCues = outputs.filter(output => output.outcome.status === "ad_added").length;
    const count = row.cue_count ?? savedCues;
    const status = row.status || (count ? "ad_added" : outputs.length ? "no_ad" : "unknown");
    const result = count ? count + (count === 1 ? " AD cue" : " AD cues") :
      status === "unknown" ? "Outcome not recorded" : "No AD";
    const detail = [
      fallback ? "Scene had no placed AD before fallback" :
        audited ? row.usable_seconds.toFixed(2) + " s usable · " + row.minimum_seconds.toFixed(2) + " s minimum" : "",
      status === "not_scanned" ? (adReasonLabels[row.reason] || row.reason) + " VLM was not called." : ""
    ].filter(Boolean).join(" · ");
    const name = fallback ? "Scene " + row.scene_id : gapIndex == null ? "Speech break" : "Gap " + gapIndex;
    return [{start: row.start, end: row.end, label: (fallback ? row.scene_id : gapIndex) ?? "", status, detail,
      overlap: row.overlaps_dialogue, title: name + " · " + clock(row.start) + "–" + clock(row.end) +
        " · " + result + (row.overlaps_dialogue ? " · Dialogue overlap" : ""), outputs: new Set(outputs)}];
  });
}

function visualFrameMarkers(groups, decisions) {
  const backend = artifacts["evaluation-report.json"]?.visual_backend;
  if (backend === "heuristic") return [];
  const markers = new Map();
  function record(timestamp, windowOutput, anchor = false) {
    if (!Number.isFinite(timestamp) || timestamp < 0) return;
    const item = markers.get(timestamp) || {start: timestamp, anchor: false, outputs: new Set()};
    item.anchor ||= anchor;
    item.outputs.add(windowOutput);
    markers.set(timestamp, item);
  }

  for (const group of groups) {
    for (const window of group.windows) {
      if (!window) continue;
      const windowOutput = savedWindowOutput(group, window, decisions);
      windowOutput.label = (window.scan === "scene_fallback" ? "Scene fallback" : "Gap " + group.gap_index) +
        " · " + windowOutput.label;
      if (Number.isFinite(window.caption_frame_timestamp)) {
        record(window.caption_frame_timestamp, windowOutput, true);
      } else if (backend !== "smolvlm" && Array.isArray(window.frames)) {
        const attempts = Array.isArray(window.attempts) ? window.attempts : null;
        if (!(attempts ? attempts.length : ["qwen25", "qwen3vl", "qwen3vl8b"].includes(backend) || typeof window.raw_output === "string")) continue;
        for (const frame of window.frames) {
          record(frame?.timestamp, windowOutput, frame?.timestamp === window.description_ready_at);
        }
      }
    }
  }

  if (!groups.length) {
    const legacyOutput = {label: "Saved frame reference", responses: [""]};
    for (const decision of decisions) {
      if (!decision) continue;
      const evidence = Array.isArray(decision.evidence_timestamps) ? decision.evidence_timestamps : [];
      const paths = Array.isArray(decision.evidence_frames) ? decision.evidence_frames : [];
      const documented = backend === "smolvlm" || !paths.length ? evidence : paths.map(path => {
        const match = typeof path === "string" && /(?:^|[/\\])frame-\d+-(\d+(?:\.\d+)?)\.(?:jpe?g|png)$/i.exec(path);
        return match ? Number(match[1]) : NaN;
      }).concat(evidence);
      for (const time of documented) {
        record(time, legacyOutput, time === decision.description_ready_at);
      }
    }
  }
  return [...markers.values()].sort((a, b) => a.start - b.start).map(item => ({
    ...item,
    title: (item.anchor ? "VLM caption or cue anchor" : "VLM input frame") + " · " + clock(item.start)
  }));
}

function showTimelineDetail(item) {
  el("timeline-selection").textContent = [item.title, item.detail].filter(Boolean).join(" · ");
  el("timeline-raw").hidden = !item.outputs?.size;
  const responses = el("timeline-responses");
  responses.replaceChildren();
  if (item.outputs) {
    for (const output of item.outputs) {
      const heading = document.createElement("h3");
      heading.textContent = output.label;
      responses.append(heading);
      if (output.outcome) {
        const result = document.createElement("div");
        const outcome = output.outcome;
        result.textContent = outcome.status === "ad_added" ?
          "AD added · " + clock(outcome.cue_start) + "–" + clock(outcome.cue_end) :
          ["No AD", adOutcomeStages[outcome.stage], adReasonLabels[outcome.reason] || outcome.reason]
            .filter(Boolean).join(" · ");
        if (outcome.overlaps_dialogue) {
          const flag = document.createElement("span");
          flag.className = "overlap-flag";
          flag.textContent = "Dialogue overlap · " + outcome.dialogue_overlap_seconds.toFixed(2) + " s";
          result.append(flag);
        }
        responses.append(result);
        for (const [index, attempt] of (outcome.tts_attempts || []).entries()) {
          const timing = document.createElement("div");
          timing.textContent = "TTS attempt " + (index + 1) + " · " + attempt.duration_seconds +
            " s spoken · " + attempt.available_seconds + " s after anchor";
          responses.append(timing);
        }
      }
      const prep = output.preparation;
      if (prep) {
        const detail = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = ["AD preparer", prep.status,
          prep.rewrite_rejection && (adReasonLabels[prep.rewrite_rejection] || prep.rewrite_rejection)]
          .filter(Boolean).join(" · ");
        const raw = document.createElement("pre");
        raw.textContent = JSON.stringify({context: prep.context, reply: prep.raw_output, retry: prep.timing_retry}, null, 2);
        detail.append(summary, raw);
        responses.append(detail);
      }
      if (typeof output.recentDialogue === "string") {
        const dialogue = document.createElement("div");
        dialogue.textContent = "Dialogue sent to VLM · " + (output.recentDialogue || "None");
        responses.append(dialogue);
      }
      output.responses.forEach((text, index) => {
        if (output.prompts?.[index]) {
          const prompt = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = "VLM prompt" + (output.responses.length > 1 ? " · Attempt " + (index + 1) : "");
          const input = document.createElement("pre");
          input.textContent = output.prompts[index];
          prompt.append(summary, input);
          responses.append(prompt);
        }
        const label = document.createElement("div");
        label.className = "raw-label";
        label.textContent = "Raw VLM output" + (output.responses.length > 1 ? " · Attempt " + (index + 1) : "");
        responses.append(label);
        const raw = document.createElement("pre");
        raw.textContent = text;
        responses.append(raw);
        const validationError = output.validationErrors?.[index];
        if (validationError) {
          const problem = document.createElement("div");
          problem.textContent = validationError;
          responses.append(problem);
        }
      });
    }
  }
  el("timeline-detail").hidden = false;
}

function renderTimeline() {
  const timeline = el("timeline");
  timeline.replaceChildren();
  const proposals = artifacts["ad-proposals.json"];
  const groups = Array.isArray(proposals) ? proposals.filter(group => Array.isArray(group?.windows)) : [];
  const savedDecisions = artifacts["ad-decisions.json"];
  const decisions = Array.isArray(savedDecisions) ? savedDecisions : [];
  const lanes = [
    ["scene", "Scenes", timelineIntervals("scenes.json", "scene")],
    ["shot", "Shots", timelineIntervals("shots.json", "shot")],
    ["gap", "AD gaps", timelineGaps(groups, decisions)],
    ["scene-fallback", "Scene fallback", timelineGaps(groups, decisions, true)],
    ["frame", "VLM frames", visualFrameMarkers(groups, decisions)],
    ...Object.entries(tracks).map(([kind, items]) => [kind, labels[kind], items])
  ];
  const mediaDuration = artifacts["media.json"]?.duration;
  let duration = Math.max(1, Number.isFinite(video.duration) ? video.duration : 0,
    Number.isFinite(mediaDuration) ? mediaDuration : 0);
  for (const [, , items] of lanes) {
    for (const item of items) duration = Math.max(duration, item.end ?? item.start);
  }
  timeline.dataset.duration = duration;
  for (const [kind, name, items] of lanes) {
    if (!items.length) continue;
    const lane = document.createElement("div");
    const label = document.createElement("span");
    const track = document.createElement("div");
    lane.className = "lane" + (kind in tracks ? "" : " analysis-lane");
    label.textContent = name + (kind in tracks ? "" : " · " + items.length);
    track.className = "lane-track";
    for (const item of items) {
      const tick = document.createElement("button");
      tick.className = "tick " + kind;
      if (item.overlap || item.dialogueOverlap > 0) tick.dataset.overlap = "true";
      if (item.status) tick.dataset.status = item.status;
      tick.style.left = item.start / duration * 100 + "%";
      if (item.end !== undefined) tick.style.width = (item.end - item.start) / duration * 100 + "%";
      if (item.anchor) tick.classList.add("anchor");
      if (item.label) tick.textContent = item.label;
      tick.title = item.title || clock(item.start) + " · " + item.text +
        (item.dialogueOverlap > 0 ? " · Dialogue overlap" : "");
      tick.setAttribute("aria-label", tick.title);
      tick.onclick = () => {
        if (kind in tracks) {
          selectCue(kind, item);
          return;
        }
        if (source) video.currentTime = item.start;
        showTimelineDetail(item);
      };
      track.append(tick);
    }
    lane.append(label, track);
    timeline.append(lane);
  }
  paintTime();
}

function renderArtifact() {
  const name = el("artifact").value;
  const artifact = artifacts[name];
  const summary = el("stage-summary");
  summary.replaceChildren();
  if (Array.isArray(artifact?.stages)) {
    for (const stage of artifact.stages) {
      const row = document.createElement("div");
      row.className = "stage-row";
      for (const value of [stage.stage, stage.compute_device || "—",
        Number.isFinite(stage.elapsed_seconds) ? stage.elapsed_seconds.toFixed(3) + " s" : "—"]) {
        const span = document.createElement("span");
        span.textContent = value;
        row.append(span);
      }
      summary.append(row);
    }
  }
  const raw = artifact === undefined ? "Import a run to inspect its stage artifacts." : JSON.stringify(artifact, null, 2);
  el("artifact-content").textContent = raw.length > 100000 ? raw.slice(0, 100000) + "\n… Preview truncated" : raw;
}

function refresh() {
  for (const rows of Object.values(tracks)) rows.sort((a, b) => a.start - b.start);
  updateCueDialogueOverlap();
  discardStaleNarration();
  const report = artifacts["evaluation-report.json"];
  const displayName = artifacts["source.json"]?.title || report?.display_name;
  if (typeof displayName === "string") el("title").textContent = displayName;
  const info = [report?.device ? "Reported device " + report.device.toUpperCase() : "",
    report?.visual_backend, tracks.ad.length ? tracks.ad.length + " AD cues" : "",
    report?.preview_audio ? "Local preview voice" : ""].filter(Boolean);
  el("run-info").textContent = info.join(" · ") || "Media and cue review";
  const names = Object.keys(artifactLabels).filter(name => name in artifacts);
  el("artifact").replaceChildren(...(names.length ? names.map(name => new Option(artifactLabels[name], name)) :
    [new Option("No artifacts", "")]));
  if (names.includes("runtime-summary.json")) el("artifact").value = "runtime-summary.json";
  renderArtifact();
  renderCues();
  renderCueInput();
  renderTimeline();
}

async function inspectAudio() {
  if (media) return media;
  if (!source) throw new Error("Choose a video first.");
  const uploaded = source;
  notify("Inspecting the audio tracks…");
  const result = await acquireMedia(uploaded);
  if (uploaded !== source) throw new Error("The selected video changed.");
  media = result;
  updateAudioSelect();
  updateControls();
  notify("Audio tracks ready. Select a mix to play or export.");
  return media;
}

function acquireMedia(asset) {
  return asset.url ? post("/api/media", {url: asset.url}) :
    request("/api/media", {method: "POST", body: asset});
}

function narrationSignature(language) {
  return JSON.stringify(fields(tracks[narrationLanguages[language][0]]));
}

function selectedNarration() {
  const selected = el("audio-track").value;
  return selected.startsWith("generated-") ? generatedNarration[selected.slice("generated-".length)] : null;
}

function updateAudioSelect(value) {
  if (!media) return;
  const select = el("audio-track");
  const previous = value ?? select.value;
  select.replaceChildren(...media.tracks.map(item => new Option(item.label, item.index)));
  if (originalSource) select.append(new Option("Original soundtrack (separate file)", "original"));
  for (const language of Object.keys(generatedNarration)) {
    select.append(new Option(narrationLanguages[language][1] + " AD (generated)", "generated-" + language));
  }
  select.value = [...select.options].some(option => option.value === previous) ? previous :
    String((media.tracks.find(item => item.ad) || media.tracks[0]).index);
}

function discardStaleNarration() {
  for (const [language, item] of Object.entries(generatedNarration)) {
    if (item.signature === narrationSignature(language)) continue;
    if (playingNarration === item) stopNarrationPlayback();
    delete generatedNarration[language];
    updateAudioSelect();
  }
}

async function translateThai() {
  const english = narrationSignature("en");
  if (tracks.adThai.length && !window.confirm("Replace the existing Thai AD cues?")) return;
  const inputSource = source;
  translating = true;
  updateControls();
  notify("Translating the English cues to Thai on this machine…");
  try {
    const result = await post("/api/translate-ad", {cues: JSON.parse(english)});
    if (source !== inputSource || english !== narrationSignature("en")) {
      throw new Error("The video or English cues changed during translation.");
    }
    tracks.adThai = importRows(result.cues);
    if (activeTrack === "adThai") fillEditor(null);
    el("caption-mode").value = "bilingual";
    refresh();
    notify("Thai captions are ready. You can edit them before generating narration.");
  } finally {
    translating = false;
    updateControls();
  }
}

async function cleanSoundtrack() {
  const input = await inspectAudio();
  const original = input.tracks.find(track => !track.ad);
  if (original) return {asset: input, audio: original.index};
  if (originalSource) {
    const asset = await originalAsset();
    const track = asset.tracks.find(item => !item.ad);
    if (track) return {asset, audio: track.index};
  }
  throw new Error("Import the original soundtrack with this run to generate new AD audio.");
}

async function generateNarration() {
  const language = el("narration-language").value;
  const signature = narrationSignature(language);
  const inputSource = source;
  generatingNarration = true;
  updateControls();
  try {
    const {asset, audio} = await cleanSoundtrack();
    notify("Generating " + narrationLanguages[language][1] + " AD audio…");
    const result = await post("/api/media/" + asset.id + "/narration", {
      language, audio, cues: JSON.parse(signature)
    });
    if (source !== inputSource || signature !== narrationSignature(language)) {
      throw new Error("The video or cues changed during audio generation.");
    }
    generatedNarration[language] = {language, signature, asset, audio, url: result.url};
    updateAudioSelect("generated-" + language);
    startNarrationPlayback(generatedNarration[language]);
    notify(narrationLanguages[language][1] + " AD audio ready · " + result.voice);
  } finally {
    generatingNarration = false;
    updateControls();
  }
}

function syncNarration() {
  if (!playingNarration || narrationAudio.readyState < 1) return;
  if (Math.abs(narrationAudio.currentTime - video.currentTime) > 0.18) {
    narrationAudio.currentTime = video.currentTime;
  }
  narrationAudio.playbackRate = video.playbackRate;
  narrationAudio.volume = video.volume;
}

function playNarrationAudio() {
  if (!playingNarration || video.paused || video.seeking) return;
  syncNarration();
  narrationAudio.play().catch(error => {
    if (playingNarration && !video.paused && error.name !== "AbortError") errorMessage(error);
  });
}

function startNarrationPlayback(item) {
  if (item === playingNarration) return;
  stopNarrationPlayback();
  playingNarration = item;
  previewMuted = video.muted;
  video.muted = true;
  narrationAudio.src = item.url;
  narrationAudio.muted = previewMuted;
  playNarrationAudio();
}

function stopNarrationPlayback() {
  if (!playingNarration) return;
  playingNarration = null;
  narrationAudio.pause();
  narrationAudio.removeAttribute("src");
  narrationAudio.load();
  video.muted = previewMuted;
}

async function originalAsset() {
  if (originalMedia) return originalMedia;
  const uploaded = originalSource;
  const result = await acquireMedia(uploaded);
  if (uploaded !== originalSource) throw new Error("The selected original video changed.");
  originalMedia = result;
  return result;
}

async function renderMP4(withCaptions) {
  const original = source;
  await inspectAudio();
  const revoice = withCaptions && el("revoice").checked && !el("revoice").disabled;
  const narration = selectedNarration();
  const alternate = narration && Object.values(generatedNarration).find(
    item => item.language !== narration.language && item.asset.id === narration.asset.id
  );
  const selected = revoice ? (originalSource ? "original" : String(media.tracks.find(item => !item.ad).index)) :
    el("audio-track").value;
  const asset = narration?.asset || (selected === "original" ? await originalAsset() : media);
  const subtitles = el("export-ad-captions").value;
  notify(withCaptions ? "Exporting MP4…" : "Preparing the selected audio mix…");
  const result = await post("/api/media/" + asset.id + "/render", {
    audio: narration?.audio ?? (selected === "original" ? 0 : Number(selected)),
    narration_language: narration?.language,
    alternate_narration_language: withCaptions ? alternate?.language : undefined,
    revoice,
    captions: withCaptions ? {
      ad: subtitles === "adThai" ? [] : fields(tracks.ad),
      adThai: subtitles === "ad" ? [] : fields(tracks.adThai),
      dialogue: fields(tracks.dialogue)
    } : {}
  });
  if (original !== source) throw new Error("The selected video changed.");
  return result.url;
}

async function previewAudio() {
  const narration = selectedNarration();
  if (narration) {
    startNarrationPlayback(narration);
    notify(narrationLanguages[narration.language][1] + " AD audio selected.");
    return;
  }
  const time = video.currentTime;
  const url = await renderMP4(false);
  setPreview({url}, time);
  notify("Playing mix · " + el("audio-track").selectedOptions[0].text);
}

function download(url, name) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}

function downloadText(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], {type}));
  download(url, name);
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

async function refreshJobs() {
  const {jobs} = await request("/api/jobs");
  const list = el("jobs-list");
  const currentId = selectedJob?.id || list.value;
  list.replaceChildren(...jobs.map(job => new Option(
    job.title + " · " + job.state + (job.state === "finished" ? "" : " · " + job.stage),
    job.id
  )));
  if (!jobs.length) list.add(new Option("No jobs yet", ""));
  if (jobs.some(job => job.id === currentId)) list.value = currentId;
  updateControls();
}

async function followJob(id, openSource = false) {
  el("job-log").hidden = false;
  let first = true;
  while (true) {
    const job = await request("/api/jobs/" + id);
    selectedJob = job;
    const devices = Object.entries(job.devices || {}).map(([role, info]) =>
      (role === "whisper" ? "Whisper" : "AD") + " " + info.device.toUpperCase() + " on " + info.host
    );
    el("job-state").textContent = [
      job.state, job.stage && job.stage !== "complete" ? job.stage : null,
      "requested " + job.requested.toUpperCase(),
      ...(devices.length ? devices : [job.host + " · selected " + job.device.toUpperCase()]),
      modelProfiles[job.profile] || job.profile, job.fallback ? "CPU fallback" : null
    ].filter(Boolean).join(" · ");
    el("load-job-dialogue").hidden = !job.dialogue_ready;
    el("job-log").textContent = [job.error, job.log].filter(Boolean).join("\n") || "Waiting for output…";
    if (job.state === "finished") {
      await openManifest(job.files);
      return;
    }
    if (first && openSource && job.original) await openFiles([job.original]);
    first = false;
    if (job.state === "failed") throw new Error(job.error || "Generation failed. See the job log.");
    await new Promise(resolve => setTimeout(resolve, 1600));
  }
}

async function inspectJobDialogue() {
  const job = selectedJob;
  if (!job?.dialogue_ready) return;
  const dialogue = await request("/api/jobs/" + job.id + "/dialogue");
  if (selectedJob?.id !== job.id) return;
  const files = [
    [dialogue.transcript, "transcript.json"], [dialogue.speech_and_gaps, "speech-and-gaps.json"],
  ].map(([data, name]) => new File([JSON.stringify(data)], name, {type: "application/json"}));
  if (selectedJob.state === "finished") {
    // Keep the reviewed or edited result open and show the saved upstream transcript.
    el("artifact").value = "transcript.json";
    renderArtifact();
    return;
  }
  await openFiles([job.original, ...files]);
  el("caption-mode").value = "dialogue";
  paintTime();
}

async function generate() {
  running = true;
  updateControls();
  try {
    const asset = originalSource ? await originalAsset() : await inspectAudio();
    const job = await post("/api/media/" + asset.id + "/generate", {
      device: el("device").value, fallback: el("fallback").checked, title: el("title").textContent
    });
    await followJob(job.id);
  } finally {
    running = false;
    updateControls();
    if (capabilities.split_workers) refreshJobs().catch(errorMessage);
  }
}

el("folder").onchange = event => {
  openFiles([...event.target.files]).catch(errorMessage);
  event.target.value = "";
};
el("files").onchange = event => {
  openFiles([...event.target.files], false).catch(errorMessage);
  event.target.value = "";
};
el("video-file").onchange = event => {
  if (event.target.files.length) openFiles([...event.target.files]).catch(errorMessage);
  event.target.value = "";
};
el("example").onclick = () => request("/api/example").then(result => openManifest(result.files)).catch(errorMessage);
el("inspect-audio").onclick = () => inspectAudio().catch(errorMessage);
el("play-audio").onclick = () => previewAudio().catch(errorMessage);
el("audio-track").onchange = () => {
  updateControls();
  if (selectedNarration()) previewAudio().catch(errorMessage);
  else stopNarrationPlayback();
};
el("narration-language").onchange = updateControls;
el("translate-ad").onclick = () => translateThai().catch(errorMessage);
el("generate-audio").onclick = () => generateNarration().catch(errorMessage);
el("download-audio").onclick = () => {
  const item = selectedNarration();
  if (item) download(item.url, "ad-" + item.language + ".m4a");
};
el("generate").onclick = () => generate().catch(errorMessage);
el("refresh-jobs").onclick = () => refreshJobs().catch(errorMessage);
el("load-job-dialogue").onclick = () => inspectJobDialogue().catch(errorMessage);
el("open-job").onclick = async () => {
  const id = el("jobs-list").value;
  running = true;
  updateControls();
  try {
    await followJob(id, true);
  } catch (error) {
    errorMessage(error);
  } finally {
    running = false;
    updateControls();
  }
};
el("artifact").onchange = renderArtifact;
el("caption-mode").onchange = paintTime;
el("revoice").onchange = updateControls;
for (const event of ["timeupdate", "seeked", "loadedmetadata"]) video.addEventListener(event, paintTime);
video.addEventListener("loadedmetadata", renderTimeline);
for (const event of ["play", "playing", "seeked"]) video.addEventListener(event, playNarrationAudio);
for (const event of ["pause", "waiting", "seeking", "ended"]) {
  video.addEventListener(event, () => narrationAudio.pause());
}
for (const event of ["timeupdate", "ratechange", "volumechange"]) video.addEventListener(event, syncNarration);
video.addEventListener("volumechange", () => {
  if (playingNarration && !video.muted) {
    narrationAudio.muted = !narrationAudio.muted;
    video.muted = true;
  }
});
narrationAudio.addEventListener("loadedmetadata", syncNarration);
narrationAudio.addEventListener("error", () => {
  if (!playingNarration) return;
  stopNarrationPlayback();
  errorMessage(new Error("The selected AD audio mix could not be played."));
});
document.querySelectorAll("[data-track]").forEach(button => {
  button.onclick = () => { activeTrack = button.dataset.track; fillEditor(null); renderCues(); };
});
el("add-cue").onclick = () => {
  const start = video.currentTime || 0;
  fillEditor(cue(start, start + 2, ""));
  el("cue-text").focus();
};
el("editor").onsubmit = event => {
  event.preventDefault();
  const start = el("start").valueAsNumber;
  const end = el("end").valueAsNumber;
  const text = el("cue-text").value.trim();
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start || !text) {
    notify("Enter cue text and an end time after the start time.", true);
    return;
  }
  Object.assign(current, {start, end, text});
  if (!tracks[activeTrack].includes(current)) tracks[activeTrack].push(current);
  refresh();
  fillEditor(current);
  notify("Cue saved. Download the edits to keep them after closing the page.");
};
el("delete-cue").onclick = () => {
  tracks[activeTrack] = tracks[activeTrack].filter(item => item !== current);
  fillEditor(null);
  refresh();
};
el("export-vtt").onclick = () => {
  const text = tracks[activeTrack].map(item =>
    clock(item.start, true) + " --> " + clock(item.end, true) + "\n" + item.text).join("\n\n");
  downloadText("WEBVTT\n\n" + text + "\n", (activeTrack === "adThai" ? "ad-th" : activeTrack) + ".vtt", "text/vtt");
};
el("export-json").onclick = () => {
  const data = Object.fromEntries(Object.entries(tracks).map(([kind, items]) => [kind, fields(items)]));
  downloadText(JSON.stringify(data, null, 2), "ad-edits.json", "application/json");
};
el("export-video").onclick = () => renderMP4(true).then(url => {
  download(url, "audio-description-review.mp4");
  notify("Export ready. The MP4 includes the selected audio and editable subtitle tracks.");
}).catch(errorMessage);

refresh();
async function loadCapabilities() {
  const result = await request("/api/capabilities");
  capabilities = result;
  el("host").textContent = result.host;
  el("worker-info").textContent = result.worker ?
    (result.split_workers ? "Separate Whisper and AD workers." :
      result.cuda ? "CPU and NVIDIA CUDA available." : "CPU available. NVIDIA CUDA unavailable.") +
    " Model profile " + (modelProfiles[result.profile] || result.profile) + "." + (result.ad_preparer ? " AD preparer enabled." : "") :
    result.detail;
  el("example").hidden = !result.example;
  el("job-history").hidden = !result.split_workers;
  updateControls();
  if (result.split_workers && !result.worker) setTimeout(() => loadCapabilities().catch(errorMessage), 3000);
}
loadCapabilities().then(() => {
  if (capabilities.split_workers) return refreshJobs();
}).catch(() => {
  el("worker-info").textContent = "Browser review mode. Start the local server for MP4 export and generation.";
});
