from __future__ import annotations

import json
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

from offline_ad_poc.pipeline import (
    Gap,
    Interval,
    Decision,
    EvidenceFrame,
    Qwen25Backend,
    RuntimeProfiler,
    Scene,
    Shot,
    Word,
    _detect_shots_opencv,
    _segments_from_words,
    _whisper_generate_kwargs,
    analysis_audio_is_silent,
    build_gaps,
    choose_event_aware_cue_start,
    create_shot_boundary_evidence,
    decision_from_proposal,
    evidence_candidate_timestamps,
    expand_qwen_image_tokens,
    is_near_blank,
    limit_words,
    make_decisions,
    merge_intervals,
    mix_described_video,
    normalize_event_payload,
    plan_qwen_calls,
    record_ad_outcomes,
    refine_gap_events,
    resize_frame_max_edge,
    sample_event_evidence_frames,
    segment_scenes,
    speech_intervals_from_words,
    synthesize_decisions,
    validate_whisper_word_timestamps,
    validate_shot_timeline,
)
from offline_ad_poc.ad_preparer import build_qwen_cue_input, prepare_ad_cues, run_qwen_cue
from offline_ad_poc.smolvlm import caption_texts
from offline_ad_poc.knowledge_db import EvidenceStore


class PipelineUnitTests(unittest.TestCase):
    def test_real_scene_helper_is_present(self) -> None:
        self.assertTrue((Path(__file__).parents[1] / "offline_ad_poc" / "prepare_real_scene.py").exists())

    def test_word_intervals_are_padded_and_merged(self) -> None:
        words = [Word(1.0, 1.3, "hello"), Word(1.5, 1.8, "there")]
        intervals = speech_intervals_from_words(words, duration=5.0, padding=0.1, merge_gap=0.3)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0].start, 0.9)
        self.assertAlmostEqual(intervals[0].end, 1.9)

    def test_safe_gaps_do_not_overlap_speech(self) -> None:
        speech = [Interval(0.0, 2.0), Interval(6.0, 8.0)]
        gaps = build_gaps(speech, duration=10.0, minimum_duration=1.5, safety_margin=0.2)
        self.assertEqual(len(gaps), 2)
        self.assertGreater(gaps[0].start, 2.0)
        self.assertLess(gaps[0].end, 6.0)
        self.assertGreater(gaps[0].word_budget, 0)

    def test_gap_audit_distinguishes_short_breaks_from_the_scan_limit(self) -> None:
        speech = [Interval(1.0, 3.0), Interval(4.75, 5.0), Interval(9.0, 10.0)]
        audit = []
        gaps = build_gaps(speech, duration=14.0, audit=audit)
        decision = Decision(gaps[0].start, gaps[0].end, "WAIT", "", "no_visual_event", 0, 8, [],
                            gap_index=1, visual_proposals=[{
                                "window_index": 1, "decision": "WAIT", "reason": "no_visual_event"
                            }])
        [group] = record_ad_outcomes(audit, [decision], multi_event=False)

        self.assertEqual(gaps, build_gaps(speech, duration=14.0))
        self.assertEqual([row["reason"] for row in audit], [
            "gap_too_short", "gap_too_short_after_padding", "no_cue_from_windows", "gap_scan_limit",
        ])
        self.assertAlmostEqual(audit[1]["usable_seconds"], 1.35)
        self.assertEqual(group["windows"][0]["outcome"]["stage"], "visual_review")

    def test_window_audit_preserves_raw_output_and_the_final_stop_reason(self) -> None:
        audit = []
        [gap] = build_gaps([], 20, audit=audit)
        raw = "A person walks along a bank."
        windows = [
            {"window_index": 1, "decision": "WAIT", "reason": "empty_or_incomplete_caption", "raw_caption": ""},
            {"window_index": 2, "decision": "AD", "raw_caption": raw},
            {"window_index": 3, "decision": "AD", "raw_caption": raw},
            {"window_index": 4, "decision": "AD", "raw_caption": raw},
            {"window_index": 5, "decision": "AD", "raw_caption": raw},
        ]
        placed = Decision(gap.start, gap.end, "AD", raw, "caption", 1, 18, [], gap_index=1,
                          selected_window=2, tts_path="cue.wav", cue_start=4, cue_end=6, visual_proposals=windows,
                          visual_followups=[{"kind": "duplicate_text", "discarded_window": 3}])
        held = replace(placed, selected_window=4, decision="WAIT", tts_path="",
                       reason="already_communicated_by_dialogue", visual_proposals=[], visual_followups=[])
        late = replace(held, selected_window=5, placement_method="rejected_no_safe_event_slot",
                       reason="TTS could not fit a safe post-event slot")
        [group] = record_ad_outcomes(audit, [placed, held, late], multi_event=True)

        self.assertEqual(audit[0]["cue_count"], 1)
        self.assertEqual(group["selected_windows"], [2, 4, 5])
        self.assertEqual([row["outcome"]["reason"] for row in windows], [
            "empty_vlm_output", "cue_placed", "duplicate_visual_event", "already_communicated_by_dialogue",
            "TTS could not fit a safe post-event slot",
        ])
        self.assertEqual(windows[1]["outcome"]["cue_start"], 4)
        self.assertEqual(windows[3]["outcome"]["stage"], "ad_preparation")
        self.assertEqual(windows[4]["outcome"]["stage"], "tts_and_placement")
        self.assertTrue(all(row["raw_caption"] == raw for row in windows[1:]))

    def test_silence_check_reads_past_a_silent_opening(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            wav_path = Path(folder) / "analysis.wav"
            samples = np.zeros(70000, dtype="<i2")
            for final_sample in (0, -1):
                with self.subTest(final_sample=final_sample):
                    samples[-1] = final_sample
                    with wave.open(str(wav_path), "wb") as audio:
                        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        audio.writeframes(samples.tobytes())
                    self.assertEqual(analysis_audio_is_silent(wav_path), final_sample == 0)

    def test_invalid_whisper_word_durations_are_rejected(self) -> None:
        words = [
            Word(1.0, 1.4, "real"),
            Word(2.0, 2.0, "zero"),
            Word(5.0, 15.0, "stretched"),
        ]
        validated, report = validate_whisper_word_timestamps(words)

        self.assertEqual(validated, [words[0]])
        self.assertEqual(report["rejected_nonpositive_duration"], 1)
        self.assertEqual(report["rejected_stretched_duration"], 1)

    def test_limit_words_respects_budget(self) -> None:
        text = limit_words("one two three four five six", 4)
        self.assertEqual(len(text.split()), 4)

    def test_ad_decision_uses_direct_visual_description(self) -> None:
        for raw, expected in (
            ("I see a glass with water on top of a table.", "A glass with water on top of a table."),
            ("In the frame, you can see that steam rises from a pot.", "Steam rises from a pot."),
            ("The image shows a person washing vegetables.", "A person washing vegetables."),
            ('A sign reads "I see you."', 'A sign reads "I see you."'),
            ("I see.", ""),
        ):
            with self.subTest(raw=raw):
                decision = decision_from_proposal(
                    Gap(0.0, 10.0, 10), {"decision": "AD", "description": raw}, [], 1, True,
                )
                self.assertEqual(decision.description, expected)
                self.assertEqual(decision.decision, "AD" if expected else "WAIT")

    def test_ad_preparer_needs_prior_visual_support_inside_the_same_shot(self) -> None:
        draft = "A person is holding a."
        full = "A person is holding a cup."
        decision = Decision(7, 15, "AD", draft, "test", 1, 10, [], description_ready_at=8)
        decision.visual_proposals = [
            {"window_index": 1, "decision": "AD", "description": full, "description_ready_at": 4},
            {"window_index": 2, "decision": "AD", "description": full, "description_ready_at": 11},
        ]
        shots = [Shot(1, 0, 5), Shot(2, 5, 15)]
        words = [Word(4, 4.5, "wrong shot"), Word(6, 6.5, "cup"), Word(9, 9.5, "future")]

        def unexpected(context):
            self.fail("Called the LLM without corroborating visual evidence")

        [held] = prepare_ad_cues([replace(decision)], shots, words, unexpected)
        self.assertEqual(held["status"], "wait")
        self.assertEqual(held["context"]["dialogue"], "cup")

        decision.visual_proposals.append(
            {"window_index": 3, "decision": "AD", "description": full, "description_ready_at": 7}
        )
        [prepared] = prepare_ad_cues([decision], shots, words, lambda context: json.dumps({
            "description": full, "dialogue_reference": "",
        }))
        self.assertEqual(prepared["completion_evidence"], [{"timestamp": 7, "cue": full}])
        self.assertEqual(decision.description, full)
        self.assertEqual(decision.description_ready_at, 8)

    def test_ad_preparer_corrects_the_object_when_same_shot_dialogue_identifies_it(self) -> None:
        draft = "A person's hand is partially visible with dirt and debris on it."
        revised = "A person shows an egg mass."
        decision = Decision(7, 15, "AD", draft, "test", 1, 18, [], description_ready_at=8)
        shots = [Shot(1, 0, 5), Shot(2, 5, 15)]
        for start, dialogue, cue, reference, expected in (
            (6, "An example of an egg mass that was recently laid.", revised, "egg mass", revised),
            (6, "An example of an egg mass that was recently laid.", revised, "", revised),
            (6, "It is the time of year when an egg mass can form.", revised, "egg mass", draft),
            (4, "An example of an egg mass.", revised, "egg mass", draft),
            (6, "An example of an egg mass.", "A person shows an egg mass in a jar.", "egg mass", draft),
        ):
            with self.subTest(start=start, dialogue=dialogue, cue=cue, reference=reference):
                [result] = prepare_ad_cues(
                    [replace(decision)], shots, [Word(start, start + 0.5, dialogue)],
                    lambda context: json.dumps({"description": cue, "dialogue_reference": reference}),
                )
                self.assertEqual(result["description"], expected)
                self.assertEqual(result["dialogue_reference"], "egg mass" if expected == revised else "")

    def test_ad_preparer_honors_wait_and_keeps_fallback_for_invalid_output(self) -> None:
        draft = "A person sets a bowl on the table."
        decision = Decision(7, 15, "AD", draft, "test", 1, 18, [], description_ready_at=8)
        shots = [Shot(1, 0, 15)]
        unchanged = replace(decision)

        [held] = prepare_ad_cues([decision], shots, [], lambda _: '{"description":"WAIT","dialogue_reference":""}')
        [failed] = prepare_ad_cues([unchanged], shots, [], lambda _: "invalid")

        self.assertEqual((decision.decision, decision.description, decision.reason), ("WAIT", "", "ad_preparer_wait"))
        self.assertEqual(held["status"], "wait")
        self.assertEqual((unchanged.decision, unchanged.description), ("AD", draft))
        self.assertEqual(failed["status"], "fallback")

    def test_ad_preparer_skips_an_action_already_spoken_with_the_same_details(self) -> None:
        draft = "A person closes the window."
        decision = Decision(7, 15, "AD", draft, "test", 1, 18, [], description_ready_at=8)
        shots = [Shot(1, 0, 15)]
        words = [Word(6, 7, "She closes the window.")]

        def unexpected(_):
            self.fail("An identical spoken action reached the model")

        [held] = prepare_ad_cues([replace(decision)], shots, words, unexpected)
        richer = replace(decision, description="A person closes the window with both hands.")
        [kept] = prepare_ad_cues([richer], shots, words, lambda _: "invalid")

        self.assertEqual(held["status"], "wait")
        self.assertEqual(held["hold_reason"], "already_communicated_by_dialogue")
        self.assertIsNone(held["llm_input"])
        self.assertEqual(kept["description"], "A person closes the window with both hands.")

    def test_ad_preparer_records_the_prompt_it_passes_to_llama_cpp(self) -> None:
        draft = "A person shows a card marked <Hi>."
        decision = Decision(7, 15, "AD", draft, "test", 1, 18, [], description_ready_at=8)
        response = SimpleNamespace(stdout=json.dumps({"description": draft, "dialogue_reference": ""}))
        with patch("offline_ad_poc.ad_preparer.subprocess.run", return_value=response) as subprocess_run:
            [record] = prepare_ad_cues(
                [decision], [Shot(1, 0, 15)], [], lambda llm_input: run_qwen_cue(
                    "llama-completion", Path("test.gguf"), llm_input,
                ),
                build_input=build_qwen_cue_input,
            )

        request = record["llm_input"]
        args = subprocess_run.call_args.args[0]
        self.assertIn(request["system_prompt"], args[args.index("-p") + 1])
        self.assertIn(request["user_prompt"], args[args.index("-p") + 1])
        self.assertEqual(json.loads(request["user_prompt"]), record["context"])
        self.assertIn(r"\u003c", request["user_prompt"])
        self.assertEqual(json.loads(args[args.index("--json-schema") + 1]), request["response_schema"])
        self.assertEqual(record["status"], "prepared")

    def test_smolvlm_keeps_complete_evidence_when_the_generation_cap_is_reached(self) -> None:
        raw = "A hand rests in shallow water.\nA cluster of eggs covers the palm.\nA person is holding a"
        evidence, fallback = caption_texts(raw, reached_token_limit=True)

        self.assertEqual(evidence, "A hand rests in shallow water. A cluster of eggs covers the palm.")
        self.assertEqual(fallback, "A hand rests in shallow water.")

    def test_ad_preparer_receives_visual_evidence_beyond_the_fallback_caption(self) -> None:
        short = "A person shows an egg mass."
        evidence = short + " " + "Reeds rise around the hands in the shallow water. " * 6 + "Small specks cover the palm."
        decision = Decision(7, 15, "AD", short, "test", 1, 18, [],
                            gap_index=1, selected_window=1, description_ready_at=8)
        decision.visual_proposals = [{"window_index": 1, "description": short, "visual_evidence": evidence}]
        [record] = prepare_ad_cues([decision], [Shot(1, 0, 15)], [], lambda _: json.dumps({
            "description": short, "dialogue_reference": "",
        }))

        self.assertEqual(record["context"]["draft"], evidence)
        self.assertEqual(record["description"], short)

    @patch("offline_ad_poc.pipeline.sample_event_evidence_frames")
    def test_qwen_defers_short_gap_word_limit_to_the_ad_preparer_when_enabled(self, sample) -> None:
        frame = EvidenceFrame("frog.jpg", 71.9, 71.9, 1, 1)
        sample.return_value = [frame]
        gap = Gap(71.75, 73.69, 4)
        backend = Qwen25Backend.__new__(Qwen25Backend)
        draft = "A small dark frog sits in a person's hands."
        event = {
            "kind": "detail", "important": True, "already_audible": False, "uncertain": False,
            "frame_ids": [1], "ready_frame_id": 1, "description": draft,
        }

        with patch.object(backend, "_generate", return_value=json.dumps(event)) as generate:
            gated = backend.decide([frame], gap, "", [], {})
            self.assertEqual(gated["decision"], "WAIT")
            self.assertEqual(generate.call_count, 2)
            [decision] = make_decisions(
                Path("frog.mp4"), [gap], [Shot(1, 70, 74)], [], [], backend, Path("frames"), 1,
                defer_word_budget=True,
            )
            self.assertEqual(generate.call_count, 3)
        self.assertEqual(decision.description, draft)

        [held] = prepare_ad_cues([replace(decision)], [], [], lambda _: "")
        [prepared] = prepare_ad_cues(
            [decision], [], [], lambda _: '{"description":"A frog sits.","dialogue_reference":""}',
        )
        self.assertEqual(held["hold_reason"], "no_complete_description_within_word_budget")
        self.assertEqual(prepared["context"]["draft"], draft)
        self.assertEqual(decision.description, "A frog sits.")

    def test_ad_preparer_keeps_earlier_cues_and_dialogue_scoped_to_the_anchor(self) -> None:
        decisions = [
            Decision(3, 7, "AD", "A bird lands.", "test", 1, 18, [], description_ready_at=4),
            Decision(12, 16, "AD", "A person opens a box.", "test", 1, 18, [], description_ready_at=13),
            Decision(14, 18, "AD", "A person lifts a glass.", "test", 1, 18, [], description_ready_at=15),
        ]
        shots = [Shot(1, 0, 10), Shot(2, 10, 20)]
        scenes = [Scene(1, 0, 10, [1], 0), Scene(2, 10, 20, [2], 0)]
        words = [Word(8, 8.5, "previous"), Word(13.5, 14, "nearby"), Word(16, 17, "future")]
        rows = prepare_ad_cues(decisions, shots, words, lambda context: json.dumps({
            "description": context["draft"], "dialogue_reference": "",
        }), scenes=scenes)
        context = rows[2]["context"]

        self.assertEqual(context["dialogue"], "nearby")
        self.assertEqual(context["previous_dialogue"], "previous")
        self.assertEqual(context["previous_ad"], "A person opens a box.")
        self.assertEqual(context["current_scene"], {"id": 2, "start": 10, "end": 20})
        self.assertEqual(context["scene_previous_ads"], [{"time": 13, "text": "A person opens a box."}])

    def test_ad_preparer_thinking_and_final_json_have_separate_limits(self) -> None:
        request = build_qwen_cue_input(
            {"draft": "A person opens a box.", "max_words": 8, "incomplete": False, "identifying_dialogue": []},
            reasoning_tokens=128,
        )
        final_json = '{"description":"A person opens a box.","dialogue_reference":""}'
        responses = [SimpleNamespace(stdout="Planning text.</think> trailing text"), SimpleNamespace(stdout=final_json)]
        with patch("offline_ad_poc.ad_preparer.subprocess.run", side_effect=responses) as subprocess_run:
            result = run_qwen_cue("llama-completion", Path("test.gguf"), request)

        thinking, final = [call.args[0] for call in subprocess_run.call_args_list]
        self.assertEqual(thinking[thinking.index("-n") + 1], "128")
        self.assertEqual(final[final.index("-n") + 1], "72")
        self.assertIn("<think>\nPlanning text.\n</think>", final[final.index("-p") + 1])
        self.assertEqual(result, final_json)

    def test_scene_split_uses_combined_boundary_score(self) -> None:
        shots = [Shot(1, 0, 2, transcript="same words"), Shot(2, 2, 4, transcript="different topic")]
        first = np.zeros(4, dtype=np.float32)
        first[0] = 1
        second = np.zeros(4, dtype=np.float32)
        second[1] = 1
        scenes = segment_scenes(shots, [first, second], boundary_threshold=0.5)
        self.assertEqual(len(scenes), 2)

    def test_shot_timeline_validation_checks_coverage_and_continuity(self) -> None:
        valid = validate_shot_timeline([Shot(1, 0.0, 2.0), Shot(2, 2.0, 5.0)], 5.0)
        invalid = validate_shot_timeline([Shot(1, 0.2, 2.0), Shot(2, 2.4, 4.0)], 5.0)

        self.assertTrue(valid["valid_timeline"])
        self.assertFalse(invalid["valid_timeline"])
        self.assertIn("timeline_does_not_start_at_zero", invalid["warnings"])
        self.assertIn("gap_before_shot_2", invalid["warnings"])

    def test_opencv_shot_detector_finds_a_hard_cut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "hard-cut.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            try:
                for value in (0,) * 10 + (255,) * 10:
                    writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            finally:
                writer.release()

            shots = _detect_shots_opencv(video_path, duration=2.0, threshold=20.0)
            report = create_shot_boundary_evidence(video_path, shots, Path(directory) / "evidence", 2.0)

        self.assertEqual(len(shots), 2)
        self.assertAlmostEqual(shots[0].end, 1.0, places=1)
        self.assertAlmostEqual(shots[1].start, 1.0, places=1)
        self.assertEqual(report["near_blank_intervals"], [{"start": 0.0, "end": 1.0}])

    def test_merge_intervals_keeps_distant_regions_separate(self) -> None:
        merged = merge_intervals([Interval(0, 1), Interval(1.2, 2), Interval(4, 5)], maximum_gap=0.3)
        self.assertEqual(len(merged), 2)

    def test_transformers_words_are_grouped_into_segments(self) -> None:
        words = [Word(0.0, 0.3, "hello"), Word(0.4, 0.8, "there"), Word(2.0, 2.3, "again")]
        segments = _segments_from_words(words)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "hello there")

    def test_english_only_whisper_omits_language_and_task(self) -> None:
        self.assertEqual(_whisper_generate_kwargs(SimpleNamespace(is_multilingual=False)), {})
        self.assertEqual(
            _whisper_generate_kwargs(SimpleNamespace(is_multilingual=True)),
            {"language": "english", "task": "transcribe"},
        )

    def test_qwen_image_placeholders_expand_from_grid(self) -> None:
        expanded = expand_qwen_image_tokens(
            "start <|image_pad|> middle <|image_pad|> end",
            "<|image_pad|>",
            [[1, 4, 4], [1, 2, 4]],
            merge_size=2,
        )
        self.assertEqual(
            expanded, f"start {'<|image_pad|>' * 4} middle {'<|image_pad|>' * 2} end"
        )
        with self.assertRaises(ValueError):
            expand_qwen_image_tokens("<|image_pad|>", "<|image_pad|>", [[1, 4, 4]] * 2, 2)

    def test_visual_event_retry_preserves_outputs_and_rejects_context_ids(self) -> None:
        backend = Qwen25Backend.__new__(Qwen25Backend)
        frames = [EvidenceFrame("context.jpg", 0, 0, 1, 1), EvidenceFrame("event.jpg", 2, 2, 1, 1)]
        proposal = {
            "kind": "action", "important": True, "already_audible": False, "uncertain": False,
            "frame_ids": [2], "ready_frame_id": 2, "description": "A woman stands.",
        }
        invalid = json.dumps({**proposal, "frame_ids": [1], "ready_frame_id": 1})
        with patch.object(backend, "_generate", side_effect=[invalid, json.dumps(proposal)]) as generate:
            result = backend.decide(frames, Gap(1, 4, 8), "", [], {})
        self.assertEqual(result["decision"], "AD")
        self.assertEqual(result["event_start"], 2)
        self.assertEqual(result["event_timing_source"], "model_frame_ids")
        self.assertEqual(result["attempts"][0]["raw_output"], invalid)
        self.assertIn("failed validation", generate.call_args.args[1][-1]["text"])

        with patch.object(backend, "_generate", side_effect=["Not JSON", invalid]) as generate:
            result = backend.decide(frames, Gap(1, 4, 8), "", [], {})
        self.assertEqual(result["reason"], "invalid_vlm_output")
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn("event_start", result)

        with patch.object(backend, "_generate", return_value=json.dumps({**proposal, "already_audible": True})):
            result = backend.decide(frames, Gap(1, 4, 8), "", [], {})
        self.assertEqual(result["reason"], "already_communicated_by_dialogue")

    def test_action_anchor_cannot_use_near_blank_lanta_frame(self) -> None:
        backend = Qwen25Backend.__new__(Qwen25Backend)
        frames = [
            EvidenceFrame("context.jpg", 31.031, 31.031, 2, 1),
            EvidenceFrame("title.jpg", 33.967267, 33.967267, 2, 1),
            EvidenceFrame("fade.jpg", 35.869167, 35.869167, 2, 1, near_blank=True),
            EvidenceFrame("room.jpg", 38.772067, 38.772067, 2, 1),
        ]
        gap = Gap(32.042, 42.656, 20)
        invalid = {
            "kind": "action", "important": True, "already_audible": False, "uncertain": False,
            "frame_ids": [3, 4], "ready_frame_id": 3, "description": "A woman crosses the room.",
        }
        corrected = {**invalid, "frame_ids": [4], "ready_frame_id": 4}
        with patch.object(backend, "_generate", side_effect=map(json.dumps, [invalid, corrected])) as generate:
            result = backend.decide(frames, gap, "", [], {})
        self.assertEqual(result["decision"], "AD")
        self.assertEqual(result["description_ready_at"], 38.772067)
        self.assertIn("Near-blank image 3", result["attempts"][0]["validation_error"])
        retry_prompt = generate.call_args.args[1][-1]["text"]
        self.assertIn("as evidence: 2, 3, 4", retry_prompt)
        self.assertIn("For an action ready_frame_id, use only these IDs: 2, 4", retry_prompt)

        with patch.object(backend, "_generate", return_value=json.dumps(invalid)) as generate:
            result = backend.decide(frames, gap, "", [], {})
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["decision"], "WAIT")
        self.assertNotIn("description_ready_at", result)

    def test_near_blank_check_preserves_visible_detail_in_dark_frame(self) -> None:
        gray = np.full((90, 160), 2, dtype=np.uint8)
        self.assertTrue(is_near_blank(gray))
        gray[:, :16] = 40
        self.assertLess(gray.mean(), 12)
        self.assertFalse(is_near_blank(gray))

    @patch("offline_ad_poc.pipeline.sample_event_evidence_frames")
    def test_long_gap_evaluates_middle_window_and_places_from_first_complete_frame(self, sample) -> None:
        def sample_midpoint(video, window, shots, scenes, boundaries, directory, **kwargs):
            midpoint = (window.start + window.end) / 2
            return [
                EvidenceFrame(str(directory / "action.jpg"), midpoint, midpoint, 1, 1),
                EvidenceFrame(str(directory / "ready.jpg"), midpoint + 2, midpoint + 2, 1, 1),
            ]

        sample.side_effect = sample_midpoint
        backend = Qwen25Backend.__new__(Qwen25Backend)
        no_event = {
            "kind": "none", "important": False, "already_audible": False, "uncertain": False,
            "frame_ids": [], "ready_frame_id": None, "description": "",
        }
        event = {
            **no_event, "kind": "action", "important": True,
            "frame_ids": [1, 2], "ready_frame_id": 2, "description": "A man exits.",
        }
        responses = [no_event, no_event, event, no_event, no_event]
        with patch.object(backend, "_generate", side_effect=map(json.dumps, responses)):
            [decision] = make_decisions(
                Path("clip.mp4"), [Gap(0.2, 53.27, 132)], [], [], [], backend, Path("frames"), 1
            )
        self.assertEqual(sample.call_count, 5)
        self.assertEqual(decision.selected_window, 3)
        self.assertTrue(any(item["start"] <= 31.72 < item["end"] for item in decision.visual_proposals))
        cue, method = choose_event_aware_cue_start(decision, 3.0)
        self.assertAlmostEqual(cue, decision.description_ready_at + 0.15)
        self.assertEqual(method, "event_aligned")

        late = replace(decision, gap_end=decision.description_ready_at + 0.5)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("offline_ad_poc.pipeline.resolve_tts_backend", return_value=("transformers", "model")),
                patch("offline_ad_poc.pipeline.synthesize_text", return_value=Path("narration.wav")),
                patch("offline_ad_poc.pipeline.probe_duration", return_value=3.0),
            ):
                synthesize_decisions([late], Path(directory), "transformers", None, "model", "cpu", "ffprobe")
        self.assertEqual(late.decision, "WAIT")
        self.assertEqual(late.reason, "TTS could not fit a safe post-event slot")
        self.assertEqual(late.visual_proposals[2]["description"], "A man exits.")

    @patch("offline_ad_poc.pipeline.sample_event_evidence_frames")
    def test_event_comparison_spends_only_the_video_wide_budget(self, sample) -> None:
        def near_window_edge(video, window, shots, scenes, boundaries, directory, **kwargs):
            t = window.end - 0.4 if window.start in {0, 30} else window.start + 0.3
            return [EvidenceFrame(str(directory / "image.jpg"), t, t, 1, 1)]

        sample.side_effect = near_window_edge
        backend = Qwen25Backend.__new__(Qwen25Backend)
        event = {
            "kind": "action", "important": True, "already_audible": False, "uncertain": False,
            "frame_ids": [1], "ready_frame_id": 1, "description": "The man leaves the desk.",
        }
        adjacent = {**event, "description": "The man leaves the room."}
        with (
            patch.object(backend, "_generate", side_effect=map(json.dumps, [event, adjacent] * 2)),
            patch.object(backend, "compare_events", return_value={"relation": "same"}) as compare,
        ):
            decisions = make_decisions(
                Path("clip.mp4"), [Gap(0, 24, 60), Gap(30, 54, 60)], [], [], [], backend,
                Path("frames"), 2, multi_event=True, followup_budget=1,
            )
        self.assertEqual(compare.call_count, 1)
        self.assertEqual([item.gap_index for item in decisions], [1, 2, 2])
        self.assertEqual(decisions[0].visual_followups[0]["relation"], "same")
        self.assertEqual(decisions[1].visual_followups, [])

    def test_qwen_call_plan_from_lanta_gap_lengths(self) -> None:
        gaps = [
            Gap(start, end) for start, end in [
                (0.20, 53.27), (60.07, 61.69), (68.77, 74.19), (80.17, 87.89),
                (119.77, 128.37), (146.93, 159.61), (164.67, 167.29),
            ]
        ]
        plan = plan_qwen_calls(gaps, 12, 2, 4.8)
        self.assertEqual(plan["initial_windows"], 12)
        self.assertEqual(plan["max_calls_including_initial_retries"], 26)
        self.assertAlmostEqual(plan["estimated_seconds_with_all_followups"], 67.2)

    def test_late_proposal_can_use_earlier_supported_description(self) -> None:
        late = {
            "decision": "AD", "description": "The man walks across the room and exits.",
            "description_ready_at": 9.6, "window_index": 1,
        }
        early = {**late, "description": "The man walks away.", "description_ready_at": 8.0, "window_index": 2}
        def inspect(window, description):
            self.assertLess(window.start, early["description_ready_at"])
            self.assertEqual(description, late["description"])
            return early

        selected, reviews = refine_gap_events(
            [late], Gap(0, 10, 25), Qwen25Backend.__new__(Qwen25Backend), inspect, 1, 6
        )
        self.assertEqual(selected, [early])
        self.assertTrue(reviews[0]["used"])

    def test_multi_event_tts_resolves_collision_and_keeps_separated_cue(self) -> None:
        early = Decision(0, 14, "AD", "A woman rises.", "action", 0.5, 20, [],
                         event_start=2, description_ready_at=2, preferred_cue_start=2.15)
        priority = replace(early, description="A man enters.", importance=1,
                           event_start=3, description_ready_at=3, preferred_cue_start=3.15)
        later = replace(early, description="He picks up a letter.",
                        event_start=8, description_ready_at=8, preferred_cue_start=8.15)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("offline_ad_poc.pipeline.resolve_tts_backend", return_value=("transformers", "model")),
                patch("offline_ad_poc.pipeline.synthesize_text", return_value=Path("narration.wav")),
                patch("offline_ad_poc.pipeline.probe_duration", return_value=4.0),
            ):
                synthesize_decisions([early, priority, later], Path(directory), "transformers",
                                     None, "model", "cpu", "ffprobe")
        self.assertEqual(early.placement_method, "rejected_cue_conflict")
        self.assertTrue(priority.tts_path and later.tts_path)
        self.assertLess(priority.cue_end, later.cue_start)
        self.assertLess(later.cue_end, later.gap_end)

    def test_evidence_frame_resize_preserves_aspect_ratio_without_upscaling(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_frame_max_edge(frame, 320)
        unchanged = resize_frame_max_edge(frame, 1280)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertIs(unchanged, frame)

    def test_event_payload_timestamps_are_snapped_to_decoded_frames(self) -> None:
        frames = [
            EvidenceFrame("one.jpg", 5.0, 5.0, 1, 1, salience=0.1),
            EvidenceFrame("two.jpg", 9.0, 9.0, 1, 1, salience=0.7),
            EvidenceFrame("three.jpg", 13.0, 13.0, 2, 1, salience=0.3),
        ]
        payload = normalize_event_payload(
            {
                "decision": "AD",
                "event_start": 8.7,
                "event_end": 12.8,
                "evidence_timestamps": [8.8, 13.2],
            },
            frames,
            Gap(4.0, 14.0, 12),
        )

        self.assertEqual(payload["event_start"], 9.0)
        self.assertEqual(payload["event_end"], 13.0)
        self.assertEqual(payload["evidence_timestamps"], [9.0, 13.0])

    def test_event_aware_placement_is_not_gap_centered(self) -> None:
        decision = Decision(
            gap_start=0.0,
            gap_end=20.0,
            decision="AD",
            description="A woman enters.",
            reason="new action",
            importance=0.8,
            word_budget=20,
            evidence_frames=[],
            event_start=10.0,
            event_end=11.0,
            preferred_cue_start=10.15,
        )

        cue_start, method = choose_event_aware_cue_start(decision, narration_duration=4.0)

        self.assertAlmostEqual(cue_start, 10.15)
        self.assertEqual(method, "event_aligned")
        self.assertNotAlmostEqual(cue_start, 8.0)

    def test_event_aware_placement_rejects_spoiler_before_late_event(self) -> None:
        decision = Decision(
            gap_start=0.0,
            gap_end=20.0,
            decision="AD",
            description="The door opens.",
            reason="new action",
            importance=0.8,
            word_budget=20,
            evidence_frames=[],
            event_start=19.0,
            event_end=19.0,
            preferred_cue_start=19.15,
        )

        cue_start, method = choose_event_aware_cue_start(decision, narration_duration=3.0)

        self.assertIsNone(cue_start)
        self.assertEqual(method, "no_post_event_fit")

    def test_candidate_timestamps_include_boundaries_independent_of_scene_split(self) -> None:
        gap = Gap(10.0, 20.0, 20)
        shots = [Shot(1, 0.0, 15.0), Shot(2, 15.0, 25.0)]
        timestamps = evidence_candidate_timestamps(gap, shots, [17.0])

        self.assertIn(15.0, timestamps)
        self.assertIn(17.0, timestamps)
        self.assertTrue(any(value < gap.start for value in timestamps))

    def test_adaptive_sampler_keeps_timestamps_and_cut_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            video_path = directory_path / "event-cut.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            try:
                for value in (0,) * 20 + (255,) * 20:
                    writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            finally:
                writer.release()
            shots = [Shot(1, 0.0, 2.0), Shot(2, 2.0, 4.0)]
            scenes = [Scene(1, 0.0, 4.0, [1, 2], 0.0)]

            frames = sample_event_evidence_frames(
                video_path,
                Gap(1.0, 3.5, 8),
                shots,
                scenes,
                [],
                directory_path / "frames",
                maximum_frames=5,
                max_edge=32,
            )

        self.assertLessEqual(len(frames), 5)
        self.assertEqual([frame.timestamp for frame in frames], sorted(frame.timestamp for frame in frames))
        self.assertTrue(any(frame.role == "confirmed_cut" for frame in frames))
        self.assertTrue(all(frame.shot_id in {1, 2} for frame in frames))

    def test_evidence_store_retrieves_timestamped_gap_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "evidence.sqlite3")
            segments = [SimpleNamespace(start=1.0, end=2.0, text="hello")]
            shots = [Shot(1, 0.0, 5.0)]
            scenes = [Scene(1, 0.0, 5.0, [1], 0.0)]
            validation = {
                "evidence": [],
                "soft_candidates_requiring_review": [
                    {"timestamp": 3.0, "frame_difference_score": 12.5}
                ],
            }
            store.index_timeline(Path("video.mp4"), 5.0, segments, shots, scenes, validation)
            store.record_frames(
                1,
                [EvidenceFrame("frame.jpg", 3.1, 3.0, 1, 1, 0.8, "soft_transition")],
            )
            store.record_decision(
                1,
                Decision(
                    gap_start=2.5,
                    gap_end=4.5,
                    decision="AD",
                    description="A door opens.",
                    reason="new action",
                    importance=0.8,
                    word_budget=8,
                    evidence_frames=["frame.jpg"],
                    evidence_timestamps=[3.1],
                    event_start=3.1,
                    event_end=3.1,
                    event_timing_source="model_timestamps_grounded_to_frames",
                    placement_method="event_aligned",
                ),
            )

            context = store.query_gap_context(2.5, 4.5)
            later_context = store.query_gap_context(4.5, 5.0)
            store.close()

        self.assertEqual(context["boundaries"][0]["kind"], "soft_candidate")
        self.assertEqual(context["frames"][0]["timestamp"], 3.1)
        self.assertEqual(context["frames"][0]["shot_id"], 1)
        self.assertEqual(later_context["previous_ad"][0]["description"], "A door opens.")

    @patch("offline_ad_poc.pipeline.run_command")
    def test_wait_only_output_copies_original_streams(self, run_command) -> None:
        decision = Decision(
            gap_start=1.0,
            gap_end=3.0,
            decision="WAIT",
            description="",
            reason="no important visual change",
            importance=0.1,
            word_budget=4,
            evidence_frames=[],
        )
        method = mix_described_video(
            Path("input.mp4"),
            Path("output.mp4"),
            [decision],
            "ffmpeg",
        )

        self.assertEqual(method, "ffmpeg_stream_copy_no_ad")
        command = run_command.call_args.args[0]
        self.assertIn("copy", command)
        self.assertNotIn("-filter_complex", command)

    def test_runtime_profiler_writes_stage_events_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiler = RuntimeProfiler(Path(directory))
            profiler.set_hardware({"cpu_count_logical": 4, "cuda_available": False})
            with profiler.stage("example", method="unit-test", compute_device="cpu") as details:
                details["items"] = 3
            summary = profiler.finalize()

            event = json.loads(profiler.events_path.read_text(encoding="utf-8"))
            self.assertEqual(event["stage"], "example")
            self.assertEqual(event["method"], "unit-test")
            self.assertEqual(event["items"], 3)
            self.assertEqual(event["status"], "completed")
            self.assertGreaterEqual(event["elapsed_seconds"], 0.0)
            self.assertGreaterEqual(event["cpu_seconds"], 0.0)
            self.assertEqual(event["compute_device"], "cpu")
            self.assertTrue(event["uses_cpu"])
            self.assertFalse(event["uses_gpu"])
            self.assertGreater(event["process_peak_rss_mb"], 0.0)
            self.assertEqual(summary["stages"], [event])
            self.assertEqual(summary["hardware"]["cpu_count_logical"], 4)
            self.assertEqual(summary["totals_by_compute_device"]["cpu"]["stages"], 1)
            self.assertTrue(profiler.summary_path.exists())


if __name__ == "__main__":
    unittest.main()

