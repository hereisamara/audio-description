from __future__ import annotations

import json
import tempfile
import unittest
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
    build_gaps,
    choose_event_aware_cue_start,
    evidence_candidate_timestamps,
    expand_qwen_image_tokens,
    limit_words,
    make_decisions,
    merge_intervals,
    mix_described_video,
    normalize_event_payload,
    plan_qwen_calls,
    refine_gap_events,
    resize_frame_max_edge,
    sample_event_evidence_frames,
    segment_scenes,
    speech_intervals_from_words,
    synthesize_decisions,
    validate_whisper_word_timestamps,
    validate_shot_timeline,
)
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

        self.assertEqual(len(shots), 2)
        self.assertAlmostEqual(shots[0].end, 1.0, places=1)
        self.assertAlmostEqual(shots[1].start, 1.0, places=1)

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

