from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import os

from MRI_Agent.agent.langgraph.loop import plan_agent_dag
from MRI_Agent.commands.dispatcher import ToolDispatcher
from MRI_Agent.core.plan_dag import AgentPlanDAG


class LangGraphPlannerDagTests(unittest.TestCase):
    def test_plan_agent_dag_is_planning_only_no_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_prostate"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")
            (case_dir / "DWI_b1400.nii.gz").write_text("dwi\n", encoding="utf-8")

            with patch.object(ToolDispatcher, "create_run", autospec=True) as mk_run, patch.object(
                ToolDispatcher, "dispatch", autospec=True
            ) as mk_dispatch:
                dag = plan_agent_dag(
                    goal="Generate a prostate MRI report with lesion review.",
                    domain="prostate",
                    case_ref=str(case_dir),
                    llm_mode="stub",
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                )

            self.assertIsInstance(dag, AgentPlanDAG)
            self.assertEqual(mk_run.call_count, 0)
            self.assertEqual(mk_dispatch.call_count, 0)
            self.assertTrue(any("planning-only mode" in n for n in dag.notes))

    def test_prostate_report_plan_marks_lesion_optional_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-019_2"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")
            (case_dir / "DWI_b1400.nii.gz").write_text("dwi\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="Generate a prostate report from this case.",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            nodes = {n.tool_name: n for n in dag.nodes}

            self.assertIn("detect_lesion_candidates", nodes)
            self.assertFalse(bool(nodes["detect_lesion_candidates"].required))
            self.assertIn("generate_report", nodes)
            self.assertTrue(bool(nodes["generate_report"].required))

    def test_segmentation_only_goal_omits_report_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-seg-only"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="Segment prostate only and output mask only; no report.",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertNotIn("generate_report", tool_names)
            self.assertNotIn("package_vlm_evidence", tool_names)

    def test_negated_report_phrase_prunes_report_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-seg-no-report"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="Just do a segmentation. I do not need a report.",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertNotIn("generate_report", tool_names)
            self.assertNotIn("package_vlm_evidence", tool_names)

    def test_negated_report_goal_overrides_planner_report_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-negated-overrides"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            mocked_plan = {"plan_text": "\n".join(["1. identify_sequences", "2. segment_prostate", "3. generate_report"])}
            with patch("MRI_Agent.agent.langgraph.loop.PlannerSubagent.run", return_value=mocked_plan):
                dag = plan_agent_dag(
                    goal="Segment only. Do not generate report.",
                    domain="prostate",
                    case_ref=str(case_dir),
                    llm_mode="stub",
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertNotIn("generate_report", tool_names)
            self.assertNotIn("package_vlm_evidence", tool_names)

    def test_register_request_type_prunes_brain_segmentation_and_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "Brats18_CBICA_AAM_1"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T1c.nii.gz").write_text("t1c\n", encoding="utf-8")
            (case_dir / "T1.nii.gz").write_text("t1\n", encoding="utf-8")
            (case_dir / "T2.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "FLAIR.nii.gz").write_text("flair\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="no i want to register brain",
                domain="brain",
                case_ref=str(case_dir),
                request_type="register",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertIn("identify_sequences", tool_names)
            self.assertIn("register_to_reference", tool_names)
            self.assertNotIn("brats_mri_segmentation", tool_names)
            self.assertNotIn("extract_roi_features", tool_names)
            self.assertNotIn("generate_report", tool_names)
            reg_nodes = [n for n in dag.nodes if n.tool_name == "register_to_reference"]
            self.assertTrue(reg_nodes)
            self.assertTrue(all(bool(n.required) for n in reg_nodes))

    def test_registration_override_swaps_brain_register_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "Brats18_CBICA_AAM_1"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T1c.nii.gz").write_text("t1c\n", encoding="utf-8")
            (case_dir / "T2.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "FLAIR.nii.gz").write_text("flair\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="could you register T1C to T2 and Flair to T2 instead?",
                domain="brain",
                case_ref=str(case_dir),
                request_type="register",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            regs = [n for n in dag.nodes if n.tool_name == "register_to_reference"]
            self.assertEqual(len(regs), 2)
            reg_args = [{k: str(v) for k, v in (r.arguments or {}).items()} for r in regs]
            combos = {(ra.get("moving"), ra.get("fixed")) for ra in reg_args}
            self.assertIn(("@seq.T1c", "@seq.T2"), combos)
            self.assertIn(("@seq.FLAIR", "@seq.T2"), combos)
            labels = [str(r.label or "") for r in regs]
            self.assertTrue(any("Register T1c to T2" in lb for lb in labels))
            self.assertTrue(any("Register FLAIR to T2" in lb for lb in labels))
            self.assertTrue(any(str((r.arguments or {}).get("output_subdir") or "").endswith("/T1c") for r in regs))
            self.assertTrue(any(str((r.arguments or {}).get("output_subdir") or "").endswith("/FLAIR") for r in regs))
            self.assertTrue(all(bool(r.required) for r in regs))

    def test_blocked_plan_has_natural_response_and_no_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_like_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T1.nii.gz").write_text("t1\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="please do full cardiac pipeline and classify",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "blocked")
            self.assertTrue(bool(str(dag.natural_language_response or "").strip()))
            self.assertIn("CINE", str(dag.natural_language_response))
            self.assertEqual(len(dag.nodes), 0)
            self.assertTrue(any("missing modality: CINE" in str(x) for x in (dag.blocking_reasons or [])))

    def test_cardiac_patient061_4d_directory_evidence_unblocks_and_binds_cine_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_multiseq_patient061_ed"
            case_dir.mkdir(parents=True, exist_ok=True)
            cine_file = case_dir / "patient061_4d.nii.gz"
            cine_file.write_text("cine\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="please do full cardiac pipeline and classify",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "ready")
            self.assertFalse(any("missing modality: CINE" in str(x) for x in (dag.blocking_reasons or [])))
            seg_nodes = [n for n in dag.nodes if n.tool_name == "segment_cardiac_cine"]
            self.assertTrue(seg_nodes)
            self.assertEqual(str((seg_nodes[0].arguments or {}).get("cine_path") or ""), str(cine_file))
            self.assertTrue(any("Directory evidence" in str(n) and "CINE" in str(n) for n in (dag.notes or [])))

    def test_cardiac_symlink_candidate_keeps_lexical_case_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_multiseq_patient061_ed"
            external_root = ws / "acdc_test_group_4_per_group"
            case_dir.mkdir(parents=True, exist_ok=True)
            external_root.mkdir(parents=True, exist_ok=True)
            external_cine = external_root / "patient061_4d.nii.gz"
            external_cine.write_text("cine\n", encoding="utf-8")
            linked_cine = case_dir / "patient061_4d.nii.gz"
            os.symlink(str(external_cine), str(linked_cine))

            dag = plan_agent_dag(
                goal="please do full cardiac pipeline and classify",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            seg_nodes = [n for n in dag.nodes if n.tool_name == "segment_cardiac_cine"]
            self.assertTrue(seg_nodes)
            self.assertEqual(str((seg_nodes[0].arguments or {}).get("cine_path") or ""), str(linked_cine))

    def test_metadata_qa_intent_builds_minimal_identify_then_rag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-meta-qa"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="I only want to read the dicom header information of prostate mri image.",
                domain="prostate",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.requested_request_type), "qa")
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertEqual(tool_names, ["identify_sequences", "rag_search"])
            ident_nodes = [n for n in dag.nodes if n.tool_name == "identify_sequences"]
            self.assertTrue(ident_nodes)
            ident_args = dict(ident_nodes[0].arguments or {})
            self.assertTrue(bool(ident_args.get("deep_dump")))
            self.assertTrue(bool(ident_args.get("require_pydicom")))
            qa_nodes = [n for n in dag.nodes if n.tool_name == "rag_search"]
            self.assertTrue(qa_nodes)
            self.assertTrue(bool(qa_nodes[0].required))
            self.assertIn("question", dict(qa_nodes[0].arguments or {}))
            self.assertEqual(
                str((qa_nodes[0].arguments or {}).get("case_state_path") or ""),
                "@runtime.case_state_path",
            )

    def test_custom_analysis_intent_builds_identify_then_sandbox_exec(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-custom-analysis"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="run custom analysis with code interpreter on this case",
                domain="prostate",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.requested_request_type), "custom_analysis")
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertEqual(tool_names, ["identify_sequences", "sandbox_exec"])
            sand = [n for n in dag.nodes if n.tool_name == "sandbox_exec"]
            self.assertTrue(sand)
            self.assertTrue(bool(sand[0].required))
            self.assertIn("cmd", dict(sand[0].arguments or {}))

    def test_planner_optional_phrase_can_override_feature_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-optional"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            mocked_plan = {
                "plan_text": "\n".join(
                    [
                        "1. identify_sequences",
                        "2. register_to_reference (ADC)",
                        "3. segment_prostate",
                        "4. extract_roi_features (optional if segmentation is enough)",
                        "5. generate_report",
                    ]
                )
            }
            with patch("MRI_Agent.agent.langgraph.loop.PlannerSubagent.run", return_value=mocked_plan):
                dag = plan_agent_dag(
                    goal="Generate a prostate report.",
                    domain="prostate",
                    case_ref=str(case_dir),
                    llm_mode="stub",
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                )
            feat_nodes = [n for n in dag.nodes if n.tool_name == "extract_roi_features"]
            self.assertTrue(feat_nodes)
            self.assertFalse(bool(feat_nodes[0].required))


if __name__ == "__main__":
    unittest.main()
