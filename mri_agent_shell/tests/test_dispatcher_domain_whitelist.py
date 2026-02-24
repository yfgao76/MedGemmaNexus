from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from MRI_Agent.commands.dispatcher import SchemaValidationError, ToolDispatcher
from MRI_Agent.commands.registry import Tool, ToolRegistry
from MRI_Agent.commands.schemas import ToolCall, ToolContext, ToolSpec


def _mk_tool(name: str) -> Tool:
    def _fn(args: dict, ctx: ToolContext) -> dict:
        return {"data": {"tool": name, "args": dict(args), "run_dir": str(ctx.run_dir)}}

    return Tool(
        spec=ToolSpec(
            name=name,
            description=f"test tool {name}",
            input_schema={"type": "object", "properties": {}},
            output_schema={},
        ),
        func=_fn,
    )


class DispatcherDomainWhitelistTests(unittest.TestCase):
    def test_utility_tools_are_allowed_in_all_clinical_domains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            reg = ToolRegistry()
            reg.register(_mk_tool("rag_search"))
            reg.register(_mk_tool("sandbox_exec"))
            dispatcher = ToolDispatcher(registry=reg, runs_root=runs_root)

            for domain in ("prostate", "brain", "cardiac"):
                state, ctx = dispatcher.create_run(case_id=f"case_{domain}", run_id="run_001")
                state.metadata["domain"] = domain

                rag_res = dispatcher.dispatch(
                    ToolCall(
                        tool_name="rag_search",
                        arguments={},
                        call_id=f"rag_{domain}",
                        case_id=state.case_id,
                    ),
                    state,
                    ctx,
                )
                self.assertTrue(bool(rag_res.ok))

                sand_res = dispatcher.dispatch(
                    ToolCall(
                        tool_name="sandbox_exec",
                        arguments={},
                        call_id=f"sandbox_{domain}",
                        case_id=state.case_id,
                    ),
                    state,
                    ctx,
                )
                self.assertTrue(bool(sand_res.ok))

    def test_domain_specific_tool_remains_blocked_outside_domain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            reg = ToolRegistry()
            reg.register(_mk_tool("detect_lesion_candidates"))
            dispatcher = ToolDispatcher(registry=reg, runs_root=runs_root)

            state, ctx = dispatcher.create_run(case_id="case_brain", run_id="run_001")
            state.metadata["domain"] = "brain"

            with self.assertRaises(SchemaValidationError):
                dispatcher.dispatch(
                    ToolCall(
                        tool_name="detect_lesion_candidates",
                        arguments={},
                        call_id="lesion_brain",
                        case_id=state.case_id,
                    ),
                    state,
                    ctx,
                )


if __name__ == "__main__":
    unittest.main()

