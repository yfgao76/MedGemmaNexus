from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Return the MRI_Agent project root.

    This repo is sometimes nested under a larger workspace. We prefer the nearest
    parent that contains config/skills.json, falling back to a nested MRI_Agent
    directory if needed.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "config" / "skills.json").exists():
            return parent
        nested = parent / "MRI_Agent" / "config" / "skills.json"
        if nested.exists():
            return parent / "MRI_Agent"
    # Fallback: assume we're in MRI_Agent/core
    return here.parents[1]
