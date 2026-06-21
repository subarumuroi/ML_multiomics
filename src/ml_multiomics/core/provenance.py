"""
provenance.py
=============
``ProvenanceTrail`` — an ordered, structured record of everything done to the
data, so a reader can follow EXACTLY what happened and reproduce it without AI.

Every preprocessing step and model fit appends a :class:`ProvenanceStep`
(step name, params, input/output shapes, whether params were fit on the training
fold only, a free-text note). The trail renders to markdown for the report and to
plain records for a JSON sidecar.

This is the data-side half of the package's self-documentation contract; the
method-side half (``describe`` / ``assumptions`` / ``divergences``) lives on
``BaseMethod``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _shape(obj) -> Optional[list]:
    """Best-effort (rows, cols) as a JSON-friendly list, else None."""
    shp = getattr(obj, "shape", None)
    if shp is None:
        try:
            return [len(obj)]
        except TypeError:
            return None
    return list(shp)


@dataclass
class ProvenanceStep:
    """One recorded action on the data."""

    step: str
    params: dict = field(default_factory=dict)
    in_shape: Optional[list] = None
    out_shape: Optional[list] = None
    fit_on_train: bool = False
    note: str = ""

    def to_record(self) -> dict:
        return {
            "step": self.step,
            "params": self.params,
            "in_shape": self.in_shape,
            "out_shape": self.out_shape,
            "fit_on_train": self.fit_on_train,
            "note": self.note,
        }


class ProvenanceTrail:
    """An ordered list of :class:`ProvenanceStep` with rendering helpers."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.steps: list[ProvenanceStep] = []

    def record(
        self,
        step: str,
        params: Optional[dict] = None,
        in_obj: Any = None,
        out_obj: Any = None,
        fit_on_train: bool = False,
        note: str = "",
    ) -> "ProvenanceTrail":
        """Append a step. ``in_obj``/``out_obj`` are inspected for their shape."""
        self.steps.append(
            ProvenanceStep(
                step=step,
                params=dict(params or {}),
                in_shape=_shape(in_obj) if in_obj is not None else None,
                out_shape=_shape(out_obj) if out_obj is not None else None,
                fit_on_train=fit_on_train,
                note=note,
            )
        )
        return self

    def extend(self, other: "ProvenanceTrail") -> "ProvenanceTrail":
        self.steps.extend(other.steps)
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    # -- rendering ---------------------------------------------------------
    def to_records(self) -> list[dict]:
        """JSON-serialisable list of step records."""
        return [s.to_record() for s in self.steps]

    def to_markdown(self) -> str:
        """A compact markdown table of the trail."""
        if not self.steps:
            return "_(no steps recorded)_"
        header = f"**Provenance{f' - {self.name}' if self.name else ''}**\n\n"
        header += "| # | step | params | in | out | fit-on-train | note |\n"
        header += "|---|------|--------|----|-----|--------------|------|\n"
        rows = []
        for i, s in enumerate(self.steps, 1):
            params = ", ".join(f"{k}={v}" for k, v in s.params.items()) or "-"
            in_s = "x".join(map(str, s.in_shape)) if s.in_shape else "-"
            out_s = "x".join(map(str, s.out_shape)) if s.out_shape else "-"
            rows.append(
                f"| {i} | {s.step} | {params} | {in_s} | {out_s} | "
                f"{'yes' if s.fit_on_train else 'no'} | {s.note} |"
            )
        return header + "\n".join(rows)
