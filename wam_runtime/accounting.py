"""M6: per-executed-action compute accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ComputeRecord:
    route: str
    model_ms: float
    vae_calls: int
    dit_forwards: int
    executed_actions: int
    obsolete: bool = False
    used_for_execution: bool = True

    def validate(self) -> None:
        if self.executed_actions < 0 or self.vae_calls < 0 or self.dit_forwards < 0:
            raise ValueError("compute and action counts must be non-negative")
        if self.obsolete and self.used_for_execution:
            raise ValueError("an obsolete computation cannot be marked used")


class ComputeLedger:
    def __init__(self) -> None:
        self._records: list[ComputeRecord] = []

    def append(self, record: ComputeRecord) -> None:
        record.validate()
        self._records.append(record)

    def summary(self) -> dict[str, float | int | list[dict[str, object]]]:
        executed = sum(record.executed_actions for record in self._records)
        model_ms = sum(record.model_ms for record in self._records)
        obsolete_ms = sum(record.model_ms for record in self._records if record.obsolete or not record.used_for_execution)
        return {
            "records": len(self._records),
            "executed_actions": executed,
            "model_ms": model_ms,
            "model_ms_per_executed_action": model_ms / executed if executed else float("nan"),
            "vae_calls": sum(record.vae_calls for record in self._records),
            "dit_forwards": sum(record.dit_forwards for record in self._records),
            "obsolete_model_ms": obsolete_ms,
            "obsolete_fraction": obsolete_ms / model_ms if model_ms else 0.0,
            "records_detail": [asdict(record) for record in self._records],
        }

    def extend(self, records: Iterable[ComputeRecord]) -> None:
        for record in records:
            self.append(record)

