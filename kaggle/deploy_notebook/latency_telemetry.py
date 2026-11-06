"""Timing extraction compatible with llama.cpp completion and SSE payloads."""
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

TIMING_KEYS = ('prompt_ms', 'prompt_per_second', 'prompt_n', 'predicted_ms', 'predicted_per_second', 'predicted_n', 'cache_n')


def extract_timings(response: Mapping[str, Any]) -> dict[str, float | int]:
    """Only accept finite numeric native timings; no fabrication from token counts."""
    import math
    source = response.get('timings', {}) if isinstance(response, Mapping) else {}
    if not isinstance(source, Mapping):
        return {}
    result = {}
    for key in TIMING_KEYS:
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            result[key] = value
    return result


@dataclass
class RequestClock:
    """Client clock records TTFT on first *content token*, not SSE headers.

    For streams, call observe_chunk() for each decoded SSE JSON payload.
    For non-streaming calls, TTFT is unavailable unless measured separately.
    """
    started_at: float = field(default_factory=perf_counter)
    first_content_at: float | None = None
    finished_at: float | None = None
    native_timings: dict[str, float | int] = field(default_factory=dict)

    def observe_chunk(self, chunk: Mapping[str, Any], *, at: float | None = None) -> None:
        stamp = perf_counter() if at is None else at
        choices = chunk.get('choices', []) if isinstance(chunk, Mapping) else []
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get('delta', {})
                if isinstance(delta, Mapping) and (delta.get('content') or delta.get('tool_calls')) and self.first_content_at is None:
                    self.first_content_at = stamp
        timings = extract_timings(chunk)
        if timings:
            self.native_timings.update(timings)

    def observe_response(self, response: Mapping[str, Any]) -> None:
        self.native_timings.update(extract_timings(response))

    def finish(self, *, at: float | None = None) -> None:
        self.finished_at = perf_counter() if at is None else at

    def result(self) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = dict(self.native_timings)
        out['ttft_ms'] = None if self.first_content_at is None else max(0.0, (self.first_content_at - self.started_at) * 1000)
        out['total_ms'] = None if self.finished_at is None else max(0.0, (self.finished_at - self.started_at) * 1000)
        return out
