"""Standalone model-aware admission helpers for a trusted local llama-server.

Call from the authenticated gateway before proxying any chat completion. Never
forward a request when token admission is unavailable or inconclusive.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


class AdmissionError(Exception):
    """Malformed or context-exceeding client request."""
    status_code = 400
    error_code = 'INVALID_REQUEST'


class ContextExceeded(AdmissionError):
    error_code = 'CONTEXT_LENGTH_EXCEEDED'


class CountingUnavailable(AdmissionError):
    status_code = 503
    error_code = 'TOKEN_COUNT_UNAVAILABLE'


@dataclass(frozen=True)
class AdmissionResult:
    input_tokens: int
    reserved_output_tokens: int
    context_limit: int

    @property
    def remaining_tokens(self) -> int:
        return self.context_limit - self.input_tokens - self.reserved_output_tokens


def effective_output_reserve(payload: Mapping[str, Any], *, default: int = 2048, max_allowed: int = 2048) -> int:
    """Reserve actual request output capacity; reject unbounded / excessive requests."""
    if 'max_tokens' in payload and 'max_completion_tokens' in payload:
        raise AdmissionError('Specify only one completion limit')
    value = payload.get('max_completion_tokens', payload.get('max_tokens', default))
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= max_allowed:
        raise AdmissionError(f'Completion token limit must be an integer between 1 and {max_allowed}')
    return value


def admission_from_count(payload: Mapping[str, Any], count_response: Mapping[str, Any], *, context_limit: int = 65536, output_cap: int = 2048) -> AdmissionResult:
    if not isinstance(payload, Mapping) or not isinstance(payload.get('messages'), list):
        raise AdmissionError('Missing chat messages')
    output_tokens = effective_output_reserve(payload, max_allowed=output_cap)
    count = count_response.get('input_tokens') if isinstance(count_response, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise CountingUnavailable('Upstream returned no valid model-aware input token count')
    result = AdmissionResult(count, output_tokens, context_limit)
    if result.remaining_tokens < 0:
        raise ContextExceeded(f'Input {count} + output reserve {output_tokens} exceeds context {context_limit}')
    return result


def count_using_local_llama_server(payload: Mapping[str, Any], *, url: str = 'http://127.0.0.1:8080/v1/chat/completions/input_tokens', timeout: float = 15.0) -> Mapping[str, Any]:
    """Count the complete request with the same model, template, tools and media.

    URL is a deployment-owned constant, never derived from untrusted client input.
    This helper should run after gateway authentication and body-size validation.
    """
    data = json.dumps(dict(payload), ensure_ascii=False).encode('utf-8')
    request = Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise CountingUnavailable(f'Upstream counter returned HTTP {response.status}')
            answer = json.loads(response.read(1024 * 1024))
    except CountingUnavailable:
        raise
    except (OSError, TimeoutError, ValueError) as exc:
        raise CountingUnavailable('Model-aware token counter unavailable') from exc
    if not isinstance(answer, dict):
        raise CountingUnavailable('Invalid token counter response')
    return answer


def validate_before_proxy(payload: Mapping[str, Any], *, counter: Callable[[Mapping[str, Any]], Mapping[str, Any]] = count_using_local_llama_server, context_limit: int = 65536, output_cap: int = 2048) -> AdmissionResult:
    """Gateway integration hook: call only AFTER authentication and body-size checks."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get('messages'), list):
        raise AdmissionError('Missing chat messages')
    effective_output_reserve(payload, max_allowed=output_cap)
    try:
        count_response = counter(payload)
    except AdmissionError:
        raise
    except Exception as exc:
        raise CountingUnavailable('Unable to count request tokens') from exc
    return admission_from_count(payload, count_response, context_limit=context_limit, output_cap=output_cap)
