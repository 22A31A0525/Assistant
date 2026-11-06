"""A conservative configuration policy; not a replacement for server_runner.py."""
from __future__ import annotations


def threading_batch_flags(*, available_vcpus: int = 4, configured_threads: int = 4) -> list[str]:
    if not isinstance(available_vcpus, int) or available_vcpus <= 0:
        raise ValueError('available_vcpus must be positive')
    if not isinstance(configured_threads, int) or not 1 <= configured_threads <= available_vcpus:
        raise ValueError('configured_threads exceeds available vCPUs')
    return ['--threads', str(configured_threads), '--threads-batch', str(configured_threads), '--batch-size', '2048', '--ubatch-size', '512']


def validate_existing_command(command: list[str]) -> None:
    """Check preservation of critical user-selected settings without rebuilding it."""
    for flag, value in [('--n-gpu-layers', '99'), ('--split-mode', 'layer'), ('--parallel', '1'), ('--flash-attn', 'on'), ('--cache-type-k', 'q8_0'), ('--cache-type-v', 'q8_0')]:
        if flag not in command or command.index(flag) == len(command) - 1 or command[command.index(flag) + 1] != value:
            raise ValueError(f'Expected {flag} {value}')
    if '--no-mmap' in command or '--no-kv-offload' in command:
        raise ValueError('Unexpected memory offload / mapping override')
