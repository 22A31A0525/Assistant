import pytest
from kaggle.deploy_notebook.context_admission import (validate_before_proxy, ContextExceeded, CountingUnavailable, AdmissionError, admission_from_count)
from kaggle.deploy_notebook.latency_telemetry import extract_timings, RequestClock
from kaggle.deploy_notebook.runtime_policy import threading_batch_flags, validate_existing_command

PAYLOAD = {'model':'qwen38-27b', 'messages':[{'role':'user','content':'hello'}], 'max_tokens':2048, 'tools':[{'type':'function', 'function':{'name':'echo','parameters':{'type':'object'}}}]}

def test_admission_boundary_pass():
    r = validate_before_proxy(PAYLOAD, counter=lambda p:{'input_tokens':63488})
    assert (r.input_tokens, r.remaining_tokens) == (63488, 0)

def test_admission_overflow():
    with pytest.raises(ContextExceeded):
        validate_before_proxy(PAYLOAD, counter=lambda p:{'input_tokens':63489})

def test_counter_fails_closed():
    with pytest.raises(CountingUnavailable):
        validate_before_proxy(PAYLOAD, counter=lambda p: {})

def test_counter_network_failure_closed():
    with pytest.raises(CountingUnavailable):
        validate_before_proxy(PAYLOAD, counter=lambda p: (_ for _ in ()).throw(OSError('offline')))

def test_invalid_completion_limit():
    with pytest.raises(AdmissionError):
        validate_before_proxy({**PAYLOAD, 'max_tokens':100000}, counter=lambda p:{'input_tokens':12})

def test_count_preserves_full_payload():
    def counter(p):
        assert p['tools'] == PAYLOAD['tools']
        return {'input_tokens':20}
    validate_before_proxy(PAYLOAD, counter=counter)

def test_native_timing_extraction():
    result = extract_timings({'timings':{'prompt_ms':1.2,'predicted_n':6,'predicted_ms':50,'predicted_per_second':120,'prompt_n':99,'prompt_per_second':222,'cache_n':3}})
    assert result['prompt_ms'] == 1.2 and result['cache_n'] == 3

def test_first_content_not_role_header():
    clock = RequestClock(started_at=10)
    clock.observe_chunk({'choices':[{'delta':{'role':'assistant'}}]},at=10.25)
    assert clock.first_content_at is None
    clock.observe_chunk({'choices':[{'delta':{'content':'hi'}}]}, at=10.4)
    clock.finish(at=11)
    assert clock.result()['ttft_ms'] == pytest.approx(400)
    assert clock.result()['total_ms'] == pytest.approx(1000)

def test_nonstream_no_fake_ttft():
    clock = RequestClock(started_at=5)
    clock.observe_response({'timings':{'prompt_ms':10, 'predicted_ms':20}})
    clock.finish(at=6)
    assert clock.result()['ttft_ms'] is None
    assert clock.result()['predicted_ms'] == 20

def test_thread_flags():
    assert threading_batch_flags() == ['--threads','4','--threads-batch','4','--batch-size','2048','--ubatch-size','512']

def test_thread_oversubscription_rejected():
    with pytest.raises(ValueError):
        threading_batch_flags(available_vcpus=2, configured_threads=4)

def test_command_policy():
    validate_existing_command(['llama-server','--n-gpu-layers','99','--split-mode','layer','--parallel','1','--flash-attn','on','--cache-type-k','q8_0','--cache-type-v','q8_0'])
