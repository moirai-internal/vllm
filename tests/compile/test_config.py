# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm
from vllm.compilation.counter import compilation_counter
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)

from .piecewise.test_simple import SillyModel


def test_use_cudagraphs_dynamic(monkeypatch):
    assert vllm.envs.VLLM_USE_V1
    vllm_config = VllmConfig()
    assert vllm_config.compilation_config.use_cudagraph

    monkeypatch.setenv('VLLM_USE_V1', '0')
    vllm_config = VllmConfig()
    assert not vllm_config.compilation_config.use_cudagraph


@pytest.mark.parametrize("enabled", [True, False])
def test_use_cudagraphs(enabled):
    assert vllm.envs.VLLM_USE_V1
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=enabled,
        cudagraph_capture_sizes=[100],
    ))
    with set_current_vllm_config(vllm_config):
        model = SillyModel(vllm_config=vllm_config, prefix='')

    inputs = torch.randn(100, device="cuda")

    with compilation_counter.expect(
            num_graphs_seen=1,  # one graph for the model
            num_cudagraph_captured=1 if enabled else 0,
    ):
        # first run is warmup
        model(inputs)
        # second run does CUDAGraphs recording (if enabled)
        model(inputs)


def test_dynamo_as_is(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

    with (
            compilation_counter.expect(dynamo_as_is_count=1),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m', compilation_config={"level": 1})
            as _):
        pass


def test_no_compilation(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

    with (
            compilation_counter.expect(num_graphs_seen=0,
                                       dynamo_as_is_count=0),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m', compilation_config={"level": 0})
            as _):
        pass


def test_enforce_eager(vllm_runner, monkeypatch):
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

    with (
            compilation_counter.expect(num_graphs_seen=0,
                                       dynamo_as_is_count=0),
            # loading the model causes compilation (if enabled) to happen
            vllm_runner('facebook/opt-125m', enforce_eager=True) as _):
        pass
