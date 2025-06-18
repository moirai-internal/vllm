# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest

from tests.models.utils import RerankModelInfo


def ping_pong_test_score_models(hf_runner,
                                vllm_runner,
                                model_info: RerankModelInfo,
                                vllm_extra_kwargs=None,
                                hf_model_callback=None):
    if not model_info.enable_test:
        # A model family has many models with the same architecture,
        # and we don't need to test each one.
        pytest.skip("Skipping test.")

    sentences = []

    vllm_extra_kwargs = vllm_extra_kwargs or {}
    # This test must use float32 to pass.
    vllm_extra_kwargs["dtype"] = "float32"

    with vllm_runner(model_info.name,
                     task="score",
                     max_model_len=None,
                     **vllm_extra_kwargs) as vllm_model:

        max_model_len = vllm_model.model.llm_engine.model_config.max_model_len

        for i in range(0, int(math.log2(max_model_len - 1))):
            sentences.append(("ping", "pong" * 2**i))

        text_1 = [x[0] for x in sentences]
        text_2 = [x[1] for x in sentences]
        vllm_outputs = vllm_model.score(text_1=text_1, text_2=text_2)

    with hf_runner(
            model_info.name,
            dtype="float32",
            is_cross_encoder=True,
    ) as hf_model:

        if hf_model_callback is not None:
            hf_model_callback(hf_model)

        # use batchsize = 1 to avoid oom
        hf_outputs = [
            hf_model.predict([sentences[i]])[0] for i in range(len(sentences))
        ]

    for i in range(len(sentences)):
        assert float(hf_outputs[i]) == pytest.approx(float(vllm_outputs[i]), rel=0.01), \
            f"Test failed at #{i}, vllm: {vllm_outputs[i]}, st: {hf_outputs[i]}"
