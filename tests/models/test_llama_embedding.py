"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_llama_embedding.py`.
"""
import pytest
import torch
import torch.nn.functional as F

MODELS = [
    "intfloat/e5-mistral-7b-instruct",
]


def compare_embeddings(embeddings1, embeddings2):
    print(torch.tensor(embeddings1).shape)
    print(torch.tensor(embeddings2).shape)
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0)
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.skip(
    "Two problems: 1. Failing correctness tests. 2. RuntimeError: expected "
    "scalar type BFloat16 but found Half (only in CI).")
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.encode(example_prompts)
    del hf_model

    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             embedding_mode=True,
                             max_model_len=32768)
    vllm_outputs = vllm_model.encode(example_prompts)
    del vllm_model

    similarities = compare_embeddings(hf_outputs, vllm_outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
