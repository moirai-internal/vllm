# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SamplingParams class.
"""

import pytest

from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.sampling_params import GuidedDecodingParams

MODEL_NAME = "Qwen/Qwen1.5-7B"


def test_max_tokens_none():
    """max_tokens=None should be allowed"""
    SamplingParams(temperature=0.01, top_p=0.1, max_tokens=None)


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(
        MODEL_NAME,
        seed=0,
        dtype="float16",
    )


@pytest.fixture(scope="module")
def default_max_tokens():
    return 4096


def test_sampling_params_from_request_with_no_guided_decoding_backend(
        model_config, default_max_tokens):
    # guided_decoding_backend is not present at request level
    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        MODEL_NAME,
        'response_format': {
            'type': 'json_object',
        },
    })

    sampling_params = request.to_sampling_params(
        default_max_tokens,
        model_config.logits_processor_pattern,
    )
    # we do not expect any backend to be present and the default
    # guided_decoding_backend at engine level will be used.
    assert sampling_params.guided_decoding.backend is None


@pytest.mark.parametrize("request_level_guided_decoding_backend,expected",
                         [("xgrammar", "xgrammar"), ("guidance", "guidance"),
                          ("outlines", "outlines")])
def test_sampling_params_from_request_with_guided_decoding_backend(
        request_level_guided_decoding_backend: str, expected: str,
        model_config, default_max_tokens):

    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        MODEL_NAME,
        'response_format': {
            'type': 'json_object',
        },
        'guided_decoding_backend':
        request_level_guided_decoding_backend,
    })

    sampling_params = request.to_sampling_params(
        default_max_tokens,
        model_config.logits_processor_pattern,
    )
    # backend correctly identified in resulting sampling_params
    assert sampling_params.guided_decoding.backend == expected


def test_sampling_params_repr_privacy_protection():
    """Test that SamplingParams.__repr__() protects sensitive guided decoding content."""
    
    # Test JSON schema privacy protection
    sensitive_schema = {
        "name": "analyze_image",
        "description": "Pull out information from image.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "Description": {
                        "type": "string",
                        "description": "Generate a narrative-style description of the image. Example: Photograph of US President Donald Trump and North Korean leader Kim Jong Un in formal attire shaking hands."
                    },
                    "People": {
                        "type": "array",
                        "description": "Identify people in the image. Examples: [\"Iranian Supreme Leader Ali Khamenei\", \"large group of people\"]",
                        "items": {"type": "string"}
                    }
                },
                "required": ["Description", "People"]
            }
        }
    }
    
    guided_params = GuidedDecodingParams(json=sensitive_schema)
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4000,
        guided_decoding=guided_params
    )
    
    params_str = str(sampling_params)
    
    # Check that sensitive content is not in the string representation
    sensitive_phrases = ["Donald Trump", "Kim Jong Un", "Ali Khamenei", "Islamic cleric"]
    for phrase in sensitive_phrases:
        assert phrase not in params_str, f"Sensitive phrase '{phrase}' found in logs!"
    
    # Check that guided decoding type is mentioned but not content
    assert "guided_decoding=GuidedDecodingParams(types=['json'])" in params_str
    assert "narrative-style description" not in params_str


def test_sampling_params_repr_regex_privacy_protection():
    """Test that regex patterns are protected in SamplingParams.__repr__()."""
    
    sensitive_regex = r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    
    guided_params = GuidedDecodingParams(regex=sensitive_regex)
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        guided_decoding=guided_params
    )
    
    params_str = str(sampling_params)
    
    # Check that sensitive regex is not in the logs
    assert sensitive_regex not in params_str
    assert "guided_decoding=GuidedDecodingParams(types=['regex'])" in params_str


def test_sampling_params_repr_choice_privacy_protection():
    """Test that choice lists are protected in SamplingParams.__repr__()."""
    
    sensitive_choices = [
        "confidential_option_1",
        "secret_choice_2", 
        "private_selection_3"
    ]
    
    guided_params = GuidedDecodingParams(choice=sensitive_choices)
    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=50,
        guided_decoding=guided_params
    )
    
    params_str = str(sampling_params)
    
    # Check that sensitive choices are not in the logs
    for choice in sensitive_choices:
        assert choice not in params_str, f"Sensitive choice '{choice}' found in logs!"
    
    assert "guided_decoding=GuidedDecodingParams(types=['choice'])" in params_str


def test_sampling_params_repr_grammar_privacy_protection():
    """Test that grammar definitions are protected in SamplingParams.__repr__()."""
    
    sensitive_grammar = """
    start: expression
    expression: "SELECT" column "FROM" table "WHERE" condition
    column: "id" | "name" | "email" | "password"
    table: "users" | "admin" | "sensitive_data"
    condition: "id" "=" number
    number: /\d+/
    """
    
    guided_params = GuidedDecodingParams(grammar=sensitive_grammar)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
        guided_decoding=guided_params
    )
    
    params_str = str(sampling_params)
    
    # Check that sensitive grammar content is not in the logs
    sensitive_content = ["password", "admin", "sensitive_data", "SELECT", "FROM", "WHERE"]
    for content in sensitive_content:
        assert content not in params_str, f"Sensitive content '{content}' found in logs!"
    
    assert "guided_decoding=GuidedDecodingParams(types=['grammar'])" in params_str


def test_sampling_params_repr_no_guided_decoding():
    """Test that normal sampling params without guided decoding work correctly."""
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=100,
        top_p=0.9
    )
    
    params_str = str(sampling_params)
    
    # Check that guided_decoding=None is in the logs
    assert "guided_decoding=None" in params_str


def test_sampling_params_repr_structural_tag_privacy_protection():
    """Test that structural tag content is protected in SamplingParams.__repr__()."""
    
    sensitive_structural_tag = '{"type": "object", "properties": {"secret_field": {"type": "string", "description": "This contains confidential information"}}}'
    
    guided_params = GuidedDecodingParams(structural_tag=sensitive_structural_tag)
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=150,
        guided_decoding=guided_params
    )
    
    params_str = str(sampling_params)
    
    # Check that sensitive structural tag content is not in the logs
    assert "secret_field" not in params_str
    assert "confidential information" not in params_str
    assert "guided_decoding=GuidedDecodingParams(types=['structural_tag'])" in params_str
