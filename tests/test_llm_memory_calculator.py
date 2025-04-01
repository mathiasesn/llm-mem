import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from llm_memory.llm_memory_calculator import LLMMemoryCalculator


class TestLLMMemoryCalculator:
    @pytest.fixture
    def calculator(self):
        return LLMMemoryCalculator()

    def test_init_with_token(self):
        with patch("llm_memory.llm_memory_calculator.login") as mock_login:
            calculator = LLMMemoryCalculator(hf_token="fake_token")
            assert calculator.hf_token == "fake_token"
            mock_login.assert_called_once_with(token="fake_token")

    def test_extract_keys(self, calculator):
        test_json = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [{"f": 4}],
            "g": {"h": {"i": 5}},
        }
        keys_to_extract = {"a", "c", "f", "i"}
        result = calculator.extract_keys(test_json, keys_to_extract)
        expected = {"a": 1, "c": 2, "f": 4, "i": 5}
        assert result == expected

    @patch("llm_memory.llm_memory_calculator.hf_hub_download")
    def test_calculate_kv_cache_memory(self, mock_download, calculator):
        mock_download.return_value = "fake_path"

        # Mock config that would be in the downloaded file
        mock_config = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            result = calculator.calculate_kv_cache_memory(
                context_size=4096, model_id="facebook/opt-1.3b", dtype="float16"
            )
            # Actual calculation:
            # layers * heads * (hidden_size/heads) * bytes * 2 * context_size / 1e9
            # 32 * 32 * (4096/32) * 2 * 2 * 4096 / 1e9 = 2.147483648 GB
            assert isinstance(result, float)
            assert result == pytest.approx(2.147483648, rel=1e-3)

    @patch("llm_memory.llm_memory_calculator.hf_hub_download")
    def test_calculate_kv_cache_memory_with_kv_heads(self, mock_download, calculator):
        mock_download.return_value = "fake_path"

        # Config with separate KV heads value
        mock_config = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # MQA with fewer KV heads
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            result = calculator.calculate_kv_cache_memory(
                context_size=4096, model_id="meta-llama/Llama-2-7b", dtype="float16"
            )
            # Actual: 32 * 8 * (4096/32) * 2 * 2 * 4096 / 1e9 = 0.536870912 GB
            assert isinstance(result, float)
            assert result == pytest.approx(0.536870912, rel=1e-3)

    @patch("llm_memory.llm_memory_calculator.hf_hub_download")
    def test_calculate_kv_cache_memory_error(self, mock_download, calculator):
        mock_download.side_effect = Exception("Network error")

        result = calculator.calculate_kv_cache_memory(
            context_size=4096, model_id="nonexistent/model", dtype="float16"
        )

        assert isinstance(result, str)
        assert "Error: Network error" in result

    def test_calculate_model_memory(self, calculator):
        # Test with float16 and 7B parameters
        parameters = 7.0  # 7B parameters
        result = calculator.calculate_model_memory(parameters, "float16")
        # Actual: (7 * 4) / (32/(2*8)) * 1.18 ≈ 16.52 GB
        assert result == pytest.approx(16.52, abs=0.1)

        # Test with int8 and 13B parameters
        parameters = 13.0  # 13B parameters
        result = calculator.calculate_model_memory(parameters, "int8")
        # For 13B model in int8, actual value
        expected_int8 = round((13 * 4) / (32 / (1 * 8)) * 1.18, 2)
        assert result == pytest.approx(expected_int8, abs=0.1)

        # Test with int4 and 70B parameters
        parameters = 70.0  # 70B parameters
        result = calculator.calculate_model_memory(parameters, "int4")
        # For 70B model in int4, actual value
        expected_int4 = round((70 * 4) / (32 / (0.5 * 8)) * 1.18, 2)
        assert result == pytest.approx(expected_int4, abs=0.1)

    @patch("llm_memory.llm_memory_calculator.get_safetensors_metadata")
    def test_get_model_size(self, mock_metadata, calculator):
        # Mocking metadata for a 7B parameter model
        mock_metadata_obj = MagicMock()
        mock_metadata_obj.parameter_count = {"model.safetensors": "7000000000"}
        mock_metadata.return_value = mock_metadata_obj

        result = calculator.get_model_size("meta-llama/Llama-2-7b", "float16")

        # Should match test_calculate_model_memory for 7B float16
        assert isinstance(result, float)
        assert result == pytest.approx(16.52, abs=0.1)

    @patch("llm_memory.llm_memory_calculator.get_safetensors_metadata")
    def test_get_model_size_error(self, mock_metadata, calculator):
        mock_metadata.side_effect = Exception("API error")

        result = calculator.get_model_size("nonexistent/model", "float16")

        assert isinstance(result, str)
        assert "Error: API error" in result

    @patch.object(LLMMemoryCalculator, "get_model_size")
    @patch.object(LLMMemoryCalculator, "calculate_kv_cache_memory")
    def test_estimate_vram(self, mock_kv_cache, mock_model_size, calculator):
        # Mock the return values for the component methods
        mock_model_size.return_value = 16.52  # 7B model in float16
        mock_kv_cache.return_value = 0.52  # Example KV cache size

        result = calculator.estimate_vram(
            model_id="meta-llama/Llama-2-7b", dtype="float16", context_size=4096
        )

        # Check the correct string format with expected values
        assert "Model VRAM: 16.52 GB" in result
        assert "Context VRAM: 0.52 GB" in result
        assert "Total VRAM: 17.04 GB" in result

    @patch.object(LLMMemoryCalculator, "get_model_size")
    @patch.object(LLMMemoryCalculator, "calculate_kv_cache_memory")
    def test_estimate_vram_error_in_model_size(
        self, mock_kv_cache, mock_model_size, calculator
    ):
        # Mock an error in get_model_size
        mock_model_size.return_value = "Error: Could not fetch metadata."
        mock_kv_cache.return_value = 0.52

        result = calculator.estimate_vram(
            model_id="nonexistent/model", dtype="float16", context_size=4096
        )

        assert result == "Error: Could not fetch metadata."

    @patch.object(LLMMemoryCalculator, "get_model_size")
    @patch.object(LLMMemoryCalculator, "calculate_kv_cache_memory")
    def test_estimate_vram_error_in_kv_cache(
        self, mock_kv_cache, mock_model_size, calculator
    ):
        # Mock an error in calculate_kv_cache_memory
        mock_model_size.return_value = 16.52
        mock_kv_cache.return_value = "Error: Missing config file."

        result = calculator.estimate_vram(
            model_id="meta-llama/Llama-2-7b", dtype="float16", context_size=4096
        )

        assert result == "Error: Missing config file."

    def test_estimate_vram_unsupported_dtype(self, calculator):
        result = calculator.estimate_vram(
            model_id="meta-llama/Llama-2-7b",
            dtype="float64",  # Unsupported dtype
            context_size=4096,
        )

        assert result == "Error: Unsupported dtype"
