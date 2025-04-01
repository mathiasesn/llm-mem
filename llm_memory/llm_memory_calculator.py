import json

from huggingface_hub import get_safetensors_metadata, hf_hub_download, login


class LLMMemoryCalculator:
    # Dictionary mapping dtype strings to their byte sizes
    bytes_per_dtype: dict[str, float] = {
        "int4": 0.5,
        "int8": 1,
        "float8": 1,
        "float16": 2,
        "float32": 4,
    }

    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        if self.hf_token:
            login(token=self.hf_token)

    @staticmethod
    def extract_keys(json_obj, keys_to_extract):
        extracted_values = {}

        def recursive_search(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in keys_to_extract:
                        extracted_values[key] = value
                    recursive_search(value)
            elif isinstance(obj, list):
                for item in obj:
                    recursive_search(item)

        recursive_search(json_obj)
        return extracted_values

    def calculate_kv_cache_memory(
        self, context_size: int, model_id: str, dtype: str
    ) -> float | str:
        try:
            file_path = hf_hub_download(
                repo_id=model_id, filename="config.json", token=self.hf_token
            )
            with open(file_path, "r") as f:
                config = json.load(f)

            keys_to_find = {
                "num_hidden_layers",
                "num_key_value_heads",
                "hidden_size",
                "num_attention_heads",
            }
            config = self.extract_keys(config, keys_to_find)

            num_layers = config["num_hidden_layers"]
            num_att_heads = config.get(
                "num_key_value_heads", config["num_attention_heads"]
            )
            dim_att_head = config["hidden_size"] // config["num_attention_heads"]
            dtype_bytes = self.bytes_per_dtype[dtype]

            memory_per_token = (
                num_layers * num_att_heads * dim_att_head * dtype_bytes * 2
            )
            context_size_memory_footprint_gb = (
                context_size * memory_per_token
            ) / 1_000_000_000

            return context_size_memory_footprint_gb
        except Exception as e:
            return f"Error: {str(e)}"

    @classmethod
    def calculate_model_memory(cls, parameters: float, dtype: str) -> float:
        bytes_val = cls.bytes_per_dtype[dtype]
        return round((parameters * 4) / (32 / (bytes_val * 8)) * 1.18, 2)

    def get_model_size(self, model_id: str, dtype: str) -> float | str:
        try:
            metadata = get_safetensors_metadata(model_id, token=self.hf_token)
            if not metadata or not metadata.parameter_count:
                return "Error: Could not fetch metadata."
            model_parameters = (
                int(list(metadata.parameter_count.values())[0]) / 1_000_000_000
            )
            return self.calculate_model_memory(model_parameters, dtype)
        except Exception as e:
            return f"Error: {str(e)}"

    def estimate_vram(
        self, model_id: str, dtype: str, context_size: int
    ) -> str | float:
        if dtype not in self.bytes_per_dtype:
            return "Error: Unsupported dtype"

        model_memory = self.get_model_size(model_id, dtype)
        context_memory = self.calculate_kv_cache_memory(context_size, model_id, dtype)

        if isinstance(model_memory, str) or isinstance(context_memory, str):
            return model_memory if isinstance(model_memory, str) else context_memory

        total_memory = model_memory + context_memory
        return (
            f"Model VRAM: {model_memory:.2f} GB\n"
            f"Context VRAM: {context_memory:.2f} GB\n"
            f"Total VRAM: {total_memory:.2f} GB"
        )
