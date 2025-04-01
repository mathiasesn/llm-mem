[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Quality Check](https://github.com/mathiasesn/llm-mem/actions/workflows/check.yaml/badge.svg)](https://github.com/mathiasesn/llm-mem/actions/workflows/check.yaml)
[![Unit Tests](https://github.com/mathiasesn/llm-mem/actions/workflows/test.yaml/badge.svg)](https://github.com/mathiasesn/llm-mem/actions/workflows/test.yaml)
![code coverage](https://raw.githubusercontent.com/mathiasesn/llm-mem/coverage-badge/coverage.svg?raw=true)
[![Publish](https://github.com/mathiasesn/llm-mem/actions/workflows/publish.yaml/badge.svg)](https://github.com/mathiasesn/llm-mem/actions/workflows/publish.yaml)

# llm-mem

llm-mem is a lightweight Python CLI tool designed to calculate the GPU VRAM requirements for models on Hugging Face. It estimates the memory usage based on the model parameters, selected data type, and desired context length. This tool is ideal for developers and researchers looking to optimize model deployment and resource allocation.

## Features

 - GPU VRAM Estimation: Calculate the VRAM needed for both the model and its key-value cache based on the provided context size.
 - Support for Multiple Data Types: Choose from various data types (int4, int8, float8, float16, float32).
 - Hugging Face Integration: Automatically fetch model configuration and metadata from Hugging Face repositories.
 - Command-Line Interface: Simple and intuitive CLI for quick usage without any additional UI dependencies.

## Usage

You can use the llm-mem command from your terminal. The CLI accepts the model ID, data type, and context size as arguments.

### Command-Line Arguments

 - -m or --model: (Required) The Hugging Face model ID.
 - -d or --data-type: The data type for the model (choices: int4, int8, float8, float16, float32). Default is float16.
 - -c or --context-size: The context size for the model. Default is 8192.

### Example

```shell
uvx llm-mem -m google/gemma-3-27b-it -d float16 -c 128000
```

The above command will display an estimation of the model VRAM, context VRAM, and the total VRAM required.

## Installation

You can install the package using Hatchling or your preferred Python package installer if the package is published on PyPI.

```shell
# Using pip (if published on PyPI)
uv venv
source .venv/bin/activate
uv pip install llm-mem

# Or clone the repository and install locally
git clone https://github.com/mathiasesn/llm-mem.git
cd llm-mem
pip install .
```

## Directory Structure

```
├── README.md
├── llm_memory
│   ├── __init__.py
│   ├── cli.py
│   └── llm_memory_calculator.py
└── pyproject.toml
```

## Development

If you wish to contribute or modify the package:

Clone the repository

```shell
git clone https://github.com/mathiasesn/llm-mem.git
cd llm-mem
```

Install development dependencies

```shell
uv venv
uv pip install -e .
```

Run tests

```
uv run pytest
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
