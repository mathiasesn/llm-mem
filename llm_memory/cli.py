import logging
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

from llm_memory.llm_memory_calculator import LLMMemoryCalculator

logger = logging.getLogger("llm-memory.cli")


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Command line interface for LLM Memory.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to use.",
    )
    parser.add_argument(
        "-d",
        "--data-type",
        type=str,
        choices=["int4", "int8", "float8", "float16", "float32"],
        default="float16",
        help="Specify the data type for the model.",
    )
    parser.add_argument(
        "-c",
        "--context-size",
        type=int,
        default=8192,
        help="Specify the context size for the model.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("Model is required. Use -m or --model to specify the model.")

    return args


def main() -> None:
    args = parse_args()
    logger.info(f"Model: {args.model}")
    logger.info(f"Data Type: {args.data_type}")
    logger.info(f"Context Size: {args.context_size}")

    calculator = LLMMemoryCalculator()
    vram_estimate = calculator.estimate_vram(
        model_id=args.model, dtype=args.data_type, context_size=args.context_size
    )

    logger.info("VRAM Requirements:")
    logger.info(vram_estimate)


if __name__ == "__main__":
    main()
