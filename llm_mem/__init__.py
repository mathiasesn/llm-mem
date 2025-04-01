import logging

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv()

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%d/%m/%Y-%H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

from llm_mem.llm_memory_calculator import LLMMemoryCalculator  # noqa: E402

__all__ = ["LLMMemoryCalculator"]
