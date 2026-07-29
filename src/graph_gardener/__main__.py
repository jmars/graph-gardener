"""CLI entry point for graph-gardener."""

import argparse
import os
import sys
from pathlib import Path

from .gardener import run


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph Gardener — LLM-powered knowledge graph maintenance"
    )
    parser.add_argument(
        "--memory-file", "-f",
        default=Path.home() / ".vibe" / "memory.jsonl",
        type=Path,
        help="Path to the memory JSONL file (default: ~/.vibe/memory.jsonl)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--apply",
        action="store_true",
        help="Apply mutations (default: dry-run)",
    )
    # --dry-run is the default (absence of --apply)

    parser.add_argument(
        "--api-url",
        default=None,
        help="LLM API base URL (default: GRAPH_GARDENER_API_URL env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: GRAPH_GARDENER_MODEL env var)",
    )
    return parser


def main() -> None:
    parser = _cli()
    args = parser.parse_args()

    memory_path = Path(os.path.expanduser(str(args.memory_file)))

    if not memory_path.is_file():
        print(
            f"ERROR: memory file not found or not a regular file: {memory_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.access(str(memory_path), os.R_OK):
        print(
            f"ERROR: memory file not readable: {memory_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(run(
        memory_path,
        apply=args.apply,
        api_url=args.api_url,
        api_key=None,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
