#!/usr/bin/env python3
"""Entry point for the standard-dataset vLLM reproduction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.benchmark.vllm import main

if __name__ == "__main__":
    main()
