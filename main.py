#!/usr/bin/env python3
"""Unified task entry. Scientific pipelines live in util and the three core scripts."""
from util.cli import main


if __name__ == "__main__":
    from util.run_context import command_exit_code
    raise SystemExit(command_exit_code(main()))
