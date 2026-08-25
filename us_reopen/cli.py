"""Command-line entry point for the US reopening experiments."""

from __future__ import annotations

import sys

from us_reopen.model import main as run_model


def main() -> None:
    run_model(sys.argv[1:])


if __name__ == "__main__":
    main()
