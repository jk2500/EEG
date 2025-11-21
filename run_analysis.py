#!/usr/bin/env python3
"""
Legacy entrypoint routed to the unified random-channel MIB runner.
Use this file the same way you would call `python random_channels_mib.py`.
"""

from random_channels_mib import cli_main


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
