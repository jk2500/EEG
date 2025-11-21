#!/usr/bin/env python3
"""
Compatibility wrapper for the streamlined MIB workflow.

All analysis now routes through random channel MIB sampling (broadband and/or
spectral). This file simply forwards to the single supported CLI.
"""

from random_channels_mib import cli_main


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
