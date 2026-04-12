"""Entry: `python main.py` (CLI) or `python main.py serve` (HTTP API)."""

import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("serve", "server", "api"):
        from mnemo.server import main as serve_main

        raise SystemExit(serve_main())
    from mnemo.cli import main as cli_main

    raise SystemExit(cli_main())
