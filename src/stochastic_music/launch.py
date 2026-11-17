"""Entry point of the `stochastic-music` command: opens the web app."""

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli
    except ImportError:
        sys.exit(
            'The web app needs Streamlit. Install it with: pip install "stochastic-music[app]"'
        )

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
