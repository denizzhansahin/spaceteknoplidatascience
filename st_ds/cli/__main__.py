"""Module execution entrypoint supporting `python -m st_ds.cli`."""

from st_ds.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
