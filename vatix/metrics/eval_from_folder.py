"""Compatibility wrapper for folder-based evaluation CLI.

Use ``vatix.scripts.eval_from_folder`` moving forward.
"""

from vatix.scripts.eval_from_folder import main


if __name__ == "__main__":
    raise SystemExit(main())

