# Contributing

Bug reports, feature ideas, documentation fixes, and code changes are welcome.

## Local setup

Use Python 3.10+ and install the project from the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[torch]"
```

## Local gate

Run the following checks before opening a pull request:

```bash
python -m compileall vatix main.py
python -c "import vatix; print(vatix.FM)"
```

## Pull requests

Keep pull requests focused, include a concise description of the change, and describe the validation you ran. For behavior changes, add or update focused tests when practical.