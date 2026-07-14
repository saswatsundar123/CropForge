# Installation

## Requirements

- Python 3.12 or newer
- Windows, macOS, or Linux
- A terminal with `pip`

## Install From PyPI

```bash
pip install cropforge
```

For GLB export support:

```bash
pip install "cropforge[export]"
```

For documentation or development work:

```bash
pip install "cropforge[docs]"
pip install "cropforge[dev,export]"
```

## Install From Source

```bash
git clone https://github.com/saswatsundar123/cropforge.git
cd cropforge
pip install -e ".[dev,export,docs]"
```

## Verify Installation

```bash
python -c "import cropforge; print(cropforge.__version__)"
# -> 1.0.0

pytest
# -> 879 passed, 1 skipped
```

## First Run

Start with the self-contained [Quickstart](quickstart.md). It does not require
CSV files and runs a small wheat simulation from Python code alone.
