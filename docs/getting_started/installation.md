# Installation

## Requirements

- Python 3.12 or later
- A modern browser (Chrome, Firefox, Edge) for the dashboard

## Install from PyPI

```bash
pip install cropforge
```

## Install from Source

```bash
git clone https://github.com/saswatsundar123/cropforge.git
cd cropforge
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "import cropforge; print(cropforge.__version__)"
# → 0.1.0

pytest tests/
# → 230 passed
```
