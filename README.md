# Mean-Reversion Stability in China A-Shares

This repository studies whether mean-reverting relationships between China
A-share stocks remain stable out of sample. The research focuses on the
reproducibility and robustness of statistical relationships across time; it
does **not** yet implement pair selection, portfolio construction, or
backtesting.

## Research pipeline

1. Place source market data in `data/raw/` (never committed).
2. Clean and standardize inputs into `data/intermediate/`.
3. Build analysis-ready datasets in `data/processed/`.
4. Implement reusable research code in `src/` and validate it in `tests/`.
5. Use `notebooks/` only for inspection, diagnostics, and analysis.
6. Write generated tables, figures, and diagnostics to `outputs/`.

Configuration defaults live in `config/baseline.yaml`. Research decisions and
methodological notes belong in `docs/`.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Run the test suite with:

```bash
pytest
```

## Data policy

Raw market data and generated datasets are intentionally excluded from Git.
Only `.gitkeep` placeholders are tracked inside `data/` and `outputs/`. Do not
force-add proprietary, licensed, or large generated data files.
