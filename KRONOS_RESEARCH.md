# Kronos research integration

LeanTrader integrates Kronos as an offline/shadow forecaster, never as an order executor.

The adapter follows the official `KronosPredictor.predict` interface with OHLCV input, historical timestamps, future timestamps, prediction length, temperature, top-p and sample count. It validates the returned forecast before exposing an expected return. If the official model is not installed, health and forecast output explicitly report that Kronos is unavailable.

## Installation boundary

Use a separate research environment or container because Kronos introduces PyTorch, Hugging Face model downloads and materially higher memory requirements than the paper runner.

1. Clone the [official Kronos repository](https://github.com/shiyu-coder/Kronos).
2. Install its pinned requirements in the research environment.
3. Put the repository on `PYTHONPATH` so its `model` package is importable.
4. Construct the adapter with `KronosForecastAdapter.from_pretrained()`.
5. Run causal walk-forward evaluation with fees, slippage and calibration.
6. Submit results to the champion/challenger governor. Do not wire raw forecasts directly to execution.

The default model identifier is `NeoQuasar/Kronos-mini`, with `NeoQuasar/Kronos-Tokenizer-2k`. Larger models should be evaluated on separate compute rather than increasing the production VPS footprint.

## Promotion requirements

- No future timestamps or candles may enter the context frame.
- Kronos must outperform the classical adaptive ensemble and gradient-boost baseline after costs.
- Brier score, expected calibration error, maximum drawdown and drift must remain inside configured gates.
- At least five complete out-of-sample windows are required before a shadow promotion decision.
- The previous champion remains available for immediate rollback.
