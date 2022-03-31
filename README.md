# math_finance

Mathematical finance in Rust

## Models

### Option pricing

    - Analytic
    - Monte Carlo
        - GBM
        [*] local vol
        [*] stochastic local vol

### Risk and Portfolio theo

    - risk figur
    [*] Markovi
    [*] Black Litterman
    [*] Deep Hedging

### Stats and Timeseries analysis

[*] planned.

## Contributions

Any contribution and help is highly welcome! Work needs to be done in general and in particular

- further model implementations
- code and design improvements
- unit testing: correctness, regression, performance
- documentation, references, example implementations
  (and many more areas)

## flamegraph

Run flamegraph for detecting hot paths.
Running unit tests via workaround in:
https://github.com/flamegraph-rs/flamegraph/issues/139

For example:

```
cargo b --tests --release
flamegraph target/release/deps/pricing-8329770a5e5551d4 -- test-name(e.g. no_drift_stock_price_simulation) [--output ./flamegraph/flamegraph.svg]
```

Find the flamegraph.svg and 'perf.data' at the project's root / flamegraph folder.
