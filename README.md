# math_finance
 Mathematical finance in Rust


# flamegraph

Run flamegraph for detecting hot paths.
Running unit tests via workaround in:
https://github.com/flamegraph-rs/flamegraph/issues/139

For example:
```
cargo b --tests --release
flamegraph target/release/deps/pricing-8329770a5e5551d4 -- test-name(e.g. no_drift_stock_price_simulation) [--output ./flamegraph/flamegraph.svg]
```

Find the flamegraph.svg and 'perf.data' at the project's root / flamegraph folder.