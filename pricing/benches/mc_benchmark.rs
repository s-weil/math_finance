// https://florianovictor.medium.com/rust-adventures-criterion-50754cb3295
// https://bheisler.github.io/criterion.rs/book/getting_started.html

extern crate pricing;
use pricing::simulation::monte_carlo::{MonteCarloPathSimulator, PathEvaluator};
use pricing::simulation::GeometricBrownianMotion;

use pricing::simulation::monte_carlo2::{MonteCarloPathSimulator as mcps, PathSampler};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand_distr::Normal;

// criterion_group!{
//     name = benches;
//     config = Criterion::default().measurement_time(std::time::Duration::from_secs(100));
//     target = criterion_stock_price_simulation;
// }
criterion_group!(benches, criterion_stock_price_simulation);
criterion_main!(benches);

pub fn criterion_stock_price_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stock price Monte Carlo simulation");

    group.bench_function(
        "apply a path function to each path directly (do not store paths)",
        |b| b.iter(|| simulate_paths_with_path_fn(black_box((30_000, 200)))),
    );
    group.bench_function("apply a path function on the stored paths", |b| {
        b.iter(|| simulate_paths_with_path_generator(black_box((30_000, 200))))
    });
    group.bench_function("apply a path function on the stored paths NEW", |b| {
        b.iter(|| simulate_paths_with_path_generator_new(black_box((30_000, 200))))
    });
    group.bench_function("apply a path function on the stored paths NEW NEW", |b| {
        b.iter(|| simulate_paths_with_path_generator_new2(black_box((30_000, 200))))
    });

    group.finish()
}

fn simulate_paths_with_path_fn((nr_paths, nr_steps): (usize, usize)) {
    let vola = 50.0 / 365.0;
    let drift = 0.01;
    let dt = 0.1;
    let s0 = 300.0;

    let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
    let paths = mc_simulator.simulate_paths_with(stock_gbm, |path| path.last().cloned());
    let sum = paths
        .iter()
        .fold(0.0, |acc, last_p| acc + last_p.unwrap_or(0.0));
    assert!(sum > 0.0);
}

fn simulate_paths_with_path_generator((nr_paths, nr_steps): (usize, usize)) {
    let vola = 50.0 / 365.0;
    let drift = 0.01;
    let dt = 0.1;
    let s0 = 300.0;

    let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
    let paths = mc_simulator.simulate_paths(stock_gbm);
    let path_eval = PathEvaluator::new(&paths);
    let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
    assert!(avg_price.is_some());
}

fn simulate_paths_with_path_generator_new((nr_paths, nr_steps): (usize, usize)) {
    let vola = 50.0 / 365.0;
    let drift = 0.01;
    let dt = 0.1;
    let s0 = 300.0;

    let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    let mc_simulator = mcps::new(nr_paths, nr_steps);

    let paths =
        mc_simulator.simulate_paths_with(42, rand_distr::StandardNormal, |random_normals| {
            stock_gbm.sample_path2(random_normals)
        });

    let path_eval = PathEvaluator::new(&paths);
    let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
    assert!(avg_price.is_some());
}

fn simulate_paths_with_path_generator_new2((nr_paths, nr_steps): (usize, usize)) {
    let vola = 50.0 / 365.0;
    let drift = 0.01;
    let dt = 0.1;
    let s0 = 300.0;

    let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    let mc_simulator = mcps::new(nr_paths, nr_steps);

    let paths =
        mc_simulator.simulate_paths_with2(42, rand_distr::StandardNormal, |random_normals| {
            stock_gbm.sample_path3(random_normals)
        });

    let path_eval = PathEvaluator::new(&paths);
    let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
    assert!(avg_price.is_some());
}
