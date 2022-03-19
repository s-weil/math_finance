// https://florianovictor.medium.com/rust-adventures-criterion-50754cb3295
// https://bheisler.github.io/criterion.rs/book/getting_started.html

extern crate pricing;
use pricing::simulation::monte_carlo::{MonteCarloPathSimulator, PathEvaluator, PathSampler};
use pricing::simulation::GeometricBrownianMotion;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand_distr::StandardNormal;

// criterion_group!{
//     name = benches;
//     config = Criterion::default().measurement_time(std::time::Duration::from_secs(100));
//     target = criterion_stock_price_simulation;
// }
criterion_group!(benches, criterion_stock_price_simulation);
criterion_main!(benches);

pub fn criterion_stock_price_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stock price Monte Carlo simulation");

    group.bench_function("apply a path function on the stored paths", |b| {
        b.iter(|| simulate_paths_with_path_generator(black_box((30_000, 200))))
    });
    group.bench_function(
        "apply a path function (in place) on the stored paths",
        |b| b.iter(|| simulate_paths_with_path_generator_in_place(black_box((30_000, 200)))),
    );
    group.bench_function("direct gbm sampler", |b| {
        b.iter(|| simulate_paths_with_path_generator_gbm(black_box((30_000, 200))))
    });

    group.finish()
}

fn simulate_paths_with_path_generator((nr_paths, nr_steps): (usize, usize)) {
    let vola = 50.0 / 365.0;
    let drift = 0.01;
    let dt = 0.1;
    let s0 = 300.0;

    let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);

    let paths = mc_simulator.simulate_paths_with(42, StandardNormal, |random_normals| {
        stock_gbm.generate_path(random_normals)
    });

    let path_eval = PathEvaluator::new(&paths);
    let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
    assert!(avg_price.is_some());
}

fn simulate_paths_with_path_generator_in_place((nr_paths, nr_steps): (usize, usize)) {
    let vola = 50.0 / 365.0;
    let drift = 0.01;
    let dt = 0.1;
    let s0 = 300.0;

    let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);

    let paths = mc_simulator.simulate_paths_apply_in_place(42, StandardNormal, |random_normals| {
        stock_gbm.generate_in_place(random_normals)
    });

    let path_eval = PathEvaluator::new(&paths);
    let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
    assert!(avg_price.is_some());
}

fn simulate_paths_with_path_generator_gbm((nr_paths, nr_steps): (usize, usize)) {
    let vola = 50.0 / 365.0;
    let drift = 0.01;
    let dt = 0.1;
    let s0 = 300.0;

    let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);

    let paths = mc_simulator.simulate_paths(42, stock_gbm);

    let path_eval = PathEvaluator::new(&paths);
    let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
    assert!(avg_price.is_some());
}
