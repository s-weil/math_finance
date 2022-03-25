// TODO: multiple stocks with correlation structure

// https://backtick.se/blog/options-mc-2/
// https://jbhender.github.io/Stats506/F18/GP/Group21.html

// use nalgebra::Cholesky;
// use nalgebra::Matrix;

use std::collections::HashMap;

use ndarray::prelude::*;
use ndarray::Array2;

use crate::common::models::{DerivativeParameter, Underlying};
use crate::simulation::monte_carlo::MonteCarloPathSimulator;
use crate::simulation::monte_carlo::Path;
use crate::simulation::multivariate_gbm::MultivariateGeometricBrownianMotion;
use crate::simulation::PathEvaluator;

/// Indices of cholesky matrix must be aligned with the indices in weights, asset_proces, rf_rates
pub struct MonteCarloBasketOption {
    /// Required for Greeks
    underlying_map: HashMap<Underlying, usize>,

    weights: Array1<f64>,
    asset_prices: Array1<f64>,
    rf_rates: Array1<f64>,
    cholesky_factor: Array2<f64>,

    /// the strike or exercise price of the basket
    strike: f64,
    /// (T - t) in years, where T is the time of the option's expiration and t is the current time
    time_to_expiration: f64,

    mc_simulator: MonteCarloPathSimulator<Array1<f64>>,
    seed_nr: u64,
}

impl MonteCarloBasketOption {
    pub fn new(
        underlying_map: HashMap<Underlying, usize>,
        weights: Array1<f64>,
        asset_prices: Array1<f64>,
        rf_rates: Array1<f64>,
        cholesky_factor: Array2<f64>,
        strike: f64,
        time_to_expiration: f64,

        nr_paths: usize,
        nr_steps: usize,
        seed_nr: u64,
    ) -> Self {
        let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
        Self {
            underlying_map,
            mc_simulator,
            time_to_expiration,
            strike,
            cholesky_factor,
            rf_rates,
            asset_prices,
            weights,
            seed_nr,
        }
    }

    fn dt(&self) -> f64 {
        self.time_to_expiration / self.mc_simulator.nr_steps as f64
    }

    fn sample_payoffs(&self, pay_off: impl Fn(&Path<Array1<f64>>) -> Option<f64>) -> Option<f64> {
        let gbm: MultivariateGeometricBrownianMotion = self.into();

        let paths = self.mc_simulator.simulate_paths(self.seed_nr, gbm);

        let path_evaluator = PathEvaluator::new(&paths);
        path_evaluator.evaluate_average(pay_off)
    }

    fn call_payoff(
        &self,
        strike: f64,
        weights: &Array1<f64>,
        path: &Path<Array1<f64>>,
    ) -> Option<f64> {
        path.last().map(|p| (p.dot(weights) - strike).max(0.0))
    }
}

impl From<&MonteCarloBasketOption> for MultivariateGeometricBrownianMotion {
    fn from(mceo: &MonteCarloBasketOption) -> Self {
        // let drifts = Array1::from(mceo.rf_rates.values().cloned().collect::<Vec<f64>>());
        // let initial_values =
        //     Array1::from(mceo.asset_prices.values().cloned().collect::<Vec<f64>>());

        MultivariateGeometricBrownianMotion::new(
            mceo.asset_prices.to_owned(),
            mceo.rf_rates.to_owned(),
            mceo.cholesky_factor.to_owned(),
            mceo.dt(),
        )
    }
}
