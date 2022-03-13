// TODO: multiple stocks with correlation structure

// https://backtick.se/blog/options-mc-2/
// https://jbhender.github.io/Stats506/F18/GP/Group21.html

// use nalgebra::Cholesky;
// use nalgebra::Matrix;

use crate::common::models::DerivativeParameter;

use crate::simulation::monte_carlo::{MonteCarloPathSimulator};

pub struct MonteCarloBasketOption {
    option_params: DerivativeParameter,
    mc_simulator: MonteCarloPathSimulator,
}

impl MonteCarloBasketOption {
    pub fn new(
        asset_price: f64,
        strike: f64,
        time_to_expiration: f64,
        rfr: f64,
        vola: f64,
        nr_paths: usize,
        nr_steps: usize,
    ) -> Self {
        let option_params =
            DerivativeParameter::new(asset_price, strike, time_to_expiration, rfr, vola);
        let mc_simulator = MonteCarloPathSimulator { nr_steps, nr_paths };
        Self {
            option_params,
            mc_simulator,
        }
    }
}
