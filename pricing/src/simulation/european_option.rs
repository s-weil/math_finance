use std::collections::HashMap;

use crate::common::models::{DerivativeParameter, ExerciseType, Greek};
use crate::simulation::gbm::GeometricBrownianMotion;
use crate::simulation::monte_carlo::{MonteCarloPathSimulator, PathEvaluator};

use super::greek_engine::Pricer;

pub struct MonteCarloEuropeanOption {
    option_params: DerivativeParameter,
    mc_simulator: MonteCarloPathSimulator<Vec<f64>>,
    seed_nr: u64,
}

impl MonteCarloEuropeanOption {
    pub fn new(
        asset_price: f64,
        strike: f64,
        time_to_expiration: f64,
        rfr: f64,
        vola: f64,
        nr_paths: usize,
        nr_steps: usize,
        seed_nr: u64,
    ) -> Self {
        let option_params =
            DerivativeParameter::new(asset_price, strike, time_to_expiration, rfr, vola);
        let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
        Self {
            option_params,
            mc_simulator,
            seed_nr,
        }
    }

    fn dt(&self) -> f64 {
        self.option_params.time_to_expiration / self.mc_simulator.nr_steps as f64
    }

    fn call_payoff(&self, strike: f64, disc_factor: f64, path: &[f64]) -> Option<f64> {
        path.last().map(|p| (p - strike).max(0.0) * disc_factor)
    }

    fn put_payoff(&self, strike: f64, disc_factor: f64, path: &[f64]) -> Option<f64> {
        path.last().map(|p| (strike - p).max(0.0) * disc_factor)
    }

    pub fn sample_payoffs(&self, pay_off: impl Fn(&Vec<f64>) -> Option<f64>) -> Option<f64> {
        let stock_gbm: GeometricBrownianMotion = self.into();
        let paths = self.mc_simulator.simulate_paths(self.seed_nr, stock_gbm);
        let path_evaluator = PathEvaluator::new(&paths);
        path_evaluator.evaluate_average(pay_off)
    }

    fn discount_factor(&self, t: f64) -> f64 {
        (-t * self.option_params.rfr).exp()
    }

    /// The price (theoretical value) of the standard European call option (optimized version).
    pub fn call(&self) -> Option<f64> {
        let disc_factor = self.discount_factor(self.option_params.time_to_expiration);
        self.sample_payoffs(|path| self.call_payoff(self.option_params.strike, disc_factor, path))
    }

    /// The price (theoretical value) of the standard European put option (optimized version).
    pub fn put(&self) -> Option<f64> {
        let disc_factor = self.discount_factor(self.option_params.time_to_expiration);
        self.sample_payoffs(|path| self.put_payoff(self.option_params.strike, disc_factor, path))
    }

    /// The greeks of the (put / call) option (optimized with respect to TODO).
    pub fn greeks(
        &self,
        _exercise_type: &ExerciseType,
        _greeks: Vec<Greek>,
    ) -> HashMap<Greek, Option<f64>> {
        // let standard_normal_paths = self
        //     .mc_simulator
        //     .simulate_paths(self.seed_nr, StandardNormal);

        // let path_evaluator = PathEvaluator::new(&standard_normal_paths);

        // // let pay_off = match exercise_type {
        // //     ExerciseType::Put => |path: &Path<f64>| self.put_payoff(self.option_params.strike, path),
        // //     ExerciseType::Call => |path: &Path<f64>| self.call_payoff(self.option_params.strike, path),
        // // };

        // let stock_gbm: GeometricBrownianMotion = self.into();

        // let _put_tv = path_evaluator.evaluate(|standard_normal_path| {
        //     let stock_prices =
        //         stock_gbm.generate_path(self.option_params.asset_price, standard_normal_path);
        //     self.put_payoff(self.option_params.strike, 0.0, &stock_prices)
        // });

        todo!("implement");
    }
}

impl From<&MonteCarloEuropeanOption> for GeometricBrownianMotion {
    fn from(mceo: &MonteCarloEuropeanOption) -> Self {
        // under the risk neutral measure we have mu = r
        let drift = mceo.option_params.rfr;
        GeometricBrownianMotion::new(
            mceo.option_params.asset_price,
            drift,
            mceo.option_params.vola,
            mceo.dt(),
        )
    }
}

// pub struct MonteCarloEuropeanputOption {
//     base: MonteCarloEuropeanOption
// }

// impl Pricer<f64, &[f64], Vec<f64>> for MonteCarloEuropeanputOption {
//     fn random_generator(&self) -> Vec<Vec<f64>> {
//         self.base.mc_simulator.simulate_paths(self.base.seed_nr, rand_distr::StandardNormal)
//     }

//     fn eval(&self, paths: &Vec<Vec<f64>>) -> Option<f64> {
//         let path_evaluator = PathEvaluator::new(&paths);
//         let disc_factor = self.base.discount_factor(self.base.option_params.time_to_expiration);
//         let pay_off = |path| self.base.put_payoff(self.base.option_params.strike, disc_factor, path);
//         path_evaluator.evaluate_average(pay_off)
//     }
// }




#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    /// compare with analytic solutions from https://goodcalculators.com/black-scholes-calculator/
    const TOLERANCE: f64 = 1.5;

    #[test]
    fn european_call() {
        let mc_option =
            MonteCarloEuropeanOption::new(300.0, 310.0, 1.0, 0.03, 0.25, 20_000, 1000, 1);
        let call_price = mc_option.call().unwrap();
        assert_eq!(call_price, 29.76722498945371);
        assert_approx_eq!(call_price, 29.47, TOLERANCE);
    }

    #[test]
    fn european_put() {
        let mc_option =
            MonteCarloEuropeanOption::new(300.0, 290.0, 1.0, 0.03, 0.12, 100_000, 100, 42);
        let put_price = mc_option.put().unwrap();
        assert_eq!(put_price, 6.4775539881225335);
        assert_approx_eq!(put_price, 6.547, TOLERANCE);
    }

    /// Reference: https://predictivehacks.com/pricing-of-european-options-with-monte-carlo/
    #[test]
    fn european_put_as_of_reference() {
        let mc_option =
            MonteCarloEuropeanOption::new(102.0, 100.0, 0.5, 0.02, 0.2, 1_000_000, 100, 42);
        let put_price = mc_option.put().unwrap();
        assert_eq!(put_price, 4.2836072940653445); // black scholes ref: 4.293135
        assert_approx_eq!(put_price, 4.294683, TOLERANCE); // monte carlo ref: 4.294683
    }

    /// Reference: https://predictivehacks.com/pricing-of-european-options-with-monte-carlo/
    #[test]
    fn european_call_as_of_reference() {
        let mc_option =
            MonteCarloEuropeanOption::new(102.0, 100.0, 0.5, 0.02, 0.2, 1_000_000, 100, 111111);
        let call_price = mc_option.call().unwrap();
        assert_eq!(call_price, 7.285406206467689); // black scholes ref: 7.288151
        assert_approx_eq!(call_price, 7.290738, TOLERANCE); // monte carlo ref: 7.290738
    }
}
