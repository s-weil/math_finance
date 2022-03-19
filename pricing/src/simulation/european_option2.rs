use rand_distr::StandardNormal;

use crate::common::models::DerivativeParameter;
use crate::simulation::gbm::GeometricBrownianMotion;
use crate::simulation::monte_carlo2::{MonteCarloPathSimulator, Path, PathEvaluator, PathSampler};

pub struct MonteCarloEuropeanOption {
    option_params: DerivativeParameter,
    mc_simulator: MonteCarloPathSimulator<f64>,
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

    fn call_payoff(&self, path: &Path<f64>) -> Option<f64> {
        path.last()
            .map(|p| (p - self.option_params.strike).max(0.0))
    }

    fn put_payoff(&self, path: &Path<f64>) -> Option<f64> {
        path.last()
            .map(|p| (self.option_params.strike - p).max(0.0))
    }

    fn sample_payoffs(&self, pay_off: impl Fn(&Path<f64>) -> Option<f64>) -> Option<f64> {
        let stock_gbm: GeometricBrownianMotion = self.into();

        let paths =
            self.mc_simulator
                .simulate_paths_with(self.seed_nr, StandardNormal, |random_normals| {
                    stock_gbm.sample_path2(random_normals)
                });

        let path_evaluator = PathEvaluator::new(&paths);
        path_evaluator.evaluate_average(pay_off)
    }

    pub fn call(&self) -> Option<f64> {
        self.sample_payoffs(|path| self.call_payoff(path))
    }

    pub fn put(&self) -> Option<f64> {
        self.sample_payoffs(|path| self.put_payoff(path))
    }
}

impl From<&MonteCarloEuropeanOption> for GeometricBrownianMotion {
    fn from(mceo: &MonteCarloEuropeanOption) -> Self {
        // under the risk neutral measure we have mu = r
        // hence, if rfr = 0.0 then we have no drift term
        let drift = mceo.option_params.rfr;
        GeometricBrownianMotion::new(
            mceo.option_params.asset_price,
            drift,
            mceo.option_params.vola,
            mceo.dt(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    const TOLERANCE: f64 = 1e-1;

    #[test]
    fn european_call_300_2() {
        let mc_option =
            MonteCarloEuropeanOption::new(300.0, 250.0, 1.0, 0.03, 0.15, 100_000, 100, 42);
        let call_price = mc_option.call().unwrap();
        assert_eq!(call_price, 60.77503163606614);
        assert_approx_eq!(call_price, 58.8197, TOLERANCE); // compare with analytic solution
    }

    #[test]
    fn european_put_300_2() {
        let mc_option =
            MonteCarloEuropeanOption::new(300.0, 250.0, 1.0, 0.03, 0.15, 100_000, 100, 42);
        let put_price = mc_option.put();
        assert_approx_eq!(put_price.unwrap(), 1.4311, TOLERANCE); // compare with analytic solution
        assert_eq!(put_price.unwrap(), 1.4311);
    }

    #[test]
    fn european_call_310_2() {
        let mc_option =
            MonteCarloEuropeanOption::new(310.0, 250.0, 3.5, 0.05, 0.25, 200_000, 500, 42);
        let call_price = mc_option.call();
        assert_approx_eq!(call_price.unwrap(), 113.4155, TOLERANCE); // compare with analytic solution
    }
}
