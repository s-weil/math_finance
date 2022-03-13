use crate::common::models::DerivativeParameter;
use crate::simulation::gbm::GeometricBrownianMotion;
use crate::simulation::monte_carlo::{MonteCarloPathSimulator, Path};

pub struct MonteCarloEuropeanOption {
    option_params: DerivativeParameter,
    mc_simulator: MonteCarloPathSimulator,
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
    ) -> Self {
        let option_params =
            DerivativeParameter::new(asset_price, strike, time_to_expiration, rfr, vola);
        let mc_simulator = MonteCarloPathSimulator { nr_steps, nr_paths };
        Self {
            option_params,
            mc_simulator,
        }
    }

    fn dt(&self) -> f64 {
        self.option_params.time_to_expiration / self.mc_simulator.nr_steps as f64
    }

    fn call_payoff(&self, path: Path) -> Option<f64> {
        path.last()
            .map(|p| (p - self.option_params.strike).max(0.0))
    }

    fn put_payoff(&self, path: Path) -> Option<f64> {
        path.last()
            .map(|p| (self.option_params.strike - p).max(0.0))
    }

    fn sample_payoffs(&self, path_fn: impl Fn(Path) -> Option<f64>) -> Vec<Option<f64>> {
        let gbm_generator: crate::simulation::gbm::GeometricBrownianMotion = self.into();
        self.mc_simulator.simulate_paths_with(
            gbm_generator,
            self.option_params.asset_price,
            path_fn,
        )
    }

    fn average(&self, payoffs: &[Option<f64>]) -> Option<f64> {
        let total = payoffs.iter().fold(0.0, |acc, po| acc + po.unwrap_or(0.0));
        Some(total / payoffs.len() as f64)
    }

    pub fn call(&self) -> Option<f64> {
        let payoff = |path| self.call_payoff(path);
        let sample_payoffs = self.sample_payoffs(payoff);
        self.average(&sample_payoffs)
    }

    pub fn put(&self) -> Option<f64> {
        let payoff = |path| self.put_payoff(path);
        let sample_payoffs = self.sample_payoffs(payoff);
        self.average(&sample_payoffs)
    }
}

impl From<&MonteCarloEuropeanOption> for GeometricBrownianMotion {
    fn from(mceo: &MonteCarloEuropeanOption) -> Self {
        // under the risk neutral measure we have mu = r
        // hence, if rfr = 0.0 then we have no drift term
        let drift = mceo.option_params.rfr;
        GeometricBrownianMotion::new(drift, mceo.option_params.vola, mceo.dt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    const TOLERANCE: f64 = 1e-1;

    #[test]
    fn european_call_300() {
        let mc_option = MonteCarloEuropeanOption::new(300.0, 250.0, 1.0, 0.03, 0.15, 500_000, 100);
        let call_price = mc_option.call();
        assert_approx_eq!(call_price.unwrap(), 58.8197, TOLERANCE); // compare with analytic solution
    }

    #[test]
    fn european_call_310() {
        let mc_option = MonteCarloEuropeanOption::new(310.0, 250.0, 3.5, 0.05, 0.25, 100_000, 100);
        let call_price = mc_option.call();
        assert_approx_eq!(call_price.unwrap(), 113.4155, TOLERANCE); // compare with analytic solution
    }
}
