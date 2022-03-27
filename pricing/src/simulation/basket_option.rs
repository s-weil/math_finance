// TODO: multiple stocks with correlation structure

// https://backtick.se/blog/options-mc-2/
// https://jbhender.github.io/Stats506/F18/GP/Group21.html

// use nalgebra::Cholesky;
// use nalgebra::Matrix;

use ndarray::prelude::*;
use ndarray::Array2;

use crate::common::models::Underlying;
use crate::simulation::monte_carlo::MonteCarloPathSimulator;
use crate::simulation::multivariate_gbm::MultivariateGeometricBrownianMotion;
use crate::simulation::PathEvaluator;

/// Indices of cholesky matrix must be aligned with the indices in weights, asset_proces, rf_rates
pub struct MonteCarloEuropeanBasketOption {
    /// Required for Greeks
    // underlying_map: HashMap<Underlying, usize>,
    weights: Array1<f64>,
    asset_prices: Array1<f64>,
    rf_rates: Array1<f64>,
    cholesky_factor: Array2<f64>,

    /// the strike or exercise price of the basket
    strike: f64,
    /// (T - t) in years, where T is the time of the option's expiration and t is the current time
    time_to_expiration: f64,

    mc_simulator: MonteCarloPathSimulator<Array2<f64>>,
    seed_nr: u64,
}

impl MonteCarloEuropeanBasketOption {
    pub fn new(
        // underlying_map: HashMap<Underlying, usize>,
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
        let weight_sum = weights.iter().fold(0.0, |acc, c| acc + c);
        assert_eq!(weight_sum, 1.0);
        let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
        Self {
            // underlying_map,
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

    fn sample_payoffs(&self, pay_off: impl Fn(&Array2<f64>) -> Option<f64>) -> Option<f64> {
        dbg!("creating gbm");
        let gbm: MultivariateGeometricBrownianMotion = self.into();

        dbg!("creating paths");
        let paths = self.mc_simulator.simulate_paths(self.seed_nr, gbm);

        dbg!("eval'ing paths");
        let path_evaluator = PathEvaluator::new(&paths);
        path_evaluator.evaluate_average(pay_off)
    }

    fn call_payoff(
        &self,
        strike: f64,
        weights: &Array1<f64>,
        disc_factor: f64,
        path: &Array2<f64>,
    ) -> Option<f64> {
        path.axis_iter(Axis(0))
            .last()
            .map(|p| (p.dot(weights) - strike).max(0.0) * disc_factor)
    }

    fn put_payoff(
        &self,
        strike: f64,
        weights: &Array1<f64>,
        disc_factor: f64,
        path: &Array2<f64>,
    ) -> Option<f64> {
        path.axis_iter(Axis(0))
            .last()
            .map(|p| (strike - p.dot(weights)).max(0.0) * disc_factor)
    }

    fn discount_factor(&self, t: f64) -> f64 {
        (-t * self.rf_rates.dot(&self.weights)).exp()
    }

    /// The price (theoretical value) of the standard European call option (optimized version).
    pub fn call(&self) -> Option<f64> {
        let disc_factor = self.discount_factor(self.time_to_expiration);
        self.sample_payoffs(|path| self.call_payoff(self.strike, &self.weights, disc_factor, path))
    }

    /// The price (theoretical value) of the standard European put option (optimized version).
    pub fn put(&self) -> Option<f64> {
        let disc_factor = self.discount_factor(self.time_to_expiration);
        self.sample_payoffs(|path| self.put_payoff(self.strike, &self.weights, disc_factor, path))
    }
}

impl From<&MonteCarloEuropeanBasketOption> for MultivariateGeometricBrownianMotion {
    fn from(mceo: &MonteCarloEuropeanBasketOption) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    /// compare with analytic solutions from https://goodcalculators.com/black-scholes-calculator/
    const TOLERANCE: f64 = 1e-1;

    #[test]
    fn european_basket_call() {
        let asset_prices = arr1(&[40.0, 60.0, 100.0]);
        let rfrs = arr1(&[0.01, 0.02, -0.01]);
        let cholesky_factor = arr2(&[[1.0, 0.05, 0.1], [0.0, 0.06, 0.17], [0.0, 0.0, 0.8]]);
        let weights = arr1(&[0.25, 0.25, 0.5]);

        let mc_option = MonteCarloEuropeanBasketOption::new(
            weights,
            asset_prices,
            rfrs,
            cholesky_factor,
            230.0,
            2.0,
            10_000,
            300,
            42,
        );
        let call_price = mc_option.call().unwrap();
        dbg!(call_price);
        assert!(call_price == 0.0);
        // assert_eq!(call_price, 5.59601793502129);
        // assert_approx_eq!(call_price, 29.47, TOLERANCE);
    }

    #[test]
    fn european_basket_call_iid() {
        let asset_prices = arr1(&[102.0, 102.0]);
        let rfrs = arr1(&[0.02, 0.02]);
        let weights = arr1(&[0.5, 0.5]);

        // no correlation between assets
        let cholesky_factor = arr2(&[[0.2, 0.0], [0.0, 0.2]]);

        let mc_option = MonteCarloEuropeanBasketOption::new(
            weights,
            asset_prices,
            rfrs,
            cholesky_factor,
            100.0,
            0.5,
            10_000,
            100,
            42,
        );
        let call_price = mc_option.call().unwrap();
        dbg!(&call_price);
        // assert_approx_eq!(call_price, 7.290738, TOLERANCE);
    }

    #[test]
    fn european_basket_put() {
        let asset_prices = arr1(&[50.0, 60.0, 100.0]);
        let rfrs = arr1(&[0.01, 0.02, -0.01]);
        let cholesky_factor = arr2(&[[1.0, 0.05, 0.1], [0.0, 0.06, 0.17], [0.0, 0.0, 0.8]]);
        let weights = arr1(&[0.25, 0.25, 0.5]);

        let mc_option = MonteCarloEuropeanBasketOption::new(
            weights,
            asset_prices,
            rfrs,
            cholesky_factor,
            180.0,
            2.0,
            10_000,
            300,
            42,
        );
        let call_price = mc_option.put().unwrap();
        assert_eq!(call_price, 8.96589328828396);
        // assert_approx_eq!(call_price, 29.47, TOLERANCE);
    }

    /// https://predictivehacks.com/pricing-of-european-options-with-monte-carlo/
    /// Example from https://ch.mathworks.com/help/fininst/basketsensbyls.html
    #[test]
    fn european_basket_put_reference() {
        let corr = arr2(&[[1.0, 0.15], [0.15, 1.0]]);

        // todo: check cholesky of corr rather than cov?
        let cholesky_factor = arr2(&[[1.0, 0.15], [0.0, 1.0 - 0.15_f64.powi(2)]]);

        let asset_prices = arr1(&[90.0, 75.0]);
        let rfrs = arr1(&[0.05, 0.05]);
        let weights = arr1(&[0.5, 0.5]);

        let mc_option = MonteCarloEuropeanBasketOption::new(
            weights,
            asset_prices,
            rfrs,
            cholesky_factor,
            80.0,
            1.0,
            10_000,
            300,
            42,
        );

        // PriceSens = 0.9822
        // Delta = -0.0995

        let call_price = mc_option.put().unwrap();
        assert_eq!(call_price, 0.9822);
        // assert_approx_eq!(call_price, 29.47, TOLERANCE);
    }
}

// % Define RateSpec
// Rate = 0.05;
// Compounding = -1;
// RateSpec = intenvset('ValuationDate', Settle, 'StartDates',...
// Settle, 'EndDates', Maturity, 'Rates', Rate, 'Compounding', Compounding);

// % Define the Correlation matrix. Correlation matrices are symmetric,
// % and have ones along the main diagonal.
// NumInst  = 2;
// InstIdx = ones(NumInst,1);
// Corr = diag(ones(NumInst,1), 0);

// % Define BasketStockSpec
// Volatility = 0.15;
// Quantity = [0.50; 0.50];
// BasketStockSpec = basketstockspec(Volatility, AssetPrice, Quantity, Corr);

// % Compute the price of the put basket option. Calculate also the delta
// % of the first stock.
// OptSpec = {'put'};
// Strike = 80;
// OutSpec = {'Price','Delta'};
// UndIdx = 1; % First element in the basket

// [PriceSens, Delta] = basketsensbyls(RateSpec, BasketStockSpec, OptSpec,...
// Strike, Settle, Maturity,'OutSpec', OutSpec,'UndIdx', UndIdx)
