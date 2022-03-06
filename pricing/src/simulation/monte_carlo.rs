use rand::{self, prelude::ThreadRng};
use rand_distr::{DistIter, Distribution};
use crate::simulation::gbm::GeometricBrownianMotion;

pub trait PathGenerator {
    type Dist;

    fn sample_path(
        &self,
        current: f64,
        nr_steps: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> Vec<f64>;

    fn distribution<'a>(
        &self,
        rng: &'a mut ThreadRng,
    ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64>;
}

pub trait SampleGenerator {
    type Dist: Distribution<f64>;

    fn sample(&self, current: f64, random_nr: f64) -> f64;

    fn distribution<'a>(
        &self,
        // rng: &'a mut ThreadRng,
    ) -> Self::Dist; // DistIter<Self::Dist, &'a mut ThreadRng, f64>;
}

// pub struct MonteCarloSamples {
//     nr_samples: usize
// }

// impl MonteCarloSamples {
//     pub fn new(nr_samples: usize) -> Self {
//         Self { nr_samples }
//     }

//     pub fn simulate(&self, generator: impl SampleGenerator, initial_value: f64) -> Vec<f64> {
//         let mut rng = rand::thread_rng();
//         let mut simulations = vec![];

//         for rnd_nr in generator.distribution().sample_iter(rng).choose_multiple(&mut rng, self.nr_samples) {
//             let res = generator.sample(initial_value, rnd_nr);
//             simulations.push(res);
//         }

//         simulations
//     }
// }

pub struct MonteCarloPathSimulator {
    nr_paths: usize,
    nr_steps: usize,
}

impl MonteCarloPathSimulator {
    pub fn new(nr_paths: usize, nr_steps: usize) -> Self {
        Self { nr_paths, nr_steps }
    }

    pub fn simulate_paths(
        &self,
        generator: impl PathGenerator,
        initial_value: f64,
    ) -> Vec<Vec<f64>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut rng = rand::thread_rng();

        for _ in 0..self.nr_paths {
            // TODO: try to pass it as ref
            // maybe parallelize if it's a hotpath
            let distr = generator.distribution(&mut rng);
            let path = generator.sample_path(initial_value, self.nr_steps, distr);
            paths.push(path);
        }
        paths
    }
}

pub type Path = Vec<f64>;

pub struct PathEvaluator<'a> {
    paths: &'a Vec<Path>,
}

impl<'a> PathEvaluator<'a> {

    pub fn new(paths: &'a Vec<Path>) -> Self {
        Self { paths }
    }

    pub fn evaluate(&self, path_fn: impl Fn(&'a Path) -> Option<f64>) -> Vec<Option<f64>> {
        self.paths.iter().map(path_fn).collect()
    }

    pub fn evaluate_average(&self, path_fn: impl Fn(&'a Path) -> Option<f64>) -> Option<f64> {
        if self.paths.is_empty() {
            return None;
        }
        if let Some(total) = self.paths.iter().fold(None, |acc, path| {
            if let Some(path_value) = path_fn(path) {
                Some(acc.unwrap_or(0.0) + path_value)
            } else {
                acc
            }
        }) {
            return Some(total / self.paths.len() as f64);
        };
        None
    }
}


pub struct DerivativeParameter {
    /// the asset's price at time t
    pub asset_price: f64,
    /// the strike or exercise price of the asset
    pub strike: f64,
    /// (T - t) in years, where T is the time of the option's expiration and t is the current time
    pub time_to_expiration: f64,
    /// the annualized risk-free interest rate
    pub rfr: f64,
    /// the annualized standard deviation of the stock's returns
    pub vola: f64,
}

impl DerivativeParameter {
    pub fn new(
        asset_price: f64,
        strike: f64,
        time_to_expiration: f64,
        rfr: f64,
        vola: f64,
    ) -> Self {
        Self {
            asset_price,
            strike,
            time_to_expiration,
            rfr,
            vola,
        }
    }
}

pub struct MonteCarloEuropeanOption {
    option_params: DerivativeParameter,
    mc_simulator: MonteCarloPathSimulator
}

impl MonteCarloEuropeanOption {

    pub fn new(asset_price: f64,
        strike: f64,
        time_to_expiration: f64,
        rfr: f64,
        vola: f64,
        nr_paths: usize,
        nr_steps: usize) -> Self {
            let option_params = DerivativeParameter::new(asset_price, strike, time_to_expiration, rfr, vola);
            let mc_simulator = MonteCarloPathSimulator { nr_steps, nr_paths };
            Self { option_params, mc_simulator }
        }

    fn dt(&self) -> f64 {
        self.option_params.time_to_expiration / self.mc_simulator.nr_steps as f64
    }

    fn call_payoff(&self, path: &Path) -> Option<f64> {
        path.last().map(|p| {
            (p - self.option_params.strike).max(0.0)
        })
    }

    fn put_payoff(&self, path: &Path) -> Option<f64> {
        path.last().map(|p| {
            (self.option_params.strike - p).max(0.0)
        })
    }

    fn create_paths(&self) -> Vec<Path> {
        let gbm_generator : crate::simulation::gbm::GeometricBrownianMotion = self.into();
        let paths = self.mc_simulator.simulate_paths(gbm_generator, self.option_params.asset_price);
        paths
    }

    pub fn call(&self) -> Option<f64> {
        let paths = self.create_paths();
        let path_eval = PathEvaluator::new(&paths);
        let payoff= |path| self.call_payoff(path);
        path_eval.evaluate_average(payoff)
    }

    pub fn put(&self) -> Option<f64> {
        let paths = self.create_paths();
        let path_eval = PathEvaluator::new(&paths);
        let payoff= |path| self.put_payoff(path);
        path_eval.evaluate_average(payoff)
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
    use std::vec;

    use super::*;
    use crate::simulation::gbm::GeometricBrownianMotion;
    use rand_distr::{DistIter, Distribution, Normal};

    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    const TOLERANCE: f64 = 1e-1;


    pub struct NormalPathGenerator;

    impl PathGenerator for NormalPathGenerator {
        type Dist = Normal<f64>;

        fn sample_path(
            &self,
            _price: f64,
            nr_steps: usize,
            dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
        ) -> Vec<f64> {
            dist_iter.take(nr_steps).collect()
        }

        fn distribution<'a>(
            &self,
            rng: &'a mut ThreadRng,
        ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64> {
            Normal::new(0.5, 1.0).unwrap().sample_iter(rng)
        }
    }

    #[test]
    fn normal_path_simulation() {
        let normal_gen = NormalPathGenerator;

        let mc_simulator = MonteCarloPathSimulator::new(50_000, 100);

        let paths_slice: Vec<Vec<f64>> = mc_simulator
            .simulate_paths(normal_gen, 0.0)
            .iter()
            .map(|path| vec![ path.iter().fold(0.0, |acc, z| acc + z) ])
            .collect();

        assert_eq!(paths_slice.len(), 50_000);

        // sum of independent normal(mu, sigma^2) RVs is a normal(n*mu, n*sigma^2) RV
        // let avg_price = try_average(&paths);
        let path_eval = PathEvaluator::new(&paths_slice);
        let avg_price = path_eval.evaluate_average(|path| path.last().cloned());

        assert_approx_eq!(0.5 * 100.0, avg_price.unwrap(), TOLERANCE);
    }

    #[test]
    fn stock_price_simulation() {
        let drift = 0.01;
        let vola = 0.03;
        let dt = 0.5;
        let s0 = 100.0;

        let stock_gbm = GeometricBrownianMotion::new(drift, vola, dt);
        let mc_simulator = MonteCarloPathSimulator::new(5_000, 100);
        let paths = mc_simulator.simulate_paths(stock_gbm, s0);
        assert_eq!(paths.len(), 5_000);

        // expected value should equal analytic solution
        let path_eval = PathEvaluator::new(&paths);
        let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
        let tte = 100.0 * dt;
        let exp_price = s0 * (tte * (drift - vola.powi(2) / 2.0)).exp();
        assert_eq!(exp_price, avg_price.unwrap());
    }

    #[test]
    fn no_drift_stock_price_simulation() {
        let vola = 50.0 / 365.0;
        let r = 0.0;
        let dt = 0.1;
        let nr_steps = 100;
        let s0 = 300.0;

        let stock_gbm = GeometricBrownianMotion::new(r, vola, dt);
        let mc_simulator = MonteCarloPathSimulator::new(100_000, nr_steps);
        let paths = mc_simulator.simulate_paths(stock_gbm, s0);
    
        let path_eval = PathEvaluator::new(&paths);
        let avg_price = path_eval.evaluate_average(|path| path.last().cloned());

        // precision depends on nr_samples and other inputs
        assert_approx_eq!(avg_price.unwrap(), s0, TOLERANCE);
    }

    #[test]
    fn european_call_300() {
        let mc_simulator = MonteCarloPathSimulator::new(10_000, 100);
        let dp = DerivativeParameter::new(300.0, 250.0, 1.0, 0.03, 0.15);
        let mcOption = MonteCarloEuropeanOption { mc_simulator, option_params: dp };
        let call_price = mcOption.call();
        assert_approx_eq!(call_price.unwrap(), 58.8197, TOLERANCE); // compare with analytic solution
    }

    #[test]
    fn european_call_310() {
        let mc_simulator = MonteCarloPathSimulator::new(10_000, 100);
        let dp = DerivativeParameter::new(310.0, 250.0, 3.5, 0.05, 0.25);
        let mcOption = MonteCarloEuropeanOption { mc_simulator, option_params: dp };
        let call_price = mcOption.call();
        assert_approx_eq!(call_price.unwrap(), 113.4155, TOLERANCE); // compare with analytic solution
    }

    #[test]
    fn path_eval() {
        let paths = vec![ vec![ 1.0, 2.0], vec![ 3.0, 4.0], vec![] ];
        let path_eval = PathEvaluator::new(&paths);
        let avg = path_eval.evaluate_average(|_| Some(1.0_f64));
        assert_eq!(avg.unwrap(), (1.0 + 1.0 + 1.0) / 3.0);

        let avg = path_eval.evaluate_average(|path| path.first().cloned());
        assert_eq!(avg.unwrap(), (1.0 + 3.0) / 3.0);

        let avg = path_eval.evaluate_average(|path| path.last().cloned());
        assert_eq!(avg.unwrap(), (2.0 + 4.0) / 3.0);
    }
}
