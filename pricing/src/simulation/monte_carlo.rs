use rand::{
    self,
    prelude::ThreadRng,
};
use rand_distr::{DistIter, Distribution};

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

pub trait PathEvaluator {
    fn evaluate_path(path: &Vec<f64>) -> Option<f64>;
    // get slice?
}




#[cfg(test)]
mod tests {
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

        let paths : Vec<f64> = 
            mc_simulator.simulate_paths(normal_gen, 0.0)
            .iter().map(|path | path.iter().fold(0.0, |acc, z| acc  + z)).collect();

        assert_eq!(paths.len(), 50_000);

        // sum of independent normal(mu, sigma^2) RVs is a normal(n*mu, n*sigma^2) RV
        let avg_price = try_average(&paths);
        assert_approx_eq!(0.5 * 100.0, avg_price.unwrap(), TOLERANCE);
    }


    pub fn get_slice_value(
        paths: &[Vec<f64>],
        slice_idx: usize,
        slice_fn: impl Fn(&[f64]) -> Option<f64>,
    ) -> Option<f64> {
        if paths.is_empty() {
            return None;
        }

        let slice: Vec<f64> = paths
            .iter()
            .flat_map(|path| {
                if path.len() < slice_idx {
                    None
                } else {
                    Some(path[slice_idx])
                }
            })
            .collect();

        slice_fn(slice.as_slice())
    }

    pub fn try_average(paths: &[f64]) -> Option<f64> {
        if paths.is_empty() {
            None
        } else {
            let sum = paths.iter().fold(0.0, |curr_sum, s| curr_sum + s);
            Some(sum / (paths.len() as f64))
        }
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
        let avg_price = get_slice_value(paths.as_slice(), 100, try_average);
        let tte = 100.0 * dt;
        let exp_price = s0 * (tte * (drift - vola.powi(2) / 2.0)).exp();
        assert_eq!(exp_price, avg_price.unwrap());
    }

    #[test]
    fn no_drift_stock_price_simulation() {
        let r = 0.0;
        let vola = 50.0 / 365.0;
        let dt = 0.1;
        let nr_steps = 100;

        let s0 = 300.0;

        let bs_params = BlackScholesModelParams { r, vola, dt };

        let stock_gbm: GeometricBrownianMotion = bs_params.into();

        let mc_simulator = MonteCarloPathSimulator::new(100_000, nr_steps);

        let paths = mc_simulator.simulate_paths(stock_gbm, s0);

        let avg_price = get_slice_value(paths.as_slice(), nr_steps, try_average);

        // precision depends on nr_samples and other inputs
        assert_approx_eq!(avg_price.unwrap(), s0, 1e-2);
    }

    pub struct BlackScholesModelParams {
        r: f64,
        vola: f64,
        dt: f64,
    }

    impl From<BlackScholesModelParams> for GeometricBrownianMotion {
        fn from(bs_params: BlackScholesModelParams) -> Self {
            // under the risk neutral measure we have mu = r
            // hence, if r = 0.0 then we have no drift term
            let drift = bs_params.r;
            GeometricBrownianMotion::new(drift, bs_params.vola, bs_params.dt)
        }
    }

    #[test]
    fn european_call_300() {
        let s0 = 300.0;
        let rfr = 0.03;
        let vola = 0.15;
        let dt = 1.0 / 365.0;
        let nr_steps = 365;
        // time to expiration corresponds to nr_steps * dt = 1.0

        let bs_params = BlackScholesModelParams { r: rfr, vola, dt };
        let stock_gbm: GeometricBrownianMotion = bs_params.into();

        let mc_simulator = MonteCarloPathSimulator::new(100_000, nr_steps);
        let paths = mc_simulator.simulate_paths(stock_gbm, s0);

        // let avg_price = get_slice_value(paths.as_slice(), nr_steps, try_average);
        // dbg!(avg_price);

        // let strike = 250.0;

        // let payoff = |last_price: f64| {
        //     let pay_off = last_price - strike;
        //     if pay_off > 0.0 {
        //         pay_off
        //     } else {
        //         0.0
        //     }
        // };

        fn apply_pay_off(slice: &[f64]) -> Option<f64> {
            if slice.is_empty() {
                return None;
            }
            let total_pay_off = slice.iter().fold(0.0, |acc, last_price| {
                let pay_off = last_price - 250.0;
                if pay_off > 0.0 {
                    acc + pay_off
                } else {
                    acc
                }
            });
            Some(total_pay_off / (slice.len() as f64))
        }

        let call_price = get_slice_value(paths.as_slice(), nr_steps, apply_pay_off);

        // compare with analytic solution in ./analytic/black_scholes.rs
        assert_approx_eq!(call_price.unwrap(), 58.8197, TOLERANCE); // from analytic solution
    }


    #[test]
    fn european_call_310() {
        let s0 = 310.0;
        let r = 0.05;
        let vola = 0.25;
        let dt = 1.0 / 365.0;
        let nr_steps = 1277;
        // time to expiration corresponds to nr_steps * dt = 3.5 years

        let bs_params = BlackScholesModelParams { r, vola, dt };

        let stock_gbm: GeometricBrownianMotion = bs_params.into();

        let mc_simulator = MonteCarloPathSimulator::new(10_000, nr_steps);

        let paths = mc_simulator.simulate_paths(stock_gbm, s0);

        fn apply_pay_off(slice: &[f64]) -> Option<f64> {
            if slice.is_empty() {
                return None;
            }
            let total_pay_off = slice.iter().fold(0.0, |acc, last_price| {
                let pay_off = last_price - 250.0;
                if pay_off > 0.0 {
                    acc + pay_off
                } else {
                    acc
                }
            });
            Some(total_pay_off / (slice.len() as f64))
        }

        let call_price = get_slice_value(paths.as_slice(), nr_steps, apply_pay_off);
        assert_approx_eq!(call_price.unwrap(), 113.4155, TOLERANCE); // from analytic solution
    }
}
