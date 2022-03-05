use rand::{self, prelude::ThreadRng};
use rand_distr::DistIter;


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
    type Dist;

    fn sample(
        &self,
        current: f64,
        nr_samples: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> f64;

    fn distribution<'a>(
        &self,
        rng: &'a mut ThreadRng,
    ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64>;
}

pub struct MonteCarloStepSimulator {
    nr_samples: usize,
}

impl MonteCarloStepSimulator {
    pub fn new(nr_samples: usize) -> Self {
        Self { nr_samples }
    }

    pub fn simulate(&self, generator: impl SampleGenerator) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut simulations = vec![];

        let initial = 100.0;

        for _ in 0..self.nr_samples {
            let distr = generator.distribution(&mut rng);
            let res = generator.sample(initial, self.nr_samples, distr);
            simulations.push(res);
        }

        simulations
    }
}



pub struct MonteCarloPathSimulator {
    nr_paths: usize,
    nr_steps: usize,
}

impl MonteCarloPathSimulator {
    pub fn new(nr_paths: usize, nr_steps: usize) -> Self {
        Self { nr_paths, nr_steps }
    }

    pub fn simulate_paths(&self, generator: impl PathGenerator) -> Vec<Vec<f64>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut rng = rand::thread_rng();
        let initial = 100.0;

        for _ in 0..self.nr_paths {
            // TODO: try to pass it as ref
            // maybe parallelize if it's a hotpath
            let distr = generator.distribution(&mut rng);
            let path = generator.sample_path(initial, self.nr_steps, distr);
            paths.push(path);
        }
        paths
    }
}

pub fn try_average(paths: &[f64]) -> Option<f64> {
    if paths.is_empty() {
        None
    } else {
        let sum = paths.iter().fold(0.0, |curr_sum, s| curr_sum + s);
        Some(sum / (paths.len() as f64))
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

    #[test]
    fn stock_price_simulation() {
        let drift = 0.001;
        let vola = 0.003;
        let dt = 3.5;

        let stock_gbm = GeometricBrownianMotion::new(drift, vola, dt);

        let mc_simulator = MonteCarloStepSimulator::new(5_000);

        let paths = mc_simulator.simulate(stock_gbm);

        let avg_price = try_average(&paths);

        dbg!(&avg_price);
        assert!(avg_price.is_some());
    }
}
