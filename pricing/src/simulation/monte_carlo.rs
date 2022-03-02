use crate::simulation::PathGenerator;

pub struct MonteCarloSimulator {
    nr_paths: usize,
    nr_steps: usize,
}

impl MonteCarloSimulator {
    pub fn new(nr_paths: usize, nr_steps: usize) -> Self {
        Self { nr_paths, nr_steps }
    }

    pub fn simulate(&self, generator: impl PathGenerator) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut simulations = vec![];

        let initial = 100.0;

        for _ in 0..self.nr_paths {
            let distr = generator.get_dist_iter(&mut rng);
            let res = generator.simulate(initial, self.nr_steps, distr);
            simulations.push(res);
        }

        simulations
    }

    pub fn simulate_paths(&self, generator: impl PathGenerator) -> Vec<Vec<f64>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut rng = rand::thread_rng();
        let initial = 100.0;

        for _ in 0..self.nr_paths {
            // TODO: try to pass it as ref
            // maybe parallelize if it's a hotpath
            let distr = generator.get_dist_iter(&mut rng);
            let path = generator.simulate_path(initial, self.nr_steps, distr);
            paths.push(path);
        }
        paths
    }

    pub fn try_average_slice(&self, paths: &Vec<Vec<f64>>, slice_idx: usize) -> Option<f64> {
        if self.nr_steps > slice_idx {
            let slice: Vec<f64> = paths.iter().map(|path| path[slice_idx]).collect();
            let sum = slice.iter().fold(0.0, |curr_sum, s| curr_sum + s);
            Some(sum / (slice.len() as f64))
        } else {
            None
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::GeometricBrownianMotion;

    #[test]
    fn test_stock_simulation() {
        let drift = 0.001;
        let vola = 0.003;
        let dt = 1.0;

        let stock_gbm = GeometricBrownianMotion::new(drift, vola, dt);

        let mc_simulator = MonteCarloSimulator::new(5_000, 100);

        let paths = mc_simulator.simulate(stock_gbm);

        let avg_price = try_average(&paths);

        dbg!(&avg_price);
        assert!(avg_price.is_some());
    }
}
