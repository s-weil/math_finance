use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use rand_hc::Hc128Rng;
use std::marker::PhantomData;

/// Inherits from the 'generating distribution'.
pub trait PathSampler<SampleType>: Distribution<SampleType> + Sized {
    fn rn_generator(&self, seed_nr: u64) -> Hc128Rng {
        rand_hc::Hc128Rng::seed_from_u64(seed_nr)
    }

    #[inline]
    fn sample_path(&self, rn_generator: &mut Hc128Rng, nr_samples: usize) -> Vec<SampleType> {
        // unoptimized but generic implementation
        rn_generator.sample_iter(self).take(nr_samples).collect()

        // let mut samples_vec = Vec::with_capacity(nr_samples);
        // for _ in 0..nr_samples {
        //     samples_vec.push(self.sample(rn_generator));
        // }
        // samples_vec
    }
}

#[derive(Debug, Clone)]
pub struct MonteCarloPathSimulator<SampleType> {
    pub nr_paths: usize,
    pub nr_steps: usize,
    _phantom: PhantomData<SampleType>,
}

impl<SampleType> MonteCarloPathSimulator<SampleType> {
    pub fn new(nr_paths: usize, nr_steps: usize) -> Self {
        Self {
            nr_paths,
            nr_steps,
            _phantom: PhantomData,
        }
    }

    pub fn simulate_paths(
        &self,
        seed_nr: u64,
        sampler: impl PathSampler<SampleType>,
    ) -> Vec<Vec<SampleType>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut generator = sampler.rn_generator(seed_nr);

        for _ in 0..self.nr_paths {
            let path = sampler.sample_path(&mut generator, self.nr_steps);
            paths.push(path);
        }
        paths
    }

    pub fn simulate_paths_with(
        &self,
        seed_nr: u64,
        sampler: impl PathSampler<SampleType>,
        path_fn: impl Fn(&PathSlice<SampleType>) -> Path<SampleType>,
    ) -> Vec<Vec<SampleType>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut generator = sampler.rn_generator(seed_nr);

        for _ in 0..self.nr_paths {
            let path = sampler.sample_path(&mut generator, self.nr_steps);
            let v = path_fn(&path);
            paths.push(v);
        }
        paths
    }

    pub fn simulate_paths_apply_in_place(
        &self,
        seed_nr: u64,
        sampler: impl PathSampler<SampleType>,
        apply_in_place_fn: impl Fn(&mut PathSlice<SampleType>),
    ) -> Vec<Vec<SampleType>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut generator = sampler.rn_generator(seed_nr);

        for _ in 0..self.nr_paths {
            let mut path = sampler.sample_path(&mut generator, self.nr_steps);
            apply_in_place_fn(&mut path);
            paths.push(path);
        }
        paths
    }

    //TODO: add a version which doesnt store paths
    // pub fn simulate_paths_with(
    //     &self,
    //     generator: impl PathGenerator,
    //     path_fn: impl Fn(Path) -> Option<f64>,
    // ) -> Vec<Option<f64>> {
    //     let mut paths = Vec::with_capacity(self.nr_paths);
    //     let mut rng = rand::thread_rng();

    //     for _ in 0..self.nr_paths {
    //         let distr = generator.distribution(&mut rng);
    //         let path = generator.sample_path(self.nr_steps, distr);
    //         let v = path_fn(path);
    //         paths.push(v);
    //     }
    //     paths
    // }
}

pub type Path<SampleType> = Vec<SampleType>;
pub type PathSlice<SampleType> = [SampleType];

pub struct PathEvaluator<'a, SampleType> {
    paths: &'a [Path<SampleType>],
}

impl<'a, SampleType> PathEvaluator<'a, SampleType> {
    pub fn new(paths: &'a [Path<SampleType>]) -> Self {
        Self { paths }
    }

    // TODO: rename apply
    pub fn evaluate(
        &self,
        path_fn: impl Fn(&'a Path<SampleType>) -> Option<f64>,
    ) -> Vec<Option<f64>> {
        self.paths.iter().map(path_fn).collect()
    }

    pub fn evaluate_average(
        &self,
        path_fn: impl Fn(&'a Path<SampleType>) -> Option<f64>,
    ) -> Option<f64> {
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

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::simulation::gbm::GeometricBrownianMotion;
    use rand_distr::{Normal, StandardNormal};

    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    const TOLERANCE: f64 = 1e-1;

    #[test]
    fn normal_path_simulation() {
        let sampler = Normal::new(0.5, 1.0).unwrap();
        let mc_simulator = MonteCarloPathSimulator::new(100_000, 100);

        let paths_slice: Vec<Vec<f64>> = mc_simulator
            .simulate_paths(42, sampler)
            .iter()
            .map(|path| vec![path.iter().fold(0.0, |acc, z| acc + z)])
            .collect();

        assert_eq!(paths_slice.len(), 100_000);

        // sum of independent normal(mu, sigma^2) RVs is a normal(n*mu, n*sigma^2) RV
        let path_eval = PathEvaluator::new(&paths_slice);
        let avg_price = path_eval.evaluate_average(|path| path.last().cloned());

        assert_approx_eq!(0.5 * 100.0, avg_price.unwrap(), TOLERANCE);
    }

    #[test]
    fn stock_price_simulation_path_fn() {
        let nr_paths = 100_000;
        let drift = -0.2;
        let vola = 0.4;
        let s0 = 100.0;
        let nr_steps = 100;
        let tte = 5.0;
        let dt = tte / nr_steps as f64;

        let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
        let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);

        let paths = mc_simulator.simulate_paths_with(42, StandardNormal, |standard_normals| {
            stock_gbm.generate_path(s0, standard_normals)
        });
        assert_eq!(paths.len(), nr_paths);

        // expected value should equal analytic solution
        let path_eval = PathEvaluator::new(&paths);
        let avg_delta =
            path_eval.evaluate_average(|path| path.last().cloned().map(|p| (p / s0).ln()));
        let exp_delta = tte * (drift - vola.powi(2) / 2.0);
        assert_approx_eq!(avg_delta.unwrap(), exp_delta, TOLERANCE);
    }

    #[test]
    fn no_drift_stock_price_simulation() {
        let nr_paths = 100_000;
        let vola: f64 = 0.4;
        let drift = vola.powi(2) / 2.0;
        let s0 = 100.0;
        let nr_steps = 100;
        let tte = 5.0;
        let dt = tte / nr_steps as f64;

        let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
        let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
        let paths = mc_simulator.simulate_paths(42, stock_gbm);

        let path_eval = PathEvaluator::new(&paths);

        let avg_delta =
            path_eval.evaluate_average(|path| path.last().cloned().map(|p| (p / s0).ln()));
        let exp_delta = tte * (drift - vola.powi(2) / 2.0);
        assert_approx_eq!(avg_delta.unwrap(), exp_delta, TOLERANCE);
    }

    #[test]
    fn path_eval() {
        let paths = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![]];
        let path_eval = PathEvaluator::new(&paths);
        let avg = path_eval.evaluate_average(|_| Some(1.0_f64));
        assert_eq!(avg.unwrap(), (1.0 + 1.0 + 1.0) / 3.0);

        let avg = path_eval.evaluate_average(|path| path.first().cloned());
        assert_eq!(avg.unwrap(), (1.0 + 3.0) / 3.0);

        let avg = path_eval.evaluate_average(|path| path.last().cloned());
        assert_eq!(avg.unwrap(), (2.0 + 4.0) / 3.0);
    }
}
