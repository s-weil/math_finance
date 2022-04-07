use rand::Rng;
use std::marker::PhantomData;

pub trait SeedRng: rand::SeedableRng + rand::RngCore /*+ rand::Rng */ {}

pub trait PathGenerator<Path> {
    fn sample_path<SRng>(&self, rn_generator: &mut SRng, nr_samples: usize) -> Path
    where
        SRng: SeedRng;
}

/// Implementations for seedable_rng are for instance:
/// rand_hc::Hc128Rng
/// rand_isaac::Isaac64Rng
#[derive(Debug)]
pub struct MonteCarloPathSimulator<PathGen, SRng, Path>
where
    PathGen: PathGenerator<Path>,
    SRng: SeedRng,
{
    path_generator: PathGen,
    // rng: SRng,
    seed_nr: Option<u64>,
    _phantom_path: PhantomData<Path>,
    _phantom_rng: PhantomData<SRng>,
}

impl<PathGen, SRng, Path> MonteCarloPathSimulator<PathGen, SRng, Path>
where
    PathGen: PathGenerator<Path>,
    SRng: SeedRng,
{
    pub fn new(path_generator: PathGen, seed_nr: Option<u64>) -> Self {
        Self {
            path_generator,
            seed_nr,
            _phantom_path: PhantomData::<Path>,
            _phantom_rng: PhantomData::<SRng>,
        }
    }

    fn rn_generator(&self) -> SRng {
        match self.seed_nr {
            Some(seed_nr) => SRng::seed_from_u64(seed_nr),
            None => {
                let random_seed =
                    rand::thread_rng().sample(rand_distr::Uniform::new(0u64, 100_000));
                SRng::seed_from_u64(random_seed)
            }
        }
    }

    pub fn simulate_paths(&self, nr_paths: usize, nr_steps: usize) -> Vec<Path> {
        let mut paths = Vec::with_capacity(nr_paths);
        let mut generator = self.rn_generator();

        for _ in 0..nr_paths {
            let path = self.path_generator.sample_path(&mut generator, nr_steps);
            paths.push(path);
        }
        paths
    }

    pub fn simulate_paths_with(
        &self,
        nr_paths: usize,
        nr_steps: usize,
        path_fn: impl Fn(&Path) -> Path,
    ) -> Vec<Path> {
        let mut paths = Vec::with_capacity(nr_paths);
        let mut generator = self.rn_generator();

        for _ in 0..nr_paths {
            let path = self.path_generator.sample_path(&mut generator, nr_steps);
            let v = path_fn(&path);
            paths.push(v);
        }
        paths
    }

    pub fn simulate_paths_apply_in_place(
        &self,
        nr_paths: usize,
        nr_steps: usize,
        apply_in_place_fn: impl Fn(&mut Path),
    ) -> Vec<Path> {
        let mut paths = Vec::with_capacity(nr_paths);
        let mut generator = self.rn_generator();

        for _ in 0..nr_paths {
            let mut path = self.path_generator.sample_path(&mut generator, nr_steps);
            apply_in_place_fn(&mut path);
            paths.push(path);
        }
        paths
    }
}

pub struct PathEvaluator<'a, Path> {
    paths: &'a [Path],
}

impl<'a, Path> PathEvaluator<'a, Path> {
    pub fn new(paths: &'a [Path]) -> Self {
        Self { paths }
    }

    // TODO: rename apply
    pub fn evaluate(&self, path_fn: impl Fn(&Path) -> Option<f64>) -> Vec<Option<f64>> {
        self.paths.iter().map(path_fn).collect()
    }

    pub fn evaluate_average(&self, path_fn: impl Fn(&Path) -> Option<f64>) -> Option<f64> {
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
    use rand::SeedableRng;
    use rand_distr::{Normal, StandardNormal};

    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    const TOLERANCE: f64 = 1e-1;

    #[test]
    fn normal_path_simulation() {
        let sampler: Normal<f64> = Normal::new(0.5, 1.0).unwrap();

        // <_, <rand_hc::Hc128Rng as SeedableRng>, _>
        let rng = rand_hc::Hc128Rng::seed_from_u64(32);
        let test = <rand_hc::Hc128Rng as SeedRng>;
        let mc_simulator: MonteCarloPathSimulator<Normal<f64>, rand_hc::Hc128Rng, Vec<f64>> =
            MonteCarloPathSimulator::new(sampler, Some(42));

        let paths_slice: Vec<Vec<f64>> = mc_simulator
            .simulate_paths(100_000, 100)
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
