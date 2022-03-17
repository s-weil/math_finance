// use rand::{self, prelude::ThreadRng};
use rand_distr::{DistIter, Distribution, Normal};


use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;



pub trait DistributionExt : Distribution<f64> + Sized {
    fn generator(&self, seed_nr: u64) -> Hc128Rng {
        rand_hc::Hc128Rng::seed_from_u64(seed_nr)
    }

    fn samples<'a>(self, generator: &'a mut Hc128Rng, nr_samples: usize) -> Vec<f64> {
        generator.sample_iter(self).take(nr_samples).collect()
    }

    fn dist_iter<'a>(
        self,
        generator: &'a mut Hc128Rng,
    ) -> DistIter<Self, &'a mut Hc128Rng, f64> {
        generator.sample_iter(self)
    }
}

impl DistributionExt for Normal<f64> {}

// pub trait McSampler {
//     type Dist: Distribution<f64>;

//     fn generator(&self, seed_nr: u64) -> Hc128Rng {
//         rand_hc::Hc128Rng::seed_from_u64(seed_nr)
//     }

//     fn distribution(&self) -> Self::Dist;

//     fn samples<'a>(&self, generator: &'a mut Hc128Rng, nr_samples: usize) -> Vec<f64> {
//         //Self::SampleType>  {
//         let dist = self.distribution();
//         generator.sample_iter(dist).take(nr_samples).collect()
//     }

//     fn dist_iter<'a>(
//         &self,
//         generator: &'a mut Hc128Rng,
//     ) -> DistIter<Self::Dist, &'a mut Hc128Rng, f64> {
//         //  Self::SampleType> {
//         generator.sample_iter(self.distribution())
//     }
// }

// impl McSampler for Normal<f64> {
//     type Dist = Self;

//     fn distribution(&self) -> Self::Dist {
//         *self
//     }
// }

// pub struct NormalNumberSampler {
//     pub mu: f64,
//     pub sigma: f64,
//     // generator: Hc128Rng,
//     // distribution: Normal<f64>,
// }

// impl NormalNumberSampler {
//     pub fn new(mu: f64, sigma: f64) -> Self {
//         // let distribution = Normal::new(mu, sigma).unwrap();
//         Self {
//             mu,
//             sigma,
//             // distribution,
//         }
//     }
// }

// impl McSampler for NormalNumberSampler {
//     type Dist = Normal<f64>;

//     fn distribution(&self) -> Self::Dist {
//         Normal::new(self.mu, self.sigma).unwrap()
//     }
// }

pub trait McPathSampler { // Do not "inherit" from DistributionExt to leave more flexibility
    type Dist : DistributionExt;
    // type SampleType;

    fn distribution(&self) -> Self::Dist;

    fn sample_path<'a>(
        &self,
        generator: &'a mut Hc128Rng,
        nr_steps: usize,
    ) -> Vec<f64>;
}

// TODO: do an implementatoin for MultivariateNormalNumberPathSampler

impl McPathSampler for Normal<f64> {
    type Dist = Self;

    fn distribution(&self) -> Self::Dist {
        *self
    }

    fn sample_path<'a>(
        &self,
        generator: &'a mut Hc128Rng,
        nr_steps: usize,
    ) -> Vec<f64> {
        self.samples(generator, nr_steps)
    }
}


// pub trait McDistIter {
//     type Dist: Distribution<f64>;

//     fn distribution<'a>(
//         &self,
//         rng: &'a mut ThreadRng,
//     ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64>;
// }

// pub trait PathGenerator: McDistIter {
//     fn sample_path(
//         &self,
//         nr_steps: usize,
//         dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
//     ) -> Vec<f64>;
// }

// pub trait SampleGenerator: McDistIter {
//     fn sample(&self, current: f64, random_nr: f64) -> f64;
// }

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
    pub nr_paths: usize,
    pub nr_steps: usize,
}

impl MonteCarloPathSimulator {
    pub fn new(nr_paths: usize, nr_steps: usize) -> Self {
        Self { nr_paths, nr_steps }
    }

    pub fn simulate_paths(&self, sampler: impl McPathSampler) -> Vec<Vec<f64>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut generator = sampler.distribution().generator(53);

        for _ in 0..self.nr_paths {
            let path = sampler.sample_path(&mut generator , self.nr_steps);
            paths.push(path);
        }
        paths
    }

    pub fn simulate_paths_with(
        &self,
        sampler: impl McPathSampler,
        path_fn: impl Fn(&PathSlice) -> Path,
    ) -> Vec<Vec<f64>> {
        let mut paths = Vec::with_capacity(self.nr_paths);
        let mut generator = sampler.distribution().generator(53);

        for _ in 0..self.nr_paths {
            let path = sampler.sample_path(&mut generator , self.nr_steps);
            let v = path_fn(&path);
            paths.push(v);
        }
        paths
    }
}

pub type Path = Vec<f64>;
pub type PathSlice = [f64];

pub struct PathEvaluator<'a> {
    paths: &'a [Path],
}

impl<'a> PathEvaluator<'a> {
    pub fn new(paths: &'a [Path]) -> Self {
        Self { paths }
    }

    // TODO: rename apply
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

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::simulation::gbm::GeometricBrownianMotion;
    use rand_distr::{DistIter, Distribution, Normal};

    use assert_approx_eq::assert_approx_eq;

    /// NOTE: the tolerance will depend on the number of samples paths and other params like steps and the volatility
    const TOLERANCE: f64 = 1e-1;




    #[test]
    fn stock_price_simulation() {
        let nr_paths = 100_000;
        let nr_steps = 100;
        let drift = -0.2;
        let vola = 0.4;
        let s0 = 100.0;
        let nr_steps = 100;
        let tte = 5.0;
        let dt = tte / nr_steps as f64;

        let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
        let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let paths = mc_simulator.simulate_paths_with(normal, |random_normals| stock_gbm.sample_path2(random_normals));
        assert_eq!(paths.len(), nr_paths);

        // expected value should equal analytic solution
        let path_eval = PathEvaluator::new(&paths);
        let avg_delta =
            path_eval.evaluate_average(|path| path.last().cloned().map(|p| (p / s0).ln()));
        let exp_delta = tte * (drift - vola.powi(2) / 2.0);
        assert_approx_eq!(avg_delta.unwrap(), exp_delta, TOLERANCE);
    }

    // #[test]
    // fn no_drift_stock_price_simulation() {
    //     let nr_paths = 100_000;
    //     let nr_steps = 100;
    //     let vola: f64 = 0.4;
    //     let drift = vola.powi(2) / 2.0;
    //     let s0 = 100.0;
    //     let nr_steps = 100;
    //     let tte = 5.0;
    //     let dt = tte / nr_steps as f64;

    //     let stock_gbm = GeometricBrownianMotion::new(s0, drift, vola, dt);
    //     let mc_simulator = MonteCarloPathSimulator::new(nr_paths, nr_steps);
    //     let paths = mc_simulator.simulate_paths(stock_gbm);

    //     let path_eval = PathEvaluator::new(&paths);

    //     let avg_delta =
    //         path_eval.evaluate_average(|path| path.last().cloned().map(|p| (p / s0).ln()));
    //     let exp_delta = 0.0; // tte * (drift - vola.powi(2) / 2.0);
    //     assert_approx_eq!(avg_delta.unwrap(), exp_delta, TOLERANCE);

    //     // let avg_price = path_eval.evaluate_average(|path| path.last().cloned());
    //     // assert_approx_eq!(avg_price.unwrap(), s0, TOLERANCE);
    // }

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


















    // #[test]
    // fn normal_sampled_paths() {
    //     let nr_steps = 100;
    //     let nr_paths = 100_000;
    //     let mut paths = Vec::with_capacity(nr_paths);

    //     // let normal_sampler = NormalNumberSampler::new(0.5, 0.3);
    //     let normal_sampler  = Normal::new(0.5, 0.3).unwrap();
    //     let mut generator = normal_sampler.generator1(41);

    //     for _ in 0..nr_paths {
    //         let path = normal_sampler.samples1(&mut generator, nr_steps);
    //         // let distr = normal_sampler.dist_iter1(&mut generator);
    //         // let path: Vec<f64> = distr.take(nr_steps).collect();
    //         // let path = generator.sample_path(self.nr_steps, distr);
    //         paths.push(path);
    //     }
    //     assert_eq!(paths.len(), nr_paths);
    // }

    // #[test]
    // fn normal_sampled_paths_end() {
    //     let nr_steps = 100;
    //     let nr_paths = 100_000;
    //     let mut paths = Vec::with_capacity(nr_paths);

    //     let normal_sampler = NormalNumberSampler::new(0.5, 0.3);
    //     let mut generator = normal_sampler.generator(41);

    //     for _ in 0..nr_paths {
    //         let distr = normal_sampler.dist_iter(&mut generator);
    //         let path: f64 = distr.take(nr_steps).fold(0.0, |acc, z| acc + z);
    //         // let path = generator.sample_path(self.nr_steps, distr);
    //         paths.push(vec![path]);
    //     }
    //     assert_eq!(paths.len(), nr_paths);

    //     let path_eval = PathEvaluator::new(&paths);
    //     let avg_price = path_eval.evaluate_average(|path| path.last().cloned());

    //     assert_approx_eq!(0.5 * 100.0, avg_price.unwrap(), TOLERANCE);
    // }

    // // TODO: should also consider to return and store the random paths (without gbm or so yet),
    // // and only then apply further functions like gbm
    // // this way greeks can be calculated with the same randoms, for which the creation is the hot path
    // // would it also work for basket? -> would have to store vec<vec<f64>>

    // pub struct MonteCarloPathSampler {
    //     pub nr_paths: usize,
    //     pub nr_steps: usize,
    // }

    // impl MonteCarloPathSampler {
    //     pub fn new(nr_paths: usize, nr_steps: usize) -> Self {
    //         Self { nr_paths, nr_steps }
    //     }

    //     pub fn simulate_paths(&self, sampler: impl DistributionExt) -> Vec<Vec<f64>> {
    //         let mut paths = Vec::with_capacity(self.nr_paths);
    //         let mut rng = rand::thread_rng();
            
    //         let mut generator = sampler.generator(41);

    //         for _ in 0..self.nr_paths {
    //             // TODO: try to pass it as ref
    //             // maybe parallelize if it's a hotpath
    //             let distr = generator.distribution(&mut rng);
    //             let path = generator.sample_path(self.nr_steps, distr);
    //             paths.push(path);
    //         }
    //         paths
    //     }

    //     pub fn simulate_paths_with(
    //         &self,
    //         generator: impl PathGenerator,
    //         path_fn: impl Fn(Path) -> Option<f64>,
    //     ) -> Vec<Option<f64>> {
    //         let mut paths = Vec::with_capacity(self.nr_paths);
    //         let mut rng = rand::thread_rng();

    //         for _ in 0..self.nr_paths {
    //             let distr = generator.distribution(&mut rng);
    //             let path = generator.sample_path(self.nr_steps, distr);
    //             let v = path_fn(path);
    //             paths.push(v);
    //         }
    //         paths
    //     }
    // }

    
    // use rand::{Rng, SeedableRng};
    // use rand_hc::Hc128Rng;

    // pub trait DistributionExt : Distribution<f64> + Sized {
    //     fn generator1(&self, seed_nr: u64) -> Hc128Rng {
    //         rand_hc::Hc128Rng::seed_from_u64(seed_nr)
    //     }

    //     fn samples1<'a>(self, generator: &'a mut Hc128Rng, nr_samples: usize) -> Vec<f64> {
    //         generator.sample_iter(self).take(nr_samples).collect()
    //     }

    //     fn dist_iter1<'a>(
    //         self,
    //         generator: &'a mut Hc128Rng,
    //     ) -> DistIter<Self, &'a mut Hc128Rng, f64> {
    //         generator.sample_iter(self)
    //     }
    // }

    // impl DistributionExt for Normal<f64> {}

    // pub trait McSampler {
    //     type Dist: Distribution<f64>;

    //     fn generator(&self, seed_nr: u64) -> Hc128Rng {
    //         rand_hc::Hc128Rng::seed_from_u64(seed_nr)
    //     }

    //     fn distribution(&self) -> Self::Dist;

    //     fn samples<'a>(&self, generator: &'a mut Hc128Rng, nr_samples: usize) -> Vec<f64> {
    //         //Self::SampleType>  {
    //         let dist = self.distribution();
    //         generator.sample_iter(dist).take(nr_samples).collect()
    //     }

    //     fn dist_iter<'a>(
    //         &self,
    //         generator: &'a mut Hc128Rng,
    //     ) -> DistIter<Self::Dist, &'a mut Hc128Rng, f64> {
    //         //  Self::SampleType> {
    //         generator.sample_iter(self.distribution())
    //     }
    // }

    // impl McSampler for Normal<f64> {
    //     type Dist = Self;

    //     fn distribution(&self) -> Self::Dist {
    //         *self
    //     }
    // }

    // pub struct NormalNumberSampler {
    //     pub mu: f64,
    //     pub sigma: f64,
    //     // generator: Hc128Rng,
    //     // distribution: Normal<f64>,
    // }

    // impl NormalNumberSampler {
    //     pub fn new(mu: f64, sigma: f64) -> Self {
    //         // let distribution = Normal::new(mu, sigma).unwrap();
    //         Self {
    //             mu,
    //             sigma,
    //             // distribution,
    //         }
    //     }
    // }

    // impl McSampler for NormalNumberSampler {
    //     type Dist = Normal<f64>;

    //     fn distribution(&self) -> Self::Dist {
    //         Normal::new(self.mu, self.sigma).unwrap()
    //     }
    // }

    // pub trait McPathSampler: McSampler {
    //     type SampleType;

    //     fn sample_path<'a>(
    //         &self,
    //         generator: &'a mut Hc128Rng,
    //         nr_steps: usize,
    //     ) -> Vec<Self::SampleType>;
    // }

    // // TODO: do an implementatoin for MultivariateNormalNumberPathSampler

    // impl McPathSampler for NormalNumberSampler {
    //     type SampleType = f64;

    //     fn sample_path<'a>(
    //         &self,
    //         generator: &'a mut Hc128Rng,
    //         nr_steps: usize,
    //     ) -> Vec<Self::SampleType> {
    //         self.samples(generator, nr_steps)
    //     }
    // }

    // #[test]
    // fn normal_sampled_paths() {
    //     let nr_steps = 100;
    //     let nr_paths = 100_000;
    //     let mut paths = Vec::with_capacity(nr_paths);

    //     // let normal_sampler = NormalNumberSampler::new(0.5, 0.3);
    //     let normal_sampler  = Normal::new(0.5, 0.3).unwrap();
    //     let mut generator = normal_sampler.generator1(41);

    //     for _ in 0..nr_paths {
    //         let path = normal_sampler.samples1(&mut generator, nr_steps);
    //         // let distr = normal_sampler.dist_iter1(&mut generator);
    //         // let path: Vec<f64> = distr.take(nr_steps).collect();
    //         // let path = generator.sample_path(self.nr_steps, distr);
    //         paths.push(path);
    //     }
    //     assert_eq!(paths.len(), nr_paths);
    // }

    // #[test]
    // fn normal_sampled_paths_end() {
    //     let nr_steps = 100;
    //     let nr_paths = 100_000;
    //     let mut paths = Vec::with_capacity(nr_paths);

    //     let normal_sampler = NormalNumberSampler::new(0.5, 0.3);
    //     let mut generator = normal_sampler.generator(41);

    //     for _ in 0..nr_paths {
    //         let distr = normal_sampler.dist_iter(&mut generator);
    //         let path: f64 = distr.take(nr_steps).fold(0.0, |acc, z| acc + z);
    //         // let path = generator.sample_path(self.nr_steps, distr);
    //         paths.push(vec![path]);
    //     }
    //     assert_eq!(paths.len(), nr_paths);

    //     let path_eval = PathEvaluator::new(&paths);
    //     let avg_price = path_eval.evaluate_average(|path| path.last().cloned());

    //     assert_approx_eq!(0.5 * 100.0, avg_price.unwrap(), TOLERANCE);
    // }

    // // TODO: should also consider to return and store the random paths (without gbm or so yet),
    // // and only then apply further functions like gbm
    // // this way greeks can be calculated with the same randoms, for which the creation is the hot path
    // // would it also work for basket? -> would have to store vec<vec<f64>>

    // pub struct MonteCarloPathSampler {
    //     pub nr_paths: usize,
    //     pub nr_steps: usize,
    // }

    // impl MonteCarloPathSampler {
    //     pub fn new(nr_paths: usize, nr_steps: usize) -> Self {
    //         Self { nr_paths, nr_steps }
    //     }

    //     pub fn simulate_paths(&self, sampler: impl DistributionExt) -> Vec<Vec<f64>> {
    //         let mut paths = Vec::with_capacity(self.nr_paths);
    //         let mut rng = rand::thread_rng();
            
    //         let mut generator = sampler.generator(41);

    //         for _ in 0..self.nr_paths {
    //             // TODO: try to pass it as ref
    //             // maybe parallelize if it's a hotpath
    //             let distr = generator.distribution(&mut rng);
    //             let path = generator.sample_path(self.nr_steps, distr);
    //             paths.push(path);
    //         }
    //         paths
    //     }

    //     pub fn simulate_paths_with(
    //         &self,
    //         generator: impl PathGenerator,
    //         path_fn: impl Fn(Path) -> Option<f64>,
    //     ) -> Vec<Option<f64>> {
    //         let mut paths = Vec::with_capacity(self.nr_paths);
    //         let mut rng = rand::thread_rng();

    //         for _ in 0..self.nr_paths {
    //             let distr = generator.distribution(&mut rng);
    //             let path = generator.sample_path(self.nr_steps, distr);
    //             let v = path_fn(path);
    //             paths.push(v);
    //         }
    //         paths
    //     }
    // }
}
