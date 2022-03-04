use rand::{self, prelude::ThreadRng};
use rand_distr::{DistIter, Distribution, Normal};

use crate::simulation::GeometricBrownianMotion;

pub trait PathGenerator {
    type Dist;

    fn simulate(
        &self,
        current: f64,
        nr_steps: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> f64;

    fn simulate_path(
        &self,
        current: f64,
        nr_steps: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> Vec<f64>;

    fn get_dist_iter<'a>(
        &self,
        rng: &'a mut ThreadRng,
    ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64>;
}

impl PathGenerator for GeometricBrownianMotion {
    type Dist = Normal<f64>;

    fn simulate(
        &self,
        price: f64,
        nr_steps: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> f64 {
        self.simulate(price, nr_steps, dist_iter)
    }

    fn simulate_path(
        &self,
        price: f64,
        nr_steps: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> Vec<f64> {
        self.simulate_path(price, nr_steps, dist_iter)
    }

    fn get_dist_iter<'a>(
        &self,
        rng: &'a mut ThreadRng,
    ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64> {
        Normal::new(0.0, 1.0).unwrap().sample_iter(rng)
    }
}
