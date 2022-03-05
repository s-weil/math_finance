// https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662

use rand::{self, prelude::ThreadRng};
use rand_distr::{DistIter, Distribution, Normal};
use crate::simulation::{PathGenerator, SampleGenerator};


pub struct BlackScholesModelParams {
    r: f64,
    vola: f64,
    dt: f64
}

impl From<BlackScholesModelParams> for GeometricBrownianMotion {
    fn from(bs_params: BlackScholesModelParams) -> Self {
        todo!("not yet implemtented");
        GeometricBrownianMotion::new(0.0, 0.0, 0.0)
    }
}

/// Model params for the SDE $dS_t / S_t = mu dt + sigma dW_t$, where $dW_t ~ N(0, sqrt(dt))$
/// https://en.wikipedia.org/wiki/Geometric_Brownian_motion
pub struct GeometricBrownianMotion {
    /// drift term
    mu: f64,
    /// volatility
    sigma: f64,
    /// change in time
    dt: f64,
}


impl GeometricBrownianMotion {
    pub fn new(drift: f64, vola: f64, dt: f64) -> Self {
        Self { mu: drift, dt, sigma: vola }
    }

    /// See the solution of https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    pub fn sample(&self, s0: f64, z: f64) -> f64 {
        let ret = self.dt * (self.mu + self.sigma.powi(2) / 2.0 * z);
        s0 * ret.exp()
    }

    pub fn get_samples(
        &self,
        price: f64,
        nr_samples: usize,
        normal_distr: DistIter<Normal<f64>, &mut ThreadRng, f64>,
    ) -> f64 {
        normal_distr
            .take(nr_samples)
            .fold(price, |curr_p, z| self.sample(curr_p, z))
    }

    pub fn sample_path(
        &self,
        price: f64,
        nr_steps: usize,
        normal_distr: DistIter<Normal<f64>, &mut ThreadRng, f64>,
    ) -> Vec<f64> {
        let mut path = Vec::with_capacity(nr_steps + 1);

        let mut curr_p = price;
        path.push(curr_p);

        for z in normal_distr.take(nr_steps) {
            curr_p = self.sample(curr_p, z);
            path.push(curr_p)
        }

        path
    }
}


impl PathGenerator for GeometricBrownianMotion {
    type Dist = Normal<f64>;

    fn sample_path(
        &self,
        price: f64,
        nr_steps: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> Vec<f64> {
        self.sample_path(price, nr_steps, dist_iter)
    }

    fn distribution<'a>(
        &self,
        rng: &'a mut ThreadRng,
    ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64> {
        Normal::new(0.0, 1.0).unwrap().sample_iter(rng)
    }
}


impl SampleGenerator for GeometricBrownianMotion {
    type Dist = Normal<f64>;

    fn sample(
        &self,
        price: f64,
        nr_samples: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> f64 {
        self.get_samples(price, nr_samples, dist_iter)
    }   

    fn distribution<'a>(
        &self,
        rng: &'a mut ThreadRng,
    ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64> {
        Normal::new(0.0, 1.0).unwrap().sample_iter(rng)
    }
}
