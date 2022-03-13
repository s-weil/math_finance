// https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662

use crate::simulation::PathGenerator;
use rand::{self, prelude::ThreadRng};
use rand_distr::{DistIter, Distribution, Normal};

use super::monte_carlo::McDistIter;

/// Model params for the SDE
/// '''math
/// dS_t / S_t = mu dt + sigma dW_t
/// ''', where '$dW_t ~ N(0, sqrt(dt))$'
/// https://en.wikipedia.org/wiki/Geometric_Brownian_motion
pub struct GeometricBrownianMotion {
    initial_value: f64,
    /// drift term
    mu: f64,
    /// volatility
    sigma: f64,
    /// change in time
    dt: f64,
}

impl GeometricBrownianMotion {
    pub fn new(initial_value: f64, drift: f64, vola: f64, dt: f64) -> Self {
        Self {
            initial_value,
            mu: drift,
            dt,
            sigma: vola,
        }
    }

    /// See https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    pub fn sample(&self, st: f64, z: f64) -> f64 {
        // let ret = self.dt * (self.mu - self.sigma.powi(2) / 2.0) + self.dt.sqrt() * self.sigma * z;
        // St * ret.exp()
        let d_st = st * (self.mu * self.dt + self.sigma * self.dt.sqrt() * z);
        st + d_st // d_St = S_t+1 - St
    }

    /*/
    fn path_value(
        &self,
        s0: f64,
        nr_steps: usize,
        dist_iter: DistIter<Normal<f64>, &mut ThreadRng, f64>,
        // normal_distr: impl Iterator<Item = f64>,
    ) -> f64 {
        dist_iter
            .take(nr_steps)
            .fold(s0, |curr_p, z| self.sample(curr_p, z))
    }
    */

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
            path.push(curr_p);
        }

        path
    }
}

impl McDistIter for GeometricBrownianMotion {
    type Dist = Normal<f64>;

    fn distribution<'a>(
        &self,
        rng: &'a mut ThreadRng,
    ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64> {
        Normal::new(0.0, 1.0).unwrap().sample_iter(rng)
    }
}

impl PathGenerator for GeometricBrownianMotion {
    fn sample_path(
        &self,
        nr_steps: usize,
        dist_iter: DistIter<Self::Dist, &mut ThreadRng, f64>,
    ) -> Vec<f64> {
        self.sample_path(self.initial_value, nr_steps, dist_iter)
    }
}

// impl SampleGenerator for GeometricBrownianMotion {
//     type Dist = Normal<f64>;

//     fn sample_value(
//         &self,
//         price: f64,
//         // dist_iter: impl Iterator<Item = f64>,
//         nr_steps: usize,
//         dist_iter: &DistIter<Self::Dist, &mut ThreadRng, f64>,
//     ) -> f64 {
//         self.sample_value(price, nr_steps, dist_iter)
//     }

//     fn distribution<'a>(
//         &self,
//         rng: &'a mut ThreadRng,
//     ) -> DistIter<Self::Dist, &'a mut ThreadRng, f64> {
//         Normal::new(0.0, 1.0).unwrap().sample_iter(rng)
//     }
// }
