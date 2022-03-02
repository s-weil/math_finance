// https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662

use rand::{self, prelude::ThreadRng};
use rand_distr::{DistIter, Normal};

pub struct GeometricBrownianMotion {
    drift: f64,
    vola: f64,
    dt: f64,
}

impl GeometricBrownianMotion {
    pub fn new(drift: f64, vola: f64, dt: f64) -> Self {
        Self { drift, dt, vola }
    }

    pub fn simulate_step(&self, price: f64, z: f64) -> f64 {
        let ret = (self.dt * (self.drift + self.vola * z)).exp();
        price * ret
    }

    pub fn simulate(
        &self,
        price: f64,
        nr_steps: usize,
        normal_distr: DistIter<Normal<f64>, &mut ThreadRng, f64>,
    ) -> f64 {
        normal_distr
            .take(nr_steps)
            .fold(price, |curr_p, z| self.simulate_step(curr_p, z))
    }

    pub fn simulate_path(
        &self,
        price: f64,
        nr_steps: usize,
        normal_distr: DistIter<Normal<f64>, &mut ThreadRng, f64>,
    ) -> Vec<f64> {
        let mut path = Vec::with_capacity(nr_steps + 1);

        let mut curr_p = price;
        path.push(curr_p);

        for z in normal_distr.take(nr_steps) {
            curr_p = self.simulate_step(curr_p, z);
            path.push(curr_p)
        }

        path
    }
}
