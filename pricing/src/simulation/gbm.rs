use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use rand_hc::Hc128Rng;

use crate::simulation::monte_carlo::PathSampler;

/// Model params for the SDE
/// '''math
/// dS_t / S_t = mu dt + sigma dW_t
/// ''', where $dW_t ~ N(0, sqrt(dt))$
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

    pub fn base_distribution(&self) -> StandardNormal {
        StandardNormal
    }

    /// See https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    pub fn step(&self, st: f64, z: f64) -> f64 {
        // let ret = self.dt * (self.mu - self.sigma.powi(2) / 2.0) + self.dt.sqrt() * self.sigma * z;
        // St * ret.exp()
        let d_st = st * (self.mu * self.dt + self.sigma * self.dt.sqrt() * z);
        st + d_st // d_St = S_t+1 - St
    }

    pub fn step_analytic(&self, st: f64, z: f64) -> f64 {
        let ret = self.dt * (self.mu - self.sigma.powi(2) / 2.0) + self.dt.sqrt() * self.sigma * z;
        st * ret.exp()
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

    pub fn generate_path(&self, initial_value: f64, standard_normals: &[f64]) -> Vec<f64> {
        let mut path = Vec::with_capacity(standard_normals.len() + 1);

        let mut curr_p = initial_value;
        path.push(curr_p);

        for z in standard_normals {
            curr_p = self.step(curr_p, *z);
            path.push(curr_p);
        }

        path
    }

    pub fn generate_in_place(&self, standard_normals: &mut [f64]) {
        let mut curr_p = self.initial_value;

        for z in standard_normals.iter_mut() {
            curr_p = self.step(curr_p, *z);
            *z = curr_p;
        }
    }
}

impl Distribution<f64> for GeometricBrownianMotion {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // TODO: be careful of initial value!
        self.step_analytic(self.initial_value, rng.sample(StandardNormal))
    }
}

impl PathSampler<Vec<f64>> for GeometricBrownianMotion {
    type Distribution = StandardNormal;

    fn base_distribution(&self) -> Self::Distribution {
        StandardNormal
    }

    #[inline]
    fn sample_path(&self, rn_generator: &mut Hc128Rng, nr_samples: usize) -> Vec<f64> {
        let distr = StandardNormal;
        let mut standard_normals: Vec<f64> =
            rn_generator.sample_iter(distr).take(nr_samples).collect();

        self.generate_in_place(&mut standard_normals);
        standard_normals
    }
}
