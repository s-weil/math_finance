use ndarray::arr1;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::simulation::monte_carlo::PathGenerator;

pub struct MultivariateGeometricBrownianMotion {
    initial_values: Array1<f64>,
    /// drift term
    drifts: Array1<f64>,
    /// volatility
    cholesky_factor: Array2<f64>,
    /// change in time
    dt: f64,
}

impl MultivariateGeometricBrownianMotion {
    pub fn new(
        initial_values: Array1<f64>,
        drifts: Array1<f64>,
        cholesky_factor: Array2<f64>,
        dt: f64,
    ) -> Self {
        let iv_shape = initial_values.shape();
        let drifts_shape = drifts.shape();
        let matrix_shape = cholesky_factor.shape();

        assert_eq!(iv_shape, drifts_shape);
        assert_eq!(matrix_shape, &[drifts_shape[0], drifts_shape[0]]);

        // TODO: add a check that cholesky_factor is triangular; oR provide only a constructor using the correlation matrix
        // https://docs.rs/ndarray-linalg/0.9.0/ndarray_linalg/cholesky/index.html
        // use ndarray_linalg::cholesky::*;

        Self {
            initial_values,
            drifts,
            cholesky_factor,
            dt,
        }
    }

    fn dim(&self) -> usize {
        self.initial_values.shape()[0]
    }

    /// See https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    pub(crate) fn step(&self, st: &Array1<f64>, std_normal_vec: &Array1<f64>) -> Array1<f64> {
        let d_st_s0: Array1<f64> =
            self.dt * &self.drifts + self.dt.sqrt() * self.cholesky_factor.dot(std_normal_vec);

        st + st * &d_st_s0
    }

    pub fn transform_path(&self, sample_matrix: &Array2<f64>, nr_samples: usize) -> Array2<f64> {
        let mut multivariate_normals = self.dt.sqrt() * self.cholesky_factor.dot(sample_matrix);
        let dim = self.dim();

        //TODO: possible to use multivariate_normals.axis_windows(Axis(0), 2)?

        // overwrite the first column by initial prices
        for idx in 0..dim {
            multivariate_normals[[idx, 0]] = self.initial_values[idx];
        }

        for idx in 1..nr_samples {
            let st = multivariate_normals.column(idx - 1);
            let rnd = multivariate_normals.column(idx);
            let d_st_s0: Array1<f64> = self.dt * &self.drifts + rnd;
            let stn = &st + &st * &d_st_s0;
            for i in 0..dim {
                multivariate_normals[[i, idx]] = stn[i];
            }
        }
        multivariate_normals
    }
}

impl Distribution<Array1<f64>> for MultivariateGeometricBrownianMotion {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let standard_normals: Vec<f64> = rng.sample_iter(StandardNormal).take(self.dim()).collect();

        // NOTE: be careful of fixed initial value!
        self.step(&self.initial_values, &Array1::from(standard_normals))
    }
}

impl PathGenerator<Array2<f64>> for MultivariateGeometricBrownianMotion {
    #[inline]
    fn sample_path<R>(&self, rn_generator: &mut R, nr_samples: usize) -> Array2<f64>
    where
        R: Rng + ?Sized,
    {
        let dim = self.dim();
        let distr = ndarray_rand::rand_distr::StandardNormal;
        // create one extra dummy column
        let sample_matrix =
            ndarray::Array::random_using((dim, 1 + nr_samples), distr, rn_generator);

        self.transform_path(&sample_matrix, 1 + nr_samples)
    }
}

// TODO: still needed?
impl PathGenerator<Vec<Array1<f64>>> for MultivariateGeometricBrownianMotion {
    #[inline]
    fn sample_path<R>(&self, rn_generator: &mut R, nr_samples: usize) -> Vec<Array1<f64>>
    where
        R: Rng,
    {
        let dim = self.dim();

        let mut path = Vec::with_capacity(nr_samples + 1);

        path.push(self.initial_values.clone());

        // create the random normal numbers for the whole path and all dimensions
        let path_std_normals: Vec<f64> = rn_generator
            .sample_iter(StandardNormal)
            .take(nr_samples * dim)
            .collect();

        for (idx, _) in path_std_normals.iter().enumerate().step_by(dim) {
            let zs_slice = arr1(&path_std_normals[idx..idx + dim]);
            let curr_p = path.last().unwrap();
            let sample = self.step(curr_p, &zs_slice);
            path.push(sample);
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use crate::simulation::{monte_carlo::MonteCarloPathSimulator, PathEvaluator};

    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn sample() {
        let initial_values = arr1(&[1.0, 2.0, 3.0]);
        let drifts = arr1(&[0.1, 0.2, 0.3]);
        let cholesky_factor = arr2(&[[1.0, 0.5, 0.1], [0.0, 0.6, 0.7], [0.0, 0.0, 0.8]]);
        let dt = 4.0;

        let mv_gbm =
            MultivariateGeometricBrownianMotion::new(initial_values, drifts, cholesky_factor, dt);

        let rand_normals = arr1(&[0.1, -0.1, 0.05]);
        let sample = mv_gbm.step(&mv_gbm.initial_values, &rand_normals);
        assert_eq!(sample, arr1(&[1.51, 3.5, 6.84]));
    }

    #[test]
    fn basket_stock_price_simulation() {
        let nr_paths = 5_000;
        let nr_steps = 100;

        let initial_values = arr1(&[110.0, 120.0, 130.0]);
        let drifts = arr1(&[0.01, 0.02, 0.03]);
        let cholesky_factor = arr2(&[[1.0, 0.05, 0.1], [0.0, 0.6, 0.07], [0.0, 0.0, 1.0]]);
        let dt = 1.0 / 100.0;

        let mv_gbm =
            MultivariateGeometricBrownianMotion::new(initial_values, drifts, cholesky_factor, dt);

        let mc_simulator: MonteCarloPathSimulator<_, rand_hc::Hc128Rng, Array2<f64>> =
            MonteCarloPathSimulator::new(mv_gbm, Some(42));

        let paths = mc_simulator.simulate_paths(nr_paths, nr_steps);
        assert_eq!(paths.len(), nr_paths);
        assert_eq!(&paths[0].shape(), &[3, nr_steps + 1]);

        dbg!(&paths[0]);
        dbg!(&paths[0].column(0));
        dbg!(&paths[0].column(nr_steps));

        let path_eval = PathEvaluator::new(&paths);
        let avg_price = path_eval
            .evaluate_average(|path| path.axis_iter(Axis(1)).last().map(|p| p.sum() / 3.0));
        assert!(avg_price.unwrap() > 0.0);
        dbg!(avg_price.unwrap());
    }
}
