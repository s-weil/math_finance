use crate::simulation::monte_carlo::PathGenerator;

use ndarray::{arr1, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use super::monte_carlo::SeedRng;

fn sample_vec_path<R, D>(rn_generator: &mut R, distr: D, nr_samples: usize) -> Vec<f64>
where
    R: SeedRng,
    D: Distribution<f64>,
{
    rn_generator.sample_iter(distr).take(nr_samples).collect()
}

impl PathGenerator<Vec<f64>> for rand_distr::StandardNormal {
    fn sample_path<R>(&self, rn_generator: &mut R, nr_samples: usize) -> Vec<f64>
    where
        R: SeedRng,
    {
        sample_vec_path(rn_generator, self, nr_samples)
    }
}

impl PathGenerator<Vec<f64>> for rand_distr::Normal<f64> {
    fn sample_path<SRng: SeedRng>(&self, rn_generator: &mut SRng, nr_samples: usize) -> Vec<f64> {
        sample_vec_path(rn_generator, self, nr_samples)
    }
}

#[derive(Clone, Debug)]
pub struct MultivariateNormalDistribution {
    /// expected values (as by coordinate)
    mu: Array1<f64>,
    /// correlation structure via the cholesky_factor $C$ which is upper triangular and satisfies
    /// $C^T*C = \Sigma$ for the covariance matrix $\Sigma$
    cholesky_factor: Array2<f64>,
}

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution
impl MultivariateNormalDistribution {
    pub fn new(mu: Array1<f64>, cholesky_factor: Array2<f64>) -> Self {
        let mu_shape = mu.shape();
        let matrix_shape = cholesky_factor.shape();

        assert_eq!(matrix_shape, &[mu_shape[0], mu_shape[0]]);

        // TODO: add a check that cholesky_factor is triangular;
        // or provide only a constructor using the correlation matrix (see https://docs.rs/ndarray-linalg/0.14.1/ndarray_linalg/cholesky/index.html)
        // for i in 0..matrix_shape[0] {
        //     for j in 0..matrix_shape[1] {
        //         assert!(cholesky_factor[[[i, j]] == 0.0);
        //     }
        // }

        Self {
            mu,
            cholesky_factor,
        }
    }

    pub fn dim(&self) -> usize {
        self.mu.shape()[0]
    }

    pub(crate) fn transform_sample(&self, standard_normals: &Array1<f64>) -> Array1<f64> {
        &self.mu + self.cholesky_factor.dot(standard_normals)
    }

    pub(crate) fn transform_path(&self, standard_normals_matrix: &Array2<f64>) -> Array2<f64> {
        let mut corr_standard_normals_path = self.cholesky_factor.dot(standard_normals_matrix);

        for mut col in corr_standard_normals_path.columns_mut() {
            let rdn = &self.mu + &col;
            col.assign(&rdn);
        }
        corr_standard_normals_path
    }
}

impl Distribution<Array1<f64>> for MultivariateNormalDistribution {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let standard_normals: Vec<f64> = rng.sample_iter(StandardNormal).take(self.dim()).collect();
        self.transform_sample(&Array1::from(standard_normals))
    }
}

// #[cfg(feature = "rand_isaac")]
impl PathGenerator<Array2<f64>> for MultivariateNormalDistribution {
    #[inline]
    fn sample_path<SRng: SeedRng>(
        &self,
        rn_generator: &mut SRng,
        nr_samples: usize,
    ) -> Array2<f64> {
        let dim = self.dim();
        let distr = ndarray_rand::rand_distr::StandardNormal;
        let sample_matrix = ndarray::Array::random_using((dim, nr_samples), distr, rn_generator);
        // let sample_matrix = Array2::from_shape_vec(
        //     (dim, nr_samples),
        //     rn_generator
        //         .sample_iter(distr)
        //         .take(nr_samples * dim)
        //         .collect(),
        // )
        // .unwrap(); // TODO deal with error

        self.transform_path(&sample_matrix)
    }
}

// TODO: Still needed?
impl PathGenerator<Vec<Array1<f64>>> for MultivariateNormalDistribution {
    /// Optimized version of
    /// ''' rn_generator.sample_iter(self).take(nr_samples).collect()'''
    #[inline]
    fn sample_path<SRng: SeedRng>(
        &self,
        rn_generator: &mut SRng,
        nr_samples: usize,
    ) -> Vec<Array1<f64>>
    where
        SRng: SeedRng,
    {
        let dim = self.dim();
        let standard_normals: Vec<f64> = StandardNormal.sample_path(rn_generator, nr_samples * dim);

        let mut path: Vec<Array1<f64>> = Vec::with_capacity(nr_samples);
        for (idx, _) in standard_normals.iter().enumerate().step_by(dim) {
            let slice = &standard_normals[idx..idx + dim];
            path.push(self.transform_sample(&arr1(slice)))
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use rand::SeedableRng;

    #[test]
    fn standard_normal_path() {
        let mut rn_generator = rand_hc::Hc128Rng::seed_from_u64(13241113);
        let samples = StandardNormal.sample_path(&mut rn_generator, 100_000);

        let mu = samples.iter().fold(0.0, |acc, z| acc + z) / 100_000.0;
        assert_eq!(mu, -0.004556843413074714);

        let variance = samples.iter().fold(0.0, |acc, z| acc + (z - mu).powi(2)) / 100_000.0;
        assert_eq!(variance, 0.9965887881497351);
    }

    #[test]
    fn sample() {
        let mut rn_generator = rand_hc::Hc128Rng::seed_from_u64(13241113);
        let mu = arr1(&[0.1, 0.2, 0.3]);

        // 'forgets' the random part
        let cholesky_factor = arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let mv_normal = MultivariateNormalDistribution::new(mu.clone(), cholesky_factor);
        let sample = mv_normal.sample(&mut rn_generator);
        assert_eq!(sample, mu);

        let cholesky_factor = arr2(&[[1.0, 0.5, 0.1], [0.0, 0.6, 0.7], [0.0, 0.0, 0.8]]);
        let mv_normal = MultivariateNormalDistribution::new(mu.to_owned(), cholesky_factor);
        let sample = mv_normal.sample(&mut rn_generator);
        assert_eq!(
            sample,
            arr1(&[-0.8743027133925354, -0.5306078148397926, 0.4544744348656101])
        );
    }

    #[test]
    fn samples_avg() {
        let mut rn_generator = rand_hc::Hc128Rng::seed_from_u64(13241114);

        let mu = arr1(&[0.1, 0.2, 0.3]);
        let cholesky_factor = arr2(&[[1.0, 0.5, 0.1], [0.0, 0.6, 0.7], [0.0, 0.0, 0.8]]);
        let mv_normal = MultivariateNormalDistribution::new(mu, cholesky_factor);
        let samples: Array2<_> = mv_normal.sample_path(&mut rn_generator, 100_000);

        assert_eq!(samples.shape(), &[3, 100_000]);

        let mut sums = arr1(&[0.0, 0.0, 0.0]);
        for c in samples.columns() {
            sums = &sums + &c;
        }

        // want approx 'assert_eq!(sums / 100_000.0, mu)'
        assert_eq!(
            sums / 100_000.0,
            arr1(&[0.09734041097783784, 0.20242533842636964, 0.3057350243384335])
        );
    }
}
