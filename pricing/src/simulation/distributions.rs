use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use rand_hc::Hc128Rng;

use crate::simulation::monte_carlo::PathSampler;
use ndarray::{Array1, Array2};

// impl DistributionExt for Normal<f64> {}

/// Inherits from the 'generating distribution'
// pub trait Sampler {
//     // : Distribution<f64> + Sized {
//     type GeneratingDistribution;
//     type SampleType;

//     fn rn_generator(&self, seed_nr: u64) -> Hc128Rng {
//         rand_hc::Hc128Rng::seed_from_u64(seed_nr)
//     }

//     fn distribution(&self) -> Self::GeneratingDistribution;

//     fn sample<'a>(&self, rn_generator: &'a mut Hc128Rng) -> Self::SampleType;
// }

// impl Sampler for Normal<f64> {
//     type GeneratingDistribution = Self;
//     type SampleType = f64;

//     fn distribution(&self) -> Self::GeneratingDistribution {
//         *self
//     }

//     fn sample<'a>(
//         &self,
//         rn_generator: &'a mut Hc128Rng,
//         // distr: &Self::GeneratingDistribution,
//     ) -> Self::SampleType {
//         rn_generator.sample(*self)
//     }
// }

impl PathSampler<f64> for Normal<f64> {
    fn sample_path<'a>(&self, rn_generator: &'a mut Hc128Rng, nr_samples: usize) -> Vec<f64> {
        rn_generator.sample_iter(*self).take(nr_samples).collect()
    }
}

impl PathSampler<f64> for StandardNormal {
    fn sample_path<'a>(&self, rn_generator: &'a mut Hc128Rng, nr_samples: usize) -> Vec<f64> {
        rn_generator.sample_iter(*self).take(nr_samples).collect()
    }
}

#[derive(Clone, Debug)]
pub struct MultivariateNormalDistribution {
    /// expected values (as by coordinate)
    mu: Array1<f64>,
    /// correlation structure via the cholesky_factor $C$ which satisfies
    /// $C*C^T = \Sigma$ for the covariance matrix $\Sigma$
    cholesky_factor: Array2<f64>,
}

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution
impl MultivariateNormalDistribution {
    // TODO: maybe rename to new unchecked

    // TODO: user https://docs.rs/ndarray-linalg/0.14.1/ndarray_linalg/cholesky/index.html
    pub fn new(mu: Array1<f64>, cholesky_factor: Array2<f64>) -> Self {
        let mu_shape = mu.shape();
        let matrix_shape = cholesky_factor.shape();

        assert_eq!(matrix_shape, &[mu_shape[0], mu_shape[0]]);

        // TODO: add a check that cholesky_factor is triangular; oR provide only a constructor using the correlation matrix
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

    // fn base_distribution(&self) -> StandardNormal {
    //     StandardNormal
    // }

    // pub fn sample_array<'a>(&self, generator: &'a mut Hc128Rng) -> Array1<f64> {
    //     let standard_normals = self.base_distribution().samples(generator, self.dim());
    //     self.transform_sample(&Array1::from(standard_normals))
    // }
}

impl Distribution<Array1<f64>> for MultivariateNormalDistribution {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let standard_normals: Vec<f64> = rng.sample_iter(StandardNormal).take(self.dim()).collect();

        // let mut standard_normals = Vec::with_capacity(self.dim());
        // for _ in 0..self.dim() {
        //     standard_normals.push(rng.sample(StandardNormal))
        // }

        self.transform_sample(&Array1::from(standard_normals))
    }
}

impl PathSampler<Array1<f64>> for MultivariateNormalDistribution {
    // fn samples<'a>(&self, rn_generator: &'a mut Hc128Rng, nr_samples: usize) -> Vec<Array1<f64>> {
    //     let dim = self.dim();

    //     let samples_vec: Vec<Array1<f64>> = Vec::with_capacity(nr_samples);

    //     for step in 0..nr_samples {
    //         let standard_normals: Vec<f64> =
    //             rn_generator.sample_iter(distr).take(nr_samples).collect();

    //         samples_vec.push(self.transform_sample(&Array1::from(standard_normals)))
    //     }

    //     samples_vec
    // }
}

// impl Sampler for MultivariateNormalDistribution {
//     type GeneratingDistribution = StandardNormal;
//     type SampleType = Array1<f64>;

//     fn distribution(&self) -> Self::GeneratingDistribution {
//         self.base_distribution()
//     }

//     fn sample<'a>(
//         self,
//         rn_generator: &'a mut Hc128Rng,
//         distr: &Self::GeneratingDistribution,
//     ) -> Self::SampleType {
//         let standard_normals: Vec<f64> = rn_generator.sample_iter(distr).take(self.dim()).collect();
//         self.transform_sample(&Array1::from(standard_normals))
//     }
// }

// impl PathSampler for MultivariateNormalDistribution {
//     fn samples<'a>(
//         &self,
//         rn_generator: &'a mut Hc128Rng,
//         distr: &Self::GeneratingDistribution,
//         nr_samples: usize,
//     ) -> Vec<Self::SampleType> {
//         let dim = self.dim();
//         // let distr = self.base_distribution();

//         let samples_vec: Vec<Self::SampleType> = Vec::with_capacity(nr_samples);

//         for step in 0..nr_samples {
//             let standard_normals: Vec<f64> =
//                 rn_generator.sample_iter(distr).take(nr_samples).collect();

//             samples_vec.push(self.transform_sample(&Array1::from(standard_normals)))
//         }

//         samples_vec
//     }
// }

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
        let mv_normal = MultivariateNormalDistribution::new(mu.clone(), cholesky_factor);
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
        let samples = mv_normal.sample_path(&mut rn_generator, 100_000);

        assert_eq!(samples.len(), 100_000);

        let sums = samples
            .iter()
            .fold(arr1(&[0.0, 0.0, 0.0]), |acc, s| &acc + s);

        // want approx 'assert_eq!(sums / 100_000.0, mu)'
        assert_eq!(
            sums / 100_000.0,
            arr1(&[0.09947180969400722, 0.2024348660185755, 0.30033038001924606])
        );
    }
}
