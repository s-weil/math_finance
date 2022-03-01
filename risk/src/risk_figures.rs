use crate::error::RiskError;

// trait Numeric
// {
//     // type N where N:std::cmp::PartialEq;
//     type N : std::ops::Div<Output=Self::N>;

//     fn not_divisible(n: Self::N) -> bool;

//     // #[doc(alias = "./")]
//     fn divide(nominator: Self::N, denominator: Self::N) -> Result<Self::N, RiskError> {
//         // TODO: check of isNan
//         if Self::not_divisible(denominator) {
//             Err(RiskError::ZeroDivision)
//         } else {
//             Ok(nominator/denominator)
//         }
//     }
// }

// impl Numeric for f32 {
//     type N = f32;
//     fn not_divisible(f: f32) -> bool {
//         f.abs() < 0.00001
//         // further checks
//     }
// }
use std::ops::{Sub, Div};

pub trait SharpeRatio {
    type N: Sub<Output = Self::N> + Div<Output = Self::N> + Clone;

    fn not_divisible(n: Self::N, threshold: Option<Self::N>) -> bool;

    /// The ratio of the expected value of the excess of the asset returns and the risk-free (or benchmark) returns,
    /// over the 'risk' (standard deviation) of the excess of asset and the risk-free (or benchmark) returns.
    /// Use the threshold for the division by 'risk'.
    /// See https://en.wikipedia.org/wiki/Sharpe_ratio
    /// resp. https://en.wikipedia.org/wiki/Information_ratio
    fn calculate(
        asset_return: Self::N,
        benchmark_return: Self::N,
        asset_std: Self::N,
        threshold: Option<Self::N>,
    ) -> Result<Self::N, RiskError> {
        if Self::not_divisible(asset_std.clone(), threshold) {
            return Err(RiskError::ZeroDivision);
        }
        let sr = (asset_return - benchmark_return) / asset_std;
        Ok(sr)
    }
}


#[macro_export]
macro_rules! impl_risk_figure {
    ($rf:ident<$impl_type:ty>) => {
        impl $rf for $impl_type {
            type N = $impl_type;
            fn not_divisible(f: Self::N, tolerance: Option<Self::N>) -> bool {
                match tolerance {
                    Some(tol) => f.abs() <= tol,
                    None => f.abs() == 0.0,
                }
            }
        }
    };
}

impl_risk_figure! { SharpeRatio<f32> }
impl_risk_figure! { SharpeRatio<f64> }

// TODO:
// use bigint with feature


#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn sharpe_ratio_f32() {
    //     let a = 0.2_f32;
    //     let res : Result<f32, RiskError> = SharpeRatio::calculate(a, 0.1_f32, 1.0_f32, None);
    //     assert_eq!(SharpeRatio::calculate(0.2_f32, 0.1_f32, 1.0_f32, None).unwrap(), 0.1_f32);
    // }
}