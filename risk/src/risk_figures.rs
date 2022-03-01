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

pub trait SharpeRatio {
    type N: std::ops::Sub<Output = Self::N> + std::ops::Div<Output = Self::N> + Clone;

    fn not_divisible(n: Self::N, tolerance: Option<Self::N>) -> bool;

    /// The ratio of the expected value of the excess of the asset return and the benchmark return,
    /// over the risk of the asset.
    /// https://en.wikipedia.org/wiki/Sharpe_ratio
    fn calculate(
        asset_return: Self::N,
        benchmark_return: Self::N,
        asset_std: Self::N,
        tolerance: Option<Self::N>,
    ) -> Result<Self::N, RiskError> {
        if Self::not_divisible(asset_std.clone(), tolerance) {
            return Err(RiskError::ZeroDivision);
        }
        let sr = (asset_return - benchmark_return) / asset_std;
        Ok(sr)
    }
}


#[macro_export]
macro_rules! impl_sharpe_ratio {
    ($impl_type:ty) => {
        impl SharpeRatio for $impl_type {
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

impl_sharpe_ratio! { f32 }
impl_sharpe_ratio! { f64 }

// TODO:
// use bigint with feature


