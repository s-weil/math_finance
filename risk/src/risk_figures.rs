use crate::error::RiskError;
use std::ops::{Add, Div, Mul, Sub};

/// Mimic the key features of a field.
pub trait PseudoField:
    Sized + Add<Output = Self> + Div<Output = Self> + Mul<Output = Self> + Sub<Output = Self>
{
    fn is_divisible(&self, threshold: Option<Self>) -> bool;
}

#[macro_export]
macro_rules! impl_numeric {
    ($impl_type:ty) => {
        impl PseudoField for $impl_type {
            fn is_divisible(&self, tolerance: Option<Self>) -> bool {
                match tolerance {
                    Some(tol) => self.abs() >= tol,
                    None => self.abs() != 0.0,
                }
            }
        }
    };
}

impl_numeric! { f32 }
impl_numeric! { f64 }
// TODO: add bigint dependency and implementation with feature flag

pub(crate) fn asset_bmk_ratio<Numeric>(
    asset_return: Numeric,
    benchmark_return: Numeric,
    excess_std: Numeric,
    threshold: Option<Numeric>,
) -> Result<Numeric, RiskError>
where
    Numeric: PseudoField,
{
    if !(excess_std.is_divisible(threshold)) {
        return Err(RiskError::ZeroDivision);
    }
    let ratio = (asset_return - benchmark_return) / excess_std;
    Ok(ratio)
}

/// The ratio of the expected value of the excess of the asset returns and the risk-free rates,
/// over the 'risk' (standard deviation) of the excess of asset and the risk-free-rate rates.
/// Use the threshold for the division by 'risk'.
/// See https://en.wikipedia.org/wiki/Sharpe_ratio
pub fn sharpe_ratio<Numeric>(
    asset_return: Numeric,
    riskfree_rate: Numeric,
    excess_std: Numeric,
    threshold: Option<Numeric>,
) -> Result<Numeric, RiskError>
where
    Numeric: PseudoField,
{
    asset_bmk_ratio(asset_return, riskfree_rate, excess_std, threshold)
}

/// The ratio of the expected value of the excess of the asset returns and the benchmark returns,
/// over the 'risk' (standard deviation) of the excess of asset and the benchmark returns.
/// Use the threshold for the division by 'risk'.
/// See https://en.wikipedia.org/wiki/Information_ratio
pub fn information_ratio<Numeric>(
    asset_return: Numeric,
    benchmark_return: Numeric,
    excess_std: Numeric,
    threshold: Option<Numeric>,
) -> Result<Numeric, RiskError>
where
    Numeric: PseudoField,
{
    asset_bmk_ratio(asset_return, benchmark_return, excess_std, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_bmk_ratio_f32() {
        assert_eq!(
            asset_bmk_ratio(0.2_f32, 0.1_f32, 1.0_f32, None).unwrap(),
            0.1_f32
        );
        assert_eq!(
            asset_bmk_ratio(0.2_f32, 0.1_f32, 0.01_f32, None).unwrap(),
            10.0_f32
        );

        assert!(asset_bmk_ratio(0.2_f32, 0.1_f32, 0.01_f32, Some(0.05)).is_err());
        assert!(asset_bmk_ratio(0.2_f32, 0.1_f32, 0.01_f32, Some(0.01)).is_ok());
    }

    #[test]
    fn asset_bmk_ratio_f64() {
        assert_eq!(
            asset_bmk_ratio(0.2_f64, 0.1_f64, 1.0_f64, None).unwrap(),
            0.1_f64
        );
        assert_eq!(
            asset_bmk_ratio(0.2_f64, 0.1_f64, 0.01_f64, None).unwrap(),
            10.0_f64
        );

        assert!(asset_bmk_ratio(0.2_f64, 0.1_f64, 0.01_f64, Some(0.05)).is_err());
        assert!(asset_bmk_ratio(0.2_f64, 0.1_f64, 0.01_f64, Some(0.01)).is_ok());
    }
}
