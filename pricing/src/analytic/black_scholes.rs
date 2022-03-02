use probability::distribution::{Distribution, Gaussian};

pub(crate) fn cdf(d: f64) -> f64 {
    let normal = Gaussian::new(0.0, 1.0);
    normal.distribution(d)
}

pub struct DerivativeParameter {
    /// the asset's price at time t
    pub asset_price: f64,
    /// the strike or exercise price of the asset
    pub strike: f64,
    /// (T - t) in years, where T is the time of the option's expiration and t is the current time
    pub time_to_expiration: f64,
    /// the annualized risk-free interest rate
    pub rfr: f64,
    /// the annualized standard deviation of the stock's returns
    pub vola: f64,
}

impl DerivativeParameter {
    pub fn new(
        asset_price: f64,
        strike: f64,
        time_to_expiration: f64,
        rfr: f64,
        vola: f64,
    ) -> Self {
        Self {
            asset_price,
            strike,
            time_to_expiration,
            rfr,
            vola,
        }
    }
}

pub trait OptionPrice {
    type Params;
    fn put(params: &Self::Params) -> f64;
    fn call(params: &Self::Params) -> f64;
}

/// European Put and Call option prices for stocks.
/// https://en.wikipedia.org/wiki/Black-Scholes_model
pub struct BlackScholes;

impl OptionPrice for BlackScholes {
    type Params = DerivativeParameter;

    fn call(dp: &DerivativeParameter) -> f64 {
        let sigma_exp = dp.vola * dp.time_to_expiration.sqrt();
        let d1 = ((dp.asset_price / dp.strike).ln()
            + (dp.rfr + dp.vola.powi(2) / 2.0) * dp.time_to_expiration)
            / sigma_exp;
        let d2 = d1 - sigma_exp;
        cdf(d1) * dp.asset_price - cdf(d2) * dp.strike * (-dp.rfr * dp.time_to_expiration).exp()
    }

    fn put(dp: &DerivativeParameter) -> f64 {
        let sigma_exp = dp.vola * dp.time_to_expiration.sqrt();
        let d1 = ((dp.asset_price / dp.strike).ln()
            + (dp.rfr + dp.vola.powi(2) / 2.0) * dp.time_to_expiration)
            / sigma_exp;
        let d2 = d1 - sigma_exp;
        cdf(-d2) * dp.strike * (-dp.rfr * dp.time_to_expiration).exp() - cdf(-d1) * dp.asset_price
    }
}

/// European Put and Call option prices for futures.
///  https://en.wikipedia.org/wiki/Black_model
pub struct Black76;

impl OptionPrice for Black76 {
    type Params = DerivativeParameter;

    fn call(dp: &DerivativeParameter) -> f64 {
        let sigma_exp = dp.vola * dp.time_to_expiration.sqrt();
        let d1 = ((dp.asset_price / dp.strike).ln() + (dp.vola.powi(2) / 2.0)) / sigma_exp;
        let d2 = d1 - sigma_exp;
        (-dp.rfr * dp.time_to_expiration).exp() * (cdf(d1) * dp.asset_price - cdf(d2) * dp.strike)
    }

    fn put(dp: &DerivativeParameter) -> f64 {
        let sigma_exp = dp.vola * dp.time_to_expiration.sqrt();
        let d1 = ((dp.asset_price / dp.strike).ln() + (dp.vola.powi(2) / 2.0)) / sigma_exp;
        let d2 = d1 - sigma_exp;
        (-dp.rfr * dp.time_to_expiration).exp() * (cdf(-d2) * dp.strike - cdf(-d1) * dp.asset_price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    const TOLERANCE: f64 = 1e-4;

    #[test]
    fn normal_cdf() {
        let center_value = cdf(0.0);
        assert_eq!(center_value, 0.5);

        let sigma_top = cdf(1.0); // mu + 1 sigma
        assert_approx_eq!(sigma_top, 0.8413, 0.0001); // table value for 1.0
    }

    #[test]
    fn european_call() {
        let dp = DerivativeParameter::new(300.0, 250.0, 1.0, 0.03, 0.15);
        assert_approx_eq!(BlackScholes::call(&dp), 58.8197, TOLERANCE);

        let dp = DerivativeParameter::new(310.0, 250.0, 3.5, 0.05, 0.25);
        assert_approx_eq!(BlackScholes::call(&dp), 113.4155, TOLERANCE);
    }

    #[test]
    fn european_put() {
        let dp = DerivativeParameter::new(300.0, 250.0, 1.0, 0.03, 0.15);
        assert_approx_eq!(BlackScholes::put(&dp), 1.4311, TOLERANCE);

        let dp = DerivativeParameter::new(310.0, 250.0, 3.5, 0.05, 0.25);
        assert_approx_eq!(BlackScholes::put(&dp), 13.2797, TOLERANCE);
    }

    #[test]
    fn european_put_call_parity() {
        let dp = DerivativeParameter::new(300.0, 250.0, 1.0, 0.03, 0.15);
        let put_call_parity = BlackScholes::call(&dp) - BlackScholes::put(&dp);
        assert_eq!(
            put_call_parity,
            dp.asset_price - dp.strike * (-dp.rfr * dp.time_to_expiration).exp()
        );
    }
}
