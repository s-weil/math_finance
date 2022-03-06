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
