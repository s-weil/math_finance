use thiserror::Error;

#[derive(Error, Debug)]
pub enum RiskError {
    #[error("division by 0")]
    ZeroDivision,
}
