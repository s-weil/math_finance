pub trait Sensitivity<Paths, Config> {
    fn randomness(&self) -> Paths;

    fn calculate(&self, randomness: &Paths, cfg: &Config) -> Option<f64>;
}

pub trait Dynamcis<RandomPath, Path> {
    fn transform(&self, randomness: &RandomPath) -> Path;
}

// TODO: maybe evals just on a single path, but that would be payoff rather -> check with american option
pub trait Pricer<Path> {
    fn eval(&self, paths: &[Path]) -> Option<f64>;
}

// theoretical value for put  /call: -> translated into payoff
// apply GBM, apply Payoff

pub struct GreekEngine<'a, Paths> {
    randomness: &'a Paths,
}
