pub trait Sensitivity<Paths, Config> {
    fn randomness(&self) -> Paths;

    fn calculate(&self, randomness: &Paths, cfg: &Config) -> Option<f64>;
}

/// Models the dynamics of the asset(s) price.
/// RandomPath represents the underlying random distribution, 
/// which is transformed to the price path.
pub trait Dynamics<Input, RandomPath, Path> {
    fn transform(&self, input: Input, rnd_path: RandomPath) -> Path;
}

// TODO: maybe evals just on a single path, but that would be payoff rather -> check with american option
// pub trait Pricer<Input, RandomPath, Path> {
//  <RandomPath>;

//     fn eval(&self, paths: &[Path]) -> Option<f64>;
// }

// theoretical value for put  /call: -> translated into payoff
// apply GBM, apply Payoff

pub struct GreekEngine<Path> {
    rnd_paths: Vec<Path>,
}

impl GreekEngine<Path> {

    pub fn new(rnd_paths: Vec<Path>)

}


