use std::hash::Hash;

use crate::simulation::PathEvaluator;

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

pub trait PathPricer<Path> {
    fn eval(&self, input: Path) -> Option<f64>;
}

pub struct GreekEngine<RandomPath, Path, OptionInput>
where
    OptionInput: Eq + Hash + Clone, // TODO: idea is to store dynamic transformations depending on input
{
    rnd_paths: Vec<RandomPath>,
    shift_size: f64, // TODO: should be configurarble for every greek
    pricer: Box<dyn PathPricer<Path>>,
    dynamics: Box<dyn Dynamics<OptionInput, RandomPath, Path>>,
}

impl<RandomPath, Path, OptionInput> GreekEngine<RandomPath, Path, OptionInput>
where
    OptionInput: Eq + Hash + Clone,
{
    // pub fn new(rnd_paths: Vec<RandomPath>, shift_size: f64) -> Self {
    //     Self {
    //         rnd_paths,
    //         shift_size,
    //     }
    // }

    /// The payoff encodes already the dynamics and the actualy payoff
    pub fn theoretical_value(&self, pay_off: impl Fn(&RandomPath) -> Option<f64>) -> Option<f64> {
        let path_evaluator = PathEvaluator::new(&self.rnd_paths);
        path_evaluator.evaluate_average(pay_off)
    }
}
