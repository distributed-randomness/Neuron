use rand::{thread_rng, Rng};

use crate::val::Val;

pub struct Neuron {
    weights: Vec<Val>,
    bias: Val,
}

impl Neuron {
    pub fn new(num_input: usize) -> Neuron {
        let mut rng = thread_rng();
        let weights = (0..num_input)
            .map(|_| Val::from(rng.gen_range(-1.0..1.0)))
            .collect::<Vec<_>>();
        let bias = Val::from(rng.gen_range(-1.0..1.0)).with_label("b");

        Self { weights, bias }
    }

    pub fn forward(&self, inputs: &[Val]) -> Val {
        inputs
            .iter()
            .zip(self.weights.iter().cloned())
            .fold(self.bias.clone(), |acc, (a, b)| acc + a * b)
            .relu()
    }
}
