use crate::{neuron::Neuron, val::Val};

/// A layer of neurons.
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_neurons: usize) -> Self {
        Self {
            neurons: (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect(),
        }
    }

    pub fn forward(&self, inputs: &[Val]) -> Vec<Val> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }
}
