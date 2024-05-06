use crate::{layer::Layer, val::Val};

pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    pub fn new(num_inputs: usize, mut layer_config: Vec<usize>) -> Self {
        layer_config.insert(0, num_inputs);
        Self {
            layers: layer_config
                .iter()
                .zip(layer_config.iter().skip(1))
                .map(|(i, o)| Layer::new(*i, *o))
                .collect(),
        }
    }

    pub fn forward(&self, xs: &[f64]) -> Vec<Val> {
        let mut input = xs.iter().map(|x| Val::from(*x)).collect::<Vec<_>>();

        for layer in &self.layers {
            input = layer.forward(&input);
            // The output of this layer becomes the input to the next layer.
        }
        input
    }
}

#[cfg(test)]
mod tests {
    use super::Mlp;

    #[test]
    fn test_mlp() {
        let x = vec![2.0, 3.0, -1.0];
        let mlp = Mlp::new(3, vec![4, 4, 1]);
        let output = mlp.forward(&x);
        println!("{output:?}");
        output[0].visualize();
    }
}
