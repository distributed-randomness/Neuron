//! This module borrows heavily from
//! https://github.com/danielway/micrograd-rs/blob/master/src/value.rs
use std::{
    cell::{Ref, RefCell},
    collections::HashSet,
    fmt::Display,
    hash::Hash,
    ops::Deref,
    rc::Rc,
};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Val(Rc<RefCell<ValInternal>>);

type PropagateGradientBackwardsFn = fn(value: &Ref<ValInternal>);

#[derive(Clone, Debug)]
pub struct ValInternal {
    data: f64,
    gradient: f64,
    label: Option<String>,
    operation: Option<String>,
    parents: Vec<Val>,
    propagate: Option<PropagateGradientBackwardsFn>,
}

impl Val {
    pub fn new(data: f64, label: &str) -> Self {
        Self::with_neuron_internal(ValInternal {
            data,
            gradient: 0.0,
            label: Some(label.to_string()),
            operation: None,
            parents: vec![],
            propagate: None,
        })
    }

    fn with_neuron_internal(value: ValInternal) -> Val {
        Val(Rc::new(RefCell::new(value)))
    }

    pub fn with_label(self, label: &str) -> Val {
        self.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn gradient(&self) -> f64 {
        self.borrow().gradient
    }

    pub fn reset_gradient(&self) {
        self.borrow_mut().gradient = 0.0;
    }

    pub fn back_prop_gradient(&self) {
        self.borrow_mut().gradient = 1.0;
        let mut visited: HashSet<Val> = HashSet::new();

        fn back_prop_internal(node: &Val, visited: &mut HashSet<Val>) {
            if !visited.contains(node) {
                visited.insert(node.clone());
                let borrowed = node.borrow();
                if let Some(f) = borrowed.propagate {
                    f(&borrowed);
                }

                for parent in &node.borrow().parents {
                    back_prop_internal(parent, visited);
                }
            }
        }

        back_prop_internal(self, &mut visited);
    }

    pub fn pow(&self, other: &Val) -> Val {
        let result = self.borrow().data.powf(other.borrow().data);

        let prop_fn: PropagateGradientBackwardsFn = |value| {
            let mut base = value.parents[0].borrow_mut();
            let power = value.parents[1].borrow();

            // d(x^(n))/dx = n . x^ (n-1)
            base.gradient += power.data * (base.data.powf(power.data - 1.0)) * value.gradient;
        };

        Val::with_neuron_internal(ValInternal::new(
            result,
            None,
            Some("^".to_string()),
            vec![self.clone(), other.clone()],
            Some(prop_fn),
        ))
    }

    pub fn relu(&self) -> Val {
        // If the value is positive, leave it as it is, if it is negative, reset it to zero.
        let result = if self.borrow().data < 0.0 {
            0.0
        } else {
            self.borrow().data
        };

        let prop_fn: PropagateGradientBackwardsFn = |value| {
            let mut first = value.parents[0].borrow_mut();

            first.gradient += if first.data > 0.0 {
                value.gradient
            } else {
                0.0
            };
        };

        Val::with_neuron_internal(ValInternal::new(
            result,
            None,
            Some("ReLU".to_string()),
            vec![self.clone()],
            Some(prop_fn),
        ))
    }

    #[cfg(feature = "notebook")]
    pub fn visualize(&self) {
        use petgraph::{graph::NodeIndex, Graph};
        use petgraph_evcxr::draw_graph;

        type GraphTy = Graph<String, String, petgraph::Directed>;

        let mut g: GraphTy = Graph::new();

        fn traverse(node: &Val, node_idx: NodeIndex, g: &mut GraphTy) {
            for parent in &node.borrow().parents {
                let parent_idx = g.add_node(parent.to_string());

                g.add_edge(parent_idx, node_idx, String::new());

                traverse(parent, parent_idx, g);
            }
        }

        let node_idx = g.add_node(self.to_string());
        traverse(self, node_idx, &mut g);

        draw_graph(&g);
    }
}

impl ValInternal {
    fn new(
        data: f64,
        label: Option<String>,
        op: Option<String>,
        prev: Vec<Val>,
        propagate: Option<PropagateGradientBackwardsFn>,
    ) -> ValInternal {
        ValInternal {
            data,
            gradient: 0.0,
            label,
            operation: op,
            parents: prev,
            propagate,
        }
    }
}

impl PartialEq for ValInternal {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.gradient == other.gradient
            && self.label == other.label
            && self.operation == other.operation
            && self.parents == other.parents
    }
}
impl Eq for ValInternal {}

impl Deref for Val {
    type Target = Rc<RefCell<ValInternal>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Add<Val> for Val {
    type Output = Val;

    fn add(self, other: Val) -> Self::Output {
        let result = self.borrow().data + other.borrow().data;

        let prop_fn: PropagateGradientBackwardsFn = |value| {
            if *value.parents[1].borrow() == *value.parents[0].borrow() {
                // The both the parent nodes are the same.
                let mut first = value.parents[0].borrow_mut();
                first.gradient += 2.0 * value.gradient;
            } else {
                let mut first = value.parents[0].borrow_mut();
                let mut second = value.parents[1].borrow_mut();

                first.gradient += value.gradient;
                second.gradient += value.gradient;
            }
        };

        Val::with_neuron_internal(ValInternal::new(
            result,
            None,
            Some("+".to_string()),
            vec![self.clone(), other.clone()],
            Some(prop_fn),
        ))
    }
}

impl std::ops::Neg for Val {
    type Output = Val;

    fn neg(self) -> Self::Output {
        Val::from(-1.0) * self
    }
}

impl From<f64> for Val {
    fn from(t: f64) -> Val {
        Val::with_neuron_internal(ValInternal::new(t, None, None, Vec::new(), None))
    }
}

impl std::ops::Mul<Val> for Val {
    type Output = Val;

    fn mul(self, other: Val) -> Self::Output {
        &self * other

        // let result = self.borrow().data * other.borrow().data;

        // let prop_fn: PropagateGradientBackwardsFn = |value| {
        //     if *value.parents[1].borrow() == *value.parents[0].borrow() {
        //         // The both the parent nodes are the same.
        //         let mut first = value.parents[0].borrow_mut();
        //         first.gradient += 2.0 * first.data;
        //     } else {
        //         let mut first = value.parents[0].borrow_mut();
        //         let mut second = value.parents[1].borrow_mut();

        //         first.gradient += second.data * value.gradient;
        //         second.gradient += first.data * value.gradient;
        //     }
        // };

        // Val::with_neuron_internal(ValInternal::new(
        //     result,
        //     None,
        //     Some("*".to_string()),
        //     vec![self.clone(), other.clone()],
        //     Some(prop_fn),
        // ))
    }
}

impl std::ops::Mul<Val> for &Val {
    type Output = Val;

    fn mul(self, other: Val) -> Self::Output {
        let result = self.borrow().data * other.borrow().data;

        let prop_fn: PropagateGradientBackwardsFn = |value| {
            if *value.parents[1].borrow() == *value.parents[0].borrow() {
                // The both the parent nodes are the same.
                let mut first = value.parents[0].borrow_mut();
                first.gradient += 2.0 * first.data;
            } else {
                let mut first = value.parents[0].borrow_mut();
                let mut second = value.parents[1].borrow_mut();

                first.gradient += second.data * value.gradient;
                second.gradient += first.data * value.gradient;
            }
        };

        Val::with_neuron_internal(ValInternal::new(
            result,
            None,
            Some("*".to_string()),
            vec![self.clone(), other.clone()],
            Some(prop_fn),
        ))
    }
}

impl Display for ValInternal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = if let Some(label) = &self.label {
            label
        } else {
            ""
        };

        let op = if let Some(op) = &self.operation {
            op
        } else {
            ""
        };
        write!(f, "{label}| op:{op}, v:{}, g:{}", self.data, self.gradient)
    }
}

impl Hash for ValInternal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        self.label.hash(state);
        self.operation.hash(state);
        self.parents.hash(state);
    }
}

impl Hash for Val {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

#[cfg(test)]
mod tests {
    use super::Val;

    #[test]
    #[cfg(feature = "notebook")]
    fn test_nn() {
        let a = Val::new(2.0, "a");
        let b = Val::new(-3.0, "b");
        let c = Val::new(10.0, "c");

        let e = a * b;
        let e = e.with_label("e");

        let d = e + c;
        let d = d.with_label("d");

        let f = Val::new(-2.0, "f");

        let l = d * f;
        let l = l.with_label("L");

        // Look here for the gradient values in the video.
        // https://youtu.be/VMj-3S1tku0?t=2984
        l.back_prop_gradient();

        l.visualize();
    }

    #[test]
    fn add_node_parents_same() {
        let a: Val = Val::new(3.0, "a");
        let b: Val = a.clone() + a;
        let b = b.with_label("b");
        b.back_prop_gradient();
    }

    #[test]
    fn mul_node_parents_same() {
        let a: Val = Val::new(3.0, "a");
        let b: Val = a.clone() * a;
        let b = b.with_label("b");
        b.back_prop_gradient();
    }
}
