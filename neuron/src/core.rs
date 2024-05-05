use std::{cell::RefCell, rc::Rc};

use petgraph::Graph;

type ValTy = f32;

type _Neuron<'a> = Rc<RefCell<Neuron<'a>>>;
type BackProp = fn(&Neuron) -> ();

#[derive(derive_more::Display, PartialEq, Eq, Debug, Clone)]
pub enum Ops {
    #[display(fmt = "+")]
    Add,
    #[display(fmt = "*")]
    Mul,

    None,
}

#[derive(derive_more::Display, PartialEq, Debug, Clone)]
#[display(fmt = "val: {val}, grad: {grad}")]
pub struct Neuron<'a> {
    val: ValTy,
    pub name: &'a str,

    /// gradiant. How changing the value of the node changes the final output.
    grad: ValTy,

    /// The parent neuron on which some operations were done to produce this neuron.
    parents: Option<(_Neuron<'a>, _Neuron<'a>)>,

    /// What operation was done on the parents that generated this node.
    parent_ops: Ops,

    backprop: Option<BackProp>,
}

// pub fn type_of<T>(_: T) -> &'static str {
//     std::any::type_name::<T>()
// }

// #[macro_export]
// macro_rules! nn {
//     ($var_name:ident, $v:expr) => {
//         println!("Type of {} is {}", $v, type_of($v));

//         let ty = type_of($v);

//         if ty == "f64" || ty == "f32" {
//             println!("{} is a neuron", stringify!($var_name));

//             let $var_name = Neuron::new($v, stringify!($var_name));
//         } else {
//             // This is a neuron.
//             println!("{} is a neuron", stringify!($var_name));
//             let $var_name = $v;
//             $var_name.name = stringify!($var_name);
//         }
//     };
// }

impl<'a> Neuron<'a> {
    pub fn new(val: ValTy, name: &'a str) -> Self {
        Self {
            val,
            name,
            grad: 0.0,
            parents: None,
            parent_ops: Ops::None,
            backprop: None,
        }
    }

    pub fn with_parents_ops(
        val: ValTy,
        parent_ops: Ops,
        parents: (_Neuron<'a>, _Neuron<'a>),
    ) -> Self {
        Self {
            val,
            name: "",
            grad: 0.0,
            parents: Some(parents),
            parent_ops,
            backprop: None,
        }
    }

    pub fn visualize(&self) -> Graph<&str, &str> {
        let mut tree: Graph<&str, &str, petgraph::Directed> = Graph::new();
        let node = self;

        let mut visited = Vec::with_capacity(10);
        let i = tree.add_node(node.name);

        visited.push((node.clone(), i));

        while let Some((node, idx)) = visited.pop() {
            println!("{}", node.name);

            if let Some((p1, p2)) = &node.parents {
                let i1 = tree.add_node(&p1.borrow().name);

                visited.push((p1.borrow().clone(), i1));

                let i2 = tree.add_node(&p2.borrow().name);
                visited.push((p2.borrow().clone(), i2));

                tree.add_edge(i1, idx, "");
                tree.add_edge(i2, idx, "");
            }
        }
        tree
    }
}

impl<'a> std::ops::Add for Neuron<'a> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut node = Neuron::with_parents_ops(
            self.val + rhs.val,
            Ops::Add,
            (Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))),
        );

        let back_prop = |v: &Neuron| {
            if let Some((first, second)) = &v.parents {
                first.borrow_mut().grad += v.grad;
                second.borrow_mut().grad += v.grad;
            };
        };
        node.backprop = Some(back_prop);

        node
    }
}

impl<'a> std::ops::Mul for Neuron<'a> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut node = Neuron::with_parents_ops(
            self.val * rhs.val,
            Ops::Mul,
            (Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))),
        );

        let back_prop = |v: &Neuron| {
            if let Some((first, second)) = &v.parents {
                first.borrow_mut().grad += v.grad * second.borrow().val;
                second.borrow_mut().grad += v.grad * first.borrow().val;
            };
        };
        node.backprop = Some(back_prop);

        node
    }
}

#[cfg(test)]
mod tests {
    use super::{type_of, Neuron};

    #[test]
    fn test_add() {
        let a = Neuron::new(1.0, "a");
        let b = Neuron::new(1.1, "b");
        let c = Neuron::new(2.1, "c");

        let mut a_plus_b = a + b;
        a_plus_b.name = "a+b";

        assert_eq!(a_plus_b.val, c.val);
        assert_eq!(a_plus_b.grad, c.grad);

        if let Some(bp) = &a_plus_b.backprop {
            bp(&a_plus_b);
        }
        println!("{a_plus_b:?}");
    }

    #[test]
    fn test_mul() {
        let a = Neuron::new(0.2, "a");
        let b = Neuron::new(1.1, "b");
        let c = Neuron::new(0.22000001, "c");

        let mut a_mul_b = a * b;
        a_mul_b.name = "a*b";

        assert_eq!(a_mul_b.val, c.val);
        assert_eq!(a_mul_b.grad, c.grad);

        if let Some(bp) = &a_mul_b.backprop {
            bp(&a_mul_b);
        }
        println!("{a_mul_b:?}");
    }

    #[test]
    fn test_expr() {
        nn!(a, 2.0);
        nn!(b, -3.0);
        // nn!(c, a * b);

        // nn!(d, c + 10.0);
    }
}