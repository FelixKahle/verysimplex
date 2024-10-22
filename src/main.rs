// Copyright 2024 Felix Kahle. All rights reserved.

use crate::problem::{Variable};

mod problem;

fn main() {
    // Define variables
    let x1 = Variable::new("x1");
    let x2 = Variable::new("x2");

    // Create constraints
    let constraint1 = (x1.clone() * 2.0 + x2.clone() * 3.0).less_than(12.0);
    let constraint2 = (x1.clone() * 1.0 - x2.clone() * 26.0).greater_than(4.0);

    println!("Constraint 1: {}", constraint1);
    println!("Constraint 2: {}", constraint2);
}
