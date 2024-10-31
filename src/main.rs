// Copyright 2024 Felix Kahle. All rights reserved.

use problem::{Constraint, LinearProgram, Objective, Problem};

mod linsys;
mod problem;
mod solver;
mod tableau;

fn main() {
    let program = LinearProgram::<f64>::builder()
        .set_objective(Objective::Maximize(vec![1.6, 2.0, 3.0].into()))
        .with_constraint(Constraint::LessOrEqual(vec![1.0, 2.0, 3.0].into(), 4.0))
        .with_constraint(Constraint::Equal(vec![130.27, -2.0, 3.0].into(), 5.0))
        .with_constraint(Constraint::GreaterOrEqual(vec![1.0, 2.0, 3.0].into(), 6.0))
        .build()
        .unwrap();
    let problem = Problem::new(program, vec!["x".into(), "y".into(), "z".into()]).unwrap();
    println!("{}", problem);

    let tableau = solver::create_initial_tableau(&problem);
    println!("{}", tableau);
}
