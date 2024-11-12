// Copyright 2024 Felix Kahle. All rights reserved.

mod problem;
mod solver;

use std::rc::Rc;

use problem::{
    Constraint, LinearProgram, LinearProgramBuilderError, Objective, Variable, VariableValue,
};

fn main() {
    // Define variable names
    let x_name = Rc::new("x".to_string());
    let y_name = Rc::new("y".to_string());

    // Create variables with unique IDs
    let x = Variable::new(10, x_name);
    let y = Variable::new(4, y_name);

    // Using the builder to construct the linear program
    let linear_program: Result<LinearProgram<f64>, LinearProgramBuilderError> =
        LinearProgram::builder()
            .set_objective(Objective::Maximize(
                vec![
                    VariableValue::new(x.clone(), 3.0).into(),
                    VariableValue::new(y.clone(), 5.0).into(),
                ]
                .into(),
            ))
            .with_constraint(Constraint::GreaterOrEqual(
                vec![
                    VariableValue::new(x.clone(), 6.0).into(),
                    VariableValue::new(y.clone(), 10.0).into(),
                ]
                .into(),
                0.0,
            ))
            .with_constraint(Constraint::GreaterOrEqual(
                vec![
                    VariableValue::new(x.clone(), 0.0).into(),
                    VariableValue::new(y.clone(), 10.0).into(),
                ]
                .into(),
                0.0,
            ))
            .build();

    // Display the linear program
    println!("{}", linear_program.unwrap());
}
