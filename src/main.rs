// Copyright (c) 2024 Felix Kahle

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

mod etam;
mod problem;
mod solution;
mod solver;
mod var;

use std::rc::Rc;

use problem::{Constraint, LinearProgram, LinearProgramBuilderError, Objective};
use var::{Variable, VariableValue};

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
            .set_objective(Objective::Maximize(<problem::LinearExpression<f64>>::from(
                vec![
                    VariableValue::new(x.clone(), 3.0).into(),
                    VariableValue::new(y.clone(), 5.0).into(),
                ],
            )))
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
