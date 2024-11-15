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

use nalgebra::{DMatrix, DVector};
use solver::{Solver, SolverVariable};
use std::collections::HashMap;

fn main() {
    // Define the constraint matrix A for the linear program.
    //
    // For the constraints:
    // 1. x1 + 2x2 <= 10
    // 2. 2x1 + 3x2 <= 15
    // The matrix form is:
    // [1  2]
    // [2  3]
    let constraint_matrix = DMatrix::from_row_slice(
        2,
        4,
        &[
            1.0, 2.0, 1.0, 0.0, // row representing x1 + 2x2 + 1s1 + 0s2
            2.0, 3.0, 0.0, 1.0, // row representing 2x1 + 3x2 + 0s1 + 1s2
        ],
    );

    // Right-hand side (b) vector representing the constraints:
    // For the given constraints:
    // 1. x1 + 2x2 <= 10
    // 2. 2x1 + 3x2 <= 15
    // The vector is:
    // [10, 15]
    let rhs = DVector::from_row_slice(&[10.0, 15.0]);

    // Objective coefficients (c) for the objective function z = 3x1 + 5x2.
    // The vector of coefficients for each variable is:
    // c = [3, 5]
    // However, the solver might also contain slack variables. We'll initially set them to 0.
    let objective_coefficients = DVector::from_row_slice(&[-3.0, -5.0]);

    // Create an initial basis and non-basic variable list.
    // Assume x1 and x2 are non-basic initially. Slack variables will form the basis.
    // For 2 constraints, we have 2 slack variables:
    // Let the indexing convention be:
    // x1 -> index 0, x2 -> index 1
    // Slack variable for constraint 1 -> index 2
    // Slack variable for constraint 2 -> index 3
    let basic_indices = vec![2, 3]; // Slack variables as basic
    let non_basic_indices = vec![0, 1]; // Original variables x1, x2 as non-basic

    // Initially, the basic slack variables are equal to the RHS (10, 15).
    // The objective function initially is 0 because x1 = x2 = 0.
    let objective_value = 0.0;

    // Map indices to solver variables (this helps in identifying which solver variable corresponds to which original variable).
    let mut index_to_variable = HashMap::new();
    index_to_variable.insert(0, SolverVariable::Free(0)); // x1
    index_to_variable.insert(1, SolverVariable::Free(1)); // x2
    index_to_variable.insert(2, SolverVariable::Free(0)); // slack variable for constraint 1
    index_to_variable.insert(3, SolverVariable::Free(1)); // slack variable for constraint 2

    // Define a small epsilon for numerical tolerance
    let epsilon = 1e-8;

    // Create a solver instance with all parameters initialized.
    let solver = Solver::new(
        objective_value,
        constraint_matrix,
        rhs,
        objective_coefficients,
        basic_indices,
        non_basic_indices,
        index_to_variable,
        epsilon,
    );

    println!("Objective Value:{}", solver.objective_value());
    println!("Constraint-Matrix:{}", solver.constraint_matrix());
    println!("RHS:{}", solver.rhs());
    println!("Objective Coefficients:{}", solver.objective_coefficients());
}
