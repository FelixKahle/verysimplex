// Copyright 2024 Felix Kahle. All rights reserved.

use std::rc::Rc;

use nalgebra::{DMatrix, Scalar};
use num_traits::Float;

use crate::{
    problem::{Coefficients, Constant, Objective, Problem},
    tableau::{Tableau, TableauVariable},
};

/// Creates the initial tableau for the given problem.
///
/// # Arguments
/// - `problem`: The problem for which the tableau should be created.
///
/// # Returns
/// The initial tableau for the given problem.
pub fn create_initial_tableau<T>(problem: &Problem<T>) -> Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    let variable_names = problem.variable_names();
    let original_count = variable_names.len();
    let num_constraints = problem.program().num_constraints();
    let total_variables = original_count + num_constraints;
    let mut all_variables = Vec::with_capacity(total_variables);
    let mut original_variables = Vec::with_capacity(original_count);

    // Create tableau variables for original variables and slack variables
    for i in 0..total_variables {
        let name = if i < original_count {
            Rc::new(variable_names[i].to_string())
        } else {
            Rc::new(format!("s{}", i - original_count + 1))
        };

        let variable = TableauVariable::new(i, name);

        if i < original_count {
            original_variables.push(variable.clone());
        }
        all_variables.push(variable);
    }

    // Create the matrix for the tableau
    let mut matrix = DMatrix::zeros(num_constraints + 1, total_variables + 1);

    // Fill the matrix by iterating over the constraints
    for (i, constraint) in problem.program().constraints().iter().enumerate() {
        // Fill the coefficients of the constraint
        for (j, &coef) in constraint.coefficients().iter().enumerate() {
            matrix[(i, j)] = coef;
        }

        // Fill the slack variable
        matrix[(i, original_count + i)] = T::one();

        // Set the constant term
        matrix[(i, total_variables)] = *constraint.constant();
    }

    // Fill the objective function
    let objective = problem.program().objective();
    for (j, &coef) in objective.coefficients().iter().enumerate() {
        matrix[(num_constraints, j)] = match objective {
            Objective::Maximize(_) => -coef,
            Objective::Minimize(_) => coef,
        };
    }

    // The slack variables are the row variables
    let row_variables = all_variables[original_count..].to_vec();
    let column_variables = all_variables;
    Tableau::new(matrix, row_variables, column_variables)
}
