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

#![allow(dead_code)]

use nalgebra_lapack::LUScalar;
use std::collections::HashSet;

use crate::var::VariableValue;

/// A solution to a linear program.
///
/// # Type parameters
/// - `T`: The number type.
#[derive(Debug, Clone)]
pub struct Solution<T>
where
    T: LUScalar,
{
    /// The objective value.
    objective_value: T,

    /// The variable values.
    variable_values: HashSet<VariableValue<T>>,
}

impl<T> Solution<T>
where
    T: LUScalar,
{
    /// Constructs a new `Solution`.
    ///
    /// # Parameters
    /// - `objective_value`: The objective value.
    /// - `variable_values`: The variable values.
    ///
    /// # Returns
    /// A new instance of `Solution`.
    pub fn new(objective_value: T, variable_values: HashSet<VariableValue<T>>) -> Self {
        Self {
            objective_value,
            variable_values,
        }
    }

    /// Gets the objective value.
    ///
    /// # Returns
    /// The objective value.
    pub fn objective_value(&self) -> T {
        self.objective_value
    }

    /// Gets the variable values.
    ///
    /// # Returns
    /// The variable values.
    pub fn variable_values(&self) -> &HashSet<VariableValue<T>> {
        &self.variable_values
    }
}

impl<'a, T> IntoIterator for &'a Solution<T>
where
    T: LUScalar,
{
    type Item = &'a VariableValue<T>;
    type IntoIter = std::collections::hash_set::Iter<'a, VariableValue<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.variable_values.iter()
    }
}

impl<T> std::fmt::Display for Solution<T>
where
    T: LUScalar + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Objective value: {}", self.objective_value)?;

        for variable_value in &self.variable_values {
            writeln!(f, "{}", variable_value)?;
        }

        Ok(())
    }
}

/// The solution status of a solver.
///
/// # Type parameters
/// - `T`: The number type.
#[derive(Debug, Clone)]
pub enum SolutionStatus<T>
where
    T: LUScalar,
{
    /// The solution is optimal.
    Optimal(Solution<T>),

    /// The solution is feasible.
    Feasible(Solution<T>),

    /// The solution is unbounded.
    Unbounded,

    /// The solution is infeasible.
    Infeasible,
}

impl<T> std::fmt::Display for SolutionStatus<T>
where
    T: LUScalar + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolutionStatus::Optimal(solution) => {
                writeln!(f, "Optimal solution")?;
                write!(f, "{}", solution)
            }
            SolutionStatus::Feasible(solution) => {
                writeln!(f, "Feasible solution")?;
                write!(f, "{}", solution)
            }
            SolutionStatus::Unbounded => write!(f, "Unbounded solution"),
            SolutionStatus::Infeasible => write!(f, "Infeasible solution"),
        }
    }
}
