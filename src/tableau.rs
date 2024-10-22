// Copyright 2024 Felix Kahle. All rights reserved.

use std::fmt::Display;
use crate::problem::{ObjectiveType, Problem, Relation};

/// A struct representing a tableau with a specified number of rows and columns.
#[derive(Debug, Clone)]
pub struct Tableau {
    /// The number of rows in the tableau.
    pub row_count: usize,
    
    /// The number of columns in the tableau.
    pub column_count: usize,
    
    /// The data of the tableau, stored as a flat vector.
    data: Vec<f64>,
}

/// Enum representing possible errors that can occur while working with a `Tableau`.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum TableauError {
    /// Indicates that the specified row or column is out of bounds.
    /// `row` and `column` specify which index caused the error.
    OutOfBounds { row: Option<usize>, column: Option<usize> },
}

#[allow(dead_code)]
impl Tableau {
    /// Creates a new `Tableau` with the specified number of rows and columns.
    ///
    /// # Parameters
    /// - `row_count`: The number of rows in the tableau.
    /// - `column_count`: The number of columns in the tableau.
    ///
    /// # Returns
    /// A `Tableau` instance with all elements initialized to `0.0`.
    pub fn new(row_count: usize, column_count: usize) -> Self {
        Self {
            row_count,
            column_count,
            data: vec![0.0; row_count * column_count],
        }
    }
    
    /// Creates a new `Tableau` from a given `Problem`.
    ///
    /// # Parameters
    /// - `problem`: The `Problem` instance to create the tableau from.
    /// 
    /// # Returns
    /// A `Tableau` instance representing the given `Problem`.
    pub fn from_problem(problem: &Problem) -> Result<Self, TableauError> {
        // Number of rows = number of constraints + 1 for the objective function
        let row_count = problem.constraints.len() + 1;

        // Number of columns = number of variables + 1 for the RHS values
        let column_count = problem.variables.len() + 1;

        // Create an empty tableau with the appropriate size
        let mut tableau = Tableau::new(row_count, column_count);

        // Fill the tableau with constraint coefficients and RHS values
        for (i, constraint) in problem.constraints.iter().enumerate() {
            let constraint_variable_multiplier = match constraint.relation {
                Relation::LessThan => 1.0,
                Relation::LessThanOrEqual => 1.0,
                Relation::GreaterThan => -1.0,
                Relation::GreaterThanOrEqual => -1.0,
                Relation::Equal => 1.0, // Treat Equal as LessThanOrEqual for now
            };
            
            // Set the variable coefficients for each constraint
            for term in &constraint.expression.terms {
                if let Some(variable_index) = problem.variables.iter().position(|v| v == &term.variable) {
                    tableau.set(i, variable_index, term.coefficient * constraint_variable_multiplier)?;
                }
            }

            // Set the RHS value in the last column of the tableau for this row
            tableau.set(i, column_count - 1, constraint.rhs)?;
        }
        
        // Normally we always maximize, so we multiply by -1 if we want to minimize.
        let objective_variable_multiplier = match problem.objective.objective_type {
            ObjectiveType::Maximize => 1.0,
            ObjectiveType::Minimize => -1.0,
        };

        // Set the objective function coefficients in the last row of the tableau
        let last_row = row_count - 1;
        for term in &problem.objective.expression.terms {
            if let Some(variable_index) = problem.variables.iter().position(|v| v == &term.variable) {
                // Use negative coefficients because of the standard form for the tableau
                tableau.set(last_row, variable_index, term.coefficient * -1.0 * objective_variable_multiplier)?;
            }
        }

        // RHS for the objective function is usually set to 0
        tableau.set(last_row, column_count - 1, 0.0)?;

        Ok(tableau)
    }

    /// Returns the value at the specified `row` and `col`.
    ///
    /// # Parameters
    /// - `row`: The row index of the desired value.
    /// - `col`: The column index of the desired value.
    ///
    /// # Returns
    /// - `Ok(f64)` if the indices are valid.
    /// - `Err(TableauError::OutOfBounds)` if the indices are invalid.
    pub fn get(&self, row: usize, col: usize) -> Result<f64, TableauError> {
        if row >= self.row_count {
            return Err(TableauError::OutOfBounds { row: Some(row), column: None });
        }
        if col >= self.column_count {
            return Err(TableauError::OutOfBounds { row: None, column: Some(col) });
        }
        
        Ok(self.data[self.index(row, col)])
    }

    /// Sets the value at the specified `row` and `col`.
    ///
    /// # Parameters
    /// - `row`: The row index where the value should be set.
    /// - `col`: The column index where the value should be set.
    /// - `value`: The value to be set.
    ///
    /// # Returns
    /// - `Ok(())` if the value is set successfully.
    /// - `Err(TableauError::OutOfBounds)` if the indices are invalid.
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), TableauError> {
        if row >= self.row_count {
            return Err(TableauError::OutOfBounds { row: Some(row), column: None });
        }
        if col >= self.column_count {
            return Err(TableauError::OutOfBounds { row: None, column: Some(col) });
        }
        
        let idx = self.index(row, col);
        self.data[idx] = value;
        Ok(())
    }
    

    /// Returns the objective value of the tableau.
    /// 
    /// # Returns
    /// - `Some(f64)` if the tableau is not empty and has at least one column.
    /// - `None` if the tableau is empty or has no columns.
    pub fn get_objective_value(&self) -> Option<f64> {
        if self.is_empty() {
            return None; // Return None if the tableau is empty or has no columns.
        }

        Some(self.data[self.index(self.row_count - 1, self.column_count - 1)])
    }
    
    /// Checks if the tableau is feasible.
    /// A tableau is feasible if the value in the last column of each row is non-negative.
    ///
    /// # Returns
    /// - `true` if the tableau is feasible.
    /// - `false` if the tableau is not feasible.
    pub fn is_feasible(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        for row in 0..self.row_count {
            if self.data[self.index(row, self.column_count - 1)] < 0.0 {
                return false;
            }
        }

        true
    }
    
    /// Checks if the tableau is optimal.
    /// A tableau is optimal if the values in the last row are non-negative.
    /// 
    /// # Returns
    /// - `true` if the tableau is optimal.
    /// - `false` if the tableau is not optimal.
    pub fn is_optimal(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        for col in 0..self.column_count {
            if self.data[self.index(self.row_count - 1, col)] < 0.0 {
                return false;
            }
        }

        true
    }
    
    /// Checks if the tableau is empty.
    ///
    /// # Returns
    /// - `true` if the tableau is empty.
    /// - `false` if the tableau is not empty.
    pub fn is_empty(&self) -> bool {
        self.row_count == 0 || self.column_count == 0
    }

    /// Helper function to calculate the flat index in the `data` vector from row and column indices.
    ///
    /// # Parameters
    /// - `row`: The row index.
    /// - `col`: The column index.
    ///
    /// # Returns
    /// The flat index corresponding to the specified row and column.
    fn index(&self, row: usize, col: usize) -> usize {
        row * self.column_count + col
    }
}

impl Display for Tableau {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.row_count {
            for col in 0..self.column_count {
                write!(f, "{}\t", self.data[self.index(row, col)])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "Objective value: {}", self.get_objective_value().unwrap_or(0.0))
    }
}

impl From<Problem> for Result<Tableau, TableauError> {
    fn from(problem: Problem) -> Self {
        Tableau::from_problem(&problem)
    }
}
