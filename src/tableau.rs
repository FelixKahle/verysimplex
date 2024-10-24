// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::fmt::{Display};

use nalgebra::{DMatrix, Dyn, MatrixView, U1};
use tabled::settings::Style;

/// A tableau that represents a linear program.
pub struct Tableau {
    /// The matrix that represents the tableau.
    matrix: DMatrix<f64>,
    
    /// The names of the rows of the tableau.
    row_names: Vec<String>,
    
    /// The names of the columns of the tableau.
    column_names: Vec<String>,
}

impl Tableau {
    /// Create a new Tableau from a matrix.
    /// 
    /// # Arguments
    /// * `matrix` - The matrix that represents the tableau.
    /// * `row_names` - The names of the rows of the tableau.
    /// * `column_names` - The names of the columns of the tableau.
    ///
    /// # Returns
    /// A new Tableau.
    pub fn new(matrix: DMatrix<f64>, row_names: Vec<String>, column_names: Vec<String>) -> Tableau {
        // Check if the number of row names matches the number of rows of the matrix.
        if matrix.nrows() != row_names.len() {
            panic!("The number of row names did not match the number of rows of the matrix.");
        }
        
        // Check if the number of column names matches the number of columns of the matrix.
        if matrix.ncols() != column_names.len() {
            panic!("The number of column names did not match the number of columns of the matrix.");
        }
        
        // Create the tableau.
        Tableau {
            matrix,
            row_names,
            column_names,
        }
    }

    /// Get the matrix of the tableau.
    /// 
    /// # Returns
    /// The matrix of the tableau.
    pub fn get_matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }
    
    /// Get the number of rows of the tableau.
    ///
    /// # Returns
    /// The number of rows of the tableau.
    pub fn rows(&self) -> usize {
        self.matrix.nrows()
    }
    
    /// Get the number of columns of the tableau.
    ///
    /// # Returns
    /// The number of columns of the tableau.
    pub fn cols(&self) -> usize {
        self.matrix.ncols()
    }
    
    /// Get the names of the columns of the tableau.
    ///
    /// # Returns
    /// The names of the columns of the tableau.
    pub fn row_names(&self) -> &Vec<String> {
        &self.row_names
    }
    
    /// Get the names of the columns of the tableau.
    ///
    /// # Returns
    /// The names of the columns of the tableau.
    pub fn column_names(&self) -> &Vec<String> {
        &self.column_names
    }

    /// Get a mutable reference to the matrix of the tableau.
    ///
    /// # Returns
    /// A mutable reference to the matrix of the tableau.
    pub fn row_names_mut(&mut self) -> &mut Vec<String> {
        &mut self.row_names
    }

    /// Get a mutable reference to the matrix of the tableau.
    ///
    /// # Returns
    /// A mutable reference to the matrix of the tableau.
    pub fn column_names_mut(&mut self) -> &mut Vec<String> {
        &mut self.column_names
    }
    
    /// Get the objective value of the tableau.
    ///
    /// # Returns
    /// The objective value of the tableau.
    pub fn get_objective_value(&self) -> f64 {
        self.matrix[(self.rows() - 1, self.cols() - 1)]
    }
    
    /// Get the rhs vector of the tableau.
    ///
    ///
    /// # Returns
    /// The rhs vector of the tableau.
    /// 
    /// # Note
    /// The rhs vector is the last column of the matrix without the last row.
    pub fn rhs_vector(&self) -> MatrixView<f64, Dyn, Dyn, U1, Dyn> {
        // The rhs vector is the last column of the matrix without the last row.
        self.matrix.view((0, self.cols() - 1), (self.rows() - 1, 1))
    }
    
    /// Get the objective coefficients of the tableau.
    /// 
    /// # Returns
    /// The objective coefficients of the tableau.
    /// 
    /// # Note
    /// The objective coefficients are the last row of the matrix without the last column.
    pub fn objective_coefficients(&self) -> MatrixView<f64, Dyn, Dyn, U1, Dyn> {
        // The objective coefficients are the last row of the matrix without the last column.
        self.matrix.view((self.rows() - 1, 0), (1, self.cols() - 1))
    }
    
    ///Check if the current tableau is feasible.
    ///
    /// # Returns
    /// - `true` if the tableau is feasible.
    /// - `false` if the tableau is not feasible.
    pub fn is_feasible(&self) -> bool {
        let rhs_vector = self.rhs_vector();
        
        // Check if all values of the rhs vector are greater or equal to zero.
        rhs_vector.iter().all(|value| *value >= 0.0)
    }
    
    /// Check if the tableau is optimal.
    ///
    /// # Returns
    /// - `true` if the tableau is optimal.
    /// - `false` if the tableau is not optimal.
    pub fn is_optimal(&self) -> bool {
        let objective_coefficients = self.objective_coefficients();
        
        // Check if all values of the objective are greater or equal to zero.
        objective_coefficients.iter().all(|value| *value >= 0.0)
    }

    /// Perform a pivot operation on the tableau.
    ///
    /// # Arguments
    /// * `pivot_row` - The index of the pivot row.
    /// * `pivot_column` - The index of the pivot column.
    ///
    /// # Note
    /// The pivot operation is performed in place using the gaussian elimination method.
    pub fn gaussian_pivot(&mut self, pivot_row: usize, pivot_column: usize) {
        // Get the pivot element.
        let pivot_element = self.matrix[(pivot_row, pivot_column)];
        
        let mut pivot_row_mut = self.matrix.row_mut(pivot_row);
        pivot_row_mut.scale_mut(1.0 / pivot_element);
        let pivot_row_copy = self.matrix.row(pivot_row).clone_owned();

        // Perform row operations to eliminate other entries in the pivot column.
        let num_rows = self.rows();
        for r in 0..num_rows {
            if r != pivot_row {
                let factor = self.matrix[(r, pivot_column)];
                let mut current_row_mut = self.matrix.row_mut(r);
                current_row_mut -= factor * &pivot_row_copy;
            }
        }
    }
}

/// Implement the Display trait for Tableau.
/// 
/// # Note
/// This implementation uses the tabled crate to display the tableau
/// in a well formatted table.
impl Display for Tableau {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut builder = tabled::builder::Builder::default();
        
        // Push the column headers to the table.
        // The first cell (0, 0) is empty, because this column is used for the row names.
        let mut column_header = vec![String::new()];
        column_header.extend(self.column_names.iter().cloned());
        builder.push_record(column_header);
        
        // Push the rows to the table.
        // The first cell of each row is the row name.
        for (i, row) in self.matrix.row_iter().enumerate() {
            let mut row_data = Vec::new();
            row_data.push(self.row_names[i].to_string());
            row_data.extend(row.iter().map(|value| value.to_string()));

            builder.push_record(row_data);
        }

        // The first column is the index column.
        let mut table = builder
            .index()
            .column(0)
            .name(None)
            .build();

        // Set the style of the table to markdown,
        // and write the table to the formatter.
        table.with(Style::markdown());
        write!(f, "{}", table)
    }
}