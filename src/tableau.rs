// Copyright 2024 Felix Kahle. All rights reserved.

use std::fmt::{Display};

use nalgebra::DMatrix;
use tabled::settings::Style;
use tabled::Table;

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
}

impl Display for Tableau {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut builder = tabled::builder::Builder::default();
        
        let mut column_header = vec![String::new()];
        column_header.extend(self.column_names.iter().cloned());
        builder.push_record(column_header);
        
        for (i, row) in self.matrix.row_iter().enumerate() {
            let mut row_data = Vec::new();
            row_data.push(self.row_names[i].to_string());
            row_data.extend(row.iter().map(|value| value.to_string()));

            builder.push_record(row_data);
        }

        let mut table = builder
            .index()
            .column(0)
            .name(None)
            .build();

        table.with(Style::markdown());
        write!(f, "{}", table)
    }
}