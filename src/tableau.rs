// Copyright 2024 Felix Kahle. All rights reserved.

/// A struct representing a tableau with a specified number of rows and columns.
#[derive(Debug, Clone)]
pub struct Tableau {
    /// The number of rows in the tableau.
    row_count: usize,
    
    /// The number of columns in the tableau.
    column_count: usize,
    
    /// The data of the tableau, stored as a flat vector.
    data: Vec<f64>,
}

/// Enum representing possible errors that can occur while working with a `Tableau`.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum TableauError {
    /// The provided vector length does not match the expected length.
    VectorLengthMismatch,

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

    /// Adds a new row at the end of the tableau, with all elements initialized to `0.0`.
    ///
    /// # Example
    /// ```
    /// let mut tableau = Tableau::new(2, 3);
    /// tableau.add_row(); // Now has 3 rows
    /// ```
    pub fn add_row(&mut self) {
        self.data.extend(std::iter::repeat(0.0).take(self.column_count));
        self.row_count += 1;
    }

    /// Inserts a new row at the specified `index`, with all elements initialized to `0.0`.
    ///
    /// # Parameters
    /// - `index`: The index at which to insert the new row.
    ///
    /// # Returns
    /// - `Ok(())` if the row is inserted successfully.
    /// - `Err(TableauError::OutOfBounds)` if the row index is invalid.
    pub fn add_row_at(&mut self, index: usize) -> Result<(), TableauError> {
        if index > self.row_count {
            return Err(TableauError::OutOfBounds { row: Some(index), column: None });
        }
        let start = index * self.column_count;
        self.data.splice(start..start, std::iter::repeat(0.0).take(self.column_count));
        self.row_count += 1;
        Ok(())
    }

    /// Adds a new row with the given `values`. The length of the vector must match the number of columns.
    ///
    /// # Parameters
    /// - `values`: A vector of values to initialize the new row.
    ///
    /// # Returns
    /// - `Ok(())` if the row is added successfully.
    /// - `Err(TableauError::VectorLengthMismatch)` if the vector size does not match the number of columns.
    pub fn add_row_with_values(&mut self, values: Vec<f64>) -> Result<(), TableauError> {
        if values.len() != self.column_count {
            return Err(TableauError::VectorLengthMismatch);
        }
        self.data.extend(values);
        self.row_count += 1;
        Ok(())
    }

    /// Removes the last row from the tableau.
    ///
    /// # Example
    /// ```
    /// let mut tableau = Tableau::new(2, 3);
    /// tableau.remove_row(); // Now has 1 row
    /// ```
    pub fn remove_row(&mut self) {
        self.data.truncate(self.data.len() - self.column_count);
        self.row_count -= 1;
    }

    /// Removes the row at the specified `index`.
    ///
    /// # Parameters
    /// - `index`: The index of the row to be removed.
    ///
    /// # Returns
    /// - `Ok(())` if the row is removed successfully.
    /// - `Err(TableauError::OutOfBounds)` if the row index is invalid.
    pub fn remove_row_at(&mut self, index: usize) -> Result<(), TableauError> {
        if index >= self.row_count {
            return Err(TableauError::OutOfBounds { row: Some(index), column: None });
        }
        let start = index * self.column_count;
        let end = start + self.column_count;
        self.data.drain(start..end);
        self.row_count -= 1;
        Ok(())
    }

    /// Adds a new column at the end of the tableau, with all elements initialized to `0.0`.
    ///
    /// # Example
    /// ```
    /// let mut tableau = Tableau::new(2, 3);
    /// tableau.add_column(); // Now has 4 columns
    /// ```
    pub fn add_column(&mut self) {
        for row in (0..self.row_count).rev() {
            let idx = self.index(row, self.column_count);
            self.data.insert(idx, 0.0);
        }
        self.column_count += 1;
    }

    /// Inserts a new column at the specified `index`, with all elements initialized to `0.0`.
    ///
    /// # Parameters
    /// - `index`: The index at which to insert the new column.
    ///
    /// # Returns
    /// - `Ok(())` if the column is inserted successfully.
    /// - `Err(TableauError::OutOfBounds)` if the column index is invalid.
    pub fn add_column_at(&mut self, index: usize) -> Result<(), TableauError> {
        if index > self.column_count {
            return Err(TableauError::OutOfBounds { row: None, column: Some(index) });
        }
        for row in (0..self.row_count).rev() {
            let idx = self.index(row, index);
            self.data.insert(idx, 0.0);
        }
        self.column_count += 1;
        Ok(())
    }

    /// Adds a new column with the given `values`. The length of the vector must match the number of rows.
    ///
    /// # Parameters
    /// - `values`: A vector of values to initialize the new column.
    ///
    /// # Returns
    /// - `Ok(())` if the column is added successfully.
    /// - `Err(TableauError::VectorLengthMismatch)` if the vector size does not match the number of rows.
    pub fn add_column_with_values(&mut self, values: Vec<f64>) -> Result<(), TableauError> {
        if values.len() != self.row_count {
            return Err(TableauError::VectorLengthMismatch);
        }
        for row in (0..self.row_count).rev() {
            let idx = self.index(row, self.column_count);
            self.data.insert(idx, values[row]);
        }
        self.column_count += 1;
        Ok(())
    }

    /// Removes the last column from the tableau.
    ///
    /// # Example
    /// ```
    /// let mut tableau = Tableau::new(2, 3);
    /// tableau.remove_column(); // Now has 2 columns
    /// ```
    pub fn remove_column(&mut self) {
        for row in (0..self.row_count).rev() {
            let idx = self.index(row, self.column_count - 1);
            self.data.remove(idx);
        }
        self.column_count -= 1;
    }

    /// Removes the column at the specified `index`.
    ///
    /// # Parameters
    /// - `index`: The index of the column to be removed.
    ///
    /// # Returns
    /// - `Ok(())` if the column is removed successfully.
    /// - `Err(TableauError::OutOfBounds)` if the column index is invalid.
    pub fn remove_column_at(&mut self, index: usize) -> Result<(), TableauError> {
        if index >= self.column_count {
            return Err(TableauError::OutOfBounds { row: None, column: Some(index) });
        }
        for row in (0..self.row_count).rev() {
            let idx = self.index(row, index);
            self.data.remove(idx);
        }
        self.column_count -= 1;
        Ok(())
    }

    /// Returns the objective value of the tableau.
    /// 
    /// # Returns
    /// - `Some(f64)` if the tableau is not empty and has at least one column.
    /// - `None` if the tableau is empty or has no columns.
    pub fn get_objective_value(&self) -> Option<f64> {
        if self.row_count == 0 || self.column_count == 0 {
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
        if self.row_count == 0 || self.column_count == 0 {
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
        if self.row_count == 0 || self.column_count == 0 {
            return false;
        }

        for col in 0..self.column_count {
            if self.data[self.index(self.row_count - 1, col)] < 0.0 {
                return false;
            }
        }

        true
    }

    /// Prints the entire tableau in a matrix-like format.
    ///
    /// Each row is printed on a new line, and columns are separated by tabs.
    ///
    /// # Example
    /// ```
    /// let mut tableau = Tableau::new(2, 2);
    /// tableau.set(0, 0, 1.0).unwrap();
    /// tableau.set(0, 1, 2.0).unwrap();
    /// tableau.print_tableau(); // Prints the tableau
    /// ```
    pub fn print_tableau(&self) {
        for row in 0..self.row_count {
            for col in 0..self.column_count {
                print!("{}\t", self.data[self.index(row, col)]);
            }
            println!();
        }
        println!("Objective value: {}", self.get_objective_value().unwrap_or(0.0));
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