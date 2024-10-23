// Copyright 2024 Felix Kahle. All rights reserved.

use std::fmt::Display;

/// A struct representing a 2D matrix with a specified number of rows and columns.
/// The matrix is stored as a flat vector.
///
/// # Type Parameters
/// - `T`: The type of the elements in the matrix.
///
/// # Note
/// The index of a entry at row `i` and column `j` is calculated as `i * columns + j`.
pub struct Matrix2D<T> {
    data: Vec<T>,
    rows: usize,
    columns: usize,
}

impl<T> Matrix2D<T> {
    /// Creates a new `Matrix2D` with the specified number of rows and columns.
    ///
    /// # Parameters
    /// - `rows`: The number of rows in the matrix.
    /// - `columns`: The number of columns in the matrix.
    /// - `default`: The default value to initialize the matrix with.
    /// 
    /// # Returns
    /// A `Matrix2D` instance with all elements initialized to `default`.
    pub fn new(rows: usize, columns: usize, default: T) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![default; rows * columns],
            rows,
            columns,
        }
    }
    
    /// Creates a new `Matrix2D` from a given `Vec<T>`.
    ///
    /// # Parameters
    /// - `rows`: The number of rows in the matrix.
    /// - `columns`: The number of columns in the matrix.
    ///
    /// # Returns
    /// - `Some(Matrix2D)` if the length of the data vector matches the specified dimensions.
    /// - `None` if the length of the data vector does not match the specified dimensions.
    pub fn from_vec(rows: usize, columns: usize, data: Vec<T>) -> Option<Self> {
        if data.len() == rows * columns {
            Some(Self { data, rows, columns })
        } else {
            None
        }
    }

    /// Returns a reference to the value at the specified `row` and `col`.
    ///
    /// # Parameters
    /// - `row`: The row index of the desired value.
    /// - `column`: The column index of the desired value.
    ///
    /// # Returns
    /// - `Some(&T)` if the indices are valid.
    /// - `None` if the indices are invalid.
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if self.indices_valid(row, column) {
            Some(&self.data[self.index(row, column)])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value at the specified `row` and `col`.
    ///
    /// # Parameters
    /// - `row`: The row index of the desired value.
    /// - `column`: The column index of the desired value.
    ///
    /// # Returns
    /// - `Some(&mut T)` if the indices are valid.
    /// - `None` if the indices are invalid.
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        if self.indices_valid(row, column) {
            let index = self.index(row, column);
            self.data.get_mut(index)
        } else {
            None
        }
    }
    
    /// Returns the number of rows in the matrix.
    ///
    /// # Returns
    /// The number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }
    
    /// Returns the number of columns in the matrix.
    ///
    /// # Returns
    /// The number of columns in the matrix.
    pub fn columns(&self) -> usize {
        self.columns
    }
    
    /// Returns the number of rows in the matrix.
    ///
    /// # Returns
    /// The number of rows in the matrix.
    pub fn entries_count(&self) -> usize {
        self.rows * self.columns
    }
    
    /// Returns a reference to the data vector.
    ///
    /// # Returns
    /// A reference to the data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
    
    /// Returns a mutable reference to the data vector.
    ///
    /// # Returns
    /// A mutable reference to the data vector.
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }
    
    /// Helper function to check if the row and column indices are valid.
    ///
    /// # Parameters
    /// - `row`: The row index.
    /// - `column`: The column index.
    ///
    /// # Returns
    /// `true` if the indices are valid, `false` otherwise.
    pub fn indices_valid(&self, row: usize, column: usize) -> bool {
        row < self.rows && column < self.columns
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
        row * self.columns + col
    }
}

impl Display for Matrix2D<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.columns {
                write!(f, "{:.2}\t", self.data[self.index(row, col)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix() {
        let rows = 3;
        let columns = 3;
        let matrix = Matrix2D::new(rows, columns, 0);

        assert_eq!(matrix.rows(), rows);
        assert_eq!(matrix.columns(), columns);
        assert_eq!(matrix.entries_count(), rows * columns);

        for i in 0..rows {
            for j in 0..columns {
                assert_eq!(matrix.get(i, j), Some(&0));
            }
        }
    }

    #[test]
    fn test_from_vec_valid() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let rows = 2;
        let columns = 3;
        let matrix = Matrix2D::from_vec(rows, columns, data.clone());

        assert!(matrix.is_some());
        let matrix = matrix.unwrap();

        assert_eq!(matrix.rows(), rows);
        assert_eq!(matrix.columns(), columns);
        assert_eq!(matrix.data(), &data);
    }

    #[test]
    fn test_from_vec_invalid() {
        let data = vec![1, 2, 3, 4, 5];
        let rows = 2;
        let columns = 3;
        let matrix = Matrix2D::from_vec(rows, columns, data);

        assert!(matrix.is_none());
    }

    #[test]
    fn test_get_valid() {
        let rows = 2;
        let columns = 2;
        let mut matrix = Matrix2D::new(rows, columns, 1);
        matrix.data_mut()[1] = 5;

        assert_eq!(matrix.get(0, 1), Some(&5));
        assert_eq!(matrix.get(1, 1), Some(&1));
    }

    #[test]
    fn test_get_invalid_indices() {
        let matrix = Matrix2D::new(2, 2, 1);
        
        assert_eq!(matrix.get(3, 0), None);
        assert_eq!(matrix.get(0, 3), None);
    }

    #[test]
    fn test_get_mut() {
        let rows = 2;
        let columns = 2;
        let mut matrix = Matrix2D::new(rows, columns, 1);

        if let Some(value) = matrix.get_mut(0, 0) {
            *value = 10;
        }

        assert_eq!(matrix.get(0, 0), Some(&10));
    }

    #[test]
    fn test_indices_valid() {
        let matrix = Matrix2D::new(2, 2, 1);

        assert!(matrix.indices_valid(1, 1));
        assert!(!matrix.indices_valid(2, 2));
    }

    #[test]
    fn test_display() {
        let rows = 2;
        let columns = 2;
        let matrix = Matrix2D::new(rows, columns, 1.0);

        let expected_output = "1.00\t1.00\t\n1.00\t1.00\t\n";
        assert_eq!(format!("{}", matrix), expected_output);
    }
}
