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

use crate::alias::{DenseColumn, DenseRow};

/// Invalid row count error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidRowCountError {
    /// Expected row count.
    pub expected: usize,

    /// Actual row count.
    pub actual: usize,
}

impl InvalidRowCountError {
    /// Create a new invalid row count error.
    ///
    /// # Arguments
    /// - `expected`: Expected row count.
    /// - `actual`: Actual row count.
    ///
    /// # Returns
    /// A new invalid row count error.
    pub fn new(expected: usize, actual: usize) -> Self {
        Self { expected, actual }
    }

    /// Get the expected row count.
    ///
    /// # Returns
    /// The expected row count.
    pub fn expected(&self) -> usize {
        self.expected
    }

    /// Get the actual row count.
    ///
    /// # Returns
    /// The actual row count.
    pub fn actual(&self) -> usize {
        self.actual
    }
}

impl std::fmt::Display for InvalidRowCountError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Invalid row count: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for InvalidRowCountError {}

/// Invalid column count error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidColumnCountError {
    /// Expected column count.
    pub expected: usize,

    /// Actual column count.
    pub actual: usize,
}

impl InvalidColumnCountError {
    /// Create a new invalid column count error.
    ///
    /// # Arguments
    /// - `expected`: Expected column count.
    /// - `actual`: Actual column count.
    ///
    /// # Returns
    /// A new invalid column count error.
    pub fn new(expected: usize, actual: usize) -> Self {
        Self { expected, actual }
    }

    /// Get the expected column count.
    ///
    /// # Returns
    /// The expected column count.
    pub fn expected(&self) -> usize {
        self.expected
    }

    /// Get the actual column count.
    ///
    /// # Returns
    /// The actual column count.
    pub fn actual(&self) -> usize {
        self.actual
    }
}

impl std::fmt::Display for InvalidColumnCountError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Invalid column count: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for InvalidColumnCountError {}

/// An error indicating a zero pivot element which is not allowed for certain operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroPivotElementError {
    /// The column index of the zero pivot element.
    column_index: usize,
    /// The row index of the zero pivot element.
    row_index: usize,
}

impl ZeroPivotElementError {
    /// Construct a new `ZeroPivotElementError`.
    ///
    /// # Parameters
    /// - `column_index`: The column index of the zero pivot element.
    /// - `row_index`: The row index of the zero pivot element.
    ///
    /// # Returns
    /// A new `ZeroPivotElementError`.
    pub fn new(column_index: usize, row_index: usize) -> Self {
        Self {
            column_index,
            row_index,
        }
    }

    /// Get the column index of the zero pivot element.
    ///
    /// # Returns
    /// The column index of the zero pivot element.
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Get the row index of the zero pivot element.
    ///
    /// # Returns
    /// The row index of the zero pivot element.
    pub fn row_index(&self) -> usize {
        self.row_index
    }
}

impl std::fmt::Display for ZeroPivotElementError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "zero pivot element at column {} and row {}",
            self.column_index, self.row_index
        )
    }
}

impl std::error::Error for ZeroPivotElementError {}

/// An error indicating that the left side of a linear system could not be solved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeftSolveError {
    /// The pivot element is zero.
    ZeroPivotElement(ZeroPivotElementError),

    /// The row count of the left side of the linear system does not match the row count of the right side.
    InvalidColumnCount(InvalidColumnCountError),
}

impl std::fmt::Display for LeftSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LeftSolveError::ZeroPivotElement(e) => write!(f, "{}", e),
            LeftSolveError::InvalidColumnCount(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for LeftSolveError {}

impl From<ZeroPivotElementError> for LeftSolveError {
    fn from(e: ZeroPivotElementError) -> Self {
        LeftSolveError::ZeroPivotElement(e)
    }
}

impl From<InvalidColumnCountError> for LeftSolveError {
    fn from(e: InvalidColumnCountError) -> Self {
        LeftSolveError::InvalidColumnCount(e)
    }
}

/// An error indicating that the right side of a linear system could not be solved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightSolveError {
    /// The pivot element is zero.
    ZeroPivotElement(ZeroPivotElementError),

    /// The row count of the right side of the linear system does not match the row count of the left side.
    InvalidRowCount(InvalidRowCountError),
}

impl std::fmt::Display for RightSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RightSolveError::ZeroPivotElement(e) => write!(f, "{}", e),
            RightSolveError::InvalidRowCount(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for RightSolveError {}

impl From<ZeroPivotElementError> for RightSolveError {
    fn from(e: ZeroPivotElementError) -> Self {
        RightSolveError::ZeroPivotElement(e)
    }
}

impl From<InvalidRowCountError> for RightSolveError {
    fn from(e: InvalidRowCountError) -> Self {
        RightSolveError::InvalidRowCount(e)
    }
}

/// A trait for solving linear systems.
///
/// # Type Parameters
/// - `T`: The type of the elements of the linear system.
pub trait LeftSolve<T> {
    /// Solves the system x * self = y, where x is the unknown to be determined.
    ///
    /// `y` will be overwritten with the solution.
    /// If an error occurs, `y` might be overwritten with garbage values.
    ///
    /// # Arguments
    /// - `y`: The right side of the linear system.
    fn left_solve_mut(&self, y: &mut DenseRow<T>) -> Result<(), LeftSolveError>;

    /// Solves the system x * self = y, where x is the unknown to be determined.
    ///
    /// # Arguments
    /// - `y`: The right side of the linear system.
    ///
    /// # Returns
    /// The solution to the linear system.
    fn left_solve(&self, y: &DenseRow<T>) -> Result<DenseRow<T>, LeftSolveError>;
}

pub trait RightSolve<T> {
    /// Solves the system self * x = y, where x is the unknown to be determined.
    ///
    /// `y` will be overwritten with the solution.
    /// If an error occurs, `y` might be overwritten with garbage values.
    ///
    /// # Arguments
    /// - `y`: The right side of the linear system.
    fn right_solve_mut(&self, y: &mut DenseColumn<T>) -> Result<(), RightSolveError>;

    /// Solves the system self * x = y, where x is the unknown to be determined.
    ///
    /// # Arguments
    /// - `y`: The right side of the linear system.
    ///
    /// # Returns
    /// The solution to the linear system.
    fn right_solve(&self, y: &DenseColumn<T>) -> Result<DenseColumn<T>, RightSolveError>;
}
