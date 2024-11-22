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

use std::ops::{DivAssign, SubAssign};

use crate::idx::ValidIndex;
use nalgebra::{iter::MatrixIter, Const, DVector, Dyn, Scalar, VecStorage};
use num_traits::Zero;

/// An eta matrix E corresponds to the identity matrix except for one column e of
/// index j. In particular, B.E is the matrix of the new basis obtained from B by
/// replacing the j-th vector of B by B.e, note that this is exactly what happens
/// during a "pivot" of the current basis in the simplex algorithm.
///
/// # Type parameters
/// - `T`: The type of the elements of the eta matrix.
///
/// # Example
///
/// To create an eta matrix `E` of the form
///
/// ```text
/// E = | 1  0  0  3  0 |
///     | 0  1  0  4  0 |
///     | 0  0  1  5  0 |
///     | 0  0  0  1  0 |
///     | 0  0  0  0  1 |
/// ```
///
/// you can use the following code:
///
/// ```rust
/// let eta = EtaMatrix::new(3, DVector::from_vec(vec![3, 4, 5, 1, 0]));
/// ```
///
/// # Math
///
/// The eta matrix is defined as:
///
/// ```text
/// E = |  1  ...  0    e_0    0  ...  0  |
///     | ... ... ...   ...   ... ... ... |
///     |  0  ...  1  e_{j-1}  0  ...  0  |
///     |  0  ...  0    e_j    0  ...  0  |
///     |  0  ...  0  e_{j+1}  1  ...  0  |
///     | ... ... ...   ...   ... ... ... |
///     |  0  ...  0  e_{n-1}  0  ...  1  |
/// ```
///
/// The inverse of the eta matrix is:
///
/// ```text
/// E^{-1} = |  1  ...  0      -e_0/e_j  0  ...  0  |
///          | ... ... ...     ...      ... ... ... |
///          |  0  ...  1  -e_{j-1}/e_j  0  ...  0  |
///          |  0  ...  0         1/e_j  0  ...  0  |
///          |  0  ...  0  -e_{j+1}/e_j  1  ...  0  |
///          | ... ... ...     ...      ... ... ... |
///          |  0  ...  0  -e_{n-1}/e_j  0  ...  1  |
/// ```
#[derive(Debug, Clone)]
pub struct EtaMatrix<T> {
    /// The index of the column that is not the identity column.
    column_index: usize,

    /// The column that is not the identity column.
    eta_column: DVector<T>,
}

/// Represents an error that occurs when attempting to construct an `EtaMatrix`
/// with an invalid column index.
///
/// This error is triggered when the specified `column_index` is not a valid index
/// for the `eta_column`, which determines the size of the eta matrix. Since the
/// eta matrix is square, the number of rows and columns is equal to the length of
/// the `eta_column`.
///
/// # Details
/// - The `column_index` must be less than the length of the `eta_column`.
/// - If this condition is violated, the `EtaMatrix::new` constructor will return
///   this error to indicate that the provided `column_index` is out of bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnOutOfBoundsError {
    /// The index of the column that is out of bounds.
    column_index: usize,

    /// The size of the eta matrix.
    /// Since the eta matrix is square, the size determines both the number of rows and columns.
    size: usize,
}

impl ColumnOutOfBoundsError {
    /// Construct a new `ColumnOutOfBoundsError`.
    ///
    /// # Parameters
    /// - `column_index`: The index of the column that is out of bounds.
    /// - `size`: The size of the eta matrix.
    ///
    /// # Returns
    /// A new `ColumnOutOfBoundsError`.
    pub fn new(column_index: usize, size: usize) -> Self {
        Self { column_index, size }
    }

    /// Get the index of the column that is out of bounds.
    ///
    /// # Returns
    /// The index of the column that is out of bounds.
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Get the size of the eta matrix.
    ///
    /// # Returns
    /// The size of the eta matrix.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl std::fmt::Display for ColumnOutOfBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "column index {} is out of bounds for eta matrix with a size of {}",
            self.column_index, self.size
        )
    }
}

impl std::error::Error for ColumnOutOfBoundsError {}

impl<T> EtaMatrix<T> {
    /// Construct a new eta matrix.
    ///
    /// # Parameters
    /// - `column_index`: The index of the column that is not the identity column.
    /// - `eta_column`: The column that is not the identity column.
    ///
    /// # Returns
    /// A new eta matrix.
    pub fn new(
        column_index: usize,
        eta_column: DVector<T>,
    ) -> Result<Self, ColumnOutOfBoundsError> {
        if column_index >= eta_column.len() {
            return Err(ColumnOutOfBoundsError::new(column_index, eta_column.len()));
        }

        Ok(Self {
            column_index,
            eta_column,
        })
    }

    /// Get the index of the column that is not the identity column.
    ///
    /// # Returns
    /// The index of the column that is not the identity column.
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Get the column that is not the identity column.
    ///
    /// # Returns
    /// The column that is not the identity column.
    pub fn eta_column(&self) -> &DVector<T> {
        &self.eta_column
    }

    /// Get the size of the eta matrix.
    ///
    /// # Note
    /// The eta matrix is a square matrix, so the size determines both the number of rows and columns.
    ///
    /// # Returns
    /// The size of the eta matrix.
    pub fn size(&self) -> usize {
        self.eta_column.len()
    }

    /// Get the number of rows of the eta matrix.
    ///
    /// # Returns
    /// The number of rows of the eta matrix.
    pub fn nrows(&self) -> usize {
        self.eta_column.len()
    }

    /// Get the number of columns of the eta matrix.
    ///
    /// # Returns
    /// The number of columns of the eta matrix.
    pub fn ncols(&self) -> usize {
        self.eta_column.len()
    }

    /// Gets an iterator over the elements of the non-identity column of the eta matrix.
    ///
    /// # Returns
    /// An iterator over the elements of the non-identity column of the eta matrix.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.eta_column.iter()
    }

    /// Get the pivot element of the eta matrix.
    ///
    /// # Returns
    /// The pivot element of the eta matrix.
    ///
    /// # Note
    /// In the eta matrix, the pivot element is located in the intersection of the
    /// specified `column_index` and the same row, `column_index`. This is because
    /// the eta matrix modifies a single column of the identity matrix, making it
    /// non-identity. The non-zero entries of this column correspond to the row indices
    /// in the eta column, and the diagonal element (which is the pivot) is located at
    /// the `column_index`th row and `column_index`th column.
    ///
    /// Mathematically, for an eta matrix E of size n, E is defined as:
    ///
    /// `E[i, j] = δ(i, j) + v[i], where δ(i, j)` is the Kronecker delta and `v[i]` is the non-identity column.
    ///
    /// Therefore, the pivot element (diagonal of `E`) is always located at `(j, j)`,
    /// where `j` is the column index, aligning the row and column indices in this case.
    pub fn pivot(&self) -> &T {
        &self.eta_column[self.column_index]
    }

    /// Get the row index of the pivot element of the eta matrix.
    ///
    /// # Returns
    /// The row index of the pivot element of the eta matrix.
    ///
    /// # Note
    /// In the eta matrix, the pivot element is located in the intersection of the
    /// specified `column_index` and the same row, `column_index`. This is because
    /// the eta matrix modifies a single column of the identity matrix, making it
    /// non-identity. The non-zero entries of this column correspond to the row indices
    /// in the eta column, and the diagonal element (which is the pivot) is located at
    /// the `column_index`th row and `column_index`th column.
    ///
    /// Mathematically, for an eta matrix E of size n, E is defined as:
    ///
    /// `E[i, j] = δ(i, j) + v[i], where δ(i, j)` is the Kronecker delta and `v[i]` is the non-identity column.
    ///
    /// Therefore, the pivot element (diagonal of `E`) is always located at `(j, j)`,
    /// where `j` is the column index, aligning the row and column indices in this case.
    pub fn pivot_row(&self) -> usize {
        self.column_index
    }

    /// Get the column index of the pivot element of the eta matrix.
    ///
    /// # Returns
    /// The column index of the pivot element of the eta matrix.
    ///
    /// # Note
    /// In the eta matrix, the pivot element is located in the intersection of the
    /// specified `column_index` and the same row, `column_index`. This is because
    /// the eta matrix modifies a single column of the identity matrix, making it
    /// non-identity. The non-zero entries of this column correspond to the row indices
    /// in the eta column, and the diagonal element (which is the pivot) is located at
    /// the `column_index`th row and `column_index`th column.
    ///
    /// Mathematically, for an eta matrix E of size n, E is defined as:
    ///
    /// `E[i, j] = δ(i, j) + v[i], where δ(i, j)` is the Kronecker delta and `v[i]` is the non-identity column.
    ///
    /// Therefore, the pivot element (diagonal of `E`) is always located at `(j, j)`,
    /// where `j` is the column index, aligning the row and column indices in this case.
    pub fn pivot_column(&self) -> usize {
        self.column_index
    }
}

impl<T> EtaMatrix<T>
where
    T: Scalar,
{
    /// Get the element of the eta matrix in the non identity column and the given row.
    ///
    /// # Parameters
    /// - `row`: The row of the element to get.
    ///
    /// # Returns
    /// The element of the eta matrix in the non identity column and the given row.
    pub fn get(&self, row: usize) -> Option<&T> {
        self.eta_column.get(row)
    }

    /// Get the element of the eta matrix in the non identity column and the given row.
    ///
    /// # Parameters
    /// - `row`: The row of the element to get.
    ///
    /// # Returns
    /// The element of the eta matrix in the non identity column and the given row.
    pub fn get_mut(&mut self, row: usize) -> Option<&mut T> {
        self.eta_column.get_mut(row)
    }
}

/// An error indicating that the size of a column vector does not match the size of an eta matrix.
/// This error occurs when attempting to perform an operation with an eta matrix and a vector
/// that have incompatible sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionMismatchError {
    /// The size of the eta matrix.
    /// Since the eta matrix is square, the size determines both the number of rows and columns.
    eta_matrix_size: usize,

    /// The length of the vector that caused the error.
    vector_len: usize,
}

impl DimensionMismatchError {
    /// Construct a new `DimensionMismatchError`.
    ///
    /// # Parameters
    /// - `eta_matrix_size`: The size of the eta matrix.
    /// - `vector_len`: The length of the vector that caused the error.
    ///
    /// # Returns
    /// A new `DimensionMismatchError`.
    pub fn new(eta_matrix_size: usize, vector_len: usize) -> Self {
        Self {
            eta_matrix_size,
            vector_len,
        }
    }

    /// Get the size of the eta matrix.
    ///
    /// # Returns
    /// The size of the eta matrix.
    pub fn eta_matrix_size(&self) -> usize {
        self.eta_matrix_size
    }

    /// Get the length of the vector that caused the error.
    ///
    /// # Returns
    /// The length of the vector that caused the error.
    pub fn vector_len(&self) -> usize {
        self.vector_len
    }
}

impl std::fmt::Display for DimensionMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "dimension mismatch: eta matrix size is {} but vector length is {}",
            self.eta_matrix_size, self.vector_len
        )
    }
}

impl std::error::Error for DimensionMismatchError {}

/// An error indicating a zero pivot element which is not allowed for certain operations.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// An error that can occur when solving a system of linear equations with an eta matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EtaMatrixSolveError {
    /// An error indicating that the size of a column vector does not match the size of an eta matrix.
    DimensionMismatch(DimensionMismatchError),

    /// Encountered a zero pivot element which is not allowed for certain operations.
    ZeroPivotElement(ZeroPivotElementError),
}

impl std::fmt::Display for EtaMatrixSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch(err) => write!(f, "{}", err),
            Self::ZeroPivotElement(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for EtaMatrixSolveError {}

impl From<DimensionMismatchError> for EtaMatrixSolveError {
    fn from(err: DimensionMismatchError) -> Self {
        Self::DimensionMismatch(err)
    }
}

impl From<ZeroPivotElementError> for EtaMatrixSolveError {
    fn from(err: ZeroPivotElementError) -> Self {
        Self::ZeroPivotElement(err)
    }
}

impl<T> EtaMatrix<T>
where
    T: Scalar
        + Copy
        + Zero
        + SubAssign
        + DivAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    /// Solves the system `y.E = c`, `c` beeing the initial value of `y`.
    /// Then `y = c.E^{-1}`, so `y` is equal to `c` except for
    /// `y_j = (c_j - \sum_{i != j}{c_i * e_i}) / e_j`.
    ///
    /// # Parameters
    /// - `y`: The vector to solve the system for.
    ///
    /// # Returns
    /// Error is an error occurs during the solve operation.
    pub fn left_solve_mut(&self, y: &mut DVector<T>) -> Result<(), EtaMatrixSolveError> {
        if self.size() != y.len() {
            return Err(DimensionMismatchError::new(self.size(), y.len()).into());
        }

        let pivot_value = *self.pivot();
        if pivot_value.is_zero() {
            return Err(ZeroPivotElementError::new(self.pivot_column(), self.pivot_row()).into());
        }

        // Fetch the value of y corresponding to the pivot column.
        let mut y_value = y[self.column_index];

        // Subtract contributions from the eta column.
        for (row_index, &eta_coefficient) in self.eta_column.iter().enumerate() {
            if row_index != self.column_index {
                y_value -= y[row_index] * eta_coefficient;
            }
        }

        // Normalize the pivot column value.
        y[self.column_index] = y_value / pivot_value;

        Ok(())
    }

    /// Solves the system `y.E = c`, `c` beeing the initial value of `y`.
    /// Then `y = c.E^{-1}`, so `y` is equal to `c` except for
    /// `y_j = (c_j - \sum_{i != j}{c_i * e_i}) / e_j`.
    ///
    /// # Parameters
    /// - `y`: The vector to solve the system for.
    ///
    /// # Returns
    /// The solution of the system.
    #[inline]
    pub fn left_solve(&self, y: &DVector<T>) -> Result<DVector<T>, EtaMatrixSolveError> {
        let mut y = y.clone();
        self.left_solve_mut(&mut y)?;
        Ok(y)
    }

    /// Solves the system `E.d = a`, `a` beeing the initial value of `d`.
    /// Then
    ///
    /// ```text
    /// d = E^{-1}.a = | a_0     - e_0   *   a_j / e_j  |
    ///                |            ...                 |
    ///                |  a_{j-1} - e_{j-1} * a_j / e_j |
    ///                |                      a_j / e_j |
    ///                |  a_{j+1} - e_{j+1} * a_j / e_j |
    ///                |            ...                 |
    ///                |  a_{n-1} - e_{n-1} * a_j / e_j |
    /// ```
    ///
    /// # Parameters
    /// - `d`: The vector to solve the system for.
    ///
    /// # Returns
    /// Error is an error occurs during the solve operation.
    pub fn right_solve_mut(&self, d: &mut DVector<T>) -> Result<(), EtaMatrixSolveError> {
        // Check the dimensions of the eta matrix and the vector.
        if self.size() != d.len() {
            return Err(DimensionMismatchError::new(self.size(), d.len()).into());
        }

        // From here it is totally safe to index into the eta matrix and the vector
        // because we have already checked that the dimensions match.
        let pivot_value = *self.pivot();

        // Check if the pivot element is zero.
        if pivot_value.is_zero() {
            return Err(ZeroPivotElementError::new(self.pivot_row(), self.pivot_column()).into());
        }

        // Normalize the pivot row by dividing the vector's corresponding entry by the pivot value.
        d[self.column_index] /= pivot_value;
        let normalized_value = d[self.column_index];

        // Update all rows except the pivot row by subtracting the scaled eta column values.
        for (row_index, &eta_coefficient) in self.eta_column.iter().enumerate() {
            if row_index != self.column_index {
                d[row_index] -= normalized_value * eta_coefficient;
            }
        }

        Ok(())
    }

    /// Solves the system `E.d = a`, `a` beeing the initial value of `d`.
    /// Then
    ///
    /// ```text
    /// d = E^{-1}.a = | a_0     - e_0   *   a_j / e_j  |
    ///                |            ...                 |
    ///                |  a_{j-1} - e_{j-1} * a_j / e_j |
    ///                |                      a_j / e_j |
    ///                |  a_{j+1} - e_{j+1} * a_j / e_j |
    ///                |            ...                 |
    ///                |  a_{n-1} - e_{n-1} * a_j / e_j |
    /// ```
    ///
    /// # Parameters
    /// - `d`: The vector to solve the system for.
    ///
    /// # Returns
    /// The solution of the system.
    pub fn right_solve(&self, d: &DVector<T>) -> Result<DVector<T>, EtaMatrixSolveError> {
        let mut d = d.clone();
        self.right_solve_mut(&mut d)?;
        Ok(d)
    }
}

impl<T> std::ops::Index<usize> for EtaMatrix<T> {
    type Output = T;

    /// Performs an indexing into the non identity column of the eta matrix.
    ///
    /// # Parameters
    /// - `index`: The index of the element to get.
    ///
    /// # Returns
    /// The element of the eta matrix in the non identity column at the given index.
    ///
    /// # Panics
    /// If the index is out of bounds.
    fn index(&self, index: usize) -> &Self::Output {
        &self.eta_column[index]
    }
}

impl<T> ValidIndex<usize> for EtaMatrix<T> {
    fn is_index_valid(&self, index: usize) -> bool {
        index < self.eta_column.len()
    }
}

impl<'a, T> IntoIterator for &'a EtaMatrix<T>
where
    T: Scalar,
{
    type Item = &'a T;
    type IntoIter = MatrixIter<'a, T, Dyn, Const<1>, VecStorage<T, Dyn, Const<1>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.eta_column.into_iter()
    }
}

impl<T> PartialEq for EtaMatrix<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.column_index == other.column_index && self.eta_column == other.eta_column
    }
}

impl<T> Eq for EtaMatrix<T> where T: Eq {}

impl<T> PartialOrd for EtaMatrix<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.column_index.partial_cmp(&other.column_index)
    }
}

impl<T> Ord for EtaMatrix<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.column_index.cmp(&other.column_index)
    }
}

impl<T> std::fmt::Display for EtaMatrix<T>
where
    T: std::fmt::Debug + std::fmt::Display + Clone + PartialEq + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}|{}", self.column_index, self.eta_column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_left_solve() {
        // Create an EtaMatrix with a non-identity column.
        let column_index = 1;
        let eta_column = DVector::from_vec(vec![0.0, 2.0, 1.0]);
        let eta_matrix = EtaMatrix::new(column_index, eta_column).unwrap();

        // Input vector to solve: `y.E = c`
        let c = DVector::from_vec(vec![4.0, 8.0, 6.0]);

        // Solve for `y`.
        let y = eta_matrix.left_solve(&c).unwrap();

        // Manually computed expected result.
        let expected_y = DVector::from_vec(vec![4.0, 1.0, 6.0]);

        // Assert the result is as expected.
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_right_solve() {
        // Create an EtaMatrix with a non-identity column.
        let column_index = 1;
        let eta_column = DVector::from_vec(vec![3.0, 2.0, 0.0]);
        let eta_matrix = EtaMatrix::new(column_index, eta_column).unwrap();

        // Input vector to solve: `E.d = a`
        let a = DVector::from_vec(vec![9.0, 4.0, 7.0]);

        // Solve for `d`.
        let d = eta_matrix.right_solve(&a).unwrap();

        // Manually computed expected result.
        let expected_d = DVector::from_vec(vec![3.0, 2.0, 7.0]);

        // Assert the result is as expected.
        assert_eq!(d, expected_d);
    }
}
