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

use crate::{
    alias::{DenseColumn, DenseRow},
    idx::ValidIndex,
    linerr::{InvalidColumnCountError, InvalidRowCountError},
    linsys::{LeftSolve, RightSolve},
};
use nalgebra::{iter::MatrixIter, Const, Dyn, Scalar, SquareMatrix, VecStorage};
use num_traits::{One, Zero};
use std::ops::{DivAssign, SubAssign};

/// An eta matrix E corresponds to the identity matrix except for one column e of
/// index j. In particular, B.E is the matrix of the new basis obtained from B by
/// replacing the j-th vector of B by B.e, note that this is exactly what happens
/// during a "pivot" of the current basis in the simplex algorithm.
///
/// Stored is only the column e and the index, t
/// the rest of the matrix is the identity matrix.
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
/// let eta = EtaMatrix::new(3, DenseColumn::from_vec(vec![3, 4, 5, 1, 0]));
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
    eta_column: DenseColumn<T>,
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
        eta_column: DenseColumn<T>,
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
    pub fn eta_column(&self) -> &DenseColumn<T> {
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

    /// Get the determinant of the eta matrix.
    ///
    /// # Returns
    /// The determinant of the eta matrix.
    pub fn determinant(&self) -> &T {
        &self.eta_column[self.column_index]
    }

    /// Returns the complete matrix representation of the eta matrix
    /// as a `SquareMatrix`.
    ///
    /// # Returns
    /// The full matrix representation of the eta matrix.
    pub fn to_full_matrix(&self) -> SquareMatrix<T, Dyn, VecStorage<T, Dyn, Dyn>>
    where
        T: Scalar + Zero + One,
    {
        let size = self.size();
        let mut eta_matrix: SquareMatrix<T, Dyn, VecStorage<T, Dyn, Dyn>> =
            SquareMatrix::<T, Dyn, VecStorage<T, Dyn, Dyn>>::identity(size, size);

        for (i, v) in self.eta_column.iter().enumerate() {
            eta_matrix[(i, self.column_index)] = v.clone();
        }

        eta_matrix
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

impl<T> EtaMatrix<T>
where
    T: Zero,
{
    /// Check if the eta matrix is singular.
    ///
    /// # Returns
    /// `true` if the eta matrix is singular, `false` otherwise.
    pub fn is_singular(&self) -> bool {
        self.eta_column[self.column_index].is_zero()
    }
}

/// Error that occurs when solving a linear system of the form `x * A = b`,
/// where `A` is the eta matrix, and `x` is the unknown to be determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtaMatrixLeftSolveError {
    /// Error that occurs when the number of rows in the eta matrix does not match
    /// the number of elements in the right-hand side vector.
    InvalidRowCount(InvalidRowCountError),

    /// Error that occurs when the eta matrix is singular.
    SingularMatrix,
}

impl std::fmt::Display for EtaMatrixLeftSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EtaMatrixLeftSolveError::InvalidRowCount(e) => write!(f, "{}", e),
            EtaMatrixLeftSolveError::SingularMatrix => write!(f, "singular matrix"),
        }
    }
}

impl std::error::Error for EtaMatrixLeftSolveError {}

impl From<InvalidRowCountError> for EtaMatrixLeftSolveError {
    fn from(e: InvalidRowCountError) -> Self {
        EtaMatrixLeftSolveError::InvalidRowCount(e)
    }
}

/// Error that occurs when solving a linear system of the form `A * x = b`,
/// where `A` is the eta matrix, and `x` is the unknown to be determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtaMatrixRightSolveError {
    /// Error that occurs when the number of columns in the eta matrix does not match
    /// the number of elements in the right-hand side vector.
    InvalidColumnCount(InvalidColumnCountError),

    /// Error that occurs when the eta matrix is singular.
    SingularMatrix,
}

impl std::fmt::Display for EtaMatrixRightSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EtaMatrixRightSolveError::InvalidColumnCount(e) => write!(f, "{}", e),
            EtaMatrixRightSolveError::SingularMatrix => write!(f, "singular matrix"),
        }
    }
}

impl std::error::Error for EtaMatrixRightSolveError {}

impl From<InvalidColumnCountError> for EtaMatrixRightSolveError {
    fn from(e: InvalidColumnCountError) -> Self {
        EtaMatrixRightSolveError::InvalidColumnCount(e)
    }
}

impl<T> LeftSolve<T> for EtaMatrix<T>
where
    T: Scalar + Copy + Zero + SubAssign + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
{
    type Error = EtaMatrixLeftSolveError;

    fn left_solve(&self, mut y: DenseRow<T>) -> Result<DenseRow<T>, Self::Error> {
        if self.size() != y.len() {
            return Err(InvalidRowCountError::new(self.size(), y.len()).into());
        }

        let pivot_value = *self.pivot();
        if pivot_value.is_zero() {
            return Err(EtaMatrixLeftSolveError::SingularMatrix);
        }

        let mut y_value = y[self.column_index];
        for (row_index, &eta_coefficient) in self.eta_column.iter().enumerate() {
            if row_index != self.column_index {
                y_value -= y[row_index] * eta_coefficient;
            }
        }

        y[self.column_index] = y_value / pivot_value;

        Ok(y)
    }
}

impl<T> RightSolve<T> for EtaMatrix<T>
where
    T: Scalar
        + Copy
        + Zero
        + SubAssign
        + DivAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    type Error = EtaMatrixRightSolveError;

    fn right_solve(&self, mut d: DenseColumn<T>) -> Result<DenseColumn<T>, Self::Error> {
        if self.size() != d.len() {
            return Err(InvalidColumnCountError::new(self.size(), d.len()).into());
        }

        let pivot_value = *self.pivot();
        if pivot_value.is_zero() {
            return Err(EtaMatrixRightSolveError::SingularMatrix);
        }

        d[self.column_index] /= pivot_value;
        let normalized_value = d[self.column_index];

        for (row_index, &eta_coefficient) in self.eta_column.iter().enumerate() {
            if row_index != self.column_index {
                d[row_index] -= normalized_value * eta_coefficient;
            }
        }

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

    #[test]
    fn test_left_solve() {
        // Create the matrix:
        // |  1  0  3  0  |
        // |  0  1  4  0  |
        // |  0  0  2  0  |
        // |  0  0  7  1  |
        let eta_matrix =
            EtaMatrix::new(2, DenseColumn::from_vec(vec![3.0, 4.0, 2.0, 7.0])).unwrap();

        // Create the vector.
        // |  5  6  7  8  |
        let a = DenseRow::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        // Solve the system:
        // |  d_0  d_1  d_2  d_3  | |  1  0  3  0  |   |  5  |
        //                          |  0  1  4  0  |   |  6  |
        //                          |  0  0  2  0  | = |  7  |
        //                          |  0  0  7  1  |   |  8  |
        let d = eta_matrix.left_solve(a).unwrap();

        // The expected solution is:
        // |  5  6  -44  8  |
        let expected_x = DenseRow::from_vec(vec![5.0, 6.0, -44.0, 8.0]);

        // Assert that the solution is correct.
        assert_eq!(d, expected_x);
    }

    #[test]
    fn test_right_solve() {
        // Create the matrix:
        // |  1  0  3  0  |
        // |  0  1  4  0  |
        // |  0  0  2  0  |
        // |  0  0  7  1  |
        let eta_matrix =
            EtaMatrix::new(2, DenseColumn::from_vec(vec![3.0, 4.0, 2.0, 7.0])).unwrap();

        // Create the vector.
        // | 5 |
        // | 6 |
        // | 7 |
        // | 8 |
        let a = DenseColumn::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        // Solve the system:
        // |  1  0  3  0  | | d_0 |   | 5 |
        // |  0  1  4  0  | | d_1 |   | 6 |
        // |  0  0  2  0  | | d_2 | = | 7 |
        // |  0  0  7  1  | | d_3 |   | 8 |
        let d = eta_matrix.right_solve(a).unwrap();

        // The expected result is:
        // |  -5.5  |
        // |  -8.0  |
        // |   5.0  |
        // |  -16.5  |
        let expected_d = DenseColumn::from_vec(vec![-5.5, -8.0, 3.5, -16.5]);

        // Assert the result is as expected.
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_to_full_matrix() {
        // Create the matrix:
        // |  1  0  3  0  |
        // |  0  1  4  0  |
        // |  0  0  2  0  |
        // |  0  0  7  1  |
        let eta_matrix =
            EtaMatrix::new(2, DenseColumn::from_vec(vec![3.0, 4.0, 2.0, 7.0])).unwrap();

        // Expected full matrix:
        // |  1  0  3  0  |
        // |  0  1  4  0  |
        // |  0  0  2  0  |
        // |  0  0  7  1  |
        let expected = SquareMatrix::<f64, Dyn, VecStorage<f64, Dyn, Dyn>>::from_row_slice(
            4,
            4,
            &[
                1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 7.0, 1.0,
            ],
        );

        // Convert the eta matrix to a full matrix.
        let full = eta_matrix.to_full_matrix();

        // Assert that the full matrix is as expected.
        assert_eq!(full, expected);
    }

    #[test]
    fn test_pivot() {
        // Create the matrix:
        // |  1  0  3  0  |
        // |  0  1  4  0  |
        // |  0  0  2  0  |
        // |  0  0  7  1  |
        let eta_matrix =
            EtaMatrix::new(2, DenseColumn::from_vec(vec![3.0, 4.0, 2.0, 7.0])).unwrap();

        // Get the pivot of the matrix.
        let pivot = eta_matrix.pivot().clone();

        // The expected pivot is 2.0.
        let expected = 2.0;

        // Assert that the pivot is as expected.
        assert_eq!(pivot, expected);
    }

    #[test]
    fn test_determinant() {
        // Create the matrix:
        // |  1  0  3  0  |
        // |  0  1  4  0  |
        // |  0  0  2  0  |
        // |  0  0  7  1  |
        let eta_matrix =
            EtaMatrix::new(2, DenseColumn::from_vec(vec![3.0, 4.0, 2.0, 7.0])).unwrap();

        // Get the determinant of the matrix.
        let determinant = eta_matrix.determinant().clone();

        // The expected determinant is 2.0.
        let expected = 2.0;

        // Assert that the determinant is as expected.
        assert_eq!(determinant, expected);
    }
}
