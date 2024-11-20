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

/// An eta matrix E corresponds to the identity matrix except for one column e of
/// index j. In particular, B.E is the matrix of the new basis obtained from B by
/// replacing the j-th vector of B by B.e, note that this is exactly what happens
/// during a "pivot" of the current basis in the simplex algorithm.
///
/// # Type parameters
/// - `T`: The type of the elements of the eta matrix.
///
/// # Math
///
/// The eta matrix is defined as:
///
/// ```text
/// E = [  1  ...  0    e_0    0  ...  0
///       ... ... ...   ...   ... ... ...
///        0  ...  1  e_{j-1}  0  ...  0
///        0  ...  0    e_j    0  ...  0
///        0  ...  0  e_{j+1}  1  ...  0
///       ... ... ...   ...   ... ... ...
///        0  ...  0  e_{n-1}  0  ...  1 ]
/// ```
///
/// The inverse of the eta matrix is:
///
/// ```text
/// E^{-1} = [  1  ...  0      -e_0/e_j  0  ...  0
///            ... ... ...     ...      ... ... ...
///             0  ...  1  -e_{j-1}/e_j  0  ...  0
///             0  ...  0         1/e_j  0  ...  0
///             0  ...  0  -e_{j+1}/e_j  1  ...  0
///            ... ... ...     ...      ... ... ...
///             0  ...  0  -e_{n-1}/e_j  0  ...  1 ]
/// ```
#[derive(Debug, Clone)]
pub struct EtaMatrix<T> {
    /// The index of the column that is not the identity column.
    column_index: usize,

    /// The column that is not the identity column.
    eta_column: DVector<T>,
}

impl<T> EtaMatrix<T> {
    /// Construct a new eta matrix.
    ///
    /// # Parameters
    /// - `column_index`: The index of the column that is not the identity column.
    /// - `eta_column`: The column that is not the identity column.
    ///
    /// # Returns
    /// A new eta matrix.
    pub fn new(column_index: usize, eta_column: DVector<T>) -> Self {
        Self {
            column_index,
            eta_column,
        }
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
    T: Scalar
        + Copy
        + SubAssign
        + DivAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    /// Perform the left solve operation with the eta matrix on a mutable vector `y`.
    ///
    /// This function modifies `y` in place.
    ///
    /// # Parameters
    /// - `y`: The mutable vector to be transformed.
    pub fn left_solve_mut(&self, y: &mut DVector<T>) {
        let pivot_value = self.eta_column[self.column_index];
        y[self.column_index] /= pivot_value;
        let normalized_value = y[self.column_index];

        for (row_index, &eta_coefficient) in self.eta_column.iter().enumerate() {
            if row_index != self.column_index {
                y[row_index] -= normalized_value * eta_coefficient;
            }
        }
    }

    /// Perform the left solve operation with the eta matrix on a vector `y`.
    ///
    /// # Parameters
    /// - `y`: The vector to be transformed.
    ///
    /// # Returns
    /// The transformed sparse vector.
    #[inline]
    pub fn left_sole(&self, y: &DVector<T>) -> DVector<T> {
        let mut y = y.clone();
        self.left_solve_mut(&mut y);
        y
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
    fn test_left_solve_basic_case() {
        // Create an EtaMatrix with a non-identity column.
        let column_index = 1;
        let eta_column = DVector::from_vec(vec![6.0, 2.0, 0.0]);
        // The eta matrix is
        //
        // | 1 6 0 |
        // | 0 2 0 |
        // | 0 0 1 |
        //
        // Thus the inverse of the eta matrix is
        //
        // | 1 -3  0 |
        // | 0 0.5 0 |
        // | 0  0  1 |
        let eta_matrix = EtaMatrix::new(column_index, eta_column);

        // Input vector to transform.
        // The vector is [1, 4, 3].
        let mut y = DVector::from_vec(vec![1.0, 4.0, 3.0]);

        // We will try to solve the equation
        //
        // | 1 6 0 |   | x1 |   | 1 |
        // | 0 2 0 | * | x2 | = | 4 |
        // | 0 0 1 |   | x3 |   | 3 |
        //
        eta_matrix.left_solve_mut(&mut y);

        // The vector should be [-11, 2, 3].
        let expected = DVector::from_vec(vec![-11.0, 2.0, 3.0]);

        // Assert equality.
        assert_eq!(y, expected);
    }
}
