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

use nalgebra::{DVector, Scalar};

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
    /// The index of the column of the eta matrix
    column_index: usize,

    /// The column of the eta matrix
    column: DVector<T>,
}

impl<T> EtaMatrix<T> {
    /// Construct a new eta matrix.
    ///
    /// # Parameters
    /// - `column_index`: The index of the column of the eta matrix.
    /// - `column`: The column of the eta matrix.
    ///
    /// # Returns
    /// A new eta matrix.
    pub fn new(column_index: usize, column: DVector<T>) -> Self {
        Self {
            column_index,
            column,
        }
    }

    /// Get the index of the column of the eta matrix.
    ///
    /// # Returns
    /// The index of the column of the eta matrix.
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Get the column of the eta matrix.
    ///
    /// # Returns
    /// The column of the eta matrix.
    pub fn column(&self) -> &DVector<T> {
        &self.column
    }
}

impl<T> std::fmt::Display for EtaMatrix<T>
where
    T: Scalar + std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}|{}", self.column_index, self.column)
    }
}

impl<T> Into<DVector<T>> for EtaMatrix<T>
where
    T: Scalar,
{
    fn into(self) -> DVector<T> {
        self.column
    }
}

impl<T> From<(usize, DVector<T>)> for EtaMatrix<T> {
    fn from((column_index, column): (usize, DVector<T>)) -> Self {
        Self::new(column_index, column)
    }
}
