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

use sprs::CsVec;
use std::fmt::Write;

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
    column: CsVec<T>,
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
    pub fn new(column_index: usize, column: CsVec<T>) -> Self {
        Self {
            column_index,
            column,
        }
    }

    /// Get the index of the column of the eta matrix.
    ///
    /// # Returns
    /// The index of the column of the eta matrix.
    #[inline]
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Get the column of the eta matrix.
    ///
    /// # Returns
    /// The column of the eta matrix.
    #[inline]
    pub fn column(&self) -> &CsVec<T> {
        &self.column
    }

    /// Perform the left solve operation with the eta matrix on a mutable vector `y`.
    ///
    /// This function modifies `y` in place.
    ///
    /// # Parameters
    /// - `y`: The mutable vector to be transformed.
    pub fn left_solve_mut(&self, y: &mut CsVec<T>)
    where
        T: Copy + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::SubAssign,
    {
        let mut y_value = y[self.column_index];
        for (row, &eta_coeff) in self.column.iter() {
            y_value -= y[row] * eta_coeff;
        }
        y[self.column_index] = y_value / self.column[self.column_index];
    }

    /// Perform the left solve operation with the eta matrix on a vector `y`.
    ///
    /// # Parameters
    /// - `y`: The vector to be transformed.
    ///
    /// # Returns
    /// The transformed sparse vector.
    #[inline]
    pub fn left_solve(&self, y: &CsVec<T>) -> CsVec<T>
    where
        T: Copy + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::SubAssign,
    {
        let mut y = y.clone();
        self.left_solve_mut(&mut y);
        y
    }

    /// Return the size of the eta matrix.
    ///
    /// This represents the number of rows or columns since the eta matrix is square.
    ///
    /// # Returns
    /// The size of the eta matrix.
    #[inline]
    pub fn size(&self) -> usize {
        self.column.dim()
    }
}

/// Converts a sparse vector to a string in the format `[a, b, c, ...]`.
///
/// # Parameters
/// - `vec`: The sparse vector to convert to a string.
///
/// # Returns
/// A string representation of the sparse vector.
#[inline(always)]
fn vec_to_string<T>(vec: &CsVec<T>) -> Result<String, std::fmt::Error>
where
    T: std::fmt::Display,
{
    let mut string = String::new();
    string.push('[');
    let mut iter = vec.iter().peekable();
    while let Some((_, value)) = iter.next() {
        write!(&mut string, "{}", value)?;
        if iter.peek().is_some() {
            string.push_str(", ");
        }
    }
    string.push(']');

    Ok(string)
}

impl<T> std::fmt::Display for EtaMatrix<T>
where
    T: std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let column_string = vec_to_string(&self.column)?;
        write!(f, "{}|{}", self.column_index, column_string)
    }
}

impl<T> Into<CsVec<T>> for EtaMatrix<T> {
    fn into(self) -> CsVec<T> {
        self.column
    }
}

impl<T> From<(usize, CsVec<T>)> for EtaMatrix<T> {
    fn from((column_index, column): (usize, CsVec<T>)) -> Self {
        Self::new(column_index, column)
    }
}
