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

use nalgebra::{DVector, Dyn, RowDVector, Scalar, U1};

/// A type alias for a dense column vector from the `nalgebra` library.
///
/// `DenseColumn<T>` represents a dynamically-sized column vector where each element
/// is of type `T`.
pub type DenseColumn<T> = DVector<T>;

/// A type alias for a dense row vector from the `nalgebra` library.
///
/// `DenseRow<T>` represents a dynamically-sized row vector where each element
/// is of type `T`.
pub type DenseRow<T> = RowDVector<T>;

/// A trait for converting a type into a dense column vector.
///
/// # Type Parameters
/// - `T`: The type of the elements in the column vector.
pub trait IntoDenseColumn<T> {
    fn into_dense_column(self) -> DenseColumn<T>;
}

/// A trait for converting a type into a dense row vector.
///
/// # Type Parameters
/// - `T`: The type of the elements in the row vector.
pub trait IntoDenseRow<T> {
    fn into_dense_row(self) -> DenseRow<T>;
}

impl<T: Scalar> IntoDenseColumn<T> for DenseRow<T> {
    fn into_dense_column(self) -> DenseColumn<T> {
        let nrows = Dyn(self.ncols());
        self.reshape_generic(nrows, U1)
    }
}

impl<T: Scalar> IntoDenseRow<T> for DenseColumn<T> {
    fn into_dense_row(self) -> DenseRow<T> {
        let ncols: Dyn = Dyn(self.nrows());
        self.reshape_generic(U1, ncols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a dense column vector with the elements 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0.
    ///
    /// # Returns
    /// A dense column vector with the elements 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0.
    fn create_dense_row() -> DenseRow<f64> {
        DenseRow::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    }

    /// Creates a dense row vector with the elements 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0.
    ///
    /// # Returns
    /// A dense row vector with the elements 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0.
    fn create_dense_column() -> DenseColumn<f64> {
        DenseColumn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    }

    #[test]
    fn test_dense_row_into_dense_column() {
        let row = create_dense_row();
        let expected_row_count = row.ncols();
        let col = row.into_dense_column();

        assert_eq!(1, col.ncols());
        assert_eq!(expected_row_count, col.nrows());
        assert_eq!(col, create_dense_column());
    }

    #[test]
    fn test_dense_column_into_dense_row() {
        let col = create_dense_column();
        let expected_col_count = col.nrows();
        let row = col.into_dense_row();

        assert_eq!(1, row.nrows());
        assert_eq!(expected_col_count, row.ncols());
        assert_eq!(row, create_dense_row());
    }
}
