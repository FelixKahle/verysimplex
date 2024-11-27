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

use crate::{linerr::InvalidColumnCountError, linsys::RightSolve};
use nalgebra::{ComplexField, Dyn, FullPivLU};

/// Error that occurs when solving a linear system of the form `A * x = b`,
/// where `A` is the LU decomposition of a matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LUMatrixRightSolveError {
    /// Error that occurs when the number of columns in the LU decomposition does not match
    /// the number of elements in the right-hand side vector.
    InvalidColumnCount(InvalidColumnCountError),

    /// The matrix the LU decomposition was computed from was not square.
    NonSquareMatrix,

    /// The matrix the LU decomposition was computed from is not invertible.
    NonInvertibleMatrix,
}

impl std::fmt::Display for LUMatrixRightSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LUMatrixRightSolveError::InvalidColumnCount(e) => write!(f, "{}", e),
            LUMatrixRightSolveError::NonSquareMatrix => write!(f, "non square matrix"),
            LUMatrixRightSolveError::NonInvertibleMatrix => write!(f, "non invertible matrix"),
        }
    }
}

impl std::error::Error for LUMatrixRightSolveError {}

impl<T: ComplexField> RightSolve<T> for FullPivLU<T, Dyn, Dyn> {
    type Error = LUMatrixRightSolveError;

    fn right_solve(
        &self,
        mut b: crate::alias::DenseColumn<T>,
    ) -> Result<crate::alias::DenseColumn<T>, Self::Error> {
        let lu = self.lu_internal();

        if lu.nrows() != b.nrows() {
            return Err(LUMatrixRightSolveError::InvalidColumnCount(
                InvalidColumnCountError {
                    expected: lu.nrows(),
                    actual: b.nrows(),
                },
            ));
        }

        if !lu.is_square() {
            return Err(LUMatrixRightSolveError::NonSquareMatrix);
        }

        if !self.is_invertible() {
            return Err(LUMatrixRightSolveError::NonInvertibleMatrix);
        }

        self.p().permute_rows(&mut b);
        let _ = lu.solve_lower_triangular_with_diag_mut(&mut b, T::one());
        let _ = lu.solve_upper_triangular_mut(&mut b);
        self.q().inv_permute_rows(&mut b);

        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alias::DenseColumn;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    #[test]
    fn test_right_solve() {
        // Create the matrix:
        // |  10  0   3   22  11 |
        // |  11  1   4   33  90 |
        // |  33  44  7   13  2  |
        // |  54  33  44  7   13 |
        let a = DMatrix::from_row_slice(
            4,
            4,
            &[
                10.0, 0.0, 3.0, 22.0, 11.0, 1.0, 4.0, 33.0, 90.0, 3.0, 2.0, 54.0, 33.0, 44.0, 7.0,
                13.0,
            ],
        );

        // Create the vector:
        // | 5 |
        // | 6 |
        // | 7 |
        // | 8 |
        let b = DenseColumn::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        // Solve the linear system:
        // |  10  0   3   22  11 |   | x1 |   | 5 |
        // |  11  1   4   33  90 |   | x2 |   | 6 |
        // |  33  44  7   13  2  | * | x3 | = | 7 |
        // |  54  33  44  7   13 |   | x4 |   | 8 |
        let lu = a.full_piv_lu();
        let x = lu.right_solve(b).unwrap();

        // We expect the solution to be:
        // | 0.08759308  |
        // | -0.17163593 |
        // | 1.95598354  |
        // | -0.07926733 |
        let x_expected =
            DenseColumn::from_vec(vec![0.08759308, -0.17163593, 1.95598354, -0.07926733]);

        // Check that the solution is correct. Use a large epsilon as we computed the test
        // with python and numpy.
        assert_relative_eq!(x_expected, x, epsilon = 1e-6);
    }
}
