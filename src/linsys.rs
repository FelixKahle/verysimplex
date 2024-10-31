// Written by Felix Kahle in 2024. All rights reserved.

#![allow(dead_code)]

use nalgebra::{
    ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, Const, DefaultAllocator, Dim, Matrix,
    RawStorageMut, Scalar,
};
use num_traits::{Float, One, Zero};
use std::fmt::Display;

/// Error occuring during Gaussian elimination due to a zero pivot element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroPivotElementError;

impl std::error::Error for ZeroPivotElementError {}

impl Display for ZeroPivotElementError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Zero pivot element encountered during Gaussian elimination."
        )
    }
}

/// A trait defining Gaussian elimination functionality for a matrix.
///
/// This trait provides a method for performing Gaussian elimination to transform a matrix
/// column into an identity-like structure. The transformation maintains the integrity of the
/// underlying linear system, and it works across various matrix dimensions and scalar types.
///
/// # Type parameters
/// - `T`: Scalar type representing the matrix's numeric components.
/// - `R`: Row dimension of the matrix (can be dynamic).
/// - `C`: Column dimension of the matrix (can be dynamic).
/// - `S`: Type of storage used to hold the matrix data (usually based on `nalgebra` storage).
pub trait GaussianElimination<T, R, C, S> {
    /// Transforms a specified column into an identity-like structure by modifying the pivot row
    /// to have a `1` at the specified position and setting all other entries in that column to `0`.
    ///
    /// # Parameters
    ///
    /// * `target_row` - The row index of the pivot element to normalize.
    /// * `target_column` - The column index to transform into an identity-like column.
    ///
    /// # Returns
    ///
    /// Result containing either:
    /// - A new matrix with the specified column in an identity-like form.
    /// - An error if the pivot element is zero, making it impossible to proceed.
    fn to_column_identity(
        &self,
        target_row: usize,
        target_column: usize,
    ) -> Result<Matrix<T, R, C, S>, ZeroPivotElementError>;

    /// Modifies a specified column to transform it into an identity-like structure by setting the pivot row
    /// to `1` at the specified position and all other entries in that column to `0`.
    ///
    /// This method operates in-place on the current matrix, applying a Gaussian elimination step to normalize
    /// the specified row and zero out the rest of the column while preserving the integrity of the linear system.
    ///
    /// # Parameters
    ///
    /// - `target_row` - The row index of the pivot element to set to `1`.
    /// - `target_column` - The column index to transform into an identity-like column by zeroing out all other entries.
    ///
    /// # Returns
    ///
    /// Result containing either:
    /// - Ok(()) if the operation was successful.
    /// - An error if the pivot element is zero, making it impossible to proceed.
    fn apply_column_identity(
        &mut self,
        target_row: usize,
        target_column: usize,
    ) -> Result<(), ZeroPivotElementError>;
}

impl<T, R, C, S> GaussianElimination<T, R, C, S> for Matrix<T, R, C, S>
where
    T: Float + Scalar + ClosedMulAssign + ClosedDivAssign + ClosedSubAssign + Zero + One, // Only allow floating-point types for Gaussian elimination.
    R: Dim,                            // Allows dynamic or fixed row dimensions.
    C: Dim,                            // Allows dynamic or fixed column dimensions.
    S: RawStorageMut<T, R, C> + Clone, // Requires mutable storage to modify the matrix.
    DefaultAllocator: nalgebra::allocator::Allocator<Const<1>, C>,
{
    fn to_column_identity(
        &self,
        target_row: usize,
        target_column: usize,
    ) -> Result<Matrix<T, R, C, S>, ZeroPivotElementError> {
        let mut matrix = self.clone();
        matrix.apply_column_identity(target_row, target_column)?;
        Ok(matrix)
    }

    fn apply_column_identity(
        &mut self,
        target_row: usize,
        target_column: usize,
    ) -> Result<(), ZeroPivotElementError> {
        // Check if the pivot element is zero. If so, return an error as we cannot proceed with elimination.
        if self[(target_row, target_column)].is_zero() {
            return Err(ZeroPivotElementError);
        }

        // Retrieve the pivot element at the specified target row and column.
        let pivot_value = self[(target_row, target_column)];

        // Calculate the scalar needed to normalize the pivot element to 1.
        // This is equivalent to dividing each element in the row by the pivot value.
        // Because we only allow floating-point types, the overflows will not occur,
        // rather the result will be positive or negative infinity or NaN.
        let scalar = T::one() / pivot_value;

        // Adjust every element in the pivot row to normalize it, scaling it by the divisor.
        // This operation sets the pivot element to 1, making it ready for elimination.
        let mut row_to_adjust = self.row_mut(target_row);
        row_to_adjust *= scalar;

        // Explicitly set the pivot element to `1` to correct for potential floating-point inaccuracies.
        self[(target_row, target_column)] = T::one();

        // Gaussian elimination loop: iterate over each row to set elements in the target column to zero.
        for row_index in 0..self.nrows() {
            // Skip the pivot row, as it is already normalized.
            if row_index == target_row {
                continue;
            }

            let factor = self[(row_index, target_column)];

            // If the factor is zero, we can skip the row as it will not affect the elimination.
            if factor.is_zero() {
                continue;
            }

            // Take a copy of the normalized pivot row for manipulation.
            let adjusted_row = self.row(target_row).into_owned();

            // Perform row transformation: subtract the pivot row multiplied by the factor
            // from the current row to set the target element to zero.
            let mut row_to_transform = self.row_mut(row_index);
            row_to_transform -= adjusted_row * factor;

            // Set the target element in the column to zero explicitly to avoid floating-point errors.
            self[(row_index, target_column)] = T::zero();
        }
        Ok(())
    }
}
