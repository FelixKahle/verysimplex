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

use crate::etam::EtaMatrix;
use nalgebra::{
    ClosedDivAssign, ClosedSubAssign, DMatrix, DVector, DVectorView, Dyn, Matrix, VecStorage,
};
use nalgebra_lapack::LUScalar;
use num_traits::{One, Signed, Zero};
use std::collections::HashMap;

/// Represents a solver variable, which can be either:
/// - `Fixed`: A variable with a fixed value or constrained directly by the solver.
/// - `Free`: A variable without such direct constraints, allowed to vary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SolverVariable {
    /// A fixed variable.
    Fixed(usize),

    /// A free variable.
    Free(usize),
}

impl std::fmt::Display for SolverVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SolverVariable::Fixed(i) => write!(f, "{}", i),
            SolverVariable::Free(i) => write!(f, "{}", i),
        }
    }
}

/// A simplex solver.
///
/// # Type parameters
/// - `T`: The number type.
#[derive(Debug, Clone)]
pub struct Solver<T>
where
    T: LUScalar + std::fmt::Display,
{
    // Small tolerance.
    epsilon: T,

    /// Constraint matrix (A)
    constraint_matrix: DMatrix<T>,

    /// Right-hand side values (b)
    rhs: DVector<T>,

    /// Objective coefficients (c) for all variables
    objective_coefficients: DVector<T>,

    /// Indices of basic variables
    basic_indices: Vec<usize>,

    /// Indices of non-basic variables
    non_basic_indices: Vec<usize>,

    /// Current value of the objective function (z)
    objective_value: T,

    /// Maps indices to original variables
    index_to_variable: HashMap<usize, SolverVariable>,

    /// List of Eta matrices for updating B inverse
    eta_matrices: Vec<EtaMatrix<T>>,
}

impl<T> Solver<T>
where
    T: LUScalar
        + Zero
        + One
        + std::iter::Sum
        + std::fmt::Display
        + std::ops::Sub
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::Neg<Output = T>
        + PartialOrd
        + ClosedSubAssign
        + ClosedDivAssign
        + Signed,
{
    /// Constructs a new `Solver`.
    ///
    /// # Parameters
    /// - `objective_value`: The objective value.
    /// - `constraint_matrix`: The constraint matrix (A).
    /// - `rhs`: The right-hand side values (b).
    /// - `objective_coefficients`: The objective coefficients (c) for all variables.
    /// - `basic_indices`: The indices of basic variables.
    /// - `non_basic_indices`: The indices of non-basic variables.
    /// - `index_to_variable`: Maps indices to original variables.
    /// - `epsilon`: A small tolerance.
    ///
    /// # Returns
    /// A new instance of `Solver`.
    pub fn new(
        objective_value: T,
        constraint_matrix: DMatrix<T>,
        rhs: DVector<T>,
        objective_coefficients: DVector<T>,
        basic_indices: Vec<usize>,
        non_basic_indices: Vec<usize>,
        index_to_variable: HashMap<usize, SolverVariable>,
        epsilon: T,
    ) -> Self {
        Self {
            epsilon,
            constraint_matrix,
            rhs,
            objective_coefficients,
            basic_indices,
            non_basic_indices,
            objective_value,
            index_to_variable,
            eta_matrices: Vec::new(),
        }
    }

    /// Gets the objective value.
    ///
    /// # Returns
    /// The objective value.
    pub fn objective_value(&self) -> T {
        self.objective_value
    }

    /// Gets the constraint matrix.
    ///
    /// # Returns
    /// The constraint matrix (A).
    pub fn constraint_matrix(&self) -> &DMatrix<T> {
        &self.constraint_matrix
    }

    /// Gets the number of rows in the constraint matrix.
    ///
    /// # Returns
    /// The number of rows in the constraint matrix.
    pub fn constraint_matrix_num_rows(&self) -> usize {
        self.constraint_matrix.nrows()
    }

    /// Gets the number of columns in the constraint matrix.
    ///
    /// # Returns
    /// The number of columns in the constraint matrix.
    pub fn constraint_matrix_num_cols(&self) -> usize {
        self.constraint_matrix.ncols()
    }

    /// Gets the right-hand side values.
    ///
    /// # Returns
    /// The right-hand side values (b).
    pub fn rhs(&self) -> &DVector<T> {
        &self.rhs
    }

    /// Gets the objective coefficients.
    ///
    /// # Returns
    /// The objective coefficients (c).
    pub fn objective_coefficients(&self) -> &DVector<T> {
        &self.objective_coefficients
    }

    /// Gets the basic indices.
    ///
    /// # Returns
    /// The indices of the basic variables.
    pub fn basic_indices(&self) -> &Vec<usize> {
        &self.basic_indices
    }

    /// Gets the non-basic indices.
    ///
    /// # Returns
    /// The indices of the non-basic variables.
    pub fn non_basic_indices(&self) -> &Vec<usize> {
        &self.non_basic_indices
    }

    /// Gets the objective value.
    ///
    /// # Returns
    /// The current value of the objective function.
    pub fn get_objective_value(&self) -> T {
        self.objective_value
    }

    /// Gets the index to variable mapping.
    ///
    /// # Returns
    /// A mapping from indices to original variables.
    pub fn index_to_variable(&self) -> &HashMap<usize, SolverVariable> {
        &self.index_to_variable
    }

    /// Extracts the basis matrix `B` from the constraint matrix.
    ///
    /// # Returns
    /// The basis matrix `B`.
    fn get_basis_matrix(&self) -> Matrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>> {
        self.constraint_matrix().select_columns(&self.basic_indices)
    }

    /// Extracts the non-basis matrix `N` from the constraint matrix.
    ///
    /// # Returns
    /// The non-basis matrix `N`.
    fn get_non_basis_matrix(&self) -> Matrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>> {
        self.constraint_matrix()
            .select_columns(&self.non_basic_indices)
    }

    /// Retrieves the objective coefficients for basic variables `cb`.
    ///
    /// # Returns
    /// The objective coefficients for basic variables.
    fn get_cb(&self) -> DVector<T> {
        DVector::from_iterator(
            self.basic_indices.len(),
            self.basic_indices
                .iter()
                .map(|&i| self.objective_coefficients[i]),
        )
    }

    /// Retrieves the objective coefficients for non-basic variables `cn`.
    ///
    /// # Returns
    /// The objective coefficients for non-basic variables.
    fn get_cn(&self) -> DVector<T> {
        DVector::from_iterator(
            self.non_basic_indices.len(),
            self.non_basic_indices
                .iter()
                .map(|&i| self.objective_coefficients[i]),
        )
    }

    /// Forward transformation (FTRAN): Solves `B * d = a_entering
    ///
    /// # Arguments
    /// * `a_entering` - The entering column vector
    ///
    /// # Returns
    /// The solution vector `d`
    fn ftran(&self, a_entering: &DVectorView<T>) -> DVector<T> {
        let mut d = a_entering.clone_owned();

        for eta in &self.eta_matrices {
            let col_idx = eta.column_index();
            let pivot_val = d[col_idx];
            d -= eta.eta_column() * pivot_val;
        }
        d
    }

    /// Backward transformation (BTRAN): Solves `y^T = c_B^T * B^{-1}`
    ///
    /// # Arguments
    /// - `c_b` - The objective coefficients for basic variables
    ///
    /// # Returns
    /// The solution vector `y`
    fn btran(&self, c_b: &DVectorView<T>) -> DVector<T> {
        let mut y = c_b.clone_owned();

        for eta in self.eta_matrices.iter().rev() {
            let col_idx = eta.column_index();
            let s = eta.eta_column().dot(&y);
            y[col_idx] -= s;
        }
        y
    }

    /// Update Eta matrix after each pivot to track the changes to the inverse.
    ///
    /// # Arguments
    /// - `pivot_row` - The row index where the pivot occurs in the basis matrix (position of the leaving variable)
    /// - `d` - The direction vector computed from `ftran`, which represents the new column entering the basis
    fn update_eta_matrix(&mut self, pivot_row: usize, d: &DVector<T>) {
        // Construct the new Eta column as u = d - e_{pivot_row}
        let mut eta_column = d.clone();
        eta_column[pivot_row] -= T::one(); // Subtract 1 from the pivot row

        // The column index in the basis matrix where the replacement occurs is pivot_row
        // Create the new Eta matrix with the computed column
        let eta = EtaMatrix::new(pivot_row, eta_column);

        // Append the new Eta matrix to the list
        self.eta_matrices.push(eta);
    }

    // Utility method to swap a variable between basic and non-basic.
    //
    // # Arguments
    // - `entering_index` - The index of the variable that is entering the basis
    // - `exiting_index` - The index of the variable that is exiting the basis
    fn swap_basis_indices(&mut self, entering_index: usize, exiting_index: usize) {
        let old_basic_index = self.basic_indices[exiting_index];
        self.basic_indices[exiting_index] = entering_index;
        self.non_basic_indices.retain(|&x| x != entering_index);
        self.non_basic_indices.push(old_basic_index);
    }

    /// Gets reduced costs for non-basic variables.
    ///
    /// # Returns
    /// The reduced costs for non-basic variables.
    fn get_reduced_costs(&self) -> DVector<T> {
        let y = self.btran(&self.get_cb().as_view());
        let non_basic_a = self.get_non_basis_matrix(); // columns of A for non-basic variables
        let c_n = self.get_cn();
        // For a minimization problem:
        // reduced_costs = c_N - (N^T * y)
        &c_n - &(non_basic_a.transpose() * y)
    }

    /// Gets the direction vector.
    ///
    /// # Returns
    /// The direction vector.
    fn get_direction_vector(&self, entering_index: usize) -> DVector<T> {
        let a_entering = self.constraint_matrix.column(entering_index);
        self.ftran(&a_entering)
    }
}
