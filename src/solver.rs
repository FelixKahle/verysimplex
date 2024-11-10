// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, Dyn, Matrix, VecStorage};
use nalgebra_lapack::{LUScalar, LU};
use num_traits::{Float, One, Zero};

use crate::problem::Variable;

/// Elementary transformation matrix.
/// Used to update the basis matrix way more efficiently than inverting it.
///
/// # Type parameters
/// - `T`: The type of the elements of the matrix.
#[derive(Clone, Debug)]
struct EtaMatrix<T>
where
    T: LUScalar,
{
    /// The index of the column that the matrix will be applied to.
    column_index: usize,

    /// The eta column.
    eta_column: DVector<T>,
}

impl<T> EtaMatrix<T>
where
    T: LUScalar,
{
    /// Constructs a new `EtaMatrix`.
    ///
    /// # Parameters
    /// - `column_index`: The index of the column that the matrix will be applied to.
    /// - `eta_column`: The eta column.
    ///
    /// # Returns
    /// A new instance of `EtaMatrix`.
    pub fn new(column_index: usize, eta_column: DVector<T>) -> Self {
        Self {
            column_index,
            eta_column,
        }
    }

    /// Gets the index of the column that the matrix will be applied to.
    ///
    /// # Returns
    /// The index of the column that the matrix will be applied to.
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Gets the eta column.
    ///
    /// # Returns
    /// The eta column.
    pub fn eta_column(&self) -> &DVector<T> {
        &self.eta_column
    }
}

impl<T> std::fmt::Display for EtaMatrix<T>
where
    T: LUScalar + Float + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}|{}]", self.column_index, self.eta_column)
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
    index_to_variable: HashMap<usize, Variable>,

    /// List of Eta matrices for updating B inverse
    eta_matrices: Vec<EtaMatrix<T>>,

    /// LU decomposition of the basis matrix B (used for refactorization)
    basis_lu: LU<T, Dyn, Dyn>,
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
        + PartialOrd
        + nalgebra::ClosedSubAssign,
{
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
    pub fn index_to_variable(&self) -> &HashMap<usize, Variable> {
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
    fn ftran(&self, a_entering: &DVector<T>) -> Option<DVector<T>> {
        // Solve B * d = a_entering.
        // This is a expensive operation, but this implementation is based on the
        // LAPACK routines which are highly optimized and may also use the GPU
        // for acceleration.
        let mut d = self.basis_lu.solve(a_entering)?;

        for eta in &self.eta_matrices {
            let idx = eta.column_index;
            let multiplier = d[idx];
            d -= &eta.eta_column * multiplier;
            d[idx] = multiplier * eta.eta_column[idx];
        }
        Some(d)
    }

    /// Backward transformation (BTRAN): Solves `y^T = c_B^T * B^{-1}`
    ///
    /// # Arguments
    /// - `c_b` - The objective coefficients for basic variables
    ///
    /// # Returns
    /// The solution vector `y`
    fn btran(&self, c_b: &DVector<T>) -> Option<DVector<T>> {
        // Solve B^T * y = c_B.
        // This is a expensive operation, but this implementation is based on the
        // LAPACK routines which are highly optimized and may also use the GPU
        // for acceleration.
        let mut y = self.basis_lu.solve_transpose(c_b)?;

        for eta in self.eta_matrices.iter().rev() {
            let idx = eta.column_index;
            let multiplier = y[idx];
            y -= &eta.eta_column * multiplier;
            y[idx] = multiplier * eta.eta_column[idx];
        }
        Some(y)
    }

    /// Finds the index of the variable that will enter the basis or None if no variable can enter.
    /// If this method returns None, the current solution is optimal.
    ///
    /// # Returns
    /// The index of the variable that will enter the basis
    /// or None if no variable can enter.
    fn enter(&self) -> Option<usize> {
        let y = self.btran(&self.get_cb())?;
        let matrix_y_product = &self.constraint_matrix * y;
        let reduced_costs = &self.objective_coefficients - matrix_y_product;
        let entering_index = reduced_costs
            .iter()
            .enumerate()
            .filter(|&(_, &rc)| rc > self.epsilon)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index);
        entering_index
    }

    /// Update Eta matrix after each pivot to track the changes to the inverse.
    ///
    /// # Arguments
    /// - `exiting_index` - The index of the variable that is exiting the basis
    fn update_eta_matrix(&mut self, exiting_index: usize) {
        // Retrieve the updated pivot column from the constraint matrix
        let pivot_column_index = self.basic_indices[exiting_index];
        let eta_column = self
            .constraint_matrix
            .column(pivot_column_index)
            .into_owned();

        let eta = EtaMatrix::new(exiting_index, eta_column);
        self.eta_matrices.push(eta);
    }
}
