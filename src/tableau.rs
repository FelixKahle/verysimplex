// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::{collections::HashMap, hash::Hash, rc::Rc};

use nalgebra::{DMatrix, DVector, Dyn, Matrix, VecStorage};
use nalgebra_lapack::{LUScalar, LU};
use num_traits::Float;

/// A named variable in the tableau.
///
/// Each variable has a unique identifier (`id`) to distinguish between
/// variables with the same `name`, which can occur when user-defined variables
/// share names with automatically generated slack variables.
#[derive(Clone, Debug)]
pub struct TableauVariable {
    /// Unique identifier for the variable.
    /// This is used to differentiate between variables with the same name.
    id: usize,

    /// Name of the variable.
    /// Using an `Rc` allows shared ownership of the name between multiple instances.
    name: Rc<String>,
}

impl TableauVariable {
    /// Constructs a new `TableauVariable`.
    ///
    /// # Parameters
    /// - `id`: Unique identifier for the variable.
    /// - `name`: Name of the variable.
    ///
    /// # Returns
    /// A new instance of `TableauVariable`.
    pub fn new(id: usize, name: Rc<String>) -> Self {
        Self { id, name }
    }

    /// Retrieves the unique identifier of the variable.
    ///
    /// # Returns
    /// The unique identifier (`id`) of the variable.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Retrieves the name of the variable.
    ///
    /// # Returns
    /// The `Rc` wrapped name of the variable.
    pub fn name(&self) -> &Rc<String> {
        &self.name
    }
}

impl std::fmt::Display for TableauVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl PartialEq for TableauVariable {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TableauVariable {}

impl Hash for TableauVariable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Elementary transformation matrix.
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

#[derive(Debug, Clone)]
pub struct Tableau<T>
where
    T: LUScalar + std::fmt::Display,
{
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
    index_to_variable: HashMap<usize, TableauVariable>,

    /// List of Eta matrices for updating B inverse
    eta_matrices: Vec<EtaMatrix<T>>,

    /// LU decomposition of the basis matrix B (used for refactorization)
    basis_lu: LU<T, Dyn, Dyn>,
}

impl<T> Tableau<T>
where
    T: LUScalar
        + Float
        + std::iter::Sum
        + std::fmt::Display
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::ops::SubAssign,
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
    pub fn index_to_variable(&self) -> &HashMap<usize, TableauVariable> {
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
}
