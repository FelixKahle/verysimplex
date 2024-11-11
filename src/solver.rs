// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, Dyn, Matrix, VecStorage};
use nalgebra_lapack::{LUScalar, LU};
use num_traits::{One, Signed, Zero};

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
    T: LUScalar + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}|{}]", self.column_index, self.eta_column)
    }
}

/// A variable value pair.
///
/// # Type parameters
/// - `T`: The type of the value.
#[derive(Clone, Debug)]
pub struct VariableValue<T> {
    /// The variable.
    variable: Variable,

    /// The value.
    value: T,
}

impl<T> VariableValue<T>
where
    T: Copy,
{
    /// Constructs a new `VariableValue`.
    ///
    /// # Parameters
    /// - `variable`: The variable.
    /// - `value`: The value.
    ///
    /// # Returns
    /// A new instance of `VariableValue`.
    pub fn new(variable: Variable, value: T) -> Self {
        Self { variable, value }
    }

    /// Gets the variable.
    ///
    /// # Returns
    /// The variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }

    /// Gets the value.
    ///
    /// # Returns
    /// The value.
    pub fn value(&self) -> T {
        self.value
    }
}

impl std::hash::Hash for VariableValue<f64> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.variable.hash(state);
    }
}

impl std::cmp::PartialEq for VariableValue<f64> {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable
    }
}

impl<T> std::fmt::Display for VariableValue<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}", self.variable, self.value)
    }
}

/// A solution to a linear program.
///
/// # Type parameters
/// - `T`: The number type.
#[derive(Debug, Clone)]
pub struct Solution<T>
where
    T: LUScalar,
{
    /// The objective value.
    objective_value: T,

    /// The variable values.
    variable_values: HashSet<VariableValue<T>>,
}

impl<T> Solution<T>
where
    T: LUScalar,
{
    /// Constructs a new `Solution`.
    ///
    /// # Parameters
    /// - `objective_value`: The objective value.
    /// - `variable_values`: The variable values.
    ///
    /// # Returns
    /// A new instance of `Solution`.
    pub fn new(objective_value: T, variable_values: HashSet<VariableValue<T>>) -> Self {
        Self {
            objective_value,
            variable_values,
        }
    }

    /// Gets the objective value.
    ///
    /// # Returns
    /// The objective value.
    pub fn objective_value(&self) -> T {
        self.objective_value
    }

    /// Gets the variable values.
    ///
    /// # Returns
    /// The variable values.
    pub fn variable_values(&self) -> &HashSet<VariableValue<T>> {
        &self.variable_values
    }
}

impl<'a, T> IntoIterator for &'a Solution<T>
where
    T: LUScalar,
{
    type Item = &'a VariableValue<T>;
    type IntoIter = std::collections::hash_set::Iter<'a, VariableValue<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.variable_values.iter()
    }
}

impl<T> std::fmt::Display for Solution<T>
where
    T: LUScalar + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Objective value: {}", self.objective_value)?;

        for variable_value in &self.variable_values {
            writeln!(f, "{}", variable_value)?;
        }

        Ok(())
    }
}

/// The solution status of a solver.
///
/// # Type parameters
/// - `T`: The number type.
#[derive(Debug)]
pub enum SolutionStatus<T>
where
    T: LUScalar,
{
    /// The solution is optimal.
    Optimal(Solution<T>),

    /// The solution is feasible.
    Feasible(Solution<T>),

    /// The solution is degenerate.
    Degenerate(Solution<T>),

    /// The solution is unbounded.
    Unbounded,
}

impl<T> std::fmt::Display for SolutionStatus<T>
where
    T: LUScalar + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolutionStatus::Optimal(solution) => {
                writeln!(f, "Optimal solution")?;
                write!(f, "{}", solution)
            }
            SolutionStatus::Feasible(solution) => {
                writeln!(f, "Feasible solution")?;
                write!(f, "{}", solution)
            }
            SolutionStatus::Degenerate(solution) => {
                writeln!(f, "Degenerate solution")?;
                write!(f, "{}", solution)
            }
            SolutionStatus::Unbounded => write!(f, "Unbounded solution"),
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
        + std::ops::Neg<Output = T>
        + PartialOrd
        + nalgebra::ClosedSubAssign
        + Signed,
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
        let mut d = a_entering.clone();

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
        let mut y = c_b.clone();

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
            .filter(|&(_, &rc)| rc > T::zero() + self.epsilon)
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

    /// Periodic refactorization step to reinitialize the basis matrix.
    /// This is very expensive and should be called when we reach a high level of
    /// numerical instability.
    fn refactorize_basis(&mut self) {
        self.basis_lu = LU::new(self.get_basis_matrix());
        self.eta_matrices.clear();
    }

    // Utility method to swap a variable between basic and non-basic.
    //
    // # Arguments
    // - `entering_index` - The index of the variable that is entering the basis
    // - `exiting_index` - The index of the variable that is exiting the basis
    fn swap_basis_indices(&mut self, entering_index: usize, exiting_index: usize) {
        self.basic_indices[exiting_index] = entering_index;
        self.non_basic_indices.retain(|&x| x != entering_index);
    }

    /// Checks if the solution has achieved optimality.
    ///
    /// # Arguments
    /// - `reduced_costs` - The reduced costs of the variables
    ///
    /// # Returns
    /// `true` if the solution is optimal, `false` otherwise.
    fn is_optimal(&self, reduced_costs: &DVector<T>) -> bool {
        reduced_costs.iter().all(|&cost| cost >= -self.epsilon)
    }

    /// Checks if the solution is unbounded.
    ///
    /// # Arguments
    /// - `direction_vector` - The direction vector
    fn is_unbounded(direction_vector: &DVector<T>) -> bool {
        direction_vector.iter().all(|&entry| entry <= T::zero())
    }

    /// Gets reduced costs for non-basic variables.
    ///
    /// # Returns
    /// The reduced costs for non-basic variables.
    fn get_reduced_costs(&self) -> Option<DVector<T>> {
        let y = self.btran(&self.get_cb())?;
        Some(&self.objective_coefficients - &self.constraint_matrix * y)
    }

    /// Gets the direction vector.
    ///
    /// # Returns
    /// The direction vector.
    fn get_direction_vector(&self) -> Option<DVector<T>> {
        self.ftran(&self.get_cn())
    }
}
