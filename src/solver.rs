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

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, DVectorView, Dyn, Matrix, VecStorage};
use nalgebra_lapack::{LUScalar, LU};
use num_traits::{One, Signed, Zero};

use crate::problem::{Variable, VariableValue};

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
#[derive(Debug, Clone)]
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
        index_to_variable: HashMap<usize, Variable>,
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
    fn ftran(&self, a_entering: &DVectorView<T>) -> Option<DVector<T>> {
        let mut d = a_entering.clone_owned();

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
    fn btran(&self, c_b: &DVectorView<T>) -> Option<DVector<T>> {
        let mut y = c_b.clone_owned();

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
    fn entering_index(&self) -> Option<usize> {
        let y = self.btran(&self.get_cb().as_view())?;
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

    /// Finds the index of the variable that will leave the basis or None if no variable can leave.
    ///
    /// # Arguments
    /// - `entering` - The index of the variable that will enter the basis
    ///
    /// # Returns
    /// The index of the variable that will leave the basis
    /// or None if no variable can leave.
    pub fn leaving_index(&self, entering: usize) -> Option<usize> {
        let a_entering = self.constraint_matrix.column(entering);
        let direction_vector = self.ftran(&a_entering)?;

        // Calculate the minimum ratio.
        let mut min_ratio = None;
        let mut leaving_index = None;

        for (i, &direction) in direction_vector.iter().enumerate() {
            // Only consider positive entries in the direction vector
            if direction > T::zero() {
                // Safe to divide here because direction is positive
                // and thus cannot be zero.
                let ratio = self.rhs[i] / direction;

                if min_ratio.is_none() || ratio < min_ratio.unwrap() {
                    min_ratio = Some(ratio);
                    leaving_index = Some(i);
                }
            }
        }

        leaving_index
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

    /// Refactorizes the basis matrix by performing an LU decomposition
    /// and repopulating `eta_matrices` from the L and U factors.
    fn refactorize_basis(&mut self) {
        self.eta_matrices.clear();
        let lu_decomp = LU::new(self.get_basis_matrix());

        let l_matrix = lu_decomp.l();
        let u_matrix = lu_decomp.u();

        for (i, column) in l_matrix.column_iter().enumerate() {
            if l_matrix[(i, i)] != T::one() {
                let eta_column = column.clone_owned();
                let eta = EtaMatrix::new(i, eta_column);
                self.eta_matrices.push(eta);
            }
        }

        for (i, column) in u_matrix.column_iter().enumerate() {
            let eta_column = column.clone_owned();
            let eta = EtaMatrix::new(i, eta_column);
            self.eta_matrices.push(eta);
        }
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
    fn is_optimal(reduced_costs: &DVector<T>, epsilon: T) -> bool {
        // Ensure epsilon is negative
        let epsilon = -epsilon.abs();

        reduced_costs.iter().all(|&cost| cost >= -epsilon)
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
        let y = self.btran(&self.get_cb().as_view())?;
        Some(&self.objective_coefficients - &self.constraint_matrix * y)
    }

    /// Gets the direction vector.
    ///
    /// # Returns
    /// The direction vector.
    fn get_direction_vector(&self) -> Option<DVector<T>> {
        self.ftran(&self.get_cn().as_view())
    }
}
