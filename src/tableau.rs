// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::{collections::HashMap, hash::Hash, ops::AddAssign, rc::Rc};

use nalgebra::{DMatrix, DVector, Scalar};
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

#[derive(Debug, Clone)]
pub struct Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
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
}

impl<T> Tableau<T>
where
    T: LUScalar + Float + std::iter::Sum + std::fmt::Display,
{
    /// Constructs a new `Tableau`.
    ///
    /// # Parameters
    /// - `constraint_matrix`: The constraint matrix (A).
    /// - `rhs`: The right-hand side values (b).
    /// - `objective_coefficients`: The objective coefficients (c).
    /// - `basic_indices`: The indices of the basic variables.
    /// - `non_basic_indices`: The indices of the non-basic variables.
    /// - `index_to_variable`: A mapping from indices to variables.
    ///
    /// # Returns
    /// A new instance of `Tableau`.
    pub fn new(
        constraint_matrix: DMatrix<T>,
        rhs: DVector<T>,
        objective_coefficients: DVector<T>,
        basic_indices: Vec<usize>,
        non_basic_indices: Vec<usize>,
        index_to_variable: HashMap<usize, TableauVariable>,
    ) -> Self {
        Self {
            constraint_matrix,
            rhs,
            objective_coefficients,
            basic_indices,
            non_basic_indices,
            objective_value: T::zero(),
            index_to_variable,
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
    pub fn index_to_variable(&self) -> &HashMap<usize, TableauVariable> {
        &self.index_to_variable
    }

    pub fn perform_basis_swap(&mut self, entering_index: usize, leaving_index: usize) {
        Self::swap_basis_indices(self, entering_index, leaving_index);
        Self::update_rhs_with_lu(self);
        Self::update_objective_value(self);
    }

    /// Swaps the basis indices.
    ///
    /// # Parameters
    /// - `tableau`: The tableau to modify.
    /// - `entering_index`: The index of the entering variable.
    fn swap_basis_indices(tableau: &mut Tableau<T>, entering_index: usize, leaving_index: usize) {
        let leaving_var = tableau.basic_indices[leaving_index];
        tableau.basic_indices[leaving_index] = entering_index;
        tableau.non_basic_indices.retain(|&i| i != entering_index);
        tableau.non_basic_indices.push(leaving_var);
    }

    fn update_rhs_with_lu(tableau: &mut Tableau<T>) {
        let basis_matrix = tableau
            .constraint_matrix()
            .select_columns(&tableau.basic_indices);

        // Perform LU decomposition on the basis matrix.
        // This is a expensive operation, but this call uses the LAPACK backend
        // which is extremely optimized and efficient, resulting in a very fast
        // LU decomposition.
        let lu = LU::new(basis_matrix);

        // Solve for the new rhs values by solving B * rhs = b
        // `rhs` represents the solution for the basic variables
        lu.solve_mut(&mut tableau.rhs);
    }

    /// Updates the objective value.
    ///
    /// # Parameters
    /// - `tableau`: The tableau to update.
    fn update_objective_value(tableau: &mut Tableau<T>) {
        tableau.objective_value = tableau
            .basic_indices
            .iter()
            .zip(&tableau.rhs)
            .map(|(&basic_index, &rhs_value)| {
                tableau.objective_coefficients()[basic_index] * rhs_value
            })
            .sum();
    }
}
