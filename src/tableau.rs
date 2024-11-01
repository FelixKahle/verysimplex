// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::{hash::Hash, rc::Rc};

use nalgebra::{DMatrix, Scalar};
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
    pub fn name(&self) -> Rc<String> {
        self.name.clone()
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

/// A row in the simplex tableau, representing a linear constraint.
///
/// # Type Parameters
/// - `T`: The numeric type of the coefficients and constants in the tableau.
pub struct TableauConstraintRow<T> {
    /// The main variable associated with this row.
    variable: TableauVariable,

    /// The constant value on the right-hand side of the constraint equation.
    constant: T,

    /// Coefficients of the variables in this row.
    coefficients: Vec<T>,
}

impl<T> TableauConstraintRow<T> {
    /// Constructs a new `TableauConstraintRow`.
    ///
    /// # Parameters
    /// - `variable`: The main variable associated with this row.
    /// - `constant`: The constant value on the right-hand side.
    /// - `coefficients`: The coefficients for each variable in this row.
    ///
    /// # Returns
    /// A new `TableauConstraintRow` instance.
    pub fn new(variable: TableauVariable, constant: T, coefficients: Vec<T>) -> Self {
        Self {
            variable,
            constant,
            coefficients,
        }
    }

    /// Retrieves the main variable of the row.
    ///
    /// # Returns
    /// A reference to the `TableauVariable` associated with this row.
    pub fn variable(&self) -> &TableauVariable {
        &self.variable
    }

    /// Retrieves the constant of the row.
    ///
    /// # Returns
    /// A reference to the constant value of the row.
    pub fn constant(&self) -> &T {
        &self.constant
    }

    /// Retrieves the coefficients of the row.
    ///
    /// # Returns
    /// A reference to the vector of coefficients in the row.
    pub fn coefficients(&self) -> &Vec<T> {
        &self.coefficients
    }
}

/// Converts a vector of coefficients to a formatted string representation.
///
/// # Parameters
/// - `vector`: A vector of coefficients.
///
/// # Returns
/// A formatted string of coefficients joined by `+`.
#[inline]
fn coefficients_vector_to_string<T>(vector: &Vec<T>) -> String
where
    T: ToString,
{
    vector
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join(" + ")
}

impl<'a, T> IntoIterator for &'a TableauConstraintRow<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.coefficients.iter()
    }
}

impl<T> std::fmt::Display for TableauConstraintRow<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coefficients_string = coefficients_vector_to_string(&self.coefficients);
        write!(f, "{} = {}", coefficients_string, self.constant)
    }
}

/// A row in the simplex tableau representing the objective function.
///
/// This row holds the objective value and the coefficients of the variables
/// involved in the objective function.
///
/// # Type Parameters
/// - `T`: The numeric type of the coefficients and the objective value.
#[derive(Debug, Clone)]
pub struct TableauObjectiveRow<T> {
    /// The objective value, representing the right-hand side constant in the objective row.
    objective_value: T,

    /// The coefficients of each variable in the objective function.
    coefficients: Vec<T>,
}

impl<T> TableauObjectiveRow<T> {
    /// Constructs a new `TableauObjectiveRow`.
    ///
    /// # Parameters
    /// - `objective_value`: The constant objective value.
    /// - `coefficients`: The coefficients of each variable in the objective function.
    ///
    /// # Returns
    /// A new `TableauObjectiveRow` instance representing the objective function row in the tableau.
    pub fn new(objective_value: T, coefficients: Vec<T>) -> Self {
        Self {
            objective_value,
            coefficients,
        }
    }

    /// Retrieves the objective value of the row.
    ///
    /// # Returns
    /// A reference to the objective value of the row.
    pub fn objective_value(&self) -> &T {
        &self.objective_value
    }

    /// Retrieves the coefficients of the variables in the row.
    ///
    /// # Returns
    /// A reference to the vector of coefficients for the variables in the objective function.
    pub fn coefficients(&self) -> &Vec<T> {
        &self.coefficients
    }
}

impl<'a, T> IntoIterator for &'a TableauObjectiveRow<T> {
    type Item = &'a T;
    type IntoIter = std::iter::Chain<std::slice::Iter<'a, T>, std::iter::Once<&'a T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.coefficients
            .iter()
            .chain(std::iter::once(&self.objective_value))
    }
}

impl<T> std::fmt::Display for TableauObjectiveRow<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coefficients_string = coefficients_vector_to_string(&self.coefficients);
        write!(f, "{} = {}", coefficients_string, self.objective_value)
    }
}

/// A tableau for performing the simplex algorithm in linear programming.
///
/// # Type Parameters
/// - `T`: The numeric type of the coefficients and constants in the tableau, which
/// must support `Scalar`, `Float`, and `Display` traits.
#[derive(Debug, Clone)]
pub struct Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    /// The matrix used for the simplex algorithm computations.
    matrix: DMatrix<T>,

    /// A list of variables used in the tableau, excluding the objective and RHS.
    variables: Vec<TableauVariable>,

    /// The row names for the tableau, represented as `TableauVariable` instances.
    rows: Vec<TableauVariable>,
}

impl<T> Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    /// Constructs a new `Tableau` with a matrix representation for the simplex algorithm.
    ///
    /// # Parameters
    /// - `variables`: The list of `TableauVariable` instances representing the variables used in the tableau.
    /// - `constraint_rows`: The list of `TableauConstraintRow` instances representing the constraints.
    /// - `objective_row`: The `TableauObjectiveRow` instance representing the objective function.
    ///
    /// # Returns
    /// A new `Tableau` instance initialized with a matrix that includes all coefficients, constants, and the objective row.
    ///
    /// # Panics
    /// Panics if the number of variables is less than the number of constraints.
    pub fn new(
        variables: Vec<TableauVariable>,
        constraint_rows: Vec<TableauConstraintRow<T>>,
        objective_row: TableauObjectiveRow<T>,
    ) -> Self {
        // Ensure that the number of variables is equal to or greater than the number of constraints.
        // This is because the row variables need to be included in the variable list.
        if variables.len() <= constraint_rows.len() {
            panic!("The number of variables must be equal to or greater than the number of constraints.");
        }

        // Determine matrix dimensions.
        // Add one to the number of rows to account for the objective row.
        // Add one to the number of columns to account for the RHS column.
        let num_rows = constraint_rows.len() + 1;
        let num_columns = variables.len() + 1;
        let mut matrix = DMatrix::<T>::zeros(num_rows, num_columns);

        for (i, row) in constraint_rows.iter().enumerate() {
            for (j, &coef) in row.coefficients().iter().enumerate() {
                matrix[(i, j)] = coef;
            }
            matrix[(i, num_columns - 1)] = *row.constant();
        }

        for (j, &coef) in objective_row.coefficients().iter().enumerate() {
            matrix[(num_rows - 1, j)] = coef;
        }
        matrix[(num_rows - 1, num_columns - 1)] = *objective_row.objective_value();

        let rows = constraint_rows
            .iter()
            .map(|row| row.variable().clone())
            .collect();

        Self {
            matrix,
            variables,
            rows,
        }
    }
}
