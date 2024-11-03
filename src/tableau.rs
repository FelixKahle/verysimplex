// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::{hash::Hash, rc::Rc};

use nalgebra::{max, DMatrix, Dyn, MatrixView, Scalar, U1};
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

/// A row in the simplex tableau, representing a linear constraint.
///
/// # Type Parameters
/// - `T`: The numeric type of the coefficients and constants in the tableau.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

/// An error indicating the number of variables is incorrect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidVariableCountError;

impl std::fmt::Display for InvalidVariableCountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The number of variables is invalid for the number of constraints."
        )
    }
}

impl std::error::Error for InvalidVariableCountError {}

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
    /// A new `Tableau` instance with the matrix representation for the simplex algorithm.
    pub fn new(
        variables: Vec<TableauVariable>,
        constraint_rows: Vec<TableauConstraintRow<T>>,
        objective_row: TableauObjectiveRow<T>,
    ) -> Result<Self, InvalidVariableCountError> {
        let constraint_len = constraint_rows.len();
        let variable_len = variables.len();

        // Ensure that the number of variables is greater than the number of constraints.
        // This will not ensure complete correctness, but it is a necessary condition.
        // If we have for example two constraints, we need at least two variables.
        if variable_len <= constraint_len {
            return Err(InvalidVariableCountError);
        }

        // Determine matrix dimensions: constraint rows + 1 (for objective row), and variables + 1 (for RHS).
        let num_rows = constraint_rows.len() + 1;
        let num_columns = variables.len() + 1;

        // Create a vector for column-major data with the correct capacity.
        let mut data: Vec<T> = vec![T::zero(); num_rows * num_columns];

        // Populate constraint row data in column-major order.
        for (i, row) in constraint_rows.iter().enumerate() {
            for (j, &coef) in row.coefficients().iter().enumerate() {
                data[j * num_rows + i] = coef; // Populate each column by iterating over rows
            }
            // Correctly populate the RHS constant in the last column for this row.
            data[(num_columns - 1) * num_rows + i] = *row.constant();
        }

        // Populate objective row data in column-major order.
        for (j, &coef) in objective_row.coefficients().iter().enumerate() {
            data[j * num_rows + num_rows - 1] = coef;
        }
        // RHS objective value in the last row of the last column.
        data[(num_columns - 1) * num_rows + num_rows - 1] = *objective_row.objective_value();

        // Create the matrix with data in column-major order.
        let matrix = DMatrix::from_vec(num_rows, num_columns, data);

        // Collect row variables (basic variables) for the constraints.
        let rows = constraint_rows
            .iter()
            .map(|row| row.variable().clone())
            .collect();

        Ok(Self {
            matrix,
            variables,
            rows,
        })
    }

    /// Returns the number of rows of the tableau.
    ///
    /// # Returns
    /// The number of rows of the tableau.
    pub fn num_rows(&self) -> usize {
        self.matrix.nrows()
    }

    /// Returns the number of columns of the tableau.
    ///
    /// # Returns
    /// The number of columns of the tableau.
    pub fn num_columns(&self) -> usize {
        self.matrix.ncols()
    }

    /// Retrieves the matrix representation of the tableau.
    ///
    /// # Returns
    /// A reference to the matrix representation of the tableau.
    pub fn matrix(&self) -> &DMatrix<T> {
        &self.matrix
    }

    /// Retrieves the list of variables used in the tableau.
    ///
    /// # Returns
    /// A reference to the list of variables used in the tableau.
    pub fn variables(&self) -> &Vec<TableauVariable> {
        &self.variables
    }

    /// Retrieves the list of row variables used in the tableau.
    ///
    /// # Returns
    /// A reference to the list of row variables used in the tableau.
    pub fn rows(&self) -> &Vec<TableauVariable> {
        &self.rows
    }

    /// Returns a view to the right-hand side vector of the tableau.
    ///
    /// This vector contains the right-hand side values of the constraints in the tableau,
    /// without the objective value.
    ///
    /// # Returns
    /// A view to the right-hand side vector of the tableau.
    pub fn rhs_vector(&self) -> MatrixView<T, Dyn, Dyn, U1, Dyn> {
        self.matrix
            .view((0, self.num_columns() - 1), (self.num_rows() - 1, 1))
    }

    /// Returns a view to the objective row of the tableau.
    ///
    /// # Arguments
    /// - `column`: The column index of thr row we take the divisor from.
    ///
    /// # Returns
    /// A view to the objective row of the tableau.
    pub fn rhs_quotients(&self, column: usize) -> Vec<Option<T>> {
        self.rhs_vector()
            .iter()
            .zip(self.matrix.column(column).iter())
            .map(|(&rhs, &value)| {
                if value.is_zero() {
                    None
                } else {
                    Some(rhs / value)
                }
            })
            .collect()
    }

    /// Returns a view to the objective coefficients vector of the tableau.
    ///
    /// This vector contains the coefficients of the objective function in the tableau,
    /// without the objective value.
    ///
    /// # Returns
    /// A view to the objective coefficients vector of the tableau.
    pub fn objective_coefficients_vector(&self) -> MatrixView<T, Dyn, Dyn, U1, Dyn> {
        self.matrix
            .view((self.num_rows() - 1, 0), (1, self.num_columns() - 1))
    }

    /// Checks if the current tableau is optimal.
    ///
    /// # Returns
    /// `true` if the tableau is optimal, `false` otherwise.
    pub fn is_optimal(&self) -> bool {
        self.objective_coefficients_vector()
            .iter()
            .all(|&coeff| coeff <= T::zero())
    }

    /// Checks if the tableau is feasible.
    ///
    /// # Returns
    /// `true` if all RHS values are non-negative, `false` otherwise.
    pub fn is_feasible(&self) -> bool {
        self.rhs_vector().iter().all(|&rhs| rhs >= T::zero())
    }
}

impl<T> std::fmt::Display for Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.matrix())
    }
}

/// An error that can occur when building a tableau.
#[derive(Debug, Clone)]
pub enum TableauBuilderError {
    /// The objective row is missing.
    MissingObjectiveRow,

    /// The number of variables is incorrect.
    InvalidVariableCountError,
}

impl std::fmt::Display for TableauBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingObjectiveRow => write!(f, "Missing objective row"),
            Self::InvalidVariableCountError => write!(
                f,
                "The number of variables is invalid for the number of constraints."
            ),
        }
    }
}

impl From<InvalidVariableCountError> for TableauBuilderError {
    fn from(_: InvalidVariableCountError) -> Self {
        Self::InvalidVariableCountError
    }
}

pub struct TableauBuilder<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    variables: Vec<TableauVariable>,
    constraint_rows: Vec<TableauConstraintRow<T>>,
    objective_row: Option<TableauObjectiveRow<T>>,
}

impl<T> TableauBuilder<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    /// Creates a new tableau builder.
    ///
    /// # Returns
    /// A new tableau builder.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            constraint_rows: Vec::new(),
            objective_row: None,
        }
    }

    /// Creates a new tableau builder with a given capacity.
    ///
    /// # Arguments
    /// - `num_variables`: The number of variables to pre-allocate space for.
    /// - `num_constraints`: The number of constraints to pre-allocate space for.
    ///
    /// # Returns
    /// A new tableau builder with the given capacity.
    pub fn with_capacity(num_variables: usize, num_constraints: usize) -> Self {
        Self {
            variables: Vec::with_capacity(max(0, num_variables)),
            constraint_rows: Vec::with_capacity(max(0, num_constraints)),
            objective_row: None,
        }
    }

    /// Adds a variable to the tableau.
    ///
    /// # Arguments
    /// - `variable`: The variable to add to the tableau.
    ///
    /// # Returns
    /// A mutable reference to the tableau builder.
    pub fn add_variable(&mut self, variable: TableauVariable) -> &mut Self {
        self.variables.push(variable);
        self
    }

    /// Adds a list of variables to the tableau.
    ///
    /// # Arguments
    /// - `variables`: The list of variables to add to the tableau.
    ///
    /// # Returns
    /// A mutable reference to the tableau builder.
    pub fn add_constraint_row(&mut self, row: TableauConstraintRow<T>) -> &mut Self {
        self.constraint_rows.push(row);
        self
    }

    /// Removes a constraint row from the tableau.
    ///
    /// # Arguments
    /// - `row`: The constraint row to remove from the tableau.
    ///
    /// # Returns
    /// A mutable reference to the tableau builder.
    pub fn remove_constraint_row(&mut self, row: &TableauConstraintRow<T>) -> &mut Self {
        let index = self.constraint_rows.iter().position(|r| r == row);
        if let Some(index) = index {
            self.constraint_rows.remove(index);
        }
        self
    }

    /// Adds a list of constraint rows to the tableau.
    ///
    /// # Arguments
    /// - `rows`: The list of constraint rows to add to the tableau.
    ///
    /// # Returns
    /// A mutable reference to the tableau builder.
    pub fn add_constraint_rows(&mut self, rows: Vec<TableauConstraintRow<T>>) -> &mut Self {
        self.constraint_rows.extend(rows);
        self
    }

    /// Removes all constraint rows from the tableau.
    ///
    /// # Returns
    /// A mutable reference to the tableau builder.
    pub fn clear_constraint_rows(&mut self) -> &mut Self {
        self.constraint_rows.clear();
        self
    }

    /// Sets the objective row of the tableau.
    ///
    /// # Arguments
    /// - `row`: The objective row of the tableau.
    ///
    /// # Returns
    /// A mutable reference to the tableau builder.
    pub fn set_objective_row(&mut self, row: TableauObjectiveRow<T>) -> &mut Self {
        self.objective_row = Some(row);
        self
    }

    /// Removes the objective row from the tableau.
    ///
    /// # Returns
    /// A mutable reference to the tableau builder.
    pub fn remove_objective_row(&mut self) -> &mut Self {
        self.objective_row = None;
        self
    }

    /// Builds the tableau.
    ///
    /// # Returns
    /// A new tableau.
    pub fn build(self) -> Result<Tableau<T>, TableauBuilderError> {
        let objective_row = self
            .objective_row
            .ok_or(TableauBuilderError::MissingObjectiveRow)?;
        let tableau = Tableau::new(self.variables, self.constraint_rows, objective_row)?;

        Ok(tableau)
    }
}
