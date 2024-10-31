// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::collections::HashMap;

use nalgebra::{iter::MatrixIter, DMatrix, Dyn, MatrixView, Scalar, VecStorage, U1};
use num_traits::Float;
use tabled::settings::Style;

/// A variable in the tableau.
///
/// A variable in the tableau is a variable that is used to represent the linear program in the
/// tableau form. It has an id, a name, and a type.
#[derive(Debug, Clone)]
pub struct TableauVariable {
    /// The id of the variable.
    id: usize,

    /// The name of the variable.
    name: String,
}

impl TableauVariable {
    /// Creates a new tableau variable.
    ///
    /// # Arguments
    /// - `id`: The id of the variable.
    /// - `name`: The name of the variable.
    ///
    /// # Returns
    /// A new tableau variable.
    pub fn new(id: usize, name: String) -> Self {
        Self { id, name }
    }

    /// Returns the id of the variable.
    ///
    /// # Returns
    /// The id of the variable.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the name of the variable.
    ///
    /// # Returns
    /// The name of the variable.
    pub fn name(&self) -> &str {
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

impl std::hash::Hash for TableauVariable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// A tableau.
///
/// A tableau is a matrix representation of a linear program. It is used to solve the linear program
/// using the simplex method.
///
/// # Type Parameters
/// - `T`: The type of the elements in the tableau.
pub struct Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    /// The matrix of the tableau.
    matrix: DMatrix<T>,

    /// The row variables of the tableau.
    row_variables: Vec<TableauVariable>,

    /// The column variables of the tableau.
    column_variables: Vec<TableauVariable>,

    /// The basic variables of the tableau.
    basic_variables: HashMap<TableauVariable, usize>,
}

impl<T> Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    /// Creates a new tableau.
    ///
    /// # Arguments
    /// - `matrix`: The matrix of the tableau.
    /// - `row_variables`: The row variables of the tableau.
    /// - `column_variables`: The column variables of the tableau.
    ///
    /// # Returns
    /// A new tableau.
    pub fn new(
        matrix: DMatrix<T>,
        row_variables: Vec<TableauVariable>,
        column_variables: Vec<TableauVariable>,
    ) -> Self {
        // The basic variables are the row variables initially.
        let basic_variables = row_variables
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        Self {
            matrix,
            row_variables,
            column_variables,
            basic_variables,
        }
    }

    /// Creates a new tableau from a slice of data.
    ///
    /// # Arguments
    /// - `rows`: The number of rows of the tableau.
    /// - `columns`: The number of columns of the tableau.
    /// - `data`: The data of the tableau.
    /// - `row_variables`: The row variables of the tableau.
    /// - `column_variables`: The column variables of the tableau.
    ///
    /// # Returns
    /// A new tableau.
    pub fn from_row_slice(
        rows: usize,
        columns: usize,
        data: &Vec<T>,
        row_variables: Vec<TableauVariable>,
        column_variables: Vec<TableauVariable>,
    ) -> Self {
        let matrix = DMatrix::from_row_slice(rows, columns, &data);
        Self::new(matrix, row_variables, column_variables)
    }

    /// Returns the number of rows of the tableau.
    ///
    /// # Returns
    /// The number of rows of the tableau.
    pub fn rows(&self) -> usize {
        self.matrix.nrows()
    }

    /// Returns the number of columns of the tableau.
    ///
    /// # Returns
    /// The number of columns of the tableau.
    pub fn columns(&self) -> usize {
        self.matrix.ncols()
    }

    /// Returns the matrix of the tableau.
    ///
    /// # Returns
    /// The matrix of the tableau.
    pub fn matrix(&self) -> &DMatrix<T> {
        &self.matrix
    }

    /// Returns a mutable reference to the matrix of the tableau.
    ///
    /// # Returns
    /// A mutable reference to the matrix of the tableau.
    pub fn matrix_mut(&mut self) -> &mut DMatrix<T> {
        &mut self.matrix
    }

    /// Returns the row variables of the tableau.
    ///
    /// # Returns
    /// The row variables of the tableau.
    pub fn row_variables(&self) -> &Vec<TableauVariable> {
        &self.row_variables
    }

    /// Returns a mutable reference to the row variables of the tableau.
    ///
    /// # Returns
    /// A mutable reference to the row variables of the tableau.
    pub fn row_variables_mut(&mut self) -> &mut Vec<TableauVariable> {
        &mut self.row_variables
    }

    /// Returns the column variables of the tableau.
    ///
    /// # Returns
    /// The column variables of the tableau.
    pub fn column_variables(&self) -> &Vec<TableauVariable> {
        &self.column_variables
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
            .view((0, self.columns() - 1), (self.rows() - 1, 1))
    }

    /// Returns the right-hand side values of the tableau divided by the values of a column.
    /// If the value of a column is zero, the quotient is `None`.
    ///
    /// # Arguments
    /// - `column`: The index of the column.
    ///
    /// # Returns
    /// The right-hand side values of the tableau divided by the values of a column.
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
            .view((self.rows() - 1, 0), (1, self.columns() - 1))
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

    /// Returns the basic variables of the tableau.
    ///
    /// # Returns
    /// The basic variables of the tableau.
    pub fn basic_variables(&self) -> &HashMap<TableauVariable, usize> {
        &self.basic_variables
    }

    /// Sets a basic variable of a row.
    ///
    /// # Arguments
    /// - `row`: The row of the basic variable.
    /// - `variable`: The basic variable.
    pub fn basic_variable(&mut self, row: usize, variable: TableauVariable) {
        // Remove the old basic variable for this row from `basic_variables`.
        if let Some(old_variable) = self.row_variables.get(row) {
            self.basic_variables.remove(old_variable);
        }

        // Update `row_variables` and insert the new variable into `basic_variables`.
        self.row_variables[row] = variable.clone();
        self.basic_variables.insert(variable, row);
    }

    /// Checks if a variable is a basic variable.
    ///
    /// # Arguments
    /// - `variable`: The variable to check.
    ///
    /// # Returns
    /// `true` if the variable is a basic variable, `false` otherwise.
    pub fn is_basic_variable(&self, variable: &TableauVariable) -> bool {
        self.basic_variables.contains_key(variable)
    }

    /// Returns the value of a variable in the tableau.
    ///
    /// If the variable is a basic variable, the value is the right-hand side value of the row.
    /// Otherwise, the value is zero.
    ///
    /// # Arguments
    /// - `variable`: The variable to get the value of.
    ///
    /// # Returns
    /// The value of the variable in the tableau.
    pub fn variable_value(&self, variable: &TableauVariable) -> T {
        if let Some(&row) = self.basic_variables.get(variable) {
            self.matrix[(row, self.columns() - 1)]
        } else {
            T::zero()
        }
    }
}

impl<'a, T> IntoIterator for &'a Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    type Item = &'a T;
    type IntoIter = MatrixIter<'a, T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>>;

    fn into_iter(self) -> Self::IntoIter {
        self.matrix.iter()
    }
}

impl<T> std::fmt::Display for Tableau<T>
where
    T: Scalar + Float + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_rows = self.rows() + 1;
        let num_columns = self.columns() + 1;
        let mut builder = tabled::builder::Builder::with_capacity(num_rows, num_columns);

        // Push the header names.
        let column_names = std::iter::once("".to_string()).chain(
            self.column_variables()
                .iter()
                .map(|variable| variable.to_string()),
        );
        builder.push_record(column_names);

        // Push the rows.
        for row in 0..self.rows() {
            let row_name = self.row_variables()[row].to_string();
            let row = self.matrix.row(row);
            let record =
                std::iter::once(row_name).chain(row.iter().map(|&value| value.to_string()));
            builder.push_record(record);
        }

        let mut table = builder.index().column(0).build();
        table.with(Style::modern_rounded());

        write!(f, "{}", table)
    }
}
