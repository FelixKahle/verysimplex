// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::fmt::Display;

/// A trait for types that provide a vector of coefficients.
///
/// This trait is implemented by types that contain a vector of coefficients,
/// such as `LinearExpression`, and allows retrieval of both the coefficients
/// and their count.
///
/// # Type Parameters
/// - `T`: The type of the coefficients.
pub trait Coefficients<T> {
    /// Returns a vector of coefficients.
    ///
    /// # Returns
    /// A reference to a slice of coefficients of type `T`.
    fn coefficients(&self) -> &[T];

    /// Returns the number of coefficients in the vector.
    ///
    /// # Returns
    /// The count of coefficients in the vector as `usize`.
    fn coefficients_len(&self) -> usize {
        self.coefficients().len()
    }
}

/// A trait for types that provide a constant value.
///
/// This trait is implemented by types that contain a constant value,
/// such as `LinearExpression`, and allows retrieval of that constant.
///
/// # Type Parameters
/// - `T`: The type of the constant.
pub trait Constant<T> {
    /// Returns the constant value.
    ///
    /// # Returns
    /// The constant value of type `T`.
    fn constant(&self) -> T;
}

/// Represents a linear expression in a linear program.
///
/// A `LinearExpression` is a collection of coefficients representing
/// a linear combination of variables. It does not store variable names
/// or constants, only the coefficients.
///
/// # Type Parameters
/// - `T`: The type of the coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearExpression<T> {
    /// Coefficients of the linear expression.
    coefficients: Vec<T>,
}

impl<T> LinearExpression<T> {
    /// Creates a new `LinearExpression` with the specified coefficients.
    ///
    /// # Arguments
    /// - `coefficients`: A vector of coefficients representing the linear combination of variables.
    ///
    /// # Returns
    /// A new `LinearExpression` instance.
    pub fn new(coefficients: Vec<T>) -> Self {
        LinearExpression { coefficients }
    }
}

/// Converts a vector of coefficients into a formatted string for display.
///
/// Joins each coefficient with " + " to produce a human-readable
/// representation of a linear combination.
///
/// # Type Parameters
/// - `T`: The type of the coefficients.
///
/// # Arguments
/// - `vector`: A vector of coefficients.
///
/// # Returns
/// A string representing the vector as a linear combination.
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

impl<T> Display for LinearExpression<T>
where
    T: ToString,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", coefficients_vector_to_string(&self.coefficients))
    }
}

impl<T> From<Vec<T>> for LinearExpression<T> {
    fn from(coefficients: Vec<T>) -> Self {
        LinearExpression::new(coefficients)
    }
}

impl<T> Coefficients<T> for LinearExpression<T> {
    fn coefficients(&self) -> &[T] {
        &self.coefficients
    }
}

/// Represents a constraint in a linear program.
///
/// Constraints define boundaries for solutions in linear programming, specifying relationships
/// between a `LinearExpression` and a constant value.
///
/// # Variants
/// - `Equal`: Specifies that the `LinearExpression` must be equal to a constant value.
/// - `LessOrEqual`: Specifies that the `LinearExpression` must be less than or equal to a constant value.
/// - `GreaterOrEqual`: Specifies that the `LinearExpression` must be greater than or equal to a constant value.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint<T> {
    /// Equality constraint: `LinearExpression = value`.
    Equal(LinearExpression<T>, T),

    /// Less-than-or-equal-to constraint: `LinearExpression <= value`.
    LessOrEqual(LinearExpression<T>, T),

    /// Greater-than-or-equal-to constraint: `LinearExpression >= value`.
    GreaterOrEqual(LinearExpression<T>, T),
}

impl<T> Display for Constraint<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Constraint::Equal(expr, value) => write!(f, "{} = {}", expr, value),
            Constraint::LessOrEqual(expr, value) => write!(f, "{} <= {}", expr, value),
            Constraint::GreaterOrEqual(expr, value) => write!(f, "{} >= {}", expr, value),
        }
    }
}

impl<T> Coefficients<T> for Constraint<T> {
    fn coefficients(&self) -> &[T] {
        match self {
            Constraint::Equal(expr, _) => expr.coefficients(),
            Constraint::LessOrEqual(expr, _) => expr.coefficients(),
            Constraint::GreaterOrEqual(expr, _) => expr.coefficients(),
        }
    }
}

impl<T> Constant<T> for Constraint<T>
where
    T: Copy,
{
    fn constant(&self) -> T {
        match self {
            Constraint::Equal(_, value) => *value,
            Constraint::LessOrEqual(_, value) => *value,
            Constraint::GreaterOrEqual(_, value) => *value,
        }
    }
}

/// Represents the objective function in a linear program.
///
/// An `Objective` specifies the goal of optimization, aiming to either maximize
/// or minimize a given `LinearExpression`.
///
/// # Type Parameters
/// - `T`: The type of the coefficients in the `LinearExpression`.
///
/// # Variants
/// - `Maximize`: Aims to find the maximum value of the `LinearExpression`.
/// - `Minimize`: Aims to find the minimum value of the `LinearExpression`.
#[derive(Debug, Clone, PartialEq)]
pub enum Objective<T> {
    /// Maximize the value of the `LinearExpression`.
    Maximize(LinearExpression<T>),

    /// Minimize the value of the `LinearExpression`.
    Minimize(LinearExpression<T>),
}

impl<T> Display for Objective<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Objective::Maximize(expr) => write!(f, "Maximize: {}", expr),
            Objective::Minimize(expr) => write!(f, "Minimize: {}", expr),
        }
    }
}

impl<T> Coefficients<T> for Objective<T> {
    fn coefficients(&self) -> &[T] {
        match self {
            Objective::Maximize(expr) => expr.coefficients(),
            Objective::Minimize(expr) => expr.coefficients(),
        }
    }
}

/// Enum representing errors that may occur in `LinearProgram`.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearProgramError {
    /// Error indicating that the number of coefficients in the linear program is inconsistent.
    InconsistentCoefficientLength,
}

impl Display for LinearProgramError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LinearProgramError::InconsistentCoefficientLength => {
                write!(
                    f,
                    "Inconsistent number of coefficients in the linear program"
                )
            }
        }
    }
}

/// Represents a linear programming problem.
///
/// A `LinearProgram` consists of an objective function and a set of constraints,
/// together defining an optimization problem.
#[derive(Debug, Clone)]
pub struct LinearProgram<T> {
    /// Objective function of the linear program.
    objective: Objective<T>,

    /// Constraints of the linear program.
    constraints: Vec<Constraint<T>>,
}

impl<T> Display for LinearProgram<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "{}", self.objective)?;
        writeln!(f, "Subject to:")?;
        for constraint in &self.constraints {
            writeln!(f, "{}", constraint)?;
        }
        Ok(())
    }
}

impl<T> LinearProgram<T> {
    /// Creates a new `LinearProgram` with the given objective and constraints.
    ///
    /// # Arguments
    /// - `objective`: The objective function of the linear program.
    /// - `constraints`: A vector of constraints for the linear program.
    ///
    /// # Returns
    /// A `Result` with the `LinearProgram` or a `LinearProgramError`
    /// if a mismatch in dimensions is detected.
    pub fn new(
        objective: Objective<T>,
        constraints: Vec<Constraint<T>>,
    ) -> Result<Self, LinearProgramError> {
        let objective_coefficients_len = objective.coefficients().len();

        for constraint in &constraints {
            if constraint.coefficients_len() != objective_coefficients_len {
                return Err(LinearProgramError::InconsistentCoefficientLength);
            }
        }

        Ok(LinearProgram {
            objective,
            constraints,
        })
    }

    /// Returns the number of variables in the linear program.
    ///
    /// # Returns
    /// The number of variables in the linear program.
    pub fn num_variables(&self) -> usize {
        self.objective.coefficients().len()
    }

    /// Returns the number of constraints in the linear program.
    ///
    /// # Returns
    /// The number of constraints in the linear program.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns the constraints of the linear program.
    ///
    /// # Returns
    /// A reference to the vector of constraints in the linear program.
    pub fn constraints(&self) -> &Vec<Constraint<T>> {
        &self.constraints
    }

    /// Returns the objective of the linear program.
    ///
    /// # Returns
    /// A reference to the objective function of the linear program.
    pub fn objective(&self) -> &Objective<T> {
        &self.objective
    }

    /// Returns a builder for creating a `LinearProgram`.
    ///
    /// # Returns
    /// A `LinearProgramBuilder` for creating a `LinearProgram`.
    pub fn builder() -> LinearProgramBuilder<T> {
        LinearProgramBuilder::new()
    }
}

/// Enum for errors that may occur when building a `LinearProgram`.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearProgramBuilderError {
    /// Error indicating that the number of coefficients in the linear program is inconsistent.
    InconsistentCoefficientLength,

    /// The objective function is missing from the builder.
    MissingObjective,
}

impl From<LinearProgramError> for LinearProgramBuilderError {
    fn from(error: LinearProgramError) -> Self {
        match error {
            LinearProgramError::InconsistentCoefficientLength => {
                LinearProgramBuilderError::InconsistentCoefficientLength
            }
        }
    }
}

impl Display for LinearProgramBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LinearProgramBuilderError::InconsistentCoefficientLength => {
                write!(
                    f,
                    "Inconsistent number of coefficients in the linear program"
                )
            }
            LinearProgramBuilderError::MissingObjective => {
                write!(f, "Objective function is missing")
            }
        }
    }
}

/// Builder for creating a `LinearProgram`.
///
/// The `LinearProgramBuilder` provides a readable and flexible way
/// to create a `LinearProgram` instead of using the constructor directly.
pub struct LinearProgramBuilder<T> {
    /// Optional objective function for the linear program.
    objective: Option<Objective<T>>,

    /// Constraints for the linear program.
    constraints: Vec<Constraint<T>>,
}

impl<T> LinearProgramBuilder<T> {
    /// Creates a new `LinearProgramBuilder`.
    ///
    /// # Returns
    /// A new `LinearProgramBuilder`.
    pub fn new() -> Self {
        LinearProgramBuilder {
            objective: None,
            constraints: Vec::new(),
        }
    }

    /// Sets the objective function for the linear program.
    ///
    /// # Arguments
    /// - `objective`: The objective function for the linear program.
    ///
    /// # Returns
    /// The `LinearProgramBuilder` with the objective function set.
    pub fn set_objective(mut self, objective: Objective<T>) -> Self {
        self.objective = Some(objective);
        self
    }

    /// Adds a constraint to the linear program.
    ///
    /// # Arguments
    /// - `constraint`: The constraint to add to the linear program.
    ///
    /// # Returns
    /// The `LinearProgramBuilder` with the constraint added.
    pub fn with_constraint(mut self, constraint: Constraint<T>) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Removes all constraints from the builder.
    ///
    /// # Returns
    /// The `LinearProgramBuilder` with all constraints removed.
    pub fn clear_constraints(mut self) -> Self {
        self.constraints.clear();
        self
    }

    /// Builds the `LinearProgram` from the builder.
    ///
    /// # Returns
    /// A `Result` containing the `LinearProgram` or an error if the objective is missing
    /// or dimensions mismatch.
    pub fn build(self) -> Result<LinearProgram<T>, LinearProgramBuilderError> {
        let objective = self
            .objective
            .ok_or(LinearProgramBuilderError::MissingObjective)?;
        let problem = LinearProgram::new(objective, self.constraints)?;

        Ok(problem)
    }
}

/// A wrapper for a linear programming problem to be solved.
///
/// The `Problem` struct contains a `LinearProgram` and the names of variables
/// as they appear in the objective function and constraints.
pub struct Problem<T> {
    /// The linear program.
    program: LinearProgram<T>,

    /// Names of variables in the program.
    /// In the same order as the variables in the program.
    variable_names: Vec<String>,
}

/// Errors that can occur when creating a `Problem`.
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemError {
    /// The number of variable names does not match the number of variables in the program.
    MismatchedVariableNamesLength,
}

impl Display for ProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ProblemError::MismatchedVariableNamesLength => {
                write!(f, "The number of variable names does not match the number of variables in the program.")
            }
        }
    }
}

impl<T> Problem<T> {
    /// Creates a new `Problem`.
    ///
    /// # Arguments
    /// - `program`: The linear program to solve.
    /// - `variable_names`: The names of the variables in the program.
    ///
    /// # Returns
    /// A new `Problem` instance if successful, or a `ProblemError` if the lengths mismatch.
    pub fn new(
        program: LinearProgram<T>,
        variable_names: Vec<String>,
    ) -> Result<Problem<T>, ProblemError> {
        if program.num_variables() != variable_names.len() {
            return Err(ProblemError::MismatchedVariableNamesLength);
        }

        Ok(Problem {
            program,
            variable_names,
        })
    }

    /// Returns the linear program.
    ///
    /// # Returns
    /// The linear program.
    pub fn program(&self) -> &LinearProgram<T> {
        &self.program
    }

    /// Returns the variable names in the program.
    ///
    /// # Returns
    /// The variable names.
    pub fn variable_names(&self) -> &Vec<String> {
        &self.variable_names
    }
}

impl<T> Display for Problem<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let objective = self.program.objective();
        let objective_terms: Vec<String> = objective
            .coefficients()
            .iter()
            .zip(&self.variable_names)
            .map(|(coef, var_name)| format!("{}{}", coef, var_name))
            .collect();
        let objective_str = match objective {
            Objective::Maximize(_) => format!("Maximize: {}", objective_terms.join(" + ")),
            Objective::Minimize(_) => format!("Minimize: {}", objective_terms.join(" + ")),
        };
        writeln!(f, "{}", objective_str)?;

        writeln!(f, "Subject to:")?;
        for constraint in self.program.constraints() {
            let constraint_terms: Vec<String> = constraint
                .coefficients()
                .iter()
                .zip(&self.variable_names)
                .map(|(coef, var_name)| format!("{}{}", coef, var_name))
                .collect();
            let constraint_str = match constraint {
                Constraint::Equal(_, value) => {
                    format!("{} = {}", constraint_terms.join(" + "), value)
                }
                Constraint::LessOrEqual(_, value) => {
                    format!("{} <= {}", constraint_terms.join(" + "), value)
                }
                Constraint::GreaterOrEqual(_, value) => {
                    format!("{} >= {}", constraint_terms.join(" + "), value)
                }
            };
            writeln!(f, "{}", constraint_str)?;
        }

        Ok(())
    }
}
