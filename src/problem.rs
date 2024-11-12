// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::{collections::BTreeSet, fmt::Display, rc::Rc};

/// A trait for types that provide a constant value.
///
/// # Type Parameters
/// - `T`: The type of the constant.
pub trait Constant<T> {
    /// Returns the constant value.
    ///
    /// # Returns
    /// The constant value of type `T`.
    fn constant(&self) -> &T;
}

/// A trait for types that provide n iterator over coefficients.
///
/// # Type Parameters
/// - `T`: The type of the coefficients.
pub trait Coefficients<'a, T>
where
    T: 'a,
{
    /// The associated iterator type for the coefficients.
    type CoefficientsIter: Iterator<Item = &'a T>;

    /// Returns an iterator over the coefficients.
    ///
    /// # Returns
    /// An iterator of coefficients of type `T`.
    fn coefficients(&'a self) -> Self::CoefficientsIter;

    /// Returns the number of coefficients.
    ///
    /// # Returns
    /// The count of coefficients as `usize`.
    fn coefficients_len(&'a self) -> usize;
}

/// A named variable.
///
/// Each variable has a unique identifier (`id`) to distinguish between
/// variables with the same `name`, which can occur when user-defined variables
/// share names with automatically generated slack variables.
#[derive(Clone, Debug)]
pub struct Variable {
    /// Unique identifier for the variable.
    /// This is used to differentiate between variables with the same name.
    id: usize,

    /// Name of the variable.
    /// Using an `Rc` allows shared ownership of the name between multiple instances.
    name: Rc<String>,
}

impl Variable {
    /// Constructs a new `Variable`.
    ///
    /// # Parameters
    /// - `id`: Unique identifier for the variable.
    /// - `name`: Name of the variable.
    ///
    /// # Returns
    /// A new instance of `Variable`.
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

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Variable {}

impl PartialOrd for Variable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Ord for Variable {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl std::hash::Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
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

impl<T> VariableValue<T> {
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
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T> PartialEq for VariableValue<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable && self.value == other.value
    }
}

impl<T> Eq for VariableValue<T> where T: Eq {}

impl<T> std::fmt::Display for VariableValue<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}", self.variable, self.value)
    }
}

impl<T> Constant<T> for VariableValue<T> {
    fn constant(&self) -> &T {
        &self.value
    }
}

//// A wrapper around `VariableValue<T>` that enforces uniqueness of each variable
/// in a `LinearExpression`. This wrapper is necessary to maintain the integrity
/// of mathematical operations within linear programming models and to ensure
/// that each variable only appears once in a given expression.
///
/// # Why `UniqueVariableValue<T>`?
///
/// In linear programming, expressions and constraints are typically represented
/// as linear combinations of variables, where each variable has a coefficient.
/// Each variable in these linear expressions should be unique within the scope
/// of a single expression, meaning that a variable should appear only once,
/// with a single coefficient value.
///
/// ## Challenges
///
/// Without this wrapper, it would be possible for a `Variable` to appear multiple
/// times in a `LinearExpression` with different `value` fields. This could lead
/// to logical inconsistencies or incorrect results during computations, as the
/// mathematical operations might implicitly assume a single occurrence per variable.
///
/// For example, without `UniqueVariableValue<T>`, adding two expressions might
/// inadvertently double-count variables that share the same name but have
/// different coefficient values. This would result in an incorrect linear
/// expression and potentially inaccurate optimization results.
///
/// ## Why Use a Wrapper Rather Than Just `VariableValue<T>`?
///
/// We could consider using `VariableValue<T>` directly within a collection,
/// such as `BTreeSet`, to enforce unique terms in a linear expression. However,
/// `VariableValue<T>` includes both the `Variable` (representing the variable's
/// identity) and the `value` (representing its coefficient), and directly
/// using it would mean the `BTreeSet` enforces uniqueness based on both the
/// variable and the coefficient value.
///
/// This is problematic because we want uniqueness to depend only on the
/// `Variable` identifier, not on its coefficient. Different coefficients for
/// the same variable should not lead to duplicate entries in the expression.
///
/// ## How `UniqueVariableValue<T>` Solves This Problem
///
/// By wrapping `VariableValue<T>` in `UniqueVariableValue<T>`, we can redefine
/// equality and ordering so that they depend only on the `Variable` (and not
/// on the coefficient `value`). In this way, `UniqueVariableValue<T>` ensures
/// that each `Variable` appears only once in a `LinearExpression`.
///
/// This design allows us to safely use `UniqueVariableValue<T>` in a `BTreeSet`
/// or other collections that require unique elements, ensuring that each
/// variable's identity is unique within a given expression without regard
/// to the value of its coefficient.
///
/// By ensuring uniqueness through `UniqueVariableValue<T>`, we make our linear
/// programming model both robust and mathematically sound, allowing accurate
/// representation and manipulation of linear expressions in optimization problems.
#[derive(Clone, Debug)]
pub struct UniqueVariableValue<T> {
    /// The variable value pair.
    variable_value: VariableValue<T>,
}

impl<T> UniqueVariableValue<T> {
    /// Creates a new `UniqueVariableValue`.
    ///
    /// # Arguments
    /// - `variable`: The variable.
    /// - `value`: The value.
    ///
    /// # Returns
    /// A new instance of `UniqueVariableValue`.
    pub fn new(variable: Variable, value: T) -> Self {
        UniqueVariableValue {
            variable_value: VariableValue::new(variable, value),
        }
    }
    /// Creates a new `UniqueVariableValue` from a `VariableValue`.
    ///
    /// # Arguments
    /// - `variable_value`: The `VariableValue` to wrap.
    pub fn from_variable_value(variable_value: VariableValue<T>) -> Self {
        UniqueVariableValue { variable_value }
    }

    /// Returns a reference to the inner `VariableValue`.
    ///
    /// # Returns
    /// A reference to the inner `VariableValue`.
    #[inline]
    pub fn inner(&self) -> &VariableValue<T> {
        &self.variable_value
    }

    /// Returns a reference to the variable.
    ///
    /// # Returns
    /// A reference to the variable.
    #[inline]
    pub fn variable(&self) -> &Variable {
        &self.variable_value.variable
    }

    /// Returns a reference to the value.
    ///
    /// # Returns
    /// A reference to the value.
    #[inline]
    pub fn value(&self) -> &T {
        &self.variable_value.value
    }
}

impl<T> PartialEq for UniqueVariableValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.variable() == other.variable()
    }
}

impl<T> Eq for UniqueVariableValue<T> {}

impl<T> PartialOrd for UniqueVariableValue<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.variable().partial_cmp(other.variable())
    }
}

impl<T> Ord for UniqueVariableValue<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.variable().cmp(other.variable())
    }
}

impl<T> std::hash::Hash for UniqueVariableValue<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.variable().hash(state);
    }
}

impl<T> std::fmt::Display for UniqueVariableValue<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}", self.variable(), self.value())
    }
}

impl<T> Constant<T> for UniqueVariableValue<T> {
    fn constant(&self) -> &T {
        self.value()
    }
}

impl<T> From<VariableValue<T>> for UniqueVariableValue<T> {
    #[inline]
    fn from(variable_value: VariableValue<T>) -> Self {
        UniqueVariableValue::from_variable_value(variable_value)
    }
}

impl<T> Into<VariableValue<T>> for UniqueVariableValue<T> {
    #[inline]
    fn into(self) -> VariableValue<T> {
        self.variable_value
    }
}

/// Represents a linear expression in a linear program.
///
/// A `LinearExpression` is a collection of coefficients representing
/// a linear combination of variables. It does not store variable names
/// or constants, only the coefficients.
///
/// # Type Parameters
/// - `T`: The type of the coefficients.
#[derive(Debug, Clone)]
pub struct LinearExpression<T> {
    /// Terms of the linear expression.
    terms: BTreeSet<UniqueVariableValue<T>>,
}

impl<'a, T> Coefficients<'a, T> for LinearExpression<T>
where
    T: 'a,
{
    type CoefficientsIter = std::iter::Map<
        std::collections::btree_set::Iter<'a, UniqueVariableValue<T>>,
        fn(&'a UniqueVariableValue<T>) -> &'a T,
    >;

    fn coefficients(&'a self) -> Self::CoefficientsIter {
        self.terms.iter().map(|term| &term.value())
    }

    fn coefficients_len(&'a self) -> usize {
        self.terms.len()
    }
}

impl<T> LinearExpression<T> {
    /// Creates a new `LinearExpression` with the specified terms.
    ///
    /// # Arguments
    /// - `terms`: A vector of `VariableValue`s.
    ///
    /// # Returns
    /// A new `LinearExpression` instance.
    pub fn new(terms: BTreeSet<UniqueVariableValue<T>>) -> Self {
        LinearExpression { terms }
    }

    /// Returns the `VariableValue`s of the linear expression.
    ///
    /// # Returns
    /// A reference to the vector of `VariableValue`s.
    pub fn terms(&self) -> &BTreeSet<UniqueVariableValue<T>> {
        &self.terms
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
#[inline(always)]
fn coefficients_vector_to_string<T>(vector: &BTreeSet<UniqueVariableValue<T>>) -> String
where
    T: std::fmt::Display,
{
    vector
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join(" + ")
}

impl<T> Display for LinearExpression<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", coefficients_vector_to_string(&self.terms))
    }
}

impl<T> From<BTreeSet<UniqueVariableValue<T>>> for LinearExpression<T> {
    fn from(terms: BTreeSet<UniqueVariableValue<T>>) -> Self {
        LinearExpression::new(terms)
    }
}

impl<T> From<Vec<UniqueVariableValue<T>>> for LinearExpression<T> {
    fn from(terms: Vec<UniqueVariableValue<T>>) -> Self {
        LinearExpression::new(terms.into_iter().collect())
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
#[derive(Debug, Clone)]
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

impl<T> Constant<T> for Constraint<T> {
    fn constant(&self) -> &T {
        match self {
            Constraint::Equal(_, value) => value,
            Constraint::LessOrEqual(_, value) => value,
            Constraint::GreaterOrEqual(_, value) => value,
        }
    }
}

impl<'a, T> Coefficients<'a, T> for Constraint<T>
where
    T: 'a,
{
    type CoefficientsIter = std::iter::Map<
        std::collections::btree_set::Iter<'a, UniqueVariableValue<T>>,
        fn(&'a UniqueVariableValue<T>) -> &'a T,
    >;

    fn coefficients(&'a self) -> Self::CoefficientsIter {
        match self {
            Constraint::Equal(expr, _) => expr.coefficients(),
            Constraint::LessOrEqual(expr, _) => expr.coefficients(),
            Constraint::GreaterOrEqual(expr, _) => expr.coefficients(),
        }
    }

    fn coefficients_len(&'a self) -> usize {
        match self {
            Constraint::Equal(expr, _) => expr.coefficients_len(),
            Constraint::LessOrEqual(expr, _) => expr.coefficients_len(),
            Constraint::GreaterOrEqual(expr, _) => expr.coefficients_len(),
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
#[derive(Debug, Clone)]
pub enum Objective<T> {
    /// Maximize the value of the `LinearExpression`.
    Maximize(LinearExpression<T>),

    /// Minimize the value of the `LinearExpression`.
    Minimize(LinearExpression<T>),
}

impl<'a, T> Coefficients<'a, T> for Objective<T>
where
    T: 'a,
{
    type CoefficientsIter = std::iter::Map<
        std::collections::btree_set::Iter<'a, UniqueVariableValue<T>>,
        fn(&'a UniqueVariableValue<T>) -> &'a T,
    >;

    fn coefficients(&'a self) -> Self::CoefficientsIter {
        match self {
            Objective::Maximize(expr) => expr.coefficients(),
            Objective::Minimize(expr) => expr.coefficients(),
        }
    }

    fn coefficients_len(&'a self) -> usize {
        match self {
            Objective::Maximize(expr) => expr.coefficients_len(),
            Objective::Minimize(expr) => expr.coefficients_len(),
        }
    }
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

/// Enum representing errors that may occur in `LinearProgram`.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearProgramError {
    /// Error indicating that the number of coefficients in the linear program is inconsistent.
    InconsistentCoefficientLength,
}

impl std::error::Error for LinearProgramError {}

impl Display for LinearProgramError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LinearProgramError::InconsistentCoefficientLength => {
                write!(
                    f,
                    "inconsistent number of coefficients in the linear program"
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

    /// Returns a builder for a `LinearProgram`.
    ///
    /// # Returns
    /// A `LinearProgramBuilder` to build a `LinearProgram`.
    #[inline]
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

impl std::error::Error for LinearProgramBuilderError {}

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
                    "inconsistent number of coefficients in the linear program"
                )
            }
            LinearProgramBuilderError::MissingObjective => {
                write!(f, "objective function is missing")
            }
        }
    }
}

/// Builder for creating a `LinearProgram`.
/// The builder allows for constructing a `LinearProgram` in a more ergonomic way.
pub struct LinearProgramBuilder<T> {
    objective: Option<Objective<T>>,
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

    /// Sets the objective function of the linear program.
    ///
    /// # Arguments
    /// - `objective`: The objective function of the linear program.
    ///
    /// # Returns
    /// A mutable reference to the builder.
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
    /// A mutable reference to the builder.
    pub fn with_constraint(mut self, constraint: Constraint<T>) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Builds the `LinearProgram` from the builder.
    ///
    /// # Returns
    /// A `Result` with the `LinearProgram` or a `LinearProgramBuilderError`
    /// if the builder is missing required fields.
    pub fn build(self) -> Result<LinearProgram<T>, LinearProgramBuilderError> {
        let objective = self
            .objective
            .ok_or(LinearProgramBuilderError::MissingObjective)?;
        let problem = LinearProgram::new(objective, self.constraints)?;

        Ok(problem)
    }
}
