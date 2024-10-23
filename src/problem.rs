// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::collections::HashSet;
use std::ops::{Add, Sub, Mul, Div};
use std::fmt::{self, Display};
use std::rc::Rc;

/// A variable in the linear program.
/// Each variable has a name, represented as a string.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Variable {
    /// The name of the variable.
    /// 
    /// # Note
    /// The name is stored as a reference-counted string to avoid copying the name when passing variables around.
    pub name: Rc<String>,
}

impl Variable {
    /// Creates a new variable with the given name.
    ///
    /// # Arguments
    /// - `name`: The name of the variable.
    ///
    /// # Returns
    /// A new `Variable` object with the given name.
    pub fn new(name: &str) -> Variable {
        Variable {
            name: Rc::new(name.to_string()),
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A linear term consisting of a variable and its coefficient.
/// Represents terms like `3x`, where `3` is the coefficient and `x` is the variable.
#[derive(Debug, Clone)]
pub struct LinearTerm {
    /// The variable in the term.
    pub variable: Variable,
    
    /// The coefficient of the variable.
    pub coefficient: f64,
}

impl Mul<f64> for Variable {
    type Output = LinearTerm;

    fn mul(self, rhs: f64) -> LinearTerm {
        LinearTerm {
            variable: self,
            coefficient: rhs,
        }
    }
}

impl Div<f64> for Variable {
    type Output = LinearTerm;

    fn div(self, rhs: f64) -> LinearTerm {
        LinearTerm {
            variable: self,
            coefficient: 1.0 / rhs,
        }
    }
}

impl Mul<f64> for &Variable {
    type Output = LinearTerm;

    fn mul(self, rhs: f64) -> LinearTerm {
        LinearTerm {
            variable: self.clone(),
            coefficient: rhs,
        }
    }
}

impl Div<f64> for &Variable {
    type Output = LinearTerm;

    fn div(self, rhs: f64) -> LinearTerm {
        LinearTerm {
            variable: self.clone(),
            coefficient: 1.0 / rhs,
        }
    }
}

impl Display for LinearTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.coefficient, self.variable)
    }
}

/// A linear expression is a sum of linear terms, used in constraints.
/// Example: `3x + 5y - 2z`
#[derive(Debug, Clone)]
pub struct LinearExpression {
    /// The terms in the linear expression.
    pub terms: Vec<LinearTerm>,
}

impl Add<LinearTerm> for LinearExpression {
    type Output = LinearExpression;

    fn add(mut self, rhs: LinearTerm) -> LinearExpression {
        self.terms.push(rhs);
        self
    }
}

impl Add<LinearTerm> for LinearTerm {
    type Output = LinearExpression;

    fn add(self, rhs: LinearTerm) -> LinearExpression {
        LinearExpression {
            terms: vec![self, rhs],
        }
    }
}

impl Sub<LinearTerm> for LinearExpression {
    type Output = LinearExpression;

    fn sub(mut self, rhs: LinearTerm) -> LinearExpression {
        self.terms.push(LinearTerm {
            variable: rhs.variable,
            coefficient: -rhs.coefficient,
        });
        self
    }
}

impl Sub<LinearTerm> for LinearTerm {
    type Output = LinearExpression;

    fn sub(self, rhs: LinearTerm) -> LinearExpression {
        LinearExpression {
            terms: vec![self, LinearTerm {
                variable: rhs.variable,
                coefficient: -rhs.coefficient,
            }],
        }
    }
}

impl Mul<f64> for LinearExpression {
    type Output = LinearExpression;

    fn mul(mut self, rhs: f64) -> LinearExpression {
        for term in &mut self.terms {
            term.coefficient *= rhs;
        }
        self
    }
}

impl Div<f64> for LinearExpression {
    type Output = LinearExpression;

    fn div(mut self, rhs: f64) -> LinearExpression {
        for term in &mut self.terms {
            term.coefficient /= rhs;
        }
        self
    }
}

impl Display for LinearExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = self.terms.iter();

        if let Some(first) = terms.next() {
            write!(f, "{}", first)?;
            for term in terms {
                write!(f, " + {}", term)?;
            }
        }
        Ok(())
    }
}

/// Enum representing the relation in a constraint (<=, =, >=).
#[derive(Debug, Clone, PartialEq)]
pub enum Relation {
    /// Less-than-or-equal relation.
    LessThanOrEqual,
    
    /// Greater-than-or-equal relation.
    GreaterThanOrEqual,
    
    /// Less-than relation.
    LessThan,
    
    /// Greater-than relation.
    GreaterThan,
    
    /// Equality relation.
    Equal,
}

/// Display implementation for `Relation`.
/// This allows printing relations as <=, >=, or =.
impl Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Relation::LessThanOrEqual => write!(f, "<="),
            Relation::GreaterThanOrEqual => write!(f, ">="),
            Relation::LessThan => write!(f, "<"),
            Relation::GreaterThan => write!(f, ">"),
            Relation::Equal => write!(f, "="),
        }
    }
}

/// A constraint in a linear program, consisting of a linear expression, a relation (<=, =, >=), and a right-hand-side constant.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// The linear expression in the constraint.
    pub expression: LinearExpression,
    
    /// The relation in the constraint.
    pub relation: Relation,
    
    /// The right-hand-side constant in the constraint.
    pub rhs: f64,
}

impl Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.expression, self.relation, self.rhs)
    }
}

impl LinearExpression {
    /// Creates a less-than-or-equal constraint from the linear expression.
    ///
    /// # Arguments
    /// - `rhs`: The right-hand-side constant of the constraint.
    ///
    /// # Returns
    /// A new `Constraint` object representing the less-than-or-equal constraint.
    pub fn less_or_equal(self, rhs: f64) -> Constraint {
        Constraint {
            expression: self,
            relation: Relation::LessThanOrEqual,
            rhs,
        }
    }
    
    /// Creates a less-than constraint from the linear expression.
    ///
    /// # Arguments
    /// - `rhs`: The right-hand-side constant of the constraint.
    /// 
    /// # Returns
    /// A new `Constraint` object representing the less-than constraint.
    pub fn less_than(self, rhs: f64) -> Constraint {
        Constraint {
            expression: self,
            relation: Relation::LessThan,
            rhs,
        }
    }

    /// Creates a greater-than-or-equal constraint from the linear expression.
    ///
    /// # Arguments
    /// - `rhs`: The right-hand-side constant of the constraint.
    ///
    /// # Returns
    /// A new `Constraint` object representing the greater-than-or-equal constraint.
    pub fn greater_or_equal(self, rhs: f64) -> Constraint {
        Constraint {
            expression: self,
            relation: Relation::GreaterThanOrEqual,
            rhs,
        }
    }
    
    /// Creates a greater-than constraint from the linear expression.
    ///
    /// # Arguments
    /// - `rhs`: The right-hand-side constant of the constraint.
    ///
    /// # Returns
    /// A new `Constraint` object representing the greater-than constraint.
    pub fn greater_than(self, rhs: f64) -> Constraint {
        Constraint {
            expression: self,
            relation: Relation::GreaterThan,
            rhs,
        }
    }

    /// Creates an equality constraint from the linear expression.
    ///
    /// # Arguments
    /// - `rhs`: The right-hand-side constant of the constraint.
    ///
    /// # Returns
    /// A new `Constraint` object representing the equality constraint.
    pub fn equal(self, rhs: f64) -> Constraint {
        Constraint {
            expression: self,
            relation: Relation::Equal,
            rhs,
        }
    }
}

/// Enum representing the type of the objective function in a linear program (minimize or maximize).
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveType {
    /// Minimize the objective function.
    Minimize,
    
    /// Maximize the objective function.
    Maximize,
}

impl Display for ObjectiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectiveType::Minimize => write!(f, "Minimize"),
            ObjectiveType::Maximize => write!(f, "Maximize"),
        }
    }
}

/// The objective function in a linear program, consisting of a linear expression and an objective type (minimize or maximize).
#[derive(Debug, Clone)]
pub struct Objective {
    /// The type of the objective function (minimize or maximize).
    pub objective_type: ObjectiveType,
    
    /// The linear expression of the objective function.
    pub expression: LinearExpression,
}

impl Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.objective_type, self.expression)
    }
}

impl Objective {
    /// Creates a new objective function with the given type and linear expression.
    ///
    /// # Arguments
    /// - `objective_type`: The type of the objective function (minimize or maximize).
    /// - `expression`: The linear expression of the objective function.
    ///
    /// # Returns
    /// A new `Objective` object with the given type and expression.
    pub fn new(objective_type: ObjectiveType, expression: LinearExpression) -> Objective {
        Objective {
            objective_type,
            expression,
        }
    }
}

/// A linear program problem, consisting of a list of constraints,
/// and an objective function that needs to be minimized or maximized.
pub struct Problem {
    /// The variables in the problem.
    pub variables: Vec<Rc<Variable>>,
    
    /// The constraints in the problem.
    pub constraints: Vec<Constraint>,
    
    /// The objective function of the problem.
    pub objective: Objective,
}

impl Problem {
    /// Creates a new linear program problem with the given constraints and objective function.
    ///
    /// # Arguments
    /// - `constraints`: The constraints in the problem.
    /// - `objective`: The objective function of the problem.
    ///
    /// # Returns
    /// A new `Problem` object with the given constraints and objective function.
    pub fn new(constraints: Vec<Constraint>, objective: Objective) -> Problem {
        let mut unique_variables: HashSet<Variable> = HashSet::new();

        for constraint in &constraints {
            for term in &constraint.expression.terms {
                unique_variables.insert(term.variable.clone());
            }
        }

        for term in &objective.expression.terms {
            unique_variables.insert(term.variable.clone());
        }

        let variables: Vec<Rc<Variable>> = unique_variables
            .into_iter()
            .map(|v| Rc::new(v))
            .collect();

        Problem {
            variables,
            constraints,
            objective,
        }
    }
    
    /// Creates a new `ProblemBuilder` instance to build a `Problem`.
    ///
    /// # Returns
    /// A new `ProblemBuilder` object to build a `Problem`.
    pub fn builder() -> ProblemBuilder {
        ProblemBuilder::new()
    }
}

impl Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for constraint in &self.constraints {
            writeln!(f, "{}", constraint)?;
        }
        writeln!(f, "Objective: {}", self.objective)
    }
}

/// Builder pattern for creating `Problem` instances.
pub struct ProblemBuilder {
    /// The constraints in the problem.
    constraints: Vec<Constraint>,
    
    /// The objective function of the problem.
    objective: Option<Objective>
}

/// Error type for when the objective is missing in the `ProblemBuilder`.
#[derive(Debug, Clone)]
pub struct MissingObjectiveError;

impl ProblemBuilder {
    /// Creates a new `ProblemBuilder` instance.
    /// 
    /// # Returns
    /// A new `ProblemBuilder` object.
    pub fn new() -> ProblemBuilder {
        ProblemBuilder {
            constraints: Vec::new(),
            objective: None,
        }
    }

    /// Adds a constraint to the builder.
    ///
    /// # Arguments
    /// - `constraint`: The constraint to add.
    ///
    /// # Returns
    /// The `ProblemBuilder` object with the constraint added.
    pub fn add_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Sets the objective function.
    ///
    /// # Arguments
    /// - `objective`: The objective function to set.
    ///
    /// # Returns
    /// The `ProblemBuilder` object with the objective function set.
    pub fn set_objective(mut self, objective: Objective) -> Self {
        self.objective = Some(objective);
        self
    }

    /// Builds the final `Problem`, returning an error if the objective is missing.
    ///
    /// # Returns
    /// A `Problem` object if the objective is set, otherwise an error message.
    pub fn build(self) -> Result<Problem, MissingObjectiveError> {
        if let Some(objective) = self.objective {
            Ok(Problem::new(self.constraints, objective))
        } else {
            Err(MissingObjectiveError)
        }
    }
}



