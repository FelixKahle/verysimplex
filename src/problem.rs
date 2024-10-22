// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use std::ops::{Add, Sub, Mul, Div};
use std::fmt::{self, Display};

/// A variable in the linear program.
/// Each variable has a name, represented as a string.
#[derive(Debug, Clone)]
pub struct Variable {
    /// The name of the variable.
    pub name: String,
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
            name: name.to_string(),
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

impl Div<f64> for LinearTerm {
    type Output = LinearTerm;

    fn div(self, rhs: f64) -> LinearTerm {
        LinearTerm {
            variable: self.variable,
            coefficient: self.coefficient / rhs,
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



