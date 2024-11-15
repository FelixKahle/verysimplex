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

use std::rc::Rc;

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

impl From<(usize, Rc<String>)> for Variable {
    fn from((id, name): (usize, Rc<String>)) -> Self {
        Self::new(id, name)
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

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
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

impl<T> From<(Variable, T)> for VariableValue<T> {
    fn from((variable, value): (Variable, T)) -> Self {
        Self::new(variable, value)
    }
}

impl<T> From<(usize, Rc<String>, T)> for VariableValue<T> {
    fn from((id, name, value): (usize, Rc<String>, T)) -> Self {
        Self::new(Variable::new(id, name), value)
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

impl<T> PartialOrd for VariableValue<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.variable.partial_cmp(&other.variable) {
            Some(std::cmp::Ordering::Equal) => self.value.partial_cmp(&other.value),
            other => other,
        }
    }
}

impl<T> Ord for VariableValue<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.variable.cmp(&other.variable) {
            std::cmp::Ordering::Equal => self.value.cmp(&other.value),
            other => other,
        }
    }
}

impl<T> std::hash::Hash for VariableValue<T>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.variable.hash(state);
        self.value.hash(state);
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
