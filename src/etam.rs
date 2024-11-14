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

use nalgebra::DVector;
use std::fmt::Write;

/// Elementary transformation matrix.
///
/// # Type parameters
/// - `T`: The type of the elements of the matrix.
#[derive(Clone, Debug)]
pub(crate) struct EtaMatrix<T> {
    /// The index of the column that the matrix will be applied to.
    column_index: usize,

    /// The eta column.
    eta_column: DVector<T>,
}

impl<T> EtaMatrix<T> {
    /// Constructs a new `EtaMatrix`.
    ///
    /// # Parameters
    /// - `column_index`: The index of the column that the matrix will be applied to.
    /// - `eta_column`: The eta column.
    ///
    /// # Returns
    /// A new instance of `EtaMatrix`.
    pub(crate) fn new(column_index: usize, eta_column: DVector<T>) -> Self {
        Self {
            column_index,
            eta_column,
        }
    }

    /// Gets the index of the column that the matrix will be applied to.
    ///
    /// # Returns
    /// The index of the column that the matrix will be applied to.
    pub(crate) fn column_index(&self) -> usize {
        self.column_index
    }

    /// Gets a mutable reference to the index of the column that the matrix will be applied to.
    ///
    /// # Returns
    /// A mutable reference to the index of the column that the matrix will be applied to.
    pub(crate) fn column_index_mut(&mut self) -> &mut usize {
        &mut self.column_index
    }

    /// Gets the eta column.
    ///
    /// # Returns
    /// The eta column.
    pub(crate) fn eta_column(&self) -> &DVector<T> {
        &self.eta_column
    }

    /// Gets a mutable reference to the eta column.
    ///
    /// # Returns
    /// A mutable reference to the eta column.
    pub(crate) fn eta_column_mut(&mut self) -> &mut DVector<T> {
        &mut self.eta_column
    }
}

impl<T> PartialEq for EtaMatrix<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.column_index == other.column_index && self.eta_column == other.eta_column
    }
}

impl<T> Eq for EtaMatrix<T> where T: Eq {}

/// Returns a heuristic for the capacity of a string representation of a vector.
///
/// # Parameters
/// - `vec_len`: The length of the vector.
/// - `estimated_element_capacity`: An estimate of the capacity of the string representation of an element.
///
/// # Returns
/// A heuristic for the capacity of a string representation of a vector.
#[inline(always)]
fn string_capacity_heuristic(vec_len: usize, estimated_element_capacity: usize) -> usize {
    vec_len * estimated_element_capacity + (vec_len - 1) * 2 + 2
}

/// Converts a vector to a string.
///
/// The string will have the following format: `[a, b, c, ...]`.
///
/// # Parameters
/// - `vector`: The vector to convert.
///
/// # Returns
/// The string representation of the vector.
#[inline(always)]
fn vector_to_string<T>(vector: &DVector<T>) -> String
where
    T: std::fmt::Display,
{
    // Preallocates a string with a heuristic capacity.
    // We use a heuristic to avoid to many reallocations.
    let mut string = String::with_capacity(string_capacity_heuristic(vector.len(), 4));

    string.push('[');
    for (i, element) in vector.iter().enumerate() {
        // It is safe to call `unwrap` here, as writing to a `String` will hardly ever fail.
        // And if it does, panicking is the correct behavior as something really bad must have happened.
        write!(&mut string, "{}", element).unwrap();
        if i < vector.len() - 1 {
            string.push_str(", ");
        }
    }
    string.push(']');
    string
}

impl<T> std::fmt::Display for EtaMatrix<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let eta_column = vector_to_string(&self.eta_column);
        write!(f, "[{}|{}]", self.column_index, eta_column)
    }
}
