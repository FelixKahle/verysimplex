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

use crate::alias::{DenseColumn, DenseRow};

/// Trait for solving linear systems of the form `self * x = b`, where `x` is the unknown to be determined.
///
/// # Type Parameters
/// - `T`: Type of the elements of the linear system.
pub trait LeftSolve<T> {
    /// Error type of the linear system.
    type Error;

    /// Solves the linear system `x * self = b`, where `x` is the unknown to be determined.
    ///
    /// # Arguments
    /// - `b`: Right-hand side of the linear system.
    ///
    /// # Returns
    /// The solution `x` of the linear system.
    fn left_solve(&self, b: Box<DenseRow<T>>) -> Result<Box<DenseRow<T>>, Self::Error>;
}

/// Trait for solving linear systems of the form `self * x = b`, where `x` is the unknown to be determined.
///
/// # Type Parameters
/// - `T`: Type of the elements of the linear system.
pub trait RightSolve<T> {
    type Error;

    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// # Arguments
    /// - `b`: Right-hand side of the linear system.
    ///
    /// # Returns
    /// The solution `x` of the linear system.
    fn right_solve(&self, b: Box<DenseColumn<T>>) -> Result<Box<DenseColumn<T>>, Self::Error>;
}
