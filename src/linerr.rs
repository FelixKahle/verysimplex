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

/// Invalid row count error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidRowCountError {
    /// Expected row count.
    pub expected: usize,

    /// Actual row count.
    pub actual: usize,
}

impl InvalidRowCountError {
    /// Create a new invalid row count error.
    ///
    /// # Arguments
    /// - `expected`: Expected row count.
    /// - `actual`: Actual row count.
    ///
    /// # Returns
    /// A new invalid row count error.
    pub fn new(expected: usize, actual: usize) -> Self {
        Self { expected, actual }
    }

    /// Get the expected row count.
    ///
    /// # Returns
    /// The expected row count.
    pub fn expected(&self) -> usize {
        self.expected
    }

    /// Get the actual row count.
    ///
    /// # Returns
    /// The actual row count.
    pub fn actual(&self) -> usize {
        self.actual
    }
}

impl std::fmt::Display for InvalidRowCountError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Invalid row count: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for InvalidRowCountError {}

/// Invalid column count error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidColumnCountError {
    /// Expected column count.
    pub expected: usize,

    /// Actual column count.
    pub actual: usize,
}

impl InvalidColumnCountError {
    /// Create a new invalid column count error.
    ///
    /// # Arguments
    /// - `expected`: Expected column count.
    /// - `actual`: Actual column count.
    ///
    /// # Returns
    /// A new invalid column count error.
    pub fn new(expected: usize, actual: usize) -> Self {
        Self { expected, actual }
    }

    /// Get the expected column count.
    ///
    /// # Returns
    /// The expected column count.
    pub fn expected(&self) -> usize {
        self.expected
    }

    /// Get the actual column count.
    ///
    /// # Returns
    /// The actual column count.
    pub fn actual(&self) -> usize {
        self.actual
    }
}

impl std::fmt::Display for InvalidColumnCountError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Invalid column count: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for InvalidColumnCountError {}
