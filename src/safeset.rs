// Copyright 2024 Felix Kahle. All rights reserved.

#![allow(dead_code)]

use num_traits::PrimInt;

/// A set of bounds.
///
/// # Type parameters
/// - `T`: The type of the bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds<T> {
    /// The lower bound.
    lower: T,

    /// The upper bound.
    upper: T,
}

impl<T> Bounds<T> {
    /// Create a new set of bounds.
    pub fn new(lower: T, upper: T) -> Self {
        Self { lower, upper }
    }
}

impl<T> std::fmt::Display for Bounds<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.lower, self.upper)
    }
}

/// Index out of bounds error.
///
/// # Type parameters
/// - `T`: The type of the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexOutOfBoundsError<T>
where
    T: PrimInt,
{
    /// The bounds.
    bounds: Bounds<T>,

    /// The index that caused the error.
    index: T,
}

impl<T> IndexOutOfBoundsError<T>
where
    T: PrimInt,
{
    /// Create a new index out of bounds error.
    ///
    /// # Arguments
    /// - `bounds`: The bounds.
    /// - `index`: The index.
    ///
    /// # Returns
    /// The new index out of bounds error.
    pub fn new(bounds: Bounds<T>, index: T) -> Self {
        Self { bounds, index }
    }
}

impl<T> std::fmt::Display for IndexOutOfBoundsError<T>
where
    T: PrimInt + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "index {} out of bounds {}", self.index, self.bounds)
    }
}

impl<T> std::error::Error for IndexOutOfBoundsError<T> where
    T: std::fmt::Debug + std::fmt::Display + PrimInt
{
}

/// A safe set.
///
/// This trait allows to set a value at a given index in a safe way.
///
/// # Type parameters
/// - `T`: The type of the index.
pub trait SafeSet<I, T>
where
    I: PrimInt,
{
    /// Set a value at the given index.
    ///
    /// # Arguments
    /// - `index`: The index.
    /// - `value`: The value.
    ///
    /// # Returns
    /// The result of the operation.
    fn set(&mut self, index: I, value: T) -> Result<(), IndexOutOfBoundsError<I>>;
}

impl<T> SafeSet<usize, T> for Vec<T> {
    fn set(&mut self, index: usize, value: T) -> Result<(), IndexOutOfBoundsError<usize>> {
        if index < self.len() {
            self[index] = value;
            return Ok(());
        }

        let bounds = Bounds::new(0, self.len() - 1);
        Err(IndexOutOfBoundsError::new(bounds, index))
    }
}

// ---------------------------------------------------------------------------
// Following code implements the SafeSet trait for some common types.
// ---------------------------------------------------------------------------

impl<T> SafeSet<usize, T> for [T] {
    fn set(&mut self, index: usize, value: T) -> Result<(), IndexOutOfBoundsError<usize>> {
        if index < self.len() {
            self[index] = value;
            return Ok(());
        }

        let bounds = Bounds::new(0, self.len() - 1);
        Err(IndexOutOfBoundsError::new(bounds, index))
    }
}

impl<T> SafeSet<usize, T> for &mut [T] {
    fn set(&mut self, index: usize, value: T) -> Result<(), IndexOutOfBoundsError<usize>> {
        if index < self.len() {
            self[index] = value;
            return Ok(());
        }

        let bounds = Bounds::new(0, self.len() - 1);
        Err(IndexOutOfBoundsError::new(bounds, index))
    }
}
