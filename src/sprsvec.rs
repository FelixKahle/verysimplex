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

use faer::{
    sparse::{CreationError, SparseColMat},
    ComplexField, Entity, Index,
};
use num_traits::Zero;

/// A sparse column vector.
///
/// # Type parameters
/// - `I`: The index type.
/// - `E`: The entity type.
#[derive(Debug, Clone)]
pub struct SparseColVec<I, E>
where
    I: Index,
    E: Entity,
{
    /// The data of the sparse column vector
    /// stored as a sparse column matrix
    /// with a single column.
    data: SparseColMat<I, E>,
}

impl<I, E> SparseColVec<I, E>
where
    I: Index + Zero,
    E: Entity + ComplexField,
{
    /// Creates a new sparse column vector from a list of pairs of row indices and values.
    ///
    /// # Parameters
    /// - `nrows`: The number of rows of the vector.
    /// - `pairs`: The list of pairs of row indices and values.
    ///
    /// # Returns
    /// A new instance of `SparseColVec`, or an error if creation fails.
    pub fn from_pairs(nrows: usize, pairs: Vec<(I, E)>) -> Result<Self, CreationError> {
        let default_column = I::zero();
        let triplets: Vec<(I, I, E)> = pairs
            .into_iter()
            .map(|(row, value)| (row, default_column, value))
            .collect();

        let data = SparseColMat::try_new_from_triplets(nrows, 1, &triplets)?;

        Ok(Self { data })
    }

    /// Returns the data of the sparse column vector.
    pub fn data(&self) -> &SparseColMat<I, E> {
        &self.data
    }

    /// Returns the number of rows of the sparse column vector.
    ///
    /// # Returns
    /// The number of rows of the sparse column vector.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }
}

impl<I, E> Into<SparseColMat<I, E>> for SparseColVec<I, E>
where
    I: Index,
    E: Entity,
{
    fn into(self) -> SparseColMat<I, E> {
        self.data
    }
}
