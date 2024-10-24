// Copyright 2024 Felix Kahle. All rights reserved.

use nalgebra::DMatrix;
use crate::tableau::Tableau;

mod tableau;

fn main() {
    let matrix = DMatrix::from_row_slice(3, 3, &[9.0, 3.0, 27.0, 2.0, 1.0, 7.0, 2.0, 2.0, 12.0]);
    let row_names = vec!["row1".to_string(), "pooopo".to_string(), "row3".to_string()];
    let column_names = vec!["x1".to_string(), "x2".to_string(), "RHS".to_string()];

    let tableau = Tableau::new(matrix, row_names, column_names);

    // Print the tableau using the Display trait
    println!("{}", tableau);
}
