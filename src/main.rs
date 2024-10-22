// Copyright 2024 Felix Kahle. All rights reserved.

use crate::problem::{Objective, ObjectiveType, ProblemBuilder, Variable};
use crate::tableau::{Tableau, TableauError};

mod problem;
mod tableau;

fn main() {
    // Define variables
    let x1 = Variable::new("x1");
    let x2 = Variable::new("x2");
    
    let problem = ProblemBuilder::new()
        .set_objective(Objective::new(ObjectiveType::Maximize, x1.clone() * 5.0 + x2.clone() * 3.0))
        .add_constraint((x1.clone() * 9.0 + x2.clone() * 3.0).less_or_equal(27.0))
        .add_constraint((x1.clone() * 2.0 + x2.clone() * 1.0).less_or_equal(7.0))
        .add_constraint((x1.clone() * 2.0 + x2.clone() * 2.0).less_or_equal(12.0))
        .build()
        .unwrap();
    

    println!("{}", problem);
    
    let tableau: Result<Tableau, TableauError> = problem.into();
    
    println!("{}", tableau.unwrap());
}
