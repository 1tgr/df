use std::borrow::Borrow;
use std::ops::*;

use crate::series::tests::*;

test_binary_ops!(bool, (BitAnd, BitOr), (bitand, bitor));
test_general!(bool, true);

test_logical_ops!(
    bool,
    (VectorEq, VectorEq, VectorCmp, VectorCmp, VectorCmp, VectorCmp),
    (eq, ne, lt, lte, gt, gte)
);

test_unary_ops!(bool, (Not), (not));

#[quickcheck]
fn any(data: Vec<Option<bool>>) -> bool {
    let expected = data.iter().filter_map(|&o| o).any(|b| b);
    let actual = from(data).any();
    expected == actual
}

#[quickcheck]
fn all(data: Vec<Option<bool>>) -> bool {
    let expected = data.iter().filter_map(|&o| o).all(|b| b);
    let actual = from(data).all();
    expected == actual
}

#[quickcheck]
fn none(data: Vec<Option<bool>>) -> bool {
    let expected = !data.iter().filter_map(|&o| o).any(|b| b);
    let actual = from(data).none();
    expected == actual
}
