use std::borrow::Borrow;
use std::ops::*;

use crate::series::tests::*;

use Loc::Usize;

test_binary_ops!(str, (Add), (add));
test_general!(str, "hello".to_string());

test_logical_ops!(
    str,
    (VectorEq, VectorEq, VectorCmp, VectorCmp, VectorCmp, VectorCmp),
    (eq, ne, lt, lte, gt, gte)
);

#[test]
fn map_in_place_string_same_length() {
    let series = from::<str>(vec![Some("hello".to_string()), None, Some("world".to_string())])
        .map_in_place(|s| *s = s.to_uppercase());

    assert_eq!(
        vec!["HELLO".to_string(), "".to_string(), "WORLD".to_string()],
        series.to_vec(String::new())
    );
}

#[test]
fn map_in_place_string_shorter() {
    let series = from::<str>(vec![Some("hello".to_string()), None, Some("world".to_string())])
        .map_in_place(|s| *s = s[0..2].to_uppercase());

    assert_eq!(
        vec!["HE".to_string(), "".to_string(), "WO".to_string()],
        series.to_vec(String::new())
    );
}

#[test]
fn map_in_place_string_longer() {
    let series = from::<str>(vec![Some("hello".to_string()), None, Some("world".to_string())])
        .map_in_place(|s| *s += s.to_uppercase().as_ref());

    assert_eq!(
        vec!["helloHELLO".to_string(), "".to_string(), "worldWORLD".to_string()],
        series.to_vec(String::new())
    );
}

#[test]
fn zip_in_place_string() {
    let series1 = from::<str>(vec![Some("hello".to_string()), None, Some("world".to_string())]);

    let series2 = vec![(Usize(1), Some("123".to_string())), (Usize(2), Some("456".to_string()))]
        .into_iter()
        .collect::<Series<str>>();

    let series = series1 + series2;
    assert_eq!(
        vec!["".to_string(), "".to_string(), "world456".to_string()],
        series.to_vec(String::new())
    );
}
