use crate::*;

macro_rules! test_binary_ops {
(
    $test_type:ident,
    ( $( $trait:ident ),* ),
    ( $( $method:ident ),* )
) => {
    $(
    #[quickcheck]
    fn $method(
        data: Vec<(Option<<$test_type as Element>::Owned>, Option<<$test_type as Element>::Owned>)>,
        scalar: <$test_type as Element>::Owned
    ) -> bool {
        let expected_scalar = data.iter()
            .map(|&(ref n, _)| Some($trait::$method(n.as_ref()?.clone(), scalar.borrow())))
            .collect::<Vec<_>>();

        let expected_series = data.iter()
            .map(|&(ref n1, ref n2)| Some($trait::$method(n1.as_ref()?.clone(), n2.as_ref()?.borrow())))
            .collect::<Vec<_>>();

        let series1 = from_left(&data);
        let series2 = from_right(&data);
        let actual_scalar = $trait::$method(series1.clone(), scalar.borrow() as &$test_type).to_option_vec();
        if expected_scalar != actual_scalar {
            return false;
        }

        let actual_series = $trait::$method(series1, series2).to_option_vec();
        expected_series == actual_series
    }
    )*
    }
}

macro_rules! test_logical_ops {
(
    $test_type:ident,
    ( $( $trait:ident ),* ),
    ( $( $method:ident ),* )
) => {
    $(
    #[quickcheck]
    fn $method(
        data: Vec<(Option<<$test_type as Element>::Owned>, Option<<$test_type as Element>::Owned>)>,
        scalar: <$test_type as Element>::Owned
    ) -> bool {
        let scalar = scalar.borrow() as &$test_type;

        let expected_scalar = data.iter()
            .map(|&(ref n, _)| Some($trait::$method(n.as_ref()?.borrow() as &$test_type, scalar)))
            .collect::<Vec<_>>();

        let expected_series = data.iter()
            .map(|&(ref n1, ref n2)| Some($trait::$method(n1.as_ref()?.borrow() as &$test_type, n2.as_ref()?.borrow())))
            .collect::<Vec<_>>();

        let series1 = from_left(&data);
        let series2 = from_right(&data);
        let actual_scalar = $trait::$method(series1.clone(), scalar).to_option_vec();
        if expected_scalar != actual_scalar {
            return false;
        }

        let actual_series = $trait::$method(series1, series2).to_option_vec();
        expected_series == actual_series
    }
    )*
}
}

macro_rules! test_general {
    ($test_type: ident, $placeholder: expr) => {
        #[quickcheck]
        fn loc(data: Vec<Option<<$test_type as Element>::Owned>>) -> bool {
            let series = from::<$test_type>(data.clone());
            let expected: Option<&$test_type> = data.get(0).unwrap_or(&None).as_ref().map(|value| value.borrow());
            let actual: Option<&$test_type> = series.loc(&Loc::Usize(0));
            expected == actual
        }

        #[quickcheck]
        fn loc_range(data: Vec<Option<<$test_type as Element>::Owned>>) -> bool {
            let series = from::<$test_type>(data.clone());
            let range = data.len().min(3)..data.len().min(10);
            let mut expected = vec![None; range.len()];
            for offset in range {
                expected[offset - data.len().min(3)] = data[offset].clone();
            }

            let actual = series.loc_range(3..10).to_option_vec();
            expected == actual
        }

        #[quickcheck]
        fn where_empty(data: Vec<(Option<<$test_type as Element>::Owned>, Option<bool>)>) -> bool {
            let series: Series<$test_type> = from_with(&data, |&(ref n, _)| n.clone());
            let condition: Series<bool> = from_with(&data, |&(_, b)| b);

            let expected_where = data.iter()
                .map(|&(ref n, b)| {
                    if b.unwrap_or(false) {
                        n.clone().unwrap_or($placeholder.clone())
                    } else {
                        $placeholder.clone()
                    }
                })
                .collect::<Vec<_>>();

            let expected_mask = data.iter()
                .map(|&(ref n, b)| {
                    if b.unwrap_or(false) {
                        $placeholder.clone()
                    } else {
                        n.clone().unwrap_or($placeholder.clone())
                    }
                })
                .collect::<Vec<_>>();

            if series.clone().where_(condition.clone()).to_vec($placeholder.clone()) != expected_where {
                return false;
            }

            series.mask(condition).to_vec($placeholder.clone()) == expected_mask
        }

        #[quickcheck]
        fn where_scalar(
            data: Vec<(Option<<$test_type as Element>::Owned>, Option<bool>)>,
            scalar: <$test_type as Element>::Owned,
        ) -> bool {
            let series: Series<$test_type> = from_with(&data, |&(ref n, _)| n.clone());
            let condition: Series<bool> = from_with(&data, |&(_, b)| b);

            let expected_where = data.iter()
                .map(|&(ref n, b)| {
                    if b.unwrap_or(false) {
                        n.clone().unwrap_or($placeholder.clone())
                    } else {
                        scalar.clone()
                    }
                })
                .collect::<Vec<_>>();

            let expected_mask = data.iter()
                .map(|&(ref n, b)| {
                    if b.unwrap_or(false) {
                        scalar.clone()
                    } else {
                        n.clone().unwrap_or($placeholder.clone())
                    }
                })
                .collect::<Vec<_>>();

            if series
                .clone()
                .where_or(condition.clone(), scalar.borrow())
                .to_vec($placeholder.clone()) != expected_where
            {
                return false;
            }

            series.mask_or(condition, scalar.borrow()).to_vec($placeholder.clone()) == expected_mask
        }

        #[quickcheck]
        fn where_series(
            data: Vec<(
                Option<<$test_type as Element>::Owned>,
                Option<bool>,
                Option<<$test_type as Element>::Owned>,
            )>,
        ) -> bool {
            let series1: Series<$test_type> = from_with(&data, |&(ref n, _, _)| n.clone());
            let condition: Series<bool> = from_with(&data, |&(_, b, _)| b);
            let series2: Series<$test_type> = from_with(&data, |&(_, _, ref m)| m.clone());

            let expected_where = data.iter()
                .map(|&(ref n, b, ref m)| {
                    if b.unwrap_or(false) {
                        n.clone()
                    } else {
                        m.clone()
                    }
                })
                .map(|opt| opt.unwrap_or($placeholder.clone()))
                .collect::<Vec<_>>();

            let expected_mask = data.iter()
                .map(|&(ref n, b, ref m)| {
                    if b.unwrap_or(false) {
                        m.clone()
                    } else {
                        n.clone()
                    }
                })
                .map(|opt| opt.unwrap_or($placeholder.clone()))
                .collect::<Vec<_>>();

            if series1
                .clone()
                .where_or(condition.clone(), series2.clone())
                .to_vec($placeholder.clone()) != expected_where
            {
                return false;
            }

            series1.mask_or(condition, series2).to_vec($placeholder.clone()) == expected_mask
        }
    };
}

macro_rules! test_unary_ops {
(
    $test_type:ident,
    ( $( $trait:ident ),* ),
    ( $( $method:ident ),* )
) => {
    $(
    #[quickcheck]
    fn $method(
        data: Vec<Option<<$test_type as Element>::Owned>>
    ) -> bool {
        let expected = data.iter().map(|n| Some($trait::$method(n.as_ref()?.borrow()))).collect::<Vec<_>>();
        let series = from::<$test_type>(data);
        let actual = $trait::$method(series).to_option_vec();
        expected == actual
    }
    )*
}
}

macro_rules! test_int {
(
    $( $test_type:ident ),*
) => {

$(
mod $test_type {
    use std::borrow::Borrow;
    use std::ops::*;

    use crate::series::tests::*;

    test_binary_ops!(
        $test_type,
        (Add, /*Div,*/ Mul, /*Rem,*/ Sub),
        (add, /*div,*/ mul, /*rem,*/ sub)
    );

    test_general!($test_type, 99);

    test_logical_ops!(
        $test_type,
        (VectorEq, VectorEq, VectorCmp, VectorCmp, VectorCmp, VectorCmp),
        (eq, ne, lt, lte, gt, gte)
    );

    test_unary_ops!($test_type, (Not), (not));

    #[quickcheck]
    fn sum(data: Vec<Option<$test_type>>) -> bool {
        let expected: $test_type = data.iter().filter_map(Option::as_ref).sum();
        let actual = from::<$test_type>(data).sum();
        expected == actual
    }

    #[test]
    #[ignore = "crashes with SIGFPE"]
    fn divide_by_scalar_zero() {
        let series = Series::<$test_type>::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(vec![None, None, None, None, None], (series / 0).to_option_vec());
    }

    #[test]
    #[ignore = "crashes with SIGFPE"]
    fn divide_by_series_zero() {
        let series1 = Series::<$test_type>::from(vec![1, 2, 3, 4, 5]);
        let series2 = Series::<$test_type>::from(vec![2, 2, 0, 2, 2]);
        assert_eq!(
            vec![Some(0), Some(1), None, Some(1), Some(1)],
            (series1 / series2).to_option_vec()
        );
    }
}
)*

}
}

macro_rules! test_float {
(
    $( $test_type:ident ),*
) => {

$(
mod $test_type {

use std::borrow::Borrow;
use std::ops::*;

use crate::series::tests::*;

test_binary_ops!($test_type, (Add, Div, Mul, Rem, Sub), (add, div, mul, rem, sub));
test_general!($test_type, -1.0);

test_logical_ops!(
    $test_type,
    (VectorEq, VectorEq, VectorCmp, VectorCmp, VectorCmp, VectorCmp),
    (eq, ne, lt, lte, gt, gte)
);

test_unary_ops!($test_type, (Neg), (neg));

#[quickcheck]
fn sum(data: Vec<Option<$test_type>>) -> bool {
    let expected: $test_type = data.iter().filter_map(Option::as_ref).sum();
    let actual = from::<$test_type>(data).sum();
    (expected - actual).abs() <= 1e-9
}

#[test]
#[should_panic(expected = "[Some(-inf), Some(-inf), Some(NaN), Some(inf), Some(inf)]")]
fn divide_by_scalar_zero() {
    let series = Series::<$test_type>::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    assert_eq!(vec![None, None, None, None], (series / 0.0).to_option_vec());
}

#[test]
#[should_panic(expected = "[Some(-1.0), Some(-inf), Some(NaN), Some(inf), Some(1.0)]")]
fn divide_by_series_zero() {
    let series1 = Series::<$test_type>::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let series2 = Series::<$test_type>::from(vec![2.0, 0.0, 0.0, 0.0, 2.0]);
    assert_eq!(
        vec![Some(-1.0), None, None, None, Some(1.0)],
        (series1 / series2).to_option_vec()
    );
}
}

)*
}
}

mod bool;

test_int!(i8, i16, i32, i64, u8, u16, u32, u64);
test_float!(f32, f64);

fn from<T: Element + ?Sized>(data: Vec<Option<T::Owned>>) -> Series<T> {
    data.into_iter()
        .enumerate()
        .map(|(offset, item)| (Loc::Usize(offset), item))
        .collect()
}

fn from_with<T: Element + ?Sized, A, F: FnMut(&A) -> Option<T::Owned>>(data: &Vec<A>, mut f: F) -> Series<T> {
    data.iter()
        .enumerate()
        .map(move |(offset, item)| (Loc::Usize(offset), f(item)))
        .collect()
}

fn from_left<T: Element + ?Sized, U>(data: &Vec<(Option<T::Owned>, U)>) -> Series<T> {
    from_with(data, |&(ref item, _)| item.clone())
}

fn from_right<T: Element + ?Sized, U>(data: &Vec<(U, Option<T::Owned>)>) -> Series<T> {
    from_with(data, |&(_, ref item)| item.clone())
}

#[test]
fn reindex_same() {
    let series = Series::<f64>::from(vec![1.0, 2.0, 3.0]).reindex(Index::from(0..3));
    assert_eq!(vec![1.0, 2.0, 3.0], series.to_vec(0.0));
}

#[test]
fn reindex_empty_to_empty() {
    let series = Series::<f64>::default().reindex(Default::default());
    assert_eq!(vec![] as Vec<f64>, series.to_vec(0.0));
}

#[test]
fn reindex_some_to_empty() {
    let series = Series::<f64>::from(vec![1.0, 2.0, 3.0]).reindex(Default::default());
    assert_eq!(vec![] as Vec<f64>, series.to_vec(0.0));
}

#[test]
fn reindex_empty_to_some() {
    let series = Series::<f64>::default().reindex(Index::from(0..3));
    assert_eq!(vec![-1.0, -1.0, -1.0] as Vec<f64>, series.to_vec(-1.0));
}

#[test]
fn reindex_some_to_subset() {
    let series = Series::<f64>::from(vec![1.0, 2.0, 3.0]).reindex(Index::from(1..2));
    assert_eq!(vec![2.0], series.to_vec(0.0));
}

#[test]
fn reindex_some_to_superset() {
    let series = Series::<f64>::from(vec![1.0, 2.0, 3.0]).reindex(Index::from(0..5));
    assert_eq!(vec![1.0, 2.0, 3.0, -1.0, -1.0] as Vec<f64>, series.to_vec(-1.0));
}

#[test]
fn filter_same() {
    let series =
        Series::<f64>::from(vec![1.0, 2.0, 3.0]).filter_with(|series| series.gte(Series::from(vec![0.0, 0.0, 0.0])));
    assert_eq!(vec![1.0, 2.0, 3.0], series.to_vec(0.0));
}

#[test]
fn filter_empty_to_empty() {
    let series = Series::<f64>::default().filter(Default::default());
    assert_eq!(vec![] as Vec<f64>, series.to_vec(0.0));
}

#[test]
fn filter_some_to_empty() {
    let series =
        Series::<f64>::from(vec![1.0, 2.0, 3.0]).filter_with(|series| series.gte(Series::from(vec![4.0, 4.0, 4.0])));
    assert_eq!(vec![] as Vec<f64>, series.to_vec(0.0));
}

#[test]
fn filter_some_to_subset() {
    let series =
        Series::<f64>::from(vec![1.0, 2.0, 3.0]).filter_with(|series| series.gte(Series::from(vec![2.0, 2.0, 2.0])));
    assert_eq!(vec![2.0, 3.0], series.to_vec(0.0));
}
