#![feature(test)]
#![deny(warnings)]
use std::iter::FromIterator;

use df::{Series, VectorSum};

extern crate test;

fn main() {
    let data = Vec::from_iter((0..100000).map(|n| n as f64));
    let mut sum1 = 0.0;
    for _ in 0..100000 {
        sum1 += data.iter().sum::<f64>();
    }

    test::black_box(sum1);

    let series = Series::<f64>::from(data.clone());
    let mut sum2 = 0.0;
    for _ in 0..100000 {
        sum2 += series.sum();
    }

    test::black_box(sum2);
}
