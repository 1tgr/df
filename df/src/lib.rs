#![deny(warnings)]
#![feature(specialization)]

use std::sync::Arc;

mod df;
mod index;
mod series;
mod simd;
mod storage;

#[cfg(test)]
#[macro_use]
extern crate quickcheck_macros;

trait RefEq {
    fn ref_eq(&self, other: &Self) -> bool;
}

impl<T> RefEq for &T {
    fn ref_eq(&self, other: &Self) -> bool {
        *self as *const T == *other as *const T
    }
}

impl<T> RefEq for Arc<T> {
    fn ref_eq(&self, other: &Self) -> bool {
        RefEq::ref_eq(&(self as &T), &(other as &T))
    }
}

pub use hlist::{Cons, Nil};

pub use crate::df::{empty, AnySeries, Col, DataFrame};
pub use crate::index::{Index, Loc};
pub use crate::series::Series;
pub use crate::simd::{VectorAny, VectorCmp, VectorEq, VectorSum, VectorWhere, VectorWhereOr};
pub use crate::storage::{AnyStorage, Element, Storage};
