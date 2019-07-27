use std::ops::*;

use crate::series::Series;
use crate::simd::Select;
use crate::storage::{Element, SimdStorage, Storage};

macro_rules! unary_op {
    ($trait: ident, $method: ident) => {
        impl<T> $trait for Series<T>
        where
            T: Element,
            for<'a> &'a T::Owned: $trait<Output = T::Owned>,
        {
            type Output = Self;

            default fn $method(self) -> Self {
                self.map_in_place(|value| *value = $trait::$method(value as &_))
            }
        }

        impl<T, S> $trait for Series<T>
        where
            T: Element<Container = S>,
            for<'a> &'a T::Owned: $trait<Output = T::Owned>, // impl overlap
            S: Storage<T> + SimdStorage<T>,
            S::Packed: Default + $trait<Output = S::Packed>,
            S::Mask: Select<S::Packed>,
        {
            fn $method(self) -> Self {
                self.map_in_place_packed(|value, mask| {
                    *value = mask.select($trait::$method(*value), Default::default())
                })
            }
        }
    };
}

unary_op!(Neg, neg);
unary_op!(Not, not);
