use std::ops::*;

use crate::series::Series;
use crate::simd::Select;
use crate::storage::{Element, SimdStorage, Storage};

macro_rules! binary_op {
    ($trait: ident, $method: ident, $assign_trait: ident, $assign_method: ident) => {
        impl<T> $trait<T> for Series<T>
        where
            T: Element,
            T::Owned: for<'a> $assign_trait<&'a T>,
        {
            type Output = Self;

            fn $method(self, rhs: T) -> Self {
                $trait::$method(self, &rhs)
            }
        }

        impl<T> $trait<&T> for Series<T>
        where
            T: Element + ?Sized,
            T::Owned: for<'a> $assign_trait<&'a T>,
        {
            type Output = Self;

            default fn $method(self, rhs: &T) -> Self {
                self.map_in_place(|value| $assign_trait::$assign_method(value, rhs))
            }
        }

        impl<T, S> $trait<&T> for Series<T>
        where
            T: Element<Container = S> + ?Sized,
            T::Owned: for<'a> $assign_trait<&'a T>, // impl overlap
            S: Storage<T> + SimdStorage<T>,
            S::Packed: Default + for<'a> $trait<&'a T, Output = S::Packed>,
            S::Mask: Select<S::Packed>,
        {
            fn $method(self, rhs: &T) -> Self {
                self.map_in_place_packed(|value, mask| {
                    *value = mask.select($trait::$method(*value, rhs), Default::default())
                })
            }
        }

        impl<T> $trait<Series<T>> for Series<T>
        where
            T: Element + ?Sized,
            T::Owned: for<'a> $assign_trait<&'a T>,
        {
            type Output = Self;

            default fn $method(self, rhs: Self) -> Self {
                self.zip_in_place(rhs, |value, rhs| $assign_trait::$assign_method(value, rhs))
            }
        }

        impl<T, S> $trait<Series<T>> for Series<T>
        where
            T: Element<Container = S> + ?Sized,
            T::Owned: for<'a> $assign_trait<&'a T>, // impl overlap
            S: Storage<T> + SimdStorage<T>,
            S::Packed: Default + $trait<Output = S::Packed>,
            S::Mask: Select<S::Packed>,
        {
            fn $method(self, rhs: Self) -> Self {
                self.zip_in_place_packed(rhs, |value, mask, rhs| {
                    *value = mask.select($trait::$method(*value, rhs), Default::default())
                })
            }
        }
    };
}

binary_op!(Add, add, AddAssign, add_assign);
binary_op!(BitAnd, bitand, BitAndAssign, bitand_assign);
binary_op!(BitOr, bitor, BitOrAssign, bitor_assign);
binary_op!(Div, div, DivAssign, div_assign);
binary_op!(Mul, mul, MulAssign, mul_assign);
binary_op!(Rem, rem, RemAssign, rem_assign);
binary_op!(Shl, shl, ShlAssign, shl_assign);
binary_op!(Shr, shr, ShrAssign, shr_assign);
binary_op!(Sub, sub, SubAssign, sub_assign);
