use crate::{VectorCmp, VectorEq};
use crate::series::Series;
use crate::storage::Element;

macro_rules! logical_op {
    (
        $trait: ident,
        (
            $( $method:ident ),*
        )
    ) => {
        impl<T> $trait<T, Series<bool>> for Series<T>
        where
            T: Element,
            for<'a> &'a T: $trait<&'a T, bool>,
        {
            $(
            fn $method(self, rhs: T) -> Series<bool> {
                $trait::$method(self, &rhs)
            }
            )*
        }

        impl<T> $trait<&T, Series<bool>> for Series<T>
        where
            T: Element + ?Sized,
            for<'a> &'a T: $trait<&'a T, bool>,
        {
            $(
            default fn $method(self, rhs: &T) -> Series<bool> {
                self.map(|value| $trait::$method(value, rhs))
            }
            )*
        }

        /*
        impl<T, S> $trait<&T, Series<bool>> for Series<T>
        where
            T: Element<Container = S> + ?Sized,
            for<'a> &'a T: $trait<&'a T, bool>, // impl overlap
            S: Storage<T> + SimdStorage<T>,
            S::Packed: for<'a> $trait<&'a T, u32>,
            S::Mask: Select<u32>,
        {
            $(
            fn $method(self, rhs: &T) -> Series<bool> {
                self.map_packed(|value, mask| mask.select($trait::$method(value, rhs), 0))
            }
            )*
        }
        */

        impl<T> $trait<Series<T>, Series<bool>> for Series<T>
        where
            T: Element + ?Sized,
            for<'a> &'a T: $trait<&'a T, bool>,
        {
            $(
            default fn $method(self, rhs: Self) -> Series<bool> {
                self.zip(rhs, |value, rhs| $trait::$method(value, rhs))
            }
            )*
        }

        /*
        impl<T, S> $trait<Series<T>, Series<bool>> for Series<T>
        where
            T: Element<Container = S> + ?Sized,
            for<'a> &'a T: $trait<&'a T, bool>, // impl overlap
            S: Storage<T> + SimdStorage<T>,
            S::Packed: $trait<S::Packed, u32>,
            S::Mask: Select<u32>,
        {
            $(
            fn $method(self, rhs: Self) -> Series<bool> {
                self.zip_packed(rhs, |value, mask, rhs| mask.select($trait::$method(value, rhs), 0))
            }
            )*
        }
        */
    };
}

logical_op!(VectorEq, (eq, ne));
logical_op!(VectorCmp, (lt, lte, gt, gte));
