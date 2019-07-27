use num_traits::{Float, PrimInt, Signed};

use crate::series::{Element, Series};

macro_rules! bool_num_ops {
    ($trait: ident, $( $method:ident ),*) => {
        impl<T> Series<T> where T: Copy + Clone + Default + Element<Owned=T> + $trait {
            $(
            pub fn $method(self) -> Series<bool> {
                self.map(|value| value.$method())
            }
            )*
        }
    };
}

macro_rules! u32_num_ops {
    ($trait: ident, $( $method:ident ),*) => {
        impl<T> Series<T> where T: Copy + Clone + Default + Element<Owned=T> + $trait {
            $(
            pub fn $method(self) -> Series<u32> {
                self.map(|value| value.$method())
            }
            )*
        }
    };
}

macro_rules! self_num_ops {
    ($trait: ident, $( $method:ident ),* ) => {
        impl<T> Series<T> where T: Copy + Clone + Default + Element<Owned=T> + $trait {
            $(
            pub fn $method(self) -> Self {
                self.map_in_place(|value| *value = value.$method())
            }
            )*
        }
    };
}

macro_rules! self2_num_ops {
    ($trait: ident, $rhs:ident, $( $method:ident ),* ) => {
        impl<T> Series<T> where T: Copy + Clone + Default + Element<Owned=T> + $trait {
            $(
            pub fn $method(self, n: $rhs) -> Self {
                self.map_in_place(|value| *value = value.$method(n))
            }
            )*
        }
    };
}

bool_num_ops!(Signed, is_positive, is_negative);

self_num_ops!(Signed, abs, signum);

bool_num_ops!(
    Float,
    is_finite,
    is_infinite,
    is_nan,
    is_normal,
    is_sign_negative,
    is_sign_positive
);

self_num_ops!(
    Float,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cbrt,
    ceil,
    cos,
    cosh,
    exp,
    exp2,
    exp_m1,
    floor,
    fract,
    ln,
    ln_1p,
    log10,
    log2,
    recip,
    round,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    to_degrees,
    to_radians,
    trunc
);

self_num_ops!(PrimInt, swap_bytes, to_be, to_le);

u32_num_ops!(PrimInt, count_ones, count_zeros, leading_zeros, trailing_zeros);

self2_num_ops!(Float, i32, powi);

self2_num_ops!(Float, T, abs_sub, atan2, hypot, log, max, min, powf);

self2_num_ops!(
    PrimInt,
    u32,
    pow,
    rotate_left,
    rotate_right,
    signed_shl,
    signed_shr,
    unsigned_shl,
    unsigned_shr
);
