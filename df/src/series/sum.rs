use std::ops::Add;

use packed_simd::{Simd, SimdArray, m64x4};

use crate::{Element, Series, Storage, VectorSum};
use crate::storage::SimdStorage;

impl<T> VectorSum<T::Owned> for &Series<T>
where
    T: Element + ?Sized,
    T::Owned: for<'a> Add<&'a T, Output = T::Owned>,
{
    default fn sum(self) -> T::Owned {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data
            .fold(index_exists, T::Owned::default(), |sum, item| sum + item)
    }
}

impl<T, S, A> VectorSum<T::Owned> for &Series<T>
where
    T: Element<Container = S> + ?Sized,
    T::Owned: for<'a> Add<&'a T, Output = T::Owned>, // impl overlap
    S: Storage<T> + SimdStorage<T, Packed = Simd<A>, Mask = m64x4>,
    A: Copy + SimdArray<NT = [u32; 4]>,
    Simd<A>: Default + Add<Simd<A>, Output = Simd<A>> + VectorSum<T::Owned>,
{
    fn sum(self) -> T::Owned {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data
            .fold_packed(index_exists, Simd::<A>::default(), |sum, item, mask| {
                sum + mask.select(item, Default::default())
            })
            .sum()
    }
}
