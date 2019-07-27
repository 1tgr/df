use std::marker::PhantomData;
use std::sync::Arc;

use bit_vec::BitVec;
use itertools::izip;
use packed_simd::{Simd, SimdArray};

use crate::simd::{AsFlat, AsFlatMut, FromBitBlock, IterMasks, Select, Splat, ToPacked};
use crate::storage::{Element, SimdStorage, Storage};
use std::ops::BitAnd;

#[derive(Clone, Debug, Default)]
pub struct VecStorage<T, M> {
    data: Vec<T>,
    exists: BitVec,
    _pd: PhantomData<M>,
}

impl<T, M> VecStorage<T, M> {
    pub fn new(data: Vec<T>, exists: BitVec) -> Self {
        VecStorage {
            data,
            exists,
            _pd: PhantomData,
        }
    }
    pub fn as_vec_mask(&self) -> (&[T], &BitVec) {
        (&self.data, &self.exists)
    }

    pub fn as_vec_mask_mut(&mut self) -> (&mut Vec<T>, &mut BitVec) {
        (&mut self.data, &mut self.exists)
    }
}

impl<A, M> VecStorage<Simd<A>, M>
where
    A: SimdArray,
{
    pub fn as_flat_vec_mask(&self) -> (&[A::T], &BitVec) {
        (&self.data.as_flat()[0..self.exists.len()], &self.exists)
    }
}

impl<A, MA> Storage<A::T> for Arc<VecStorage<Simd<A>, Simd<MA>>>
where
    A: AsMut<[<A as SimdArray>::T]> + Clone + Copy + Default + Into<Simd<A>> + SimdArray,
    A::T: Clone + Copy + Default + Element<Owned = A::T>,
    Simd<A>: Splat<A::T>,
    MA: Copy + SimdArray,
    Simd<MA>: BitAnd<Simd<MA>, Output = Simd<MA>> + FromBitBlock<u32> + Select<Simd<A>>,
{
    fn from_vec(data: Vec<A::T>, exists: BitVec) -> Self {
        let data = ToPacked::<A>::to_packed(&data[..]);
        Arc::new(VecStorage::new(data, exists))
    }

    fn len(&self) -> usize {
        self.exists.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Option<&A::T>> + '_> {
        let (data, exists) = self.as_flat_vec_mask();
        Box::new(
            data.into_iter()
                .zip(exists)
                .map(|(item, exists)| if exists { Some(item) } else { None }),
        )
    }

    fn fold<B, F: FnMut(B, &A::T) -> B>(&self, index_exists: &BitVec, initial: B, mut f: F) -> B {
        let (data, exists) = self.as_flat_vec_mask();
        izip!(data, exists, index_exists).fold(
            initial,
            move |state, (item, exists, index_exists)| if exists && index_exists { f(state, item) } else { state },
        )
    }

    fn get(&self, offset: usize) -> Option<&A::T> {
        let (data, exists) = self.as_flat_vec_mask();
        if exists.get(offset)? {
            Some(&data[offset])
        } else {
            None
        }
    }

    fn map_in_place<F: FnMut(&mut A::T)>(mut self, index_exists: &BitVec, mut f: F) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        for (item, exists, index_exists) in izip!(data.as_flat_mut(), exists as &_, index_exists) {
            if exists && index_exists {
                f(item);
            }
        }

        self
    }

    fn zip_in_place<F: FnMut(&mut A::T, &A::T)>(mut self, index_exists: &BitVec, other: Self, mut f: F) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let (other_data, other_exists) = other.as_flat_vec_mask();
        exists.intersect(other_exists);

        for (item, exists, index_exists, other_item) in
            izip!(data.as_flat_mut(), exists as &_, index_exists, other_data)
        {
            if exists && index_exists {
                f(item, other_item);
            }
        }

        self
    }

    fn where_(mut self, index_exists: &BitVec, condition: &BitVec, other: Self) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let (other_data, other_exists) = other.as_vec_mask();

        for (exists, other_exists, condition) in izip!(
            unsafe { exists.storage_mut().iter_mut() },
            other_exists.blocks(),
            condition.blocks()
        ) {
            *exists = condition.select(*exists, other_exists);
        }

        for (item, exists, index_exists, &other_item, condition) in izip!(
            data,
            exists.blocks().masks(),
            index_exists.blocks().masks(),
            other_data,
            condition.blocks().masks()
        ) {
            *item = (exists & index_exists & condition).select(*item, other_item);
        }

        self
    }

    fn where_scalar(mut self, index_exists: &BitVec, condition: &BitVec, other: Option<&A::T>) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        if let Some(&other) = other {
            let other = Simd::<A>::splat(other);

            for (exists, condition) in izip!(unsafe { exists.storage_mut().iter_mut() }, condition.blocks()) {
                *exists |= !condition;
            }

            for (item, exists, index_exists, condition) in izip!(
                data,
                exists.blocks().masks(),
                index_exists.blocks().masks(),
                condition.blocks().masks()
            ) {
                *item = (exists & index_exists & condition).select(*item, other);
            }
        } else {
            exists.intersect(condition);
        }

        self
    }

    fn mask_scalar(mut self, index_exists: &BitVec, condition: &BitVec, other: Option<&A::T>) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        if let Some(&other) = other {
            let other = Simd::<A>::splat(other);
            exists.union(condition);

            for (item, exists, index_exists, condition) in izip!(
                data,
                exists.blocks().masks(),
                index_exists.blocks().masks(),
                condition.blocks().masks()
            ) {
                *item = (exists & index_exists & condition).select(other, *item);
            }
        } else {
            exists.difference(condition);
        }

        self
    }
}

impl<A, MA> SimdStorage<A::T> for Arc<VecStorage<Simd<A>, Simd<MA>>>
where
    A: Copy + SimdArray,
    A::T: Clone + Default + Element<Owned = A::T>,
    MA: Copy + SimdArray,
    Simd<MA>: BitAnd<Simd<MA>, Output = Simd<MA>> + FromBitBlock<u32> + Select<Simd<A>>,
{
    type Packed = Simd<A>;
    type Mask = Simd<MA>;

    fn iter_packed<'a>(&'a self) -> Box<dyn Iterator<Item = (Simd<A>, Simd<MA>)> + 'a> {
        let (data, exists) = self.as_vec_mask();
        Box::new(
            data.into_iter()
                .zip(exists.blocks().masks())
                .map(|(&item, exists)| (item, exists)),
        )
    }

    fn fold_packed<State, F: FnMut(State, Self::Packed, Self::Mask) -> State>(
        &self,
        index_exists: &BitVec<u32>,
        initial: State,
        mut f: F,
    ) -> State {
        let (data, exists) = self.as_vec_mask();
        let mut state = initial;
        for (&item, exists, index_exists) in izip!(data, exists.blocks().masks(), index_exists.blocks().masks()) {
            state = f(state, item, exists & index_exists);
        }

        state
    }

    fn try_fold_packed<State, Err, F: FnMut(State, Self::Packed, Self::Mask) -> Result<State, Err>>(
        &self,
        index_exists: &BitVec<u32>,
        initial: State,
        mut f: F,
    ) -> Result<State, Err> {
        let (data, exists) = self.as_vec_mask();
        let mut state = initial;
        for (&item, exists, index_exists) in izip!(data, exists.blocks().masks(), index_exists.blocks().masks()) {
            state = f(state, item, exists & index_exists)?;
        }

        Ok(state)
    }

    fn map_in_place_packed<F: FnMut(&mut Simd<A>, Simd<MA>)>(mut self, index_exists: &BitVec, mut f: F) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        for (item, exists, index_exists) in izip!(data, exists.blocks().masks(), index_exists.blocks().masks()) {
            f(item, exists & index_exists);
        }

        self
    }

    fn zip_in_place_packed<F: FnMut(&mut Simd<A>, Simd<MA>, Simd<A>)>(
        mut self,
        index_exists: &BitVec,
        other: Self,
        mut f: F,
    ) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let (other_data, other_exists) = other.as_vec_mask();
        exists.intersect(other_exists);

        for (item, exists, index_exists, other_item) in
            izip!(data, exists.blocks().masks(), index_exists.blocks().masks(), other_data)
        {
            f(item, exists & index_exists, *other_item);
        }

        self
    }
}
