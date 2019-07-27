use std::iter::FromIterator;
use std::ops::*;

use bit_vec::BitVec;
use itertools::izip;
use packed_simd::m64x4;

use crate::RefEq;
use crate::index::{Index, Loc};
use crate::simd::IterMasks;
use crate::storage::{Element, SimdStorage, Storage};

mod any;
mod binary_ops;
mod logical_ops;
mod num_ops;
mod sum;
mod unary_ops;
mod where_;

#[cfg(test)]
mod tests;

#[derive(Debug, Default)]
pub struct Series<T: Element + ?Sized> {
    index: Index,
    data: T::Container,
}

impl<T> Clone for Series<T>
where
    T: Element + ?Sized,
    T::Container: Clone,
{
    fn clone(&self) -> Self {
        Series::new(self.index.clone(), self.data.clone())
    }
}

impl<T> Series<T>
where
    T: Element + ?Sized,
{
    pub fn new(index: Index, data: T::Container) -> Self {
        Series { index, data }
    }

    pub fn into_inner(self) -> (Index, T::Container) {
        (self.index, self.data)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, offset: usize) -> Option<&T> {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        if index_exists.get(offset).unwrap() {
            self.data.get(offset)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Loc, Option<&T>)> {
        let (index_data, index_exists) = self.index.as_vec_mask();
        index_data
            .into_iter()
            .zip(self.data.iter())
            .zip(index_exists)
            .filter_map(|(pair, index_exists)| if index_exists { Some(pair) } else { None })
    }

    pub fn fold<State, F: FnMut(State, &T) -> State>(&self, initial: State, f: F) -> State {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data.fold(index_exists, initial, f)
    }

    pub fn loc(&self, loc: &Loc) -> Option<&T> {
        let offset = self.index.get(loc)?;
        self.get(offset)
    }

    pub fn to_vec(&self, default: T::Owned) -> Vec<T::Owned> {
        self.iter()
            .map(|(_, item)| item.map(T::to_owned).unwrap_or_else(|| default.clone()))
            .collect()
    }

    pub fn to_option_vec(&self) -> Vec<Option<T::Owned>> {
        self.iter().map(|(_, item)| item.map(T::to_owned)).collect()
    }

    pub fn to_index_vec(&self, default: T::Owned) -> Vec<(Loc, T::Owned)> {
        self.iter()
            .map(|(loc, item)| (loc.clone(), item.map(T::to_owned).unwrap_or_else(|| default.clone())))
            .collect()
    }

    pub fn to_index_option_vec(&self) -> Vec<(Loc, Option<T::Owned>)> {
        self.iter()
            .map(|(loc, item)| (loc.clone(), item.map(T::to_owned)))
            .collect()
    }

    pub fn into_aligned<U>(self, other: Series<U>) -> (Index, T::Container, U::Container)
    where
        U: Element + ?Sized,
    {
        let (prev_index, data) = self.into_inner();
        let (other_index, other_data) = other.into_inner();

        if prev_index.ref_eq(&other_index) {
            return (prev_index, data, other_data);
        }

        let index = prev_index.clone().union(&other_index);
        let data = data.reindex(&prev_index, &index);
        let other_data = other_data.reindex(&other_index, &index);
        (index, data, other_data)
    }

    pub fn into_aligned3<U, V>(
        self,
        other1: Series<U>,
        other2: Series<V>,
    ) -> (Index, T::Container, U::Container, V::Container)
    where
        U: Element + ?Sized,
        V: Element + ?Sized,
    {
        let (prev_index, data) = self.into_inner();
        let (other1_index, other1_data) = other1.into_inner();
        let (other2_index, other2_data) = other2.into_inner();

        if prev_index.ref_eq(&other1_index) && prev_index.ref_eq(&other2_index) {
            return (prev_index, data, other1_data, other2_data);
        }

        let index = prev_index.clone().union(&other1_index).union(&other2_index);
        let data = data.reindex(&prev_index, &index);
        let other1_data = other1_data.reindex(&other1_index, &index);
        let other2_data = other2_data.reindex(&other2_index, &index);
        (index, data, other1_data, other2_data)
    }

    pub fn align<U>(self, other: Series<U>) -> (Self, Series<U>)
    where
        U: Element + ?Sized,
    {
        let (index, data, other_data) = self.into_aligned(other);
        (Series::new(index.clone(), data), Series::new(index, other_data))
    }

    pub fn align2<U, V>(self, other1: Series<U>, other2: Series<V>) -> (Self, Series<U>, Series<V>)
    where
        U: Element + ?Sized,
        V: Element + ?Sized,
    {
        let (index, data, other1_data, other2_data) = self.into_aligned3(other1, other2);
        (
            Series::new(index.clone(), data),
            Series::new(index.clone(), other1_data),
            Series::new(index.clone(), other2_data),
        )
    }

    pub fn reindex(self, index: Index) -> Self {
        let (prev_index, data) = self.into_inner();
        let data = data.reindex(&prev_index, &index);
        Series::new(index, data)
    }

    pub fn loc_range(self, range: Range<usize>) -> Self {
        let (index, data) = self.into_inner();
        Series::new(index.loc_range(range), data)
    }

    pub fn map_in_place<F: FnMut(&mut T::Owned)>(self, f: F) -> Self {
        let (index, data) = self.into_inner();
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.map_in_place(index_exists, f);
        Series::new(index, data)
    }

    pub fn zip_in_place<F: FnMut(&mut T::Owned, &T)>(self, other: Series<T>, f: F) -> Self {
        let (index, data, other_data) = self.into_aligned(other);
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.zip_in_place(index_exists, other_data, f);
        Series::new(index, data)
    }

    pub fn map<U: Element + ?Sized, F: FnMut(&T) -> U::Owned>(self, mut f: F) -> Series<U> {
        let (index, prev_data) = self.into_inner();
        let (_, index_exists) = index.as_vec_mask();
        assert_eq!(prev_data.len(), index_exists.len());

        let mut data = Vec::with_capacity(prev_data.len());
        let mut exists = BitVec::with_capacity(prev_data.len());
        for (item, index_exists) in prev_data.iter().zip(index_exists) {
            if index_exists {
                if let Some(item) = item {
                    data.push(f(item));
                    exists.push(true);
                    continue;
                }
            }

            data.push(Default::default());
            exists.push(false);
        }

        let data = U::Container::from_vec(data, exists);
        Series::new(index, data)
    }

    pub fn zip<U: Element + ?Sized, F: FnMut(&T, &T) -> U::Owned>(self, other: Self, mut f: F) -> Series<U> {
        let (index, prev_data, other_data) = self.into_aligned(other);
        let (_, index_exists) = index.as_vec_mask();
        assert_eq!(prev_data.len(), index_exists.len());
        assert_eq!(prev_data.len(), other_data.len());

        let mut data = Vec::with_capacity(prev_data.len());
        let mut exists = BitVec::with_capacity(prev_data.len());
        for (item, other_item, index_exists) in izip!(prev_data.iter(), other_data.iter(), index_exists) {
            if index_exists {
                if let Some(item) = item {
                    if let Some(other_item) = other_item {
                        data.push(f(item, other_item));
                        exists.push(true);
                        continue;
                    }
                }
            }

            data.push(Default::default());
            exists.push(false);
        }

        let data = U::Container::from_vec(data, exists);
        Series::new(index, data)
    }

    pub fn filter(self, other: Series<bool>) -> Self {
        let (index, data) = self.into_inner();
        Series::new(index.filter(other), data)
    }
}

impl<T, S> Series<T>
where
    T: Element<Container = S> + ?Sized,
    S: Storage<T> + SimdStorage<T>,
{
    pub fn map_in_place_packed<F>(self, f: F) -> Self
    where
        F: FnMut(&mut S::Packed, S::Mask),
    {
        let (index, data) = self.into_inner();
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.map_in_place_packed(index_exists, f);
        Series::new(index, data)
    }

    pub fn zip_in_place_packed<F>(self, other: Series<T>, f: F) -> Self
    where
        F: FnMut(&mut S::Packed, S::Mask, S::Packed),
    {
        let (index, data, other_data) = self.into_aligned(other);
        let (_index_data, index_exists) = index.as_vec_mask();
        let data = data.zip_in_place_packed(index_exists, other_data, f);
        Series::new(index, data)
    }
}

impl<T, S> Series<T>
where
    T: Element<Container = S> + ?Sized,
    S: Storage<T> + SimdStorage<T, Mask = m64x4>,
{
    pub fn iter_packed<'a>(&'a self) -> impl Iterator<Item = (S::Packed, S::Mask)> + 'a
    where
        S: 'a,
    {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data
            .iter_packed()
            .zip(index_exists.blocks().masks())
            .map(|((data, mask), index_mask)| (data, mask & index_mask))
    }

    pub fn fold_packed<State, F: FnMut(State, S::Packed, S::Mask) -> State>(&self, initial: State, f: F) -> State {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data.fold_packed(index_exists, initial, f)
    }

    pub fn try_fold_packed<State, Err, F: FnMut(State, S::Packed, S::Mask) -> Result<State, Err>>(
        &self,
        initial: State,
        f: F,
    ) -> Result<State, Err> {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        self.data.try_fold_packed(index_exists, initial, f)
    }

    pub fn map_packed<U, SU, F>(self, _f: F) -> Series<U>
    where
        U: Element<Container = SU> + ?Sized,
        SU: Storage<U> + SimdStorage<U>,
        F: FnMut(S::Packed, S::Mask) -> SU::Packed,
    {
        unimplemented!()
    }

    pub fn zip_packed<U, SU, F>(self, _other: Series<T>, _f: F) -> Series<U>
    where
        U: Element<Container = SU> + ?Sized,
        SU: Storage<U> + SimdStorage<U>,
        F: FnMut(S::Packed, S::Mask, S::Packed) -> SU::Packed,
    {
        unimplemented!()
    }
}

impl<T> Series<T>
where
    T: Element + ?Sized,
    T::Container: Clone,
{
    pub fn filter_with<F: FnOnce(Series<T>) -> Series<bool>>(self, f: F) -> Self {
        self.clone().filter(f(self))
    }
}

impl<T> From<Vec<T::Owned>> for Series<T>
where
    T: Element + ?Sized,
{
    fn from(data: Vec<T::Owned>) -> Self {
        let index = Index::from(0..data.len());
        let (index_data, _index_exists) = index.as_vec_mask();
        assert_eq!(index_data.len(), data.len());

        let exists = BitVec::from_elem(data.len(), true);
        Series::new(index, T::Container::from_vec(data, exists))
    }
}

impl<T> FromIterator<(Loc, Option<T::Owned>)> for Series<T>
where
    T: Element + ?Sized,
{
    fn from_iter<I: IntoIterator<Item = (Loc, Option<T::Owned>)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (capacity, _) = iter.size_hint();
        let mut index = Vec::with_capacity(capacity);
        let mut data = Vec::with_capacity(capacity);
        let mut exists = BitVec::with_capacity(capacity);
        for (loc, item) in iter {
            index.push(loc);
            if let Some(item) = item {
                data.push(item);
                exists.push(true);
            } else {
                data.push(Default::default());
                exists.push(false);
            }
        }

        Series::new(index.into(), T::Container::from_vec(data, exists))
    }
}
