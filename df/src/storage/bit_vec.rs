use std::borrow::Cow;
use std::sync::Arc;

use bit_vec::BitVec;
use itertools::izip;

use crate::simd::Select;
use crate::storage::{SimdStorage, Storage};

#[derive(Clone, Debug, Default)]
pub struct BitVecStorage {
    data: BitVec,
    exists: BitVec,
}

impl BitVecStorage {
    pub fn new(data: BitVec, exists: BitVec) -> Self {
        BitVecStorage { data, exists }
    }

    pub fn into_inner(self) -> (BitVec, BitVec) {
        (self.data, self.exists)
    }

    pub fn as_vec_mask(&self) -> (&BitVec, &BitVec) {
        (&self.data, &self.exists)
    }

    pub fn as_vec_mask_mut(&mut self) -> (&mut BitVec, &mut BitVec) {
        (&mut self.data, &mut self.exists)
    }

    pub fn as_vec(&self, default: bool) -> Cow<'_, BitVec> {
        let (data, exists) = self.as_vec_mask();
        if default {
            if exists.none() {
                Cow::Borrowed(data)
            } else {
                let mut data = data.clone();
                data.union(exists);
                Cow::Owned(data)
            }
        } else {
            if exists.all() {
                Cow::Borrowed(data)
            } else {
                let mut data = data.clone();
                data.intersect(exists);
                Cow::Owned(data)
            }
        }
    }
}

fn as_ref(b: bool) -> &'static bool {
    if b {
        &true
    } else {
        &false
    }
}

impl Storage<bool> for Arc<BitVecStorage> {
    fn from_vec(data: Vec<bool>, exists: BitVec) -> Self {
        let data = data.into_iter().collect();
        Arc::new(BitVecStorage::new(data, exists))
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Option<&bool>> + '_> {
        let (data, exists) = self.as_vec_mask();
        Box::new(
            data.into_iter()
                .zip(exists)
                .map(|(item, exists)| if exists { Some(as_ref(item)) } else { None }),
        )
    }

    fn fold<A, F: FnMut(A, &bool) -> A>(&self, index_exists: &BitVec, initial: A, mut f: F) -> A {
        let (data, exists) = self.as_vec_mask();
        izip!(data, exists, index_exists).fold(
            initial,
            move |state, (item, exists, index_exists)| if exists && index_exists { f(state, &item) } else { state },
        )
    }

    fn get(&self, offset: usize) -> Option<&bool> {
        let (data, exists) = self.as_vec_mask();
        if exists.get(offset)? {
            Some(as_ref(data.get(offset)?))
        } else {
            None
        }
    }

    fn map_in_place<F: FnMut(&mut bool)>(mut self, index_exists: &BitVec, mut f: F) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        for (offset, (exists, index_exists)) in izip!(exists as &BitVec, index_exists).enumerate() {
            if exists && index_exists {
                let mut item = data.get(offset).unwrap();
                f(&mut item);
                data.set(offset, item);
            }
        }

        self
    }

    fn zip_in_place<F: FnMut(&mut bool, &bool)>(mut self, index_exists: &BitVec, other: Self, mut f: F) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let (other_data, other_exists) = other.as_vec_mask();
        exists.intersect(other_exists);

        for (offset, (exists, index_exists, other_item)) in
            izip!(exists as &BitVec, index_exists, other_data).enumerate()
        {
            if exists && index_exists {
                let mut item = data.get(offset).unwrap();
                f(&mut item, &other_item);
                data.set(offset, item);
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

        for (item, exists, index_exists, other_item, condition) in izip!(
            unsafe { data.storage_mut() },
            exists.blocks(),
            index_exists.blocks(),
            other_data.blocks(),
            condition.blocks()
        ) {
            *item = (exists & index_exists & condition).select(*item, other_item);
        }

        self
    }

    fn where_scalar(mut self, index_exists: &BitVec, condition: &BitVec, other: Option<&bool>) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        if let Some(&other) = other {
            let other = if other { !0 } else { 0 };

            for (exists, condition) in izip!(unsafe { exists.storage_mut().iter_mut() }, condition.blocks()) {
                *exists |= !condition;
            }

            for (item, exists, index_exists, condition) in izip!(
                unsafe { data.storage_mut().iter_mut() },
                exists.blocks(),
                index_exists.blocks(),
                condition.blocks()
            ) {
                *item = (exists & index_exists & condition).select(*item, other);
            }
        } else {
            exists.intersect(condition);
        }

        self
    }

    fn mask_scalar(mut self, index_exists: &BitVec, condition: &BitVec, other: Option<&bool>) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        if let Some(&other) = other {
            let other = if other { !0 } else { 0 };
            exists.union(condition);

            for (item, exists, index_exists, condition) in izip!(
                unsafe { data.storage_mut().iter_mut() },
                exists.blocks(),
                index_exists.blocks(),
                condition.blocks()
            ) {
                *item = (exists & index_exists & condition).select(other, *item);
            }
        } else {
            exists.difference(condition);
        }

        self
    }
}

impl SimdStorage<bool> for Arc<BitVecStorage> {
    type Packed = u32;
    type Mask = u32;

    fn iter_packed(&self) -> Box<dyn Iterator<Item = (u32, u32)> + '_> {
        let (data, exists) = self.as_vec_mask();
        Box::new(data.blocks().zip(exists.blocks()))
    }

    fn fold_packed<State, F: FnMut(State, u32, u32) -> State>(
        &self,
        index_exists: &BitVec<u32>,
        initial: State,
        mut f: F,
    ) -> State {
        let (data, exists) = self.as_vec_mask();
        let mut state = initial;
        for (item, exists, index_exists) in izip!(data.blocks(), exists.blocks(), index_exists.blocks()) {
            state = f(state, item, exists & index_exists);
        }

        state
    }

    fn try_fold_packed<State, Err, F: FnMut(State, u32, u32) -> Result<State, Err>>(
        &self,
        index_exists: &BitVec<u32>,
        initial: State,
        mut f: F,
    ) -> Result<State, Err> {
        let (data, exists) = self.as_vec_mask();
        let mut state = initial;
        for (item, exists, index_exists) in izip!(data.blocks(), exists.blocks(), index_exists.blocks()) {
            state = f(state, item, exists & index_exists)?;
        }

        Ok(state)
    }

    fn map_in_place_packed<F: FnMut(&mut u32, u32)>(mut self, index_exists: &BitVec, mut f: F) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        for (block, exists, index_exists) in
            izip!(unsafe { data.storage_mut() }, exists.blocks(), index_exists.blocks())
        {
            f(block, exists & index_exists);
        }

        self
    }

    fn zip_in_place_packed<F: FnMut(&mut u32, u32, u32)>(
        mut self,
        index_exists: &BitVec,
        other: Self,
        mut f: F,
    ) -> Self {
        let (data, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let (other_data, other_exists) = other.as_vec_mask();
        exists.intersect(other_exists);

        for (block, exists, index_exists, other_block) in izip!(
            unsafe { data.storage_mut() },
            exists.blocks(),
            index_exists.blocks(),
            other_data.blocks()
        ) {
            f(block, exists & index_exists, other_block);
        }

        self
    }
}
