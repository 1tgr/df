use std::f64;
use std::sync::Arc;

use packed_simd::*;

use crate::RefEq;
use crate::index::Index;
use crate::simd::{FromBitBlock, Splat};
use crate::storage::bit_vec::BitVecStorage;
use crate::storage::string::StringStorage;
use crate::storage::vec::VecStorage;

pub mod bit_vec;
pub mod string;
pub mod vec;

type BitVec = ::bit_vec::BitVec;

pub trait Storage<T: Element + ?Sized>: Sized {
    fn from_vec(data: Vec<T::Owned>, exists: BitVec) -> Self;
    fn len(&self) -> usize;
    fn iter(&self) -> Box<dyn Iterator<Item = Option<&T>> + '_>;

    fn fold<State, F: FnMut(State, &T) -> State>(&self, index_exists: &BitVec, initial: State, f: F) -> State;

    fn get(&self, offset: usize) -> Option<&T>;
    fn map_in_place<F: FnMut(&mut T::Owned)>(self, index_exists: &BitVec, f: F) -> Self;
    fn zip_in_place<F: FnMut(&mut T::Owned, &T)>(self, index_exists: &BitVec, other: Self, f: F) -> Self;
    fn where_(self, index_exists: &BitVec, condition: &BitVec, other: Self) -> Self;
    fn where_scalar(self, index_exists: &BitVec, condition: &BitVec, other: Option<&T>) -> Self;
    fn mask_scalar(self, index_exists: &BitVec, condition: &BitVec, other: Option<&T>) -> Self;

    fn reindex(self, prev_index: &Index, index: &Index) -> Self {
        if prev_index.ref_eq(index) {
            return self;
        }

        let (index, _index_exists) = index.as_vec_mask();
        let prev_data = self;
        let mut data = Vec::with_capacity(index.len());
        let mut exists = BitVec::with_capacity(index.len());
        for loc in index.iter() {
            if let Some(prev_offset) = prev_index.get(loc) {
                if let Some(item) = prev_data.get(prev_offset) {
                    data.push(item.to_owned());
                    exists.push(true);
                    continue;
                }
            }

            data.push(Default::default());
            exists.push(false);
        }

        Self::from_vec(data, exists)
    }
}

pub trait SimdStorage<T: Element + ?Sized> {
    type Packed: Copy;
    type Mask: Copy + FromBitBlock<u32>;

    fn iter_packed(&self) -> Box<dyn Iterator<Item = (Self::Packed, Self::Mask)> + '_>;

    fn fold_packed<State, F: FnMut(State, Self::Packed, Self::Mask) -> State>(
        &self,
        index_exists: &BitVec,
        initial: State,
        f: F,
    ) -> State;

    fn try_fold_packed<State, Err, F: FnMut(State, Self::Packed, Self::Mask) -> Result<State, Err>>(
        &self,
        index_exists: &BitVec,
        initial: State,
        f: F,
    ) -> Result<State, Err>;

    fn map_in_place_packed<F: FnMut(&mut Self::Packed, Self::Mask)>(self, index_exists: &BitVec, f: F) -> Self;

    fn zip_in_place_packed<F: FnMut(&mut Self::Packed, Self::Mask, Self::Packed)>(
        self,
        index_exists: &BitVec,
        other: Self,
        f: F,
    ) -> Self;
}

pub trait Element {
    type Container: Storage<Self>;
    type Owned: Clone + Default + Sized;

    fn to_owned(&self) -> Self::Owned;
    fn into_any(container: Self::Container) -> AnyStorage;
    fn from_any(any: AnyStorage) -> Option<Self::Container>;
}

macro_rules! storage {
    (
        ( $( $type:ident ),* ),
        ( $( $packed:ident ),* ),
        ( $( $mask:ident ),* ),
        ( $( $case:ident ),* ),
        ( $( $lanes:expr ),* )
    ) => {

#[derive(Clone, Debug)]
pub enum AnyStorage {
    Bool(Arc<BitVecStorage>),

    $(
    $case(Arc<VecStorage<$packed, $mask>>),
    )*

    Str(Arc<StringStorage>),
}

impl AnyStorage {
    pub fn reindex(self, prev_index: &Index, index: &Index) -> AnyStorage {
        use AnyStorage::*;

        match self {
            Bool(data) => Bool(data.reindex(prev_index, index)),

            $(
            $case(data) => $case(data.reindex(prev_index, index)),
            )*

            Str(data) => Str(data.reindex(prev_index, index)),
        }
    }
}

impl Element for bool {
    type Container = Arc<BitVecStorage>;
    type Owned = Self;

    fn to_owned(&self) -> Self {
        *self
    }

    fn into_any(container: Self::Container) -> AnyStorage {
        AnyStorage::Bool(container)
    }

    fn from_any(any: AnyStorage) -> Option<Self::Container> {
        if let AnyStorage::Bool(container) = any {
            Some(container)
        } else {
            None
        }
    }
}

$(

impl Splat<$type> for $packed {
    fn splat(a: $type) -> Self {
        $packed::splat(a)
    }
}

impl Element for $type {
    type Container = Arc<VecStorage<$packed, $mask>>;
    type Owned = Self;

    fn to_owned(&self) -> Self {
        *self
    }

    fn into_any(container: Self::Container) -> AnyStorage {
        AnyStorage::$case(container)
    }

    fn from_any(any: AnyStorage) -> Option<Self::Container> {
        if let AnyStorage::$case(container) = any {
            Some(container)
        } else {
            None
        }
    }
}
)*

impl Element for str {
    type Container = Arc<StringStorage>;
    type Owned = String;

    fn to_owned(&self) -> String {
        self.to_string()
    }

    fn into_any(container: Self::Container) -> AnyStorage {
        AnyStorage::Str(container)
    }

    fn from_any(any: AnyStorage) -> Option<Self::Container> {
        if let AnyStorage::Str(container) = any {
            Some(container)
        } else {
            None
        }
    }
}

    }
}

storage!(
    (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64),
    (i8x4, i16x4, i32x4, i64x4, u8x4, u16x4, u32x4, u64x4, f32x4, f64x4),
    (m8x4, m16x4, m32x4, m64x4, m8x4, m16x4, m32x4, m64x4, m32x4, m64x4),
    (I8, I16, I32, I64, U8, U16, U32, U64, F32, F64),
    (4, 4, 4, 4, 4,46, 4, 4, 4, 4)
);
