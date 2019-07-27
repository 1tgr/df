use std::iter;
use std::ops::Range;
use std::str;
use std::sync::Arc;

use bit_vec::BitVec;
use itertools::izip;

use crate::simd::Select;
use crate::storage::Storage;

#[derive(Clone, Debug, Default)]
pub struct StringStorage {
    ends: Vec<usize>,
    bytes: Vec<u8>,
    exists: BitVec,
}

impl StringStorage {
    pub fn new(ends: Vec<usize>, bytes: Vec<u8>, exists: BitVec) -> Self {
        StringStorage { ends, bytes, exists }
    }

    pub fn as_vec_mask(&self) -> (&Vec<usize>, &Vec<u8>, &BitVec) {
        (&self.ends, &self.bytes, &self.exists)
    }

    pub fn as_vec_mask_mut(&mut self) -> (&mut Vec<usize>, &mut Vec<u8>, &mut BitVec) {
        (&mut self.ends, &mut self.bytes, &mut self.exists)
    }
}

fn get_str<'a>(ends: &'a Vec<usize>, bytes: &'a Vec<u8>, offset: usize) -> (Range<usize>, &'a str) {
    let start = if offset == 0 { 0 } else { ends[offset - 1] };
    let end = ends[offset];
    (start..end, unsafe { str::from_utf8_unchecked(&bytes[start..end]) })
}

fn process<F: FnMut(&mut String)>(
    buffer: &mut String,
    ends: &mut Vec<usize>,
    bytes: &mut Vec<u8>,
    offset: usize,
    mut f: F,
) {
    let prev_range = {
        let (prev_range, s) = get_str(ends, bytes, offset);
        buffer.clear();
        buffer.push_str(s);
        prev_range
    };

    let prev_len = buffer.len();
    f(buffer);

    if let Some(delta) = buffer.len().checked_sub(prev_len) {
        for end in &mut ends[offset..] {
            *end += delta;
        }
    } else if let Some(delta) = prev_len.checked_sub(buffer.len()) {
        for end in &mut ends[offset..] {
            *end -= delta;
        }
    }

    bytes.splice(prev_range, buffer.as_bytes().iter().copied());
}

impl Storage<str> for Arc<StringStorage> {
    fn from_vec(data: Vec<String>, exists: BitVec) -> Self {
        let mut ends = Vec::with_capacity(data.len());
        let mut bytes = Vec::new();
        for s in data {
            bytes.extend_from_slice(s.as_bytes());
            ends.push(bytes.len());
        }

        Arc::new(StringStorage::new(ends, bytes, exists))
    }

    fn len(&self) -> usize {
        self.ends.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Option<&str>> + '_> {
        let (ends, bytes, exists) = self.as_vec_mask();
        let starts = iter::once(&0).chain(ends);
        Box::new(izip!(starts, ends, exists).map(move |(&start, &end, exists)| {
            if exists {
                Some(unsafe { str::from_utf8_unchecked(&bytes[start..end]) })
            } else {
                None
            }
        }))
    }

    fn fold<A, F: FnMut(A, &str) -> A>(&self, _index_exists: &BitVec, _initial: A, _f: F) -> A {
        unimplemented!()
    }

    fn get(&self, offset: usize) -> Option<&str> {
        let (ends, bytes, exists) = self.as_vec_mask();
        if exists.get(offset)? {
            let (_, s) = get_str(ends, bytes, offset);
            Some(s)
        } else {
            None
        }
    }

    fn map_in_place<F: FnMut(&mut String)>(mut self, index_exists: &BitVec, mut f: F) -> Self {
        let (ends, bytes, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let mut buffer = String::new();

        for (offset, (exists, index_exists)) in izip!(exists as &BitVec, index_exists).enumerate() {
            if exists && index_exists {
                process(&mut buffer, ends, bytes, offset, &mut f);
            }
        }

        self
    }

    fn zip_in_place<F: FnMut(&mut String, &str)>(mut self, index_exists: &BitVec, other: Self, mut f: F) -> Self {
        let (ends, bytes, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let (other_ends, other_bytes, other_exists) = other.as_vec_mask();
        exists.intersect(other_exists);

        let mut buffer = String::new();

        for (offset, (exists, index_exists)) in izip!(exists as &BitVec, index_exists).enumerate() {
            if exists && index_exists {
                let (_, other) = get_str(other_ends, other_bytes, offset);
                process(&mut buffer, ends, bytes, offset, |buffer| f(buffer, other));
            }
        }

        self
    }

    fn where_(mut self, index_exists: &BitVec, condition: &BitVec, other: Self) -> Self {
        let (ends, bytes, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();
        let (other_ends, other_bytes, other_exists) = other.as_vec_mask();

        for (exists, other_exists, condition) in izip!(
            unsafe { exists.storage_mut().iter_mut() },
            other_exists.blocks(),
            condition.blocks()
        ) {
            *exists = condition.select(*exists, other_exists);
        }

        let mut buffer = String::new();

        for (offset, (exists, index_exists, condition)) in izip!(exists as &BitVec, index_exists, condition).enumerate()
        {
            if exists && index_exists && !condition {
                let (_, other) = get_str(other_ends, other_bytes, offset);
                process(&mut buffer, ends, bytes, offset, |buffer| {
                    buffer.clear();
                    buffer.push_str(other);
                });
            }
        }

        self
    }

    fn where_scalar(mut self, index_exists: &BitVec, condition: &BitVec, other: Option<&str>) -> Self {
        let (ends, bytes, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        if let Some(other) = other {
            for (exists, condition) in izip!(unsafe { exists.storage_mut().iter_mut() }, condition.blocks()) {
                *exists |= !condition;
            }

            let mut buffer = String::new();

            for (offset, (exists, index_exists, condition)) in
                izip!(exists as &BitVec, index_exists, condition).enumerate()
            {
                if exists && index_exists && !condition {
                    process(&mut buffer, ends, bytes, offset, |buffer| {
                        buffer.clear();
                        buffer.push_str(other);
                    });
                }
            }
        } else {
            exists.intersect(condition);
        }

        self
    }

    fn mask_scalar(mut self, index_exists: &BitVec, condition: &BitVec, other: Option<&str>) -> Self {
        let (ends, bytes, exists) = Arc::make_mut(&mut self).as_vec_mask_mut();

        if let Some(other) = other {
            exists.union(condition);

            let mut buffer = String::new();

            for (offset, (exists, index_exists, condition)) in
                izip!(exists as &BitVec, index_exists, condition).enumerate()
            {
                if exists && index_exists && condition {
                    process(&mut buffer, ends, bytes, offset, |buffer| {
                        buffer.clear();
                        buffer.push_str(other);
                    });
                }
            }
        } else {
            exists.difference(condition);
        }

        self
    }
}
