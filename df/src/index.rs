use std::any::TypeId;
use std::collections::BTreeMap;
use std::iter::FromIterator;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;

use crate::RefEq;
use crate::series::Series;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Loc {
    String(String),
    TypeId(TypeId),
    Usize(usize),
}

#[derive(Clone, Debug, Default)]
struct IndexInner {
    data: Vec<Loc>,
    map: BTreeMap<Loc, usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Index {
    inner: Arc<IndexInner>,
    exists: Arc<BitVec>,
}

impl FromIterator<Loc> for Index {
    fn from_iter<T: IntoIterator<Item = Loc>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<Loc>>().into()
    }
}

impl From<Vec<Loc>> for Index {
    fn from(data: Vec<Loc>) -> Self {
        let mut map = BTreeMap::new();
        for (offset, loc) in data.iter().enumerate() {
            map.entry(loc.clone()).or_insert(offset);
        }

        let exists = BitVec::from_elem(data.len(), true);

        Index {
            inner: Arc::new(IndexInner { data, map }),
            exists: Arc::new(exists),
        }
    }
}

impl From<Range<usize>> for Index {
    fn from(range: Range<usize>) -> Self {
        range.map(Loc::Usize).collect()
    }
}

impl RefEq for Index {
    fn ref_eq(&self, other: &Self) -> bool {
        self.inner.ref_eq(&other.inner) && self.exists.ref_eq(&other.exists)
    }
}

impl Index {
    pub fn as_vec_mask(&self) -> (&Vec<Loc>, &BitVec) {
        (&self.inner.data, &self.exists)
    }

    pub fn insert(mut self, loc: Loc) -> (Index, usize) {
        let inner = Arc::make_mut(&mut self.inner);
        let exists = Arc::make_mut(&mut self.exists);
        let map = &mut inner.map;
        let data = &mut inner.data;
        let offset = *map.entry(loc.clone()).or_insert_with(move || {
            let offset = data.len();
            data.push(loc);
            exists.push(true);
            offset
        });

        (self, offset)
    }

    pub fn get(&self, loc: &Loc) -> Option<usize> {
        let &offset = self.inner.map.get(loc)?;
        if self.exists[offset] {
            Some(offset)
        } else {
            None
        }
    }

    pub fn union(mut self, other: &Index) -> Index {
        if self.ref_eq(other) {
            return self;
        }

        let inner = Arc::make_mut(&mut self.inner);
        let exists = Arc::make_mut(&mut self.exists);
        let map = &mut inner.map;
        let data = &mut inner.data;
        for loc in other.inner.data.iter() {
            map.entry(loc.clone()).or_insert_with(|| {
                let offset = data.len();
                data.push(loc.clone());
                exists.push(true);
                offset
            });
        }

        self
    }

    pub fn loc_range(&self, range: Range<usize>) -> Index {
        let range = Loc::Usize(range.start)..Loc::Usize(range.end);
        let mut exists = BitVec::from_elem(self.inner.data.len(), false);
        for (_, &offset) in self.inner.map.range(range) {
            exists.set(offset, true);
        }

        Index {
            inner: self.inner.clone(),
            exists: Arc::new(exists),
        }
    }

    pub fn filter(self, series: Series<bool>) -> Index {
        let series = series.reindex(self.clone());
        let (filter_index, filter_data) = series.into_inner();
        assert!(self.ref_eq(&filter_index));

        Index {
            inner: self.inner.clone(),
            exists: Arc::new(filter_data.as_vec(false).into_owned()),
        }
    }
}
