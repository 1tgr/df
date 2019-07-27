use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::Arc;

use hlist::{Cons, Find, Nil};
use take_mut;

use crate::index::{Index, Loc};
use crate::series::Series;
use crate::storage::{AnyStorage, Element};

#[derive(Clone, Debug)]
pub struct AnySeries {
    index: Index,
    data: AnyStorage,
}

impl AnySeries {
    pub fn new(index: Index, data: AnyStorage) -> Self {
        AnySeries { index, data }
    }

    pub fn into_inner(self) -> (Index, AnyStorage) {
        (self.index, self.data)
    }

    pub fn from_series<T: Element + ?Sized>(series: Series<T>) -> Self {
        let (index, data) = series.into_inner();
        let data = T::into_any(data);
        AnySeries::new(index, data)
    }

    pub fn into_series<T: Element + ?Sized>(self) -> Option<Series<T>> {
        let (index, data) = self.into_inner();
        let data = T::from_any(data)?;
        Some(Series::new(index, data))
    }
}

pub trait Col {
    type Value: Element + ?Sized;
}

#[derive(Default)]
pub struct DataFrame<Cols> {
    columns: Index,
    index: Index,
    data: Arc<Vec<AnyStorage>>,
    _pd: PhantomData<Cols>,
}

impl<Cols> Clone for DataFrame<Cols> {
    fn clone(&self) -> Self {
        DataFrame {
            columns: self.columns.clone(),
            index: self.index.clone(),
            data: self.data.clone(),
            _pd: PhantomData,
        }
    }
}

impl<Cols> DataFrame<Cols> {
    pub fn len(&self) -> usize {
        let (_index_data, index_exists) = self.index.as_vec_mask();
        index_exists.iter().filter(|&b| b).count()
    }

    fn insert_loc<NewCols>(self, col: Loc, series: AnySeries) -> DataFrame<NewCols> {
        let DataFrame {
            columns,
            index,
            mut data,
            _pd,
        } = self;

        let (columns, offset) = columns.insert(col);
        assert_eq!(offset, data.len());

        let (series_index, series_data) = series.into_inner();
        let prev_series_index = series_index.clone();
        let prev_index = index.clone();
        let index = series_index.union(&prev_index);

        {
            let data = Arc::make_mut(&mut data);
            for prev_data in data.iter_mut() {
                take_mut::take(prev_data, |prev_data| prev_data.reindex(&prev_index, &index))
            }

            data.push(series_data.reindex(&prev_series_index, &index));
        }

        DataFrame {
            columns,
            index,
            data,
            _pd: PhantomData,
        }
    }

    fn get_loc(&self, col: &Loc) -> Option<AnySeries> {
        let offset = self.columns.get(col)?;
        let data = self.data.get(offset)?;
        Some(AnySeries::new(self.index.clone(), data.clone()))
    }

    pub fn insert(self, name: String, series: AnySeries) -> Self {
        self.insert_loc(Loc::String(name), series)
    }

    pub fn insert_with<F: FnOnce(&Self) -> AnySeries>(self, name: String, f: F) -> Self {
        let series = f(&self);
        self.insert(name, series)
    }

    pub fn assign<T: Col + 'static>(self, series: Series<T::Value>) -> DataFrame<Cons<T, Cols>> {
        let (index, data) = series.into_inner();
        let data = T::Value::into_any(data);
        self.insert_loc(Loc::TypeId(TypeId::of::<T>()), AnySeries::new(index, data))
    }

    pub fn assign_with<T: Col + 'static, F: FnOnce(&Self) -> Series<T::Value>>(self, f: F) -> DataFrame<Cons<T, Cols>> {
        let series = f(&self);
        self.assign::<T>(series)
    }

    pub fn col<T: Col + 'static, Index>(&self) -> Series<T::Value>
    where
        Cols: Find<T, Index>,
    {
        self.get_loc(&Loc::TypeId(TypeId::of::<T>()))
            .and_then(AnySeries::into_series)
            .unwrap()
    }

    pub fn get(&self, name: &str) -> Option<AnySeries> {
        self.get_loc(&Loc::String(name.to_string()))
    }

    pub fn filter(self, series: Series<bool>) -> Self {
        DataFrame {
            index: self.index.filter(series),
            ..self
        }
    }

    pub fn filter_with<F: FnOnce(&Self) -> Series<bool>>(self, f: F) -> Self {
        let series = f(&self);
        self.filter(series)
    }
}

pub fn empty() -> DataFrame<Nil> {
    DataFrame::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub struct Price;

    pub struct Price2;

    pub struct Price3;

    impl Col for Price {
        type Value = f64;
    }

    impl Col for Price2 {
        type Value = f64;
    }

    impl Col for Price3 {
        type Value = f64;
    }

    #[test]
    fn assign() {
        let df = empty()
            .assign::<Price>(vec![1.0, 2.0, 3.0].into())
            .assign_with::<Price2, _>(|df| df.col::<Price, _>() + 1.0)
            .assign_with::<Price3, _>(|df| df.col::<Price, _>() + df.col::<Price2, _>());

        assert_eq!(df.col::<Price, _>().to_vec(0.0), vec![1.0, 2.0, 3.0]);
        assert_eq!(df.col::<Price2, _>().to_vec(0.0), vec![2.0, 3.0, 4.0]);
        assert_eq!(df.col::<Price3, _>().to_vec(0.0), vec![3.0, 5.0, 7.0]);
    }
}
