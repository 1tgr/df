#![deny(warnings)]

use df::{AnySeries, Col, Cons, DataFrame, Nil, Series, VectorAny, VectorCmp, VectorEq, VectorSum, VectorWhere};
use pyo3::buffer::{Element, PyBuffer};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::{mem, slice};

struct SepalLength;
struct SepalWidth;
struct Species;

impl Col for SepalLength {
    type Value = f64;
}

impl Col for SepalWidth {
    type Value = f64;
}

impl Col for Species {
    type Value = str;
}

#[pyclass]
struct Bench {
    df: DataFrame<Cons<Species, Cons<SepalWidth, Cons<SepalLength, Nil>>>>,
}

fn as_slice<'a>(buffer: &'a PyBuffer, _py: Python<'a>) -> Option<&'a [PyObject]> {
    if mem::size_of::<PyObject>() == buffer.item_size()
        && (buffer.buf_ptr() as usize) % mem::align_of::<PyObject>() == 0 && buffer.is_c_contiguous()
    {
        unsafe {
            Some(slice::from_raw_parts(
                buffer.buf_ptr() as *mut PyObject,
                buffer.item_count(),
            ))
        }
    } else {
        None
    }
}

#[pymethods]
impl Bench {
    #[new]
    fn new(obj: &PyRawObject, df_dict: &PyDict) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let mut df = DataFrame::default();

        for (key, value) in df_dict.iter() {
            let key = key.cast_as::<PyString>()?.to_string()?;
            let buffer = PyBuffer::get(py, value)?;
            let format = buffer.format();

            let series = if f64::is_compatible_format(format) {
                AnySeries::from_series::<f64>(buffer.to_vec(py)?.into())
            } else if i64::is_compatible_format(format) {
                AnySeries::from_series::<i64>(buffer.to_vec(py)?.into())
            } else if format.to_bytes() == b"O" {
                let data = as_slice(&buffer, py)
                    .unwrap()
                    .iter()
                    .map(|obj| Ok(obj.cast_as::<PyString>(py)?.to_string()?.into_owned()))
                    .collect::<PyResult<Vec<String>>>()?;

                AnySeries::from_series::<str>(data.into())
            } else {
                panic!("buffer format {:?} not supported", format);
            };

            df = df.insert(key.into_owned(), series);
        }

        let df = df.assign_with::<SepalLength, _>(|df| df.get("sepal_length").unwrap().into_series().unwrap());
        let df = df.assign_with::<SepalWidth, _>(|df| df.get("sepal_width").unwrap().into_series().unwrap());
        let df = df.assign_with::<Species, _>(|df| df.get("species").unwrap().into_series().unwrap());

        obj.init(Bench { df });
        Ok(())
    }

    fn noop(&self) {}

    fn len(&self) -> usize {
        self.df.len()
    }

    fn filter_float(&self) -> usize {
        self.df
            .clone()
            .filter_with(|df| df.col::<SepalLength, _>().gt(5.0))
            .len()
    }

    fn filter_two_floats(&self) -> usize {
        self.df
            .clone()
            .filter_with(|df| df.col::<SepalWidth, _>().gt(4.0) & df.col::<SepalLength, _>().gt(5.0))
            .len()
    }

    fn filter_str(&self) -> usize {
        self.df
            .clone()
            .filter_with(|df| df.col::<Species, _>().eq("setosa"))
            .len()
    }

    fn add_scalar(&self) {
        let _: Series<f64> = self.df.col::<SepalLength, _>() + 1.0;
    }

    fn add_series(&self) {
        let _: Series<f64> = self.df.col::<SepalLength, _>() + self.df.col::<SepalWidth, _>();
    }

    fn sum(&self) -> f64 {
        self.df.col::<SepalLength, _>().sum()
    }

    fn any(&self) -> bool {
        self.df.col::<SepalLength, _>().gt(0.0).any()
    }

    fn all(&self) -> bool {
        self.df.col::<SepalLength, _>().gt(0.0).all()
    }

    fn where_(&self) -> f64 {
        self.df
            .col::<SepalLength, _>()
            .where_(self.df.col::<Species, _>().eq("setosa"))
            .sum()
    }
}

#[pymodule]
fn df_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Bench>()?;
    Ok(())
}
