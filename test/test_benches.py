import os
import pandas as pd
import pytest
from df_py import Bench

df_iris = (pd.read_csv(os.path.join(os.path.dirname(__file__), 'iris.csv'))
    .pipe(lambda df: pd.concat([df] * 50))
)

bench = (df_iris
    .pipe(lambda df: Bench({col: series.values for col, series in df.to_dict(orient='series').items()}))
)


@pytest.mark.benchmark(group='len')
def test_len_pandas(benchmark):
    assert benchmark(lambda: len(df_iris)) == 7500


@pytest.mark.benchmark(group='len')
def test_len_rust(benchmark):
    assert benchmark(bench.len) == 7500


@pytest.mark.benchmark(group='filter_float')
def test_filter_float_pandas(benchmark):
    assert benchmark(lambda: len(df_iris[lambda df: df.sepal_length > 5])) == 5900


@pytest.mark.benchmark(group='filter_float')
def test_filter_float_rust(benchmark):
    assert benchmark(bench.filter_float) == 5900


@pytest.mark.benchmark(group='filter_float')
def test_query_float_pandas(benchmark):
    assert benchmark(lambda: len(df_iris.query('sepal_length > 5'))) == 5900


@pytest.mark.benchmark(group='filter_two_floats')
def test_filter_two_floats_pandas(benchmark):
    assert benchmark(lambda: len(df_iris[lambda df: (df.sepal_width > 4) & (df.sepal_length > 5)])) == 150


@pytest.mark.benchmark(group='filter_two_floats')
def test_filter_two_floats_rust(benchmark):
    assert benchmark(bench.filter_two_floats) == 150


@pytest.mark.benchmark(group='filter_str')
def test_filter_str_pandas(benchmark):
    assert benchmark(lambda: len(df_iris[lambda df: df.species == 'setosa'])) == 2500


@pytest.mark.benchmark(group='filter_str')
def test_filter_str_rust(benchmark):
    assert benchmark(bench.filter_str) == 2500


@pytest.mark.benchmark(group='filter_str')
def test_query_str_pandas(benchmark):
    assert benchmark(lambda: len(df_iris.query('species == "setosa"'))) == 2500


@pytest.mark.benchmark(group='add_scalar')
def test_add_scalar_pandas(benchmark):
    benchmark(lambda: df_iris.sepal_length + 1)


@pytest.mark.benchmark(group='add_scalar')
def test_add_scalar_rust(benchmark):
    benchmark(bench.add_scalar)


@pytest.mark.benchmark(group='add_series')
def test_add_series_pandas(benchmark):
    benchmark(lambda: df_iris.sepal_length + df_iris.sepal_width)


@pytest.mark.benchmark(group='add_series')
def test_add_series_rust(benchmark):
    benchmark(bench.add_series)


@pytest.mark.benchmark(group='sum')
def test_sum_pandas(benchmark):
    assert int(benchmark(lambda: df_iris.sepal_length.sum())) == 43825


@pytest.mark.benchmark(group='sum')
def test_sum_rust(benchmark):
    assert int(benchmark(bench.sum)) == 43825


@pytest.mark.benchmark(group='any')
def test_any_pandas(benchmark):
    assert benchmark(lambda: (df_iris.sepal_length > 0).any())


@pytest.mark.benchmark(group='any')
def test_any_rust(benchmark):
    assert benchmark(bench.any)


@pytest.mark.benchmark(group='all')
def test_any_pandas(benchmark):
    assert benchmark(lambda: (df_iris.sepal_length > 0).all())


@pytest.mark.benchmark(group='all')
def test_any_rust(benchmark):
    assert benchmark(bench.all)


@pytest.mark.benchmark(group='where')
def test_where_pandas(benchmark):
    assert int(benchmark(lambda: df_iris.sepal_length.where(df_iris.species == "setosa").sum())) == 12515


@pytest.mark.benchmark(group='where')
def test_where_rust(benchmark):
    assert int(benchmark(bench.where_)) == 12515


def test_noop_rust(benchmark):
    benchmark(bench.noop)
