#![deny(warnings)]
use std::iter::FromIterator;

use criterion::{BatchSize, Criterion};
use criterion::{black_box, criterion_group, criterion_main};
use df::{Series, VectorAny, VectorCmp, VectorSum, VectorWhereOr};

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = Vec::from_iter((0..1000).map(|n| n as f64));

    {
        let mut c = c.benchmark_group("add scalar");
        c.bench_function("no SIMD", |b| {
            b.iter_batched(
                || Series::<f64>::from(data.clone()),
                |series: Series<f64>| black_box(series.map_in_place(|f| *f += 1.0)),
                BatchSize::SmallInput,
            )
        });

        c.bench_function("SIMD", |b| {
            b.iter_batched(
                || Series::<f64>::from(data.clone()),
                |series: Series<f64>| black_box(series + 1.0),
                BatchSize::SmallInput,
            )
        });

        c.bench_function("vec", |b| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    for item in data.iter_mut() {
                        *item += 1.0;
                    }

                    black_box(data);
                },
                BatchSize::SmallInput,
            )
        });
    }

    {
        let mut c = c.benchmark_group("add series");
        let series2 = Series::<f64>::from(data.clone());
        c.bench_function("no SIMD", |b| {
            b.iter_batched(
                || Series::<f64>::from(data.clone()).align(series2.clone()),
                |(series, series2)| black_box(series.zip_in_place(series2, |f, g| *f += g)),
                BatchSize::SmallInput,
            )
        });

        c.bench_function("SIMD", |b| {
            b.iter_batched(
                || Series::<f64>::from(data.clone()).align(series2.clone()),
                |(series, series2)| black_box(series + series2),
                BatchSize::SmallInput,
            )
        });
    }

    {
        let mut c = c.benchmark_group("where scalar");
        c.bench_function("no SIMD", |b| {
            b.iter_batched(
                || Series::<f64>::from(data.clone()),
                |series: Series<f64>| {
                    black_box(series.map_in_place(|f| {
                        if *f >= 500.0 {
                            *f = -1.0;
                        }
                    }))
                },
                BatchSize::SmallInput,
            )
        });

        c.bench_function("SIMD", |b| {
            b.iter_batched(
                || {
                    let series = Series::<f64>::from(data.clone());
                    let condition = series.clone().gte(500.0);
                    (series, condition)
                },
                |(series, condition)| black_box(series.mask_or(condition, -1.0)),
                BatchSize::SmallInput,
            )
        });

        c.bench_function("vec", |b| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    for item in data.iter_mut() {
                        if *item >= 500.0 {
                            *item = -1.0;
                        }
                    }

                    black_box(data);
                },
                BatchSize::SmallInput,
            )
        });
    }

    {
        let mut c = c.benchmark_group("sum");
        let series = Series::<f64>::from(data.clone());
        c.bench_function("no SIMD", |b| {
            b.iter(|| black_box::<f64>(series.fold(0.0, |a, b| a + b)))
        });

        c.bench_function("SIMD", |b| b.iter(|| black_box::<f64>(series.sum())));

        c.bench_function("vec", |b| {
            b.iter(|| black_box::<f64>(black_box(&data).iter().sum::<f64>()))
        });
    }

    {
        let mut c = c.benchmark_group("bool_any");
        let series = Series::<f64>::from(data.clone()).gte(0.0);
        c.bench_function("no SIMD", |b| {
            b.iter(|| black_box(series.fold(false, |any, &item| any || item)))
        });

        c.bench_function("SIMD", |b| b.iter(|| black_box(series.any())));

        c.bench_function("vec", |b| {
            b.iter(|| black_box(black_box(&data).iter().any(|&n| n >= 0.0)))
        });
    }

    {
        let mut c = c.benchmark_group("bool_all");
        let series = Series::<f64>::from(data.clone()).gte(0.0);
        c.bench_function("no SIMD", |b| {
            b.iter(|| black_box(series.fold(true, |all, &item| all && item)))
        });

        c.bench_function("SIMD", |b| b.iter(|| black_box(series.all())));

        c.bench_function("vec", |b| {
            b.iter(|| black_box(black_box(&data).iter().all(|&n| n >= 0.0)))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
