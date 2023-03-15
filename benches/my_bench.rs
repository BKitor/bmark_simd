use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use rsimd::init_mat_n;
use rsimd::mat_mul::{bk_mat_mul_n, bk_mat_mul_simd_256_n, bk_mat_mul_simd_512_n};

fn mat_mul_n_bmark(criterion: &mut Criterion) {
    for n in [256, 512, 1024] {
        let mut a: Vec<f64> = vec![0.0; n * n];
        let mut b: Vec<f64> = vec![0.0; n * n];
        let mut c: Vec<f64> = vec![0.0; n * n];
        init_mat_n(&mut a, &mut b, n);

        let mut g = criterion.benchmark_group("bk_mm");
        
        g.bench_with_input(format!("base_n{:?}",n),&n, |bencher, &n|{
            bencher.iter(|| bk_mat_mul_n(&a, &b, &mut c, n))
        });

        g.bench_with_input(format!("simd_256_n{:?}",n),&n, |bencher, &n|{
            bencher.iter(|| bk_mat_mul_simd_256_n(&a, &b, &mut c, n))
        });

        g.bench_with_input(format!("simd_512_n{:?}",n),&n, |bencher, &n|{
            bencher.iter(|| bk_mat_mul_simd_512_n(&a, &b, &mut c, n))
        });
        g.finish()
    }
}

criterion_group!(benches, mat_mul_n_bmark);
criterion_main!(benches);

