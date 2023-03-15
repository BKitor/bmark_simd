use std::simd::Simd;
use std::simd::SimdFloat;

use crate::{init_mat_n, print_mat_n};

pub fn print_reg_f64x8(r: Simd<f64, 8>) {
    let tmp = r.as_array();
    for i in 0..8 {
        print!("{:?},\t", tmp[i]);
    }
    println!("");
}

pub fn bk_mat_mul_n(a: &Vec<f64>, b: &Vec<f64>, c: &mut Vec<f64>, n: usize) {
    for j in 0..n {
        for i in 0..n {
            c[i + j * n] = 0.0;
            for k in 0..n {
                c[i + j * n] += a[k + j * n] * b[i + k * n];
            }
        }
    }
}

pub fn bk_mat_mul_simd_256_n(a: &Vec<f64>, b: &Vec<f64>, c: &mut Vec<f64>, n: usize) {
    let m = n / 4;

    let mut bidxarr: [usize; 4] = [0; 4];
    for i in 0..4 {
        bidxarr[i] = i * n;
    }
    let bidxv = Simd::from_array(bidxarr);

    for i in 0..n {
        for j in 0..n {
            c[i + j * n] = 0.0;
            for k in 0..m {
                let va: Simd<f64, 4> = Simd::from_slice(&a[(k * 4) + (i * n)..]);
                let vb: Simd<f64, 4> = Simd::gather_or_default(&b[(j + k * n * 4)..], bidxv);

                c[i + j * n] += (va * vb).reduce_sum();
            }
        }
    }
}

pub fn bk_mat_mul_simd_512_n(a: &Vec<f64>, b: &Vec<f64>, c: &mut Vec<f64>, n: usize) {
    let m = n / 8;

    let mut bidxarr: [usize; 8] = [0; 8];
    for i in 0..8 {
        bidxarr[i] = i * n;
    }
    let bidxv = Simd::from_array(bidxarr);

    for i in 0..n {
        for j in 0..n {
            c[i + j * n] = 0.0;
            for k in 0..m {
                let va: Simd<f64, 8> = Simd::from_slice(&a[(k * 8) + (i * n)..]);
                let vb: Simd<f64, 8> = Simd::gather_or_default(&b[(j + k * n * 8)..], bidxv);

                c[i + j * n] += (va * vb).reduce_sum();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmp_mat_n;

    use test_case::test_case;

    #[test_case(8)]
    // #[test_case(16)]
    #[test_case(32)]
    // #[test_case(64)]
    #[test_case(128)]
    // #[test_case(256)]
    #[test_case(512)]
    // #[test_case(1024)]
    fn matmul_8(n: usize) {
        // let n: usize = 8;
        let mut a: Vec<f64> = vec![0.0; n * n];
        let mut b: Vec<f64> = vec![0.0; n * n];
        let mut c: Vec<f64> = vec![0.0; n * n];
        let mut simd_256: Vec<f64> = vec![0.0; n * n];
        let mut simd_512: Vec<f64> = vec![0.0; n * n];

        init_mat_n(&mut a, &mut b, n);

        bk_mat_mul_n(&a, &b, &mut c, n);
        bk_mat_mul_simd_256_n(&a, &b, &mut simd_256, n);
        bk_mat_mul_simd_512_n(&a, &b, &mut simd_512, n);

        assert!(cmp_mat_n(&c, &simd_512, n));
        assert!(cmp_mat_n(&c, &simd_256, n));
    }
}
