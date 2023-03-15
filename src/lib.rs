#![feature(portable_simd, test)]

pub mod mat_mul;

pub fn print_mat_n(m: &Vec<f64>, n: usize) {
    for i in 0..n * n {
        print!("{:.1},\t", m[i]);
        if 0 == (i + 1) % n {
            println!("");
        }
    }
}

pub fn init_mat_n(a: &mut Vec<f64>, b: &mut Vec<f64>, n: usize) {
    let frac = 1.0;
    for i in 0..n {
        for j in 0..n {
            a[i + j * n] = ((i + j * n) as f64) / frac;
            b[i + j * n] = ((j + i * n) as f64) / frac;
        }
    }
}

pub fn cmp_mat_n(a: &Vec<f64>, b: &Vec<f64>, n: usize) -> bool {
    let delta = 1e-9;
    for i in 0..n * n {
        if (a[i] - b[i]).abs() > delta {
			println!("i) {}, a) {} b) {}", i, a[i], b[i]);
            return false;
        }
    }
    true
}
