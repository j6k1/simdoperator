//! Comprehensive SIMD operation tests
//! 
//! Test targets:
//! - Basic Vector / Matrix operations (Add, Sub, Mul, ScalarMul)
//! - Bitwise operations (And, Or, Xor, Not, Shl, Shr)
//! - Type conversion, promotion, and demotion (Promote, Demote, Convert)
//! - Matrix operations (AddAssign, ScalarMulAssign, ScalarMul, Convert)
//! - Product operations (Dot, OuterProduct, MatVec, VMat, MatMul)
//! 
//! Size design rationale (AVX2 Backend):
//! f64: LANES=4, ROWS=1, COLS=1
//! i32: LANES=8, ROWS=2, COLS=2
//! Test sizes: 1 (<LANES), 4 (=f64 LANES), 5 (>LANES and not divisible), 8 (i32 LANES / multiple of f64 LANES)

use simdoperator::{Vector, VectorMut, OwnedVector, AccVector, Matrix, MatrixMut, OwnedMatrix, AccMatrix, ColumnMajorMatrix};

// Helper: Generate OwnedVector from a fixed-size array
fn make_vec<T: Copy + Default, const N: usize>(vals: &[T]) -> OwnedVector<T, N> {
    let mut v = OwnedVector::default();
    for (i, val) in vals.iter().enumerate() {
        if i < N { v[i] = *val; }
    }
    v
}

// Helper: Generate OwnedMatrix from a fixed-size array (Row-major)
fn make_mat<T: Copy + Default, const N: usize, const M: usize>(vals: &[T]) -> OwnedMatrix<T, N, M> {
    let mut m = OwnedMatrix::default();
    for (i, val) in vals.iter().enumerate() {
        if i < N * M { m[(i / M, i % M)] = *val; }
    }
    m
}

// ================= Vector Operation Tests =================

fn run_vec_add<const N: usize>() {
    let a = make_vec::<f64, N>(&[1.0; 8]);
    let b = make_vec::<f64, N>(&[2.0; 8]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vb = <Vector<f64, N>>::try_from(&b).unwrap();
    let vc = va + vb;
    for i in 0..N { assert_eq!(vc[i], 3.0); }
}

fn run_vec_sub<const N: usize>() {
    let a = make_vec::<f64, N>(&[5.0; 8]);
    let b = make_vec::<f64, N>(&[2.0; 8]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vb = <Vector<f64, N>>::try_from(&b).unwrap();
    let vc = va - vb;
    for i in 0..N { assert_eq!(vc[i], 3.0); }
}

fn run_vec_mul<const N: usize>() {
    // Vector-vector multiplication for f64 is not implemented, so test with i32 (LANES=8)
    let a = make_vec::<i32, N>(&[2; 8]);
    let b = make_vec::<i32, N>(&[3; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va * vb;
    for i in 0..N { assert_eq!(vc[i], 6); }
}

fn run_vec_scalar_mul<const N: usize>() {
    let a = make_vec::<f64, N>(&[2.0; 8]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vc = va * 3.0;
    for i in 0..N { assert_eq!(vc[i], 6.0); }
}

fn run_vec_bit_and<const N: usize>() {
    let a = make_vec::<i32, N>(&[0xFF; 8]);
    let b = make_vec::<i32, N>(&[0xF0; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va & vb;
    for i in 0..N { assert_eq!(vc[i], 0xF0); }
}

fn run_vec_bit_or<const N: usize>() {
    let a = make_vec::<i32, N>(&[0x0F; 8]);
    let b = make_vec::<i32, N>(&[0xF0; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va | vb;
    for i in 0..N { assert_eq!(vc[i], 0xFF); }
}

fn run_vec_bit_xor<const N: usize>() {
    let a = make_vec::<i32, N>(&[0xFF; 8]);
    let b = make_vec::<i32, N>(&[0xF0; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va ^ vb;
    for i in 0..N { assert_eq!(vc[i], 0x0F); }
}

fn run_vec_bit_not<const N: usize>() {
    let a = make_vec::<i32, N>(&[0xFF; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc = !va;
    for i in 0..N { assert_eq!(vc[i], !0xFF); }
}

fn run_vec_shl<const N: usize>() {
    let a = make_vec::<i32, N>(&[1; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc = va << 4;
    for i in 0..N { assert_eq!(vc[i], 16); }
}

fn run_vec_shr<const N: usize>() {
    let a = make_vec::<i32, N>(&[16; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc = va >> 4;
    for i in 0..N { assert_eq!(vc[i], 1); }
}

// ================= Type Conversion, Promotion, and Demotion Tests =================

fn run_promote_i8_to_i16<const N: usize>() {
    let a = make_vec::<i8, N>(&[5; 32]);
    let va = <Vector<i8, N>>::try_from(&a).unwrap();
    let vc: AccVector<i16, N> = va.promotion();
    for i in 0..N { assert_eq!(vc[i], 5); }
}

fn run_demote_i32_to_i16<const N: usize>() {
    let a = make_vec::<i32, N>(&[10; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc: AccVector<i16, N> = va.demotion();
    for i in 0..N { assert_eq!(vc[i], 10); }
}

fn run_convert_i32_to_f32<const N: usize>() {
    let a = make_vec::<i32, N>(&[7; 8]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc: AccVector<f32, N> = va.into();
    for i in 0..N { assert_eq!(vc[i], 7.0); }
}

// ================= Matrix Operation Tests =================

fn run_mat_add_assign<const N: usize, const M: usize>() {
    let mut a = make_mat::<f64, N, M>(&[1.0; 32]);
    let b = make_mat::<f64, N, M>(&[2.0; 32]);
    let ma = <MatrixMut<f64, N, M>>::try_from(&mut a).unwrap();
    let mb = <Matrix<f64, N, M>>::try_from(&b).unwrap();
    ma += mb;
    for i in 0..N {
        for j in 0..M {
            assert_eq!(a[(i,j)], 3.0);
        }
    }
}

fn run_mat_scalar_mul_assign<const N: usize, const M: usize>() {
    let mut a = make_mat::<f64, N, M>(&[2.0; 32]);
    let ma = <MatrixMut<f64, N, M>>::try_from(&mut a).unwrap();
    ma *= 3.0;
    for i in 0..N {
        for j in 0..M {
            assert_eq!(a[(i,j)], 6.0);
        }
    }
}

fn run_mat_scalar_mul<const N: usize, const M: usize>() {
    let a = make_mat::<f64, N, M>(&[2.0; 32]);
    let ma = <Matrix<f64, N, M>>::try_from(&a).unwrap();
    let mc = ma * 3.0;
    for i in 0..N {
        for j in 0..M {
            assert_eq!(mc[(i,j)], 6.0);
        }
    }
}

fn run_mat_convert<const N: usize, const M: usize>() {
    let a = make_mat::<f32, N, M>(&[5.0; 32]);
    let ma = <Matrix<f32, N, M>>::try_from(&a).unwrap();
    let mc: AccMatrix<f64, N, M> = ma.into();
    for i in 0..N {
        for j in 0..M {
            assert_eq!(mc[(i,j)], 5.0);
        }
    }
}

// ================= Product Operation Tests (Dot, Outer, MatVec, VMat, MatMul) =================

fn run_dot<const N: usize>() {
    let a = make_vec::<f64, N>(&[1.0; 8]);
    let b = make_vec::<f64, N>(&[2.0; 8]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vb = <Vector<f64, N>>::try_from(&b).unwrap();
    let res = va.dot(vb);
    assert_eq!(res, (N as f64) * 2.0);
}

fn run_outer_product<const N: usize, const M: usize>() {
    let a = make_vec::<f64, N>(&[1.0; 8]);
    let b = make_vec::<f64, M>(&[2.0; 8]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vb = <Vector<f64, M>>::try_from(&b).unwrap();
    let mc = va.product(vb);
    for i in 0..N {
        for j in 0..M {
            assert_eq!(mc[(i,j)], 2.0);
        }
    }
}

fn run_matvec<const N: usize, const K: usize>() {
    let a = make_mat::<f64, N, K>(&[1.0; 32]);
    let b = make_vec::<f64, K>(&[2.0; 8]);
    let ma = <Matrix<f64, N, K>>::try_from(&a).unwrap();
    let vb = <Vector<f64, K>>::try_from(&b).unwrap();
    let vc = ma.product(vb);
    for i in 0..N {
        assert_eq!(vc[i], (K as f64) * 2.0);
    }
}

fn run_vmat<const K: usize, const M: usize>() {
    let a = make_vec::<f64, K>(&[1.0; 8]);
    // Column-major data (order does not matter since all values are the same)
    let col_data = vec![2.0; K * M];
    let cm: ColumnMajorMatrix<f64, K, M> = <ColumnMajorMatrix<f64, K, M>>::try_from(&col_data).unwrap();
    
    let va = <Vector<f64, K>>::try_from(&a).unwrap();
    let vc = va.product(cm);
    for i in 0..M {
        assert_eq!(vc[i], (K as f64) * 2.0);
    }
}

fn run_matmul<const N: usize, const K: usize, const M: usize>() {
    let a = make_mat::<f64, N, K>(&[1.0; 32]);
    let col_data = vec![2.0; K * M];
    let cm: ColumnMajorMatrix<f64, K, M> = <ColumnMajorMatrix<f64, K, M>>::try_from(&col_data).unwrap();
    
    let ma = <Matrix<f64, N, K>>::try_from(&a).unwrap();
    let mc = ma.product(cm);
    for i in 0..N {
        for j in 0..M {
            assert_eq!(mc[(i,j)], (K as f64) * 2.0);
        }
    }
}

// ================= Test Case Generation Macros =================

/// Generate size variation tests for vector operations
/// Sizes: 1 (<LANES), 4 (=f64 LANES), 5 (>LANES and not divisible), 8 (i32 LANES / multiple of f64 LANES)
macro_rules! gen_vec_tests {
    ($($n:expr),*) => {
        $(
            #[test] fn [<vec_add_n $n>]() { run_vec_add::<$n>(); }
            #[test] fn [<vec_sub_n $n>]() { run_vec_sub::<$n>(); }
            #[test] fn [<vec_mul_n $n>]() { run_vec_mul::<$n>(); }
            #[test] fn [<vec_scalar_mul_n $n>]() { run_vec_scalar_mul::<$n>(); }
            #[test] fn [<vec_bit_and_n $n>]() { run_vec_bit_and::<$n>(); }
            #[test] fn [<vec_bit_or_n $n>]() { run_vec_bit_or::<$n>(); }
            #[test] fn [<vec_bit_xor_n $n>]() { run_vec_bit_xor::<$n>(); }
            #[test] fn [<vec_bit_not_n $n>]() { run_vec_bit_not::<$n>(); }
            #[test] fn [<vec_shl_n $n>]() { run_vec_shl::<$n>(); }
            #[test] fn [<vec_shr_n $n>]() { run_vec_shr::<$n>(); }
            #[test] fn [<promote_i8_to_i16_n $n>]() { run_promote_i8_to_i16::<$n>(); }
            #[test] fn [<demote_i32_to_i16_n $n>]() { run_demote_i32_to_i16::<$n>(); }
            #[test] fn [<convert_i32_to_f32_n $n>]() { run_convert_i32_to_f32::<$n>(); }
            #[test] fn [<dot_n $n>]() { run_dot::<$n>(); }
        )*
    };
}

/// Generate size variation tests for matrix operations (N, M)
macro_rules! gen_mat_tests {
    ($(($n:expr, $m:expr)),*) => {
        $(
            #[test] fn [<mat_add_assign_n $n _m $m>]() { run_mat_add_assign::<$n, $m>(); }
            #[test] fn [<mat_scalar_mul_assign_n $n _m $m>]() { run_mat_scalar_mul_assign::<$n, $m>(); }
            #[test] fn [<mat_scalar_mul_n $n _m $m>]() { run_mat_scalar_mul::<$n, $m>(); }
            #[test] fn [<mat_convert_n $n _m $m>]() { run_mat_convert::<$n, $m>(); }
        )*
    };
}

/// Generate size variation tests for product operations (N, K, M)
macro_rules! gen_prod_tests {
    ($(($n:expr, $k:expr, $m:expr)),*) => {
        $(
            #[test] fn [<outer_product_n $n _m $m>]() { run_outer_product::<$n, $m>(); }
            #[test] fn [<matvec_n $n _k $k>]() { run_matvec::<$n, $k>(); }
            #[test] fn [<vmat_k $k _m $m>]() { run_vmat::<$k, $m>(); }
            #[test] fn [<matmul_n $n _k $k _m $m>]() { run_matmul::<$n, $k, $m>(); }
        )*
    };
}

// Test execution expansion
gen_vec_tests!(1, 4, 5, 8);
gen_mat_tests!((1,1), (4,4), (5,6), (8,8));
gen_prod_tests!((1,1,1), (4,4,4), (5,6,7), (8,8,8));
