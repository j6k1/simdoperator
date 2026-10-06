//! Comprehensive SIMD operation tests
//! 
//! Test targets:
//! - Basic Vector / Matrix operations (Add, Sub, Mul, ScalarMul)
//! - Bitwise operations (And, Or, Xor, Not, Shl, Shr)
//! - Type conversion, promotion, and demotion (Promote, Demote, Convert)
//! - Matrix operations (AddAssign, ScalarMulAssign, ScalarMul)
//! - Product operations (Dot, OuterProduct, MatVec, VMat, MatMul)
//! 
//! Size design rationale (AVX2 Backend):
//! f64: LANES=4, ROWS=1, COLS=1
//! i32: LANES=8, ROWS=2, COLS=2
//! Test sizes: 1 (<LANES), 4 (=f64 LANES), 5 (>LANES and not divisible), 8 (i32 LANES / multiple of f64 LANES)

use simdoperator::{Vector, OwnedVector, AccVector, Matrix, MatrixMut, OwnedMatrix, ColumnMajorMatrix, AccMatrix};
use simdoperator::traits::{Demote, Promote, Dot, Product};
use simdoperator::backend::avx2::Avx2;

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
    let a = make_vec::<f64, N>(&vec![1.0; N]);
    let b = make_vec::<f64, N>(&vec![2.0; N]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vb = <Vector<f64, N>>::try_from(&b).unwrap();
    let vc = va + vb;
    for i in 0..N { assert_eq!(vc[i], 3.0); }
}

fn run_vec_sub<const N: usize>() {
    let a = make_vec::<f64, N>(&vec![5.0; N]);
    let b = make_vec::<f64, N>(&vec![2.0; N]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vb = <Vector<f64, N>>::try_from(&b).unwrap();
    let vc = va - vb;
    for i in 0..N { assert_eq!(vc[i], 3.0); }
}

fn run_vec_mul<const N: usize>() {
    // Vector-vector multiplication for f64 is not implemented, so test with i32 (LANES=8)
    let a = make_vec::<i32, N>(&vec![2; N]);
    let b = make_vec::<i32, N>(&vec![3; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va * vb;
    for i in 0..N { assert_eq!(vc[i], 6); }
}

fn run_vec_scalar_mul<const N: usize>() {
    let a = make_vec::<f64, N>(&vec![2.0; N]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vc = va * 3.0;
    for i in 0..N { assert_eq!(vc[i], 6.0); }
}

fn run_vec_bit_and<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![0xFF; N]);
    let b = make_vec::<i32, N>(&vec![0xF0; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va & vb;
    for i in 0..N { assert_eq!(vc[i], 0xF0); }
}

fn run_vec_bit_or<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![0x0F; N]);
    let b = make_vec::<i32, N>(&vec![0xF0; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va | vb;
    for i in 0..N { assert_eq!(vc[i], 0xFF); }
}

fn run_vec_bit_xor<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![0xFF; N]);
    let b = make_vec::<i32, N>(&vec![0xF0; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vb = <Vector<i32, N>>::try_from(&b).unwrap();
    let vc = va ^ vb;
    for i in 0..N { assert_eq!(vc[i], 0x0F); }
}

fn run_vec_bit_not<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![0xFF; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc = !va;
    for i in 0..N { assert_eq!(vc[i], !0xFF); }
}

fn run_vec_shl<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![1; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc = va << 4;
    for i in 0..N { assert_eq!(vc[i], 16); }
}

fn run_vec_shr<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![16; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc = va >> 4;
    for i in 0..N { assert_eq!(vc[i], 1); }
}

// ================= Type Conversion, Promotion, and Demotion Tests =================

fn run_promote_i8_to_i16<const N: usize>() {
    let a = make_vec::<i8, N>(&vec![5; N]);
    let va = <Vector<i8, N>>::try_from(&a).unwrap();
    let vc: AccVector<i16, N> = va.promotion();
    for i in 0..N { assert_eq!(vc[i], 5); }
}

fn run_demote_i32_to_i16<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![10; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc: AccVector<i16, N> = va.demotion();
    for i in 0..N { assert_eq!(vc[i], 10); }
}

fn run_convert_i32_to_f32<const N: usize>() {
    let a = make_vec::<i32, N>(&vec![7; N]);
    let va = <Vector<i32, N>>::try_from(&a).unwrap();
    let vc: AccVector<f32, N> = va.into();
    for i in 0..N { assert_eq!(vc[i], 7.0); }
}

// ================= Matrix Operation Tests =================

fn run_mat_add_assign<const N: usize, const M: usize>() {
    let mut a = make_mat::<f64, N, M>(&vec![1.0; N * M]);
    let b = make_mat::<f64, N, M>(&vec![2.0; N * M]);

    let mut acc_a: AccMatrix<f64, N, M, Avx2> = <AccMatrix<f64, N, M, Avx2>>::try_from(a).unwrap();

    // Use AccMatrix to avoid lifetime issues with borrowed data
    {
        let mb: Matrix<f64, N, M, Avx2> = <Matrix<f64, N, M, Avx2>>::try_from(&b).unwrap();

        // Perform add_assign operation
        use std::ops::AddAssign;
        acc_a += mb;
    }
    
    for i in 0..N {
        for j in 0..M {
            assert_eq!(acc_a[(i,j)], 3.0);
        }
    }
}

fn run_mat_scalar_mul_assign<const N: usize, const M: usize>() {
    let mut a = make_mat::<f64, N, M>(&vec![2.0; N * M]);
    let mut acc_a: AccMatrix<f64, N, M, Avx2> = <AccMatrix<f64, N, M, Avx2>>::try_from(a).unwrap();

    // Use AccMatrix to avoid lifetime issues with borrowed data
    {

        // Perform scalar mul assign operation
        use std::ops::MulAssign;
        acc_a *= 3.0;
    }
    
    for i in 0..N {
        for j in 0..M {
            assert_eq!(acc_a[(i,j)], 6.0);
        }
    }
}

fn run_mat_scalar_mul<const N: usize, const M: usize>() {
    let a = make_mat::<f64, N, M>(&vec![2.0; N * M]);
    
    // Use Avx2 backend explicitly to avoid AutoSelect trait issues
    let ma: Matrix<f64, N, M, Avx2> = <Matrix<f64, N, M, Avx2>>::try_from(&a).unwrap();
    let mc = ma * 3.0;
    for i in 0..N {
        for j in 0..M {
            assert_eq!(mc[(i,j)], 6.0);
        }
    }
}

// ================= Product Operation Tests (Dot, Outer, MatVec, VMat, MatMul) =================

fn run_dot<const N: usize>() {
    let a = make_vec::<f64, N>(&vec![1.0; N]);
    let b = make_vec::<f64, N>(&vec![2.0; N]);
    let va = <Vector<f64, N>>::try_from(&a).unwrap();
    let vb = <Vector<f64, N>>::try_from(&b).unwrap();
    let res = va.dot(vb);
    assert_eq!(res, (N as f64) * 2.0);
}

fn run_outer_product<const N: usize, const M: usize>() {
    let a = make_vec::<f64, N>(&vec![1.0; N]);
    let b = make_vec::<f64, M>(&vec![2.0; M]);
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
    let a = make_mat::<f64, N, K>(&vec![1.0; N * K]);
    let b = make_vec::<f64, K>(&vec![2.0; K]);
    let ma = <Matrix<f64, N, K>>::try_from(&a).unwrap();
    let vb = <Vector<f64, K>>::try_from(&b).unwrap();
    let vc = ma.product(vb);
    for i in 0..N {
        assert_eq!(vc[i], (K as f64) * 2.0);
    }
}

fn run_vmat<const K: usize, const M: usize>() {
    let a = make_vec::<f64, K>(&vec![1.0; K]);
    // Column-major data (order does not matter since all values are the same)
    let col_data = vec![2.0; K * M];
    let cm: ColumnMajorMatrix<f64, K, M> = <ColumnMajorMatrix<f64, K, M>>::try_from(col_data.as_slice()).unwrap();
    
    let va = <Vector<f64, K>>::try_from(&a).unwrap();
    let vc = va.product(cm);
    for i in 0..M {
        assert_eq!(vc[i], (K as f64) * 2.0);
    }
}

fn run_matmul<const N: usize, const K: usize, const M: usize>() {
    let a = make_mat::<f64, N, K>(&vec![1.0; N * K]);
    let col_data = vec![2.0; K * M];
    let cm: ColumnMajorMatrix<f64, K, M> = <ColumnMajorMatrix<f64, K, M>>::try_from(col_data.as_slice()).unwrap();
    
    let ma = <Matrix<f64, N, K>>::try_from(a.as_ref()).unwrap();
    let mc = ma.product(cm);
    for i in 0..N {
        for j in 0..M {
            assert_eq!(mc[(i,j)], (K as f64) * 2.0);
        }
    }
}

// ================= Test Cases (Expanded from macros for stable Rust compatibility) =================

// Vector tests for N=1
#[test] fn vec_add_n1() { run_vec_add::<1>(); }
#[test] fn vec_sub_n1() { run_vec_sub::<1>(); }
#[test] fn vec_mul_n1() { run_vec_mul::<1>(); }
#[test] fn vec_scalar_mul_n1() { run_vec_scalar_mul::<1>(); }
#[test] fn vec_bit_and_n1() { run_vec_bit_and::<1>(); }
#[test] fn vec_bit_or_n1() { run_vec_bit_or::<1>(); }
#[test] fn vec_bit_xor_n1() { run_vec_bit_xor::<1>(); }
#[test] fn vec_bit_not_n1() { run_vec_bit_not::<1>(); }
#[test] fn vec_shl_n1() { run_vec_shl::<1>(); }
#[test] fn vec_shr_n1() { run_vec_shr::<1>(); }
#[test] fn promote_i8_to_i16_n1() { run_promote_i8_to_i16::<1>(); }
#[test] fn demote_i32_to_i16_n1() { run_demote_i32_to_i16::<1>(); }
#[test] fn convert_i32_to_f32_n1() { run_convert_i32_to_f32::<1>(); }
#[test] fn dot_n1() { run_dot::<1>(); }

// Vector tests for N=4
#[test] fn vec_add_n4() { run_vec_add::<4>(); }
#[test] fn vec_sub_n4() { run_vec_sub::<4>(); }
#[test] fn vec_mul_n4() { run_vec_mul::<4>(); }
#[test] fn vec_scalar_mul_n4() { run_vec_scalar_mul::<4>(); }
#[test] fn vec_bit_and_n4() { run_vec_bit_and::<4>(); }
#[test] fn vec_bit_or_n4() { run_vec_bit_or::<4>(); }
#[test] fn vec_bit_xor_n4() { run_vec_bit_xor::<4>(); }
#[test] fn vec_bit_not_n4() { run_vec_bit_not::<4>(); }
#[test] fn vec_shl_n4() { run_vec_shl::<4>(); }
#[test] fn vec_shr_n4() { run_vec_shr::<4>(); }
#[test] fn promote_i8_to_i16_n4() { run_promote_i8_to_i16::<4>(); }
#[test] fn demote_i32_to_i16_n4() { run_demote_i32_to_i16::<4>(); }
#[test] fn convert_i32_to_f32_n4() { run_convert_i32_to_f32::<4>(); }
#[test] fn dot_n4() { run_dot::<4>(); }

// Vector tests for N=5
#[test] fn vec_add_n5() { run_vec_add::<5>(); }
#[test] fn vec_sub_n5() { run_vec_sub::<5>(); }
#[test] fn vec_mul_n5() { run_vec_mul::<5>(); }
#[test] fn vec_scalar_mul_n5() { run_vec_scalar_mul::<5>(); }
#[test] fn vec_bit_and_n5() { run_vec_bit_and::<5>(); }
#[test] fn vec_bit_or_n5() { run_vec_bit_or::<5>(); }
#[test] fn vec_bit_xor_n5() { run_vec_bit_xor::<5>(); }
#[test] fn vec_bit_not_n5() { run_vec_bit_not::<5>(); }
#[test] fn vec_shl_n5() { run_vec_shl::<5>(); }
#[test] fn vec_shr_n5() { run_vec_shr::<5>(); }
#[test] fn promote_i8_to_i16_n5() { run_promote_i8_to_i16::<5>(); }
#[test] fn demote_i32_to_i16_n5() { run_demote_i32_to_i16::<5>(); }
#[test] fn convert_i32_to_f32_n5() { run_convert_i32_to_f32::<5>(); }
#[test] fn dot_n5() { run_dot::<5>(); }

// Vector tests for N=8
#[test] fn vec_add_n8() { run_vec_add::<8>(); }
#[test] fn vec_sub_n8() { run_vec_sub::<8>(); }
#[test] fn vec_mul_n8() { run_vec_mul::<8>(); }
#[test] fn vec_scalar_mul_n8() { run_vec_scalar_mul::<8>(); }
#[test] fn vec_bit_and_n8() { run_vec_bit_and::<8>(); }
#[test] fn vec_bit_or_n8() { run_vec_bit_or::<8>(); }
#[test] fn vec_bit_xor_n8() { run_vec_bit_xor::<8>(); }
#[test] fn vec_bit_not_n8() { run_vec_bit_not::<8>(); }
#[test] fn vec_shl_n8() { run_vec_shl::<8>(); }
#[test] fn vec_shr_n8() { run_vec_shr::<8>(); }
#[test] fn promote_i8_to_i16_n8() { run_promote_i8_to_i16::<8>(); }
#[test] fn demote_i32_to_i16_n8() { run_demote_i32_to_i16::<8>(); }
#[test] fn convert_i32_to_f32_n8() { run_convert_i32_to_f32::<8>(); }
#[test] fn dot_n8() { run_dot::<8>(); }

// Matrix tests for (N,M) = (1,1), (4,4), (5,6), (8,8)
#[test] fn mat_add_assign_n1_m1() { run_mat_add_assign::<1, 1>(); }
#[test] fn mat_scalar_mul_assign_n1_m1() { run_mat_scalar_mul_assign::<1, 1>(); }
#[test] fn mat_scalar_mul_n1_m1() { run_mat_scalar_mul::<1, 1>(); }

#[test] fn mat_add_assign_n4_m4() { run_mat_add_assign::<4, 4>(); }
#[test] fn mat_scalar_mul_assign_n4_m4() { run_mat_scalar_mul_assign::<4, 4>(); }
#[test] fn mat_scalar_mul_n4_m4() { run_mat_scalar_mul::<4, 4>(); }

#[test] fn mat_add_assign_n5_m6() { run_mat_add_assign::<5, 6>(); }
#[test] fn mat_scalar_mul_assign_n5_m6() { run_mat_scalar_mul_assign::<5, 6>(); }
#[test] fn mat_scalar_mul_n5_m6() { run_mat_scalar_mul::<5, 6>(); }

#[test] fn mat_add_assign_n8_m8() { run_mat_add_assign::<8, 8>(); }
#[test] fn mat_scalar_mul_assign_n8_m8() { run_mat_scalar_mul_assign::<8, 8>(); }
#[test] fn mat_scalar_mul_n8_m8() { run_mat_scalar_mul::<8, 8>(); }

// Product tests for (N,K,M) = (1,1,1), (4,4,4), (5,6,7), (8,8,8)
#[test] fn outer_product_n1_m1() { run_outer_product::<1, 1>(); }
#[test] fn matvec_n1_k1() { run_matvec::<1, 1>(); }
#[test] fn vmat_k1_m1() { run_vmat::<1, 1>(); }
#[test] fn matmul_n1_k1_m1() { run_matmul::<1, 1, 1>(); }

#[test] fn outer_product_n4_m4() { run_outer_product::<4, 4>(); }
#[test] fn matvec_n4_k4() { run_matvec::<4, 4>(); }
#[test] fn vmat_k4_m4() { run_vmat::<4, 4>(); }
#[test] fn matmul_n4_k4_m4() { run_matmul::<4, 4, 4>(); }

#[test] fn outer_product_n5_m7() { run_outer_product::<5, 7>(); }
#[test] fn matvec_n5_k6() { run_matvec::<5, 6>(); }
#[test] fn vmat_k6_m7() { run_vmat::<6, 7>(); }
#[test] fn matmul_n5_k6_m7() { run_matmul::<5, 6, 7>(); }

#[test] fn outer_product_n8_m8() { run_outer_product::<8, 8>(); }
#[test] fn matvec_n8_k8() { run_matvec::<8, 8>(); }
#[test] fn vmat_k8_m8() { run_vmat::<8, 8>(); }
#[test] fn matmul_n8_k8_m8() { run_matmul::<8, 8, 8>(); }
