//! Tests for VectorMut left-hand side AddAssign/SubAssign operations with various types.
//!
//! Test targets:
//! - AddAssign<Vector<i32>> for VectorMut<i32> (AutoSelect and Avx2)
//! - SubAssign<Vector<i32>> for VectorMut<i32> (AutoSelect and Avx2)
//! - AddAssign<Vector<f64>> for VectorMut<f64> with explicit Avx2 backend
//! - SubAssign<Vector<f64>> for VectorMut<f64> with explicit Avx2 backend

use simdoperator::{Vector, VectorMut};
use simdoperator::backend::avx2::Avx2;

/// Test AddAssign<Vector<i32>> for VectorMut<i32> (AutoSelect) — size N=8
#[test]
fn test_vector_mut_add_assign_i32_n8() {
    let mut data_a: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let expected: [i32; 8] = [101, 104, 107, 110, 113, 116, 119, 122];

    {
        let mut vm: VectorMut<i32, 8> = <VectorMut<i32, 8>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [i32; 8] = [100, 102, 104, 106, 108, 110, 112, 114];
        let vb: Vector<i32, 8> = <Vector<i32, 8>>::try_from(&data_b).unwrap();

        vm += vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test AddAssign<Vector<i32>> for VectorMut<i32> (AutoSelect) — size N=16
#[test]
fn test_vector_mut_add_assign_i32_n16() {
    let mut data_a: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let expected: [i32; 16] = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130];

    {
        let mut vm: VectorMut<i32, 16> = <VectorMut<i32, 16>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [i32; 16] = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115];
        let vb: Vector<i32, 16> = <Vector<i32, 16>>::try_from(&data_b).unwrap();

        vm += vb;
    }

    for i in 0..16 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test SubAssign<Vector<i32>> for VectorMut<i32> (AutoSelect) — size N=8
#[test]
fn test_vector_mut_sub_assign_i32_n8() {
    let mut data_a: [i32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    let expected: [i32; 8] = [9, 18, 27, 36, 45, 54, 63, 72];

    {
        let mut vm: VectorMut<i32, 8> = <VectorMut<i32, 8>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let vb: Vector<i32, 8> = <Vector<i32, 8>>::try_from(&data_b).unwrap();

        vm -= vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test SubAssign<Vector<i32>> for VectorMut<i32> (AutoSelect) — size N=10 (non-lane-multiple)
#[test]
fn test_vector_mut_sub_assign_i32_n10() {
    let mut data_a: [i32; 10] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
    let expected: [i32; 10] = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49];

    {
        let mut vm: VectorMut<i32, 10> = <VectorMut<i32, 10>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [i32; 10] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let vb: Vector<i32, 10> = <Vector<i32, 10>>::try_from(&data_b).unwrap();

        vm -= vb;
    }

    for i in 0..10 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test AddAssign<Vector<f64>> for VectorMut<f64> with explicit Avx2 backend — size N=8
#[test]
fn test_vector_mut_add_assign_f64_avx2_n8() {
    let mut data_a: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: [f64; 8] = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0];

    {
        let mut vm: VectorMut<f64, 8, Avx2> = <VectorMut<f64, 8, Avx2>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f64; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let vb: Vector<f64, 8, Avx2> = <Vector<f64, 8, Avx2>>::try_from(&data_b).unwrap();

        vm += vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test SubAssign<Vector<f64>> for VectorMut<f64> with explicit Avx2 backend — size N=5 (non-lane-multiple)
#[test]
fn test_vector_mut_sub_assign_f64_avx2_n5() {
    let mut data_a: [f64; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];
    let expected: [f64; 5] = [9.0, 18.0, 27.0, 36.0, 45.0];

    {
        let mut vm: VectorMut<f64, 5, Avx2> = <VectorMut<f64, 5, Avx2>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let vb: Vector<f64, 5, Avx2> = <Vector<f64, 5, Avx2>>::try_from(&data_b).unwrap();

        vm -= vb;
    }

    for i in 0..5 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test AddAssign<Vector<f32>> for VectorMut<f32> — size N=8
#[test]
fn test_vector_mut_add_assign_f32_n8() {
    let mut data_a: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: [f32; 8] = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0];

    {
        let mut vm: VectorMut<f32, 8> = <VectorMut<f32, 8>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f32; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let vb: Vector<f32, 8> = <Vector<f32, 8>>::try_from(&data_b).unwrap();

        vm += vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test SubAssign<Vector<f32>> for VectorMut<f32> — size N=5 (non-lane-multiple)
#[test]
fn test_vector_mut_sub_assign_f32_n5() {
    let mut data_a: [f32; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];
    let expected: [f32; 5] = [9.0, 18.0, 27.0, 36.0, 45.0];

    {
        let mut vm: VectorMut<f32, 5> = <VectorMut<f32, 5>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let vb: Vector<f32, 5> = <Vector<f32, 5>>::try_from(&data_b).unwrap();

        vm -= vb;
    }

    for i in 0..5 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}
