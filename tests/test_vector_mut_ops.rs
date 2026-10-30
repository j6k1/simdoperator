//! Tests for VectorMut left-hand side operations that are not yet covered.
//!
//! Test targets:
//! - AddAssign<Vector> for VectorMut (f64)
//! - SubAssign<Vector> for VectorMut (f64)
//! - MulAssign<Vector> for VectorMut (i32, f32, f64)
//! - MulAssign<scalar> for VectorMut (i32, f32, f64)

use simdoperator::{Vector, VectorMut};

/// Test AddAssign<Vector<f64>> for VectorMut<f64> — size N=8
#[test]
fn test_vector_mut_add_assign_f64_n8() {
    let mut data_a: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: [f64; 8] = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0];

    {
        let mut vm: VectorMut<f64, 8> = <VectorMut<f64, 8>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f64; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let vb: Vector<f64, 8> = <Vector<f64, 8>>::try_from(&data_b).unwrap();

        vm += vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test AddAssign<Vector<f64>> for VectorMut<f64> — size N=10 (non-lane-multiple)
#[test]
fn test_vector_mut_add_assign_f64_n10() {
    let mut data_a: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let expected: [f64; 10] = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];

    {
        let mut vm: VectorMut<f64, 10> = <VectorMut<f64, 10>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f64; 10] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let vb: Vector<f64, 10> = <Vector<f64, 10>>::try_from(&data_b).unwrap();

        vm += vb;
    }

    for i in 0..10 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test SubAssign<Vector<f64>> for VectorMut<f64> — size N=8
#[test]
fn test_vector_mut_sub_assign_f64_n8() {
    let mut data_a: [f64; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let expected: [f64; 8] = [9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0];

    {
        let mut vm: VectorMut<f64, 8> = <VectorMut<f64, 8>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vb: Vector<f64, 8> = <Vector<f64, 8>>::try_from(&data_b).unwrap();

        vm -= vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test SubAssign<Vector<f64>> for VectorMut<f64> — size N=1 (smaller than LANES)
#[test]
fn test_vector_mut_sub_assign_f64_n1() {
    let mut data_a: [f64; 1] = [5.0];

    {
        let mut vm: VectorMut<f64, 1> = <VectorMut<f64, 1>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f64; 1] = [3.0];
        let vb: Vector<f64, 1> = <Vector<f64, 1>>::try_from(&data_b).unwrap();

        vm -= vb;
    }

    assert_eq!(data_a[0], 2.0);
}

/// Test MulAssign<Vector<i32>> for VectorMut<i32> — size N=8 (LANES)
#[test]
fn test_vector_mut_mul_assign_i32_n8() {
    let mut data_a: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let expected: [i32; 8] = [2, 6, 12, 20, 30, 42, 56, 72];

    {
        let mut vm: VectorMut<i32, 8> = <VectorMut<i32, 8>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [i32; 8] = [2, 3, 4, 5, 6, 7, 8, 9];
        let vb: Vector<i32, 8> = <Vector<i32, 8>>::try_from(&data_b).unwrap();

        vm *= vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test MulAssign<Vector<i32>> for VectorMut<i32> — size N=10 (non-lane-multiple)
#[test]
fn test_vector_mut_mul_assign_i32_n10() {
    let mut data_a: [i32; 10] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let expected: [i32; 10] = [6, 12, 20, 30, 42, 56, 72, 90, 110, 132];

    {
        let mut vm: VectorMut<i32, 10> = <VectorMut<i32, 10>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [i32; 10] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let vb: Vector<i32, 10> = <Vector<i32, 10>>::try_from(&data_b).unwrap();

        vm *= vb;
    }

    for i in 0..10 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test MulAssign<i32> scalar for VectorMut<i32> — size N=8
#[test]
fn test_vector_mut_scalar_mul_assign_i32_n8() {
    let mut data_a: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let expected: [i32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

    {
        let mut vm: VectorMut<i32, 8> = <VectorMut<i32, 8>>::try_from(&mut data_a[..]).unwrap();

        vm *= 10;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test MulAssign<Vector<f32>> for VectorMut<f32> — size N=8
#[test]
fn test_vector_mut_mul_assign_f32_n8() {
    let mut data_a: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected: [f32; 8] = [2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0];

    {
        let mut vm: VectorMut<f32, 8> = <VectorMut<f32, 8>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f32; 8] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let vb: Vector<f32, 8> = <Vector<f32, 8>>::try_from(&data_b).unwrap();

        vm *= vb;
    }

    for i in 0..8 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test MulAssign<f32> scalar for VectorMut<f32> — size N=5 (non-lane-multiple)
#[test]
fn test_vector_mut_scalar_mul_assign_f32_n5() {
    let mut data_a: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected: [f32; 5] = [2.0, 4.0, 6.0, 8.0, 10.0];

    {
        let mut vm: VectorMut<f32, 5> = <VectorMut<f32, 5>>::try_from(&mut data_a[..]).unwrap();

        vm *= 2.0;
    }

    for i in 0..5 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test MulAssign<Vector<f64>> for VectorMut<f64> — size N=4 (LANES)
#[test]
fn test_vector_mut_mul_assign_f64_n4() {
    let mut data_a: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let expected: [f64; 4] = [2.0, 6.0, 12.0, 20.0];

    {
        let mut vm: VectorMut<f64, 4> = <VectorMut<f64, 4>>::try_from(&mut data_a[..]).unwrap();

        let data_b: [f64; 4] = [2.0, 3.0, 4.0, 5.0];
        let vb: Vector<f64, 4> = <Vector<f64, 4>>::try_from(&data_b).unwrap();

        vm *= vb;
    }

    for i in 0..4 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test MulAssign<f64> scalar for VectorMut<f64> — size N=5 (non-lane-multiple)
#[test]
fn test_vector_mut_scalar_mul_assign_f64_n5() {
    let mut data_a: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected: [f64; 5] = [3.0, 6.0, 9.0, 12.0, 15.0];

    {
        let mut vm: VectorMut<f64, 5> = <VectorMut<f64, 5>>::try_from(&mut data_a[..]).unwrap();

        vm *= 3.0;
    }

    for i in 0..5 {
        assert_eq!(data_a[i], expected[i], "Mismatch at index {}", i);
    }
}
