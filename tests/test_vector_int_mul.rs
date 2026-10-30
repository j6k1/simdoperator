//! Tests for Vector left-hand side integer multiplication operations that are not yet covered.
//!
//! Test targets:
//! - Mul<Vector<i8>> for Vector<i8> (outputs i32)
//! - Mul<Vector<i16>> for Vector<i16> (outputs i32)
//! - Mul<i8> scalar for Vector<i8> (outputs i32)
//! - Mul<i16> scalar for Vector<i16> (outputs i32)

use simdoperator::Vector;

/// Test Mul<Vector<i8>> for Vector<i8> — size N=8
#[test]
fn test_vector_mul_i8_n8() {
    let data_a: [i8; 8] = [2, 3, 4, 5, 6, 7, 8, 9];
    let va: Vector<i8, 8> = <Vector<i8, 8>>::try_from(&data_a).unwrap();

    let data_b: [i8; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
    let vb: Vector<i8, 8> = <Vector<i8, 8>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    // Result is i32 because i8*i8 promotes to i32
    for i in 0..8 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}

/// Test Mul<Vector<i8>> for Vector<i8> — size N=16 (multiple of LANES)
#[test]
fn test_vector_mul_i8_n16() {
    let data_a: [i8; 16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let va: Vector<i8, 16> = <Vector<i8, 16>>::try_from(&data_a).unwrap();

    let data_b: [i8; 16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let vb: Vector<i8, 16> = <Vector<i8, 16>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    for i in 0..16 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}

/// Test Mul<Vector<i8>> for Vector<i8> — size N=10 (non-lane-multiple)
#[test]
fn test_vector_mul_i8_n10() {
    let data_a: [i8; 10] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    let va: Vector<i8, 10> = <Vector<i8, 10>>::try_from(&data_a).unwrap();

    let data_b: [i8; 10] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let vb: Vector<i8, 10> = <Vector<i8, 10>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    for i in 0..10 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}

/// Test Mul<Vector<i16>> for Vector<i16> — size N=8
#[test]
fn test_vector_mul_i16_n8() {
    let data_a: [i16; 8] = [2, 3, 4, 5, 6, 7, 8, 9];
    let va: Vector<i16, 8> = <Vector<i16, 8>>::try_from(&data_a).unwrap();

    let data_b: [i16; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
    let vb: Vector<i16, 8> = <Vector<i16, 8>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    // Result is i32 because i16*i16 promotes to i32
    for i in 0..8 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}

/// Test Mul<Vector<i16>> for Vector<i16> — size N=16
#[test]
fn test_vector_mul_i16_n16() {
    let data_a: [i16; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    let va: Vector<i16, 16> = <Vector<i16, 16>>::try_from(&data_a).unwrap();

    let data_b: [i16; 16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let vb: Vector<i16, 16> = <Vector<i16, 16>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    for i in 0..16 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}

/// Test Mul<Vector<i16>> for Vector<i16> — size N=10 (non-lane-multiple)
#[test]
fn test_vector_mul_i16_n10() {
    let data_a: [i16; 10] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
    let va: Vector<i16, 10> = <Vector<i16, 10>>::try_from(&data_a).unwrap();

    let data_b: [i16; 10] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let vb: Vector<i16, 10> = <Vector<i16, 10>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    for i in 0..10 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}

/// Test Mul<i8> scalar for Vector<i8> — size N=8
#[test]
fn test_vector_scalar_mul_i8_n8() {
    let data_a: [i8; 8] = [2, 3, 4, 5, 6, 7, 8, 9];
    let va: Vector<i8, 8> = <Vector<i8, 8>>::try_from(&data_a).unwrap();

    let vc = va * 5i8;

    // Result is i32
    for i in 0..8 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * 5, "Mismatch at index {}", i);
    }
}

/// Test Mul<i8> scalar for Vector<i8> — size N=16
#[test]
fn test_vector_scalar_mul_i8_n16() {
    let data_a: [i8; 16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let va: Vector<i8, 16> = <Vector<i8, 16>>::try_from(&data_a).unwrap();

    let vc = va * 3i8;

    for i in 0..16 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * 3, "Mismatch at index {}", i);
    }
}

/// Test Mul<i8> scalar for Vector<i8> — size N=10 (non-lane-multiple)
#[test]
fn test_vector_scalar_mul_i8_n10() {
    let data_a: [i8; 10] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    let va: Vector<i8, 10> = <Vector<i8, 10>>::try_from(&data_a).unwrap();

    let vc = va * 2i8;

    for i in 0..10 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * 2, "Mismatch at index {}", i);
    }
}

/// Test Mul<i16> scalar for Vector<i16> — size N=8
#[test]
fn test_vector_scalar_mul_i16_n8() {
    let data_a: [i16; 8] = [2, 3, 4, 5, 6, 7, 8, 9];
    let va: Vector<i16, 8> = <Vector<i16, 8>>::try_from(&data_a).unwrap();

    let vc = va * 10i16;

    // Result is i32
    for i in 0..8 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * 10, "Mismatch at index {}", i);
    }
}

/// Test Mul<i16> scalar for Vector<i16> — size N=16
#[test]
fn test_vector_scalar_mul_i16_n16() {
    let data_a: [i16; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    let va: Vector<i16, 16> = <Vector<i16, 16>>::try_from(&data_a).unwrap();

    let vc = va * 2i16;

    for i in 0..16 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * 2, "Mismatch at index {}", i);
    }
}

/// Test Mul<i16> scalar for Vector<i16> — size N=10 (non-lane-multiple)
#[test]
fn test_vector_scalar_mul_i16_n10() {
    let data_a: [i16; 10] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
    let va: Vector<i16, 10> = <Vector<i16, 10>>::try_from(&data_a).unwrap();

    let vc = va * 3i16;

    for i in 0..10 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * 3, "Mismatch at index {}", i);
    }
}

/// Test Mul<i8>> negative values — size N=8
#[test]
fn test_vector_mul_i8_negative_n8() {
    let data_a: [i8; 8] = [-2, -3, -4, 5, -6, 7, -8, 9];
    let va: Vector<i8, 8> = <Vector<i8, 8>>::try_from(&data_a).unwrap();

    let data_b: [i8; 8] = [-3, 4, -5, 6, -7, 8, -9, 10];
    let vb: Vector<i8, 8> = <Vector<i8, 8>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    for i in 0..8 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}

/// Test Mul<i16>> negative values — size N=8
#[test]
fn test_vector_mul_i16_negative_n8() {
    let data_a: [i16; 8] = [-2, -3, -4, 5, -6, 7, -8, 9];
    let va: Vector<i16, 8> = <Vector<i16, 8>>::try_from(&data_a).unwrap();

    let data_b: [i16; 8] = [-3, 4, -5, 6, -7, 8, -9, 10];
    let vb: Vector<i16, 8> = <Vector<i16, 8>>::try_from(&data_b).unwrap();

    let vc = va * vb;

    for i in 0..8 {
        assert_eq!(vc[i] as i64, (data_a[i] as i64) * (data_b[i] as i64), "Mismatch at index {}", i);
    }
}
