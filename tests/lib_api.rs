use simdoperator::{AccVector, Vector};
use simdoperator::traits::Demote;

#[test]
fn test_vector_add_auto() {
    // Direct construction from fixed-size array (removes .bind())
    let data_a: [f64; 3] = [1.0, 2.0, 3.0];
    let a: Vector<f64, 3> = <Vector<f64, 3>>::try_from(&data_a).unwrap();

    let data_b: [f64; 3] = [4.0, 5.0, 6.0];
    let b: Vector<f64, 3> = <Vector<f64, 3>>::try_from(&data_b).unwrap();

    // Test Addition trait implementation (SimdAddVector)
    let c = a + b;
    
    assert_eq!(c.as_ref(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_vector_sub_auto() {
    let data_a: [f64; 3] = [4.0, 5.0, 6.0];
    let a: Vector<f64, 3> = <Vector<f64, 3>>::try_from(&data_a).unwrap();

    let data_b: [f64; 3] = [1.0, 2.0, 3.0];
    let b: Vector<f64, 3> = <Vector<f64, 3>>::try_from(&data_b).unwrap();

    // Test Subtraction trait implementation (SimdSubVector)
    let c = a - b;
    
    assert_eq!(c.as_ref(), &[3.0, 3.0, 3.0]);
}

#[test]
fn test_vector_scalar_mul_auto() {
    let data_a: [f64; 3] = [2.0, 3.0, 4.0];
    let a: Vector<f64, 3> = <Vector<f64, 3>>::try_from(&data_a).unwrap();

    // Test Scalar Multiplication trait implementation (SimdScalarMulVector)
    // Note: Vector-Vector mul for f64 is not currently implemented in src/lib.rs
    let c = a * 2.0;
    
    assert_eq!(c.as_ref(), &[4.0, 6.0, 8.0]);
}
#[test]
fn test_demote_i32_to_i16_layout() {
    let data: [i32; 16] = [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15
    ];

    let v = <Vector<i32, 16>>::try_from(&data).unwrap();

    let result: AccVector<i16, 16> = v.demotion();

    for i in 0..16 {
        assert_eq!(result[i], data[i] as i16, "Index {}: expected {}, got {}", i, data[i], result[i]);
    }
}
#[test]
fn test_demote_i16_to_i8_layout() {
    let data: [i16; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    ];

    let v = <Vector<i16, 32>>::try_from(&data).unwrap();

    let result: AccVector<i8, 32> = v.demotion();

    for i in 0..16 {
        assert_eq!(result[i], data[i] as i8, "Index {}: expected {}, got {}", i, data[i], result[i]);
    }
}
#[test]
fn test_vector_shr_avx2_n16() {
    let data_a: [i32; 16] = [
        16, 32, 48, 64, 80, 96, 112, 128,
        144, 160, 176, 192, 208, 224, 240, 256
    ];
    let a: Vector<i32, 16> = <Vector<i32, 16>>::try_from(&data_a).unwrap();

    let c = a >> 2;

    let expected: [i32; 16] = [
        4, 8, 12, 16, 20, 24, 28, 32,
        36, 40, 44, 48, 52, 56, 60, 64
    ];

    assert_eq!(c.as_ref(), &expected);
}
#[test]
fn test_vector_shr_avx2_n10() {
    let data_a: [i32; 10] = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160];
    let a: Vector<i32, 10> = <Vector<i32, 10>>::try_from(&data_a).unwrap();

    let c = a >> 2;

    let expected: [i32; 10] = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40];
    assert_eq!(c.as_ref(), &expected);
}
#[test]
fn test_vector_add_i32_avx2_n16() {
    let data_a: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let a: Vector<i32, 16> = <Vector<i32, 16>>::try_from(&data_a).unwrap();

    let data_b: [i32; 16] = [100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115];
    let b: Vector<i32, 16> = <Vector<i32, 16>>::try_from(&data_b).unwrap();

    let c = a + b;

    let expected: [i32; 16] = [100, 102, 104, 106, 108, 110, 112, 114,
        116, 118, 120, 122, 124, 126, 128, 130];
    assert_eq!(c.as_ref(), &expected);
}
#[test]
fn test_vector_add_i32_avx2_n10() {
    let data_a: [i32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let a: Vector<i32, 10> = <Vector<i32, 10>>::try_from(&data_a).unwrap();

    let data_b: [i32; 10] = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109];
    let b: Vector<i32, 10> = <Vector<i32, 10>>::try_from(&data_b).unwrap();

    let c = a + b;

    let expected: [i32; 10] = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118];
    assert_eq!(c.as_ref(), &expected);
}
#[test]
fn test_vector_shr_negative_i32_n16() {
    let data_a: [i32; 16] = [-4, -8, -16, -32, -64, -128, -256, -512,
        -1024, -2048, -4096, -8192, -16384, -32768, -65536, -131072];
    let a: Vector<i32, 16> = <Vector<i32, 16>>::try_from(&data_a).unwrap();

    let c = a >> 2;

    // -4 >> 2 = -1, -8 >> 2 = -2, etc.
    let expected: [i32; 16] = [-1, -2, -4, -8, -16, -32, -64, -128,
        -256, -512, -1024, -2048, -4096, -8192, -16384, -32768];

    assert_eq!(c.as_ref(), &expected);
}
#[test]
fn test_vector_shr_negative_i32_n10() {
    let data_a: [i32; 10] = [-4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048];
    let a: Vector<i32, 10> = <Vector<i32, 10>>::try_from(&data_a).unwrap();

    let c = a >> 2;

    let expected: [i32; 10] = [-1, -2, -4, -8, -16, -32, -64, -128, -256, -512];

    assert_eq!(c.as_ref(), &expected);
}