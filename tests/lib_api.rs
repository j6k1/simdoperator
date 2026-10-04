use simdoperator::{Vector, Matrix};

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

/*
#[test]
fn test_matrix_mul_auto() {
    // Matrix multiplication is not yet implemented, so this remains commented out.
    let data_m1: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; 
    let m1: Matrix<f64, 3, 3> = <Matrix<f64, 3, 3>>::try_from(data_m1.as_ref()).unwrap();

    let data_m2: [f64; 9] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let m2: Matrix<f64, 3, 3> = <Matrix<f64, 3, 3>>::try_from(data_m2.as_ref()).unwrap();

    // Multiply Identity by M2 -> Result should be M2
    let res = m1 * m2;

    assert_eq!(res, &data_m2);
}
*/
