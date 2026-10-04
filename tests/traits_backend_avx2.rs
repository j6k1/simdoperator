use simdoperator::{Vector, Matrix};

#[test]
fn test_vector_add_avx2() {
    let data_a: [f64; 3] = [1.0, 2.0, 3.0];
    // Direct view construction for AVX2 backend tests
    let a: Vector<f64, 3> = <Vector<f64, 3>>::try_from(&data_a).unwrap();

    let data_b: [f64; 3] = [4.0, 5.0, 6.0];
    let b: Vector<f64, 3> = <Vector<f64, 3>>::try_from(&data_b).unwrap();

    let c = a + b;
    assert_eq!(c.as_ref(), &[5.0, 7.0, 9.0]);
}
/*
#[test]
fn test_matrix_mul_avx2() {
    let data_m1: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; 
    let m1: Matrix<f64, 3, 3> = <Matrix<f64, 3, 3>>::try_from(data_m1.as_ref()).unwrap();

    let data_m2: [f64; 9] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let m2: Matrix<f64, 3, 3> = <Matrix<f64, 3, 3>>::try_from(data_m2.as_ref()).unwrap();

    let res = m1 * m2;
    assert_eq!(res, &data_m2);
}
*/