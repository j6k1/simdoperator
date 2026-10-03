#![cfg(target_arch = "x86_64")]

use simdoperator::backend::autoselect::AutoSelect;
use simdoperator::backend::avx2::Avx2;
use simdoperator::traits::{BindBackend, Dot, Product, SimdDot, ToColumnMajor, Transpose};
use simdoperator::{AccMatrix, AccVector, ColumnMajorMatrix, Matrix, MatrixMut, MatrixView, OwnedColumnMajorMatrix, OwnedMatrix, OwnedVector, Scalar, Vector, VectorMut, VectorMutView, VectorView};
use simdoperator::backend::common::Backend;

fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn acc_vector_array<T, BE, const N: usize>(v: AccVector<T,N,BE>) -> [T; N] where BE: Backend {
    *Box::<[T; N]>::from(v)
}

fn acc_matrix_vec<T, BE, const N: usize, const M: usize>(m: AccMatrix<T,N,M,BE>) -> Vec<T> where BE: Backend {
    Box::<[T]>::from(m).into_vec()
}

fn owend_matrix_vec<T, const N: usize, const M: usize>(m: OwnedMatrix<T,N,M>) -> Vec<T> {
    Box::<[T]>::from(m).into_vec()
}

fn owned_column_major_vec<T, const N: usize, const M: usize>(
    m: OwnedColumnMajorMatrix<T, N, M>,
) -> Vec<T> {
    Box::<[T]>::from(m).into_vec()
}

#[test]
fn vector_matrix_view_and_owned_type_methods() {
    if !avx2_available() {
        return;
    }

    let vector_data = [1_i32, 2, 3, 4];
    let vector = Vector::<i32, 4>::try_from(&vector_data[..]).unwrap();
    assert_eq!(vector[2], 3);
    assert_eq!(vector.as_ref(), &vector_data);
    assert_eq!(*Box::<[i32; 4]>::from(&vector), vector_data);
    assert_eq!(vector.as_vertical()[2], [3]);
    assert_eq!(vector.as_horizontal()[0], [1, 2, 3, 4]);
    assert!(Vector::<i32, 5>::try_from(&vector_data[..]).is_err());

    let vector_view = VectorView::<i32, 4>::try_from(&vector_data[..]).unwrap();
    assert_eq!(vector_view[1], 2);
    assert_eq!(vector_view.as_ref(), &vector_data);
    assert_eq!(*Box::<[i32; 4]>::from(&vector_view), vector_data);
    assert_eq!(vector_view.as_vertical()[3], [4]);
    assert_eq!(vector_view.as_horizontal()[0], [1, 2, 3, 4]);
    assert!(VectorView::<i32, 5>::try_from(&vector_data[..]).is_err());

    let mut acc_vector = AccVector::<_,_,AutoSelect>::try_from(Box::new([1_i32, 2, 3, 4])).unwrap();
    {
        acc_vector[1] = 20;
        acc_vector.as_mut()[2] = 30;
        assert_eq!(acc_vector.as_ref(), &[1, 20, 30, 4]);
    }
    assert_eq!(acc_vector_array(acc_vector), [1, 20, 30, 4]);

    let matrix_data = [1_i32, 2, 3, 4, 5, 6];
    let matrix = Matrix::<i32, 2, 3>::try_from(&matrix_data[..]).unwrap();
    assert_eq!(matrix[0], [1, 2, 3]);
    assert_eq!(matrix[1], [4, 5, 6]);
    assert_eq!(matrix.row(1).as_ref(), &[4, 5, 6]);
    assert_eq!(owend_matrix_vec(matrix.transpose()), vec![1, 4, 2, 5, 3, 6]);

    let matrix = Matrix::<i32, 2, 3>::try_from(&matrix_data[..]).unwrap();
    assert_eq!(
        owned_column_major_vec(matrix.to_column_major()),
        vec![1, 4, 2, 5, 3, 6]
    );
    assert!(Matrix::<i32, 2, 4>::try_from(&matrix_data[..]).is_err());

    let matrix_view = MatrixView::<i32, 2, 3>::try_from(&matrix_data[..]).unwrap();
    assert_eq!(matrix_view[0], [1, 2, 3]);
    assert_eq!(matrix_view.row(0).as_ref(), &[1, 2, 3]);
    assert_eq!(
        owend_matrix_vec(matrix_view.transpose()),
        vec![1, 4, 2, 5, 3, 6]
    );

    let matrix_view = MatrixView::<i32, 2, 3>::try_from(&matrix_data[..]).unwrap();
    assert_eq!(
        owned_column_major_vec(matrix_view.to_column_major()),
        vec![1, 4, 2, 5, 3, 6]
    );
    assert!(MatrixView::<i32, 2, 4>::try_from(&matrix_data[..]).is_err());

    let mut owned_matrix = OwnedMatrix::<i32, 2, 3>::default();
    owned_matrix[(0, 1)] = 7;
    owned_matrix[(1, 2)] = 9;
    owned_matrix.as_mut()[0] = 5;
    {
        let mut matrix_mut = MatrixMut::from(&mut owned_matrix);
        matrix_mut[(1, 0)] = 11;
        matrix_mut.as_mut()[4] = 13;
    }
    assert_eq!(owend_matrix_vec(owned_matrix), vec![5, 7, 0, 11, 13, 9]);

    let col_major = ColumnMajorMatrix::<i32, 2, 3>::try_from(&[1, 2, 3, 4, 5, 6][..]).unwrap();
    assert_eq!(col_major.col(0).as_ref(), &[1, 2]);
    assert_eq!(col_major.col(2).as_ref(), &[5, 6]);
    assert!(ColumnMajorMatrix::<i32, 2, 4>::try_from(&[1, 2, 3, 4, 5, 6][..]).is_err());

    let owned_col_major =
        OwnedColumnMajorMatrix::<i32, 2, 3>::from(vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    let col_major = ColumnMajorMatrix::from(&owned_col_major);
    assert_eq!(col_major.col(1).as_ref(), &[3, 4]);
    assert_eq!(
        owned_column_major_vec(OwnedColumnMajorMatrix::from(&col_major)),
        vec![1, 2, 3, 4, 5, 6]
    );
}

#[test]
fn vector_operator_traits_with_unspecified_backend() {
    if !avx2_available() {
        return;
    }

    let l_data = [1_i32, 2, 3, 4, 5, 6, 7, 8, 9];
    let r_data = [9_i32, 8, 7, 6, 5, 4, 3, 2, 1];
    let l = Vector::<i32, 9>::try_from(&l_data[..]).unwrap();
    let r = Vector::<i32, 9>::try_from(&r_data[..]).unwrap();

    assert_eq!(acc_vector_array(l.clone() + r.clone()), [10; 9]);
    assert_eq!(acc_vector_array(l.clone() - r.clone()), [-8, -6, -4, -2, 0, 2, 4, 6, 8]);
    assert_eq!(
        acc_vector_array(l.clone() * r.clone()),
        [9, 16, 21, 24, 25, 24, 21, 16, 9]
    );
    assert_eq!(
        acc_vector_array(l.clone() * 3_i32),
        [3, 6, 9, 12, 15, 18, 21, 24, 27]
    );
    assert_eq!(l.dot(r.clone()), 165);

    let mut acc = AccVector::try_from(Box::new(l_data)).unwrap();
    {
        acc += r.clone();
        acc -= r.clone();
        acc *= r.clone();
        acc *= 2;
    }
    assert_eq!(
        acc_vector_array(acc),
        [18, 32, 42, 48, 50, 48, 42, 32, 18]
    );
}

#[test]
fn vector_operator_traits_with_avx2_backend() {
    if !avx2_available() {
        return;
    }

    let l_data = [1_i32, 2, 3, 4, 5, 6, 7, 8, 9];
    let r_data = [9_i32, 8, 7, 6, 5, 4, 3, 2, 1];
    let l = Vector::<i32, 9, Avx2>::try_from(&l_data[..]).unwrap();
    let r = Vector::<i32, 9, Avx2>::try_from(&r_data[..]).unwrap();

    assert_eq!(acc_vector_array(l.clone() + r.clone()), [10; 9]);
    assert_eq!(acc_vector_array(l.clone() - r.clone()), [-8, -6, -4, -2, 0, 2, 4, 6, 8]);
    assert_eq!(
        acc_vector_array(l.clone() * r.clone()),
        [9, 16, 21, 24, 25, 24, 21, 16, 9]
    );
    assert_eq!(
        acc_vector_array(l.clone() * 3_i32),
        [3, 6, 9, 12, 15, 18, 21, 24, 27]
    );
    assert_eq!(l.dot(r.clone()), 165);

    let mut acc = AccVector::<_,_,Avx2>::try_from(Box::new(l_data)).unwrap();
    {
        acc += r.clone();
        acc -= r.clone();
        acc *= r.clone();
        acc *= 2;
    }
    assert_eq!(
        acc_vector_array(acc),
        [18, 32, 42, 48, 50, 48, 42, 32, 18]
    );
}

#[test]
fn vector_operator_traits_with_autoselect_backend() {
    if !avx2_available() {
        return;
    }

    let l_data = [1_i32, 2, 3, 4, 5, 6, 7, 8, 9];
    let r_data = [9_i32, 8, 7, 6, 5, 4, 3, 2, 1];
    let l = Vector::<i32, 9, AutoSelect>::try_from(&l_data[..]).unwrap();
    let r = Vector::<i32, 9, AutoSelect>::try_from(&r_data[..]).unwrap();

    assert_eq!(acc_vector_array(l.clone() + r.clone()), [10; 9]);
    assert_eq!(acc_vector_array(l.clone() - r.clone()), [-8, -6, -4, -2, 0, 2, 4, 6, 8]);
    assert_eq!(
        acc_vector_array(l.clone() * r.clone()),
        [9, 16, 21, 24, 25, 24, 21, 16, 9]
    );
    assert_eq!(
        acc_vector_array(l.clone() * 3_i32),
        [3, 6, 9, 12, 15, 18, 21, 24, 27]
    );
    assert_eq!(l.dot(r), 165);
}

#[test]
fn matrix_product_traits_with_unspecified_backend() {
    if !avx2_available() {
        return;
    }

    let matrix_data = [1_i32, 2, 3, 4, 5, 6];
    let vector_data = [7_i32, 8, 9];
    let col_major_data = [1_i32, 2, 3, 4, 5, 6];
    let matrix = Matrix::<i32, 2, 3>::try_from(&matrix_data[..]).unwrap();
    let vector = Vector::<i32, 3>::try_from(&vector_data[..]).unwrap();
    let col_major = ColumnMajorMatrix::<i32, 3, 2>::try_from(&col_major_data[..]).unwrap();

    let matvec: AccVector<i32,2,AutoSelect> = matrix.product(vector.clone());
    assert_eq!(acc_vector_array(matvec), [50, 122]);

    let vmat: AccVector<i32,2,AutoSelect> = vector.product(col_major.clone());
    assert_eq!(acc_vector_array(vmat), [50, 122]);

    let matmul: AccMatrix<i32,2, 2, AutoSelect> = matrix.product(col_major.clone());
    assert_eq!(acc_matrix_vec(matmul), vec![14, 32, 32, 77]);
}

#[test]
fn matrix_product_and_assign_traits_with_avx2_backend() {
    if !avx2_available() {
        return;
    }

    let matrix_data = [1_i32, 2, 3, 4, 5, 6];
    let rhs_matrix_data = [6_i32, 5, 4, 3, 2, 1];
    let vector_data = [7_i32, 8, 9];
    let col_major_data = [1_i32, 2, 3, 4, 5, 6];
    let matrix = Matrix::<i32, 2, 3, Avx2>::try_from(&matrix_data[..]).unwrap();
    let vector = Vector::<i32, 3, Avx2>::try_from(&vector_data[..]).unwrap();
    let col_major = ColumnMajorMatrix::<i32, 3, 2>::try_from(&col_major_data[..]).unwrap();

    let matvec: AccVector<i32,2,Avx2> = matrix.product(vector);
    assert_eq!(acc_vector_array(matvec), [50, 122]);

    let vmat: AccVector<i32,2,Avx2> = vector.product(col_major.clone());
    assert_eq!(acc_vector_array(vmat), [50, 122]);

    let matmul: AccMatrix<i32, 2, 2, Avx2> = matrix.product(col_major);
    assert_eq!(acc_matrix_vec(matmul), vec![14, 32, 32, 77]);

    let scaled: AccMatrix<i32, 2, 3, Avx2> = matrix.clone() * 2;
    assert_eq!(acc_matrix_vec(scaled), vec![2, 4, 6, 8, 10, 12]);

    let converted: AccMatrix<f32, 2, 3, Avx2> = AccMatrix::from(matrix);
    assert_eq!(
        acc_matrix_vec(converted),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let rhs = Matrix::<i32, 2, 3, Avx2>::try_from(&rhs_matrix_data[..]).unwrap();
    let mut acc = AccMatrix::<i32, 2, 3, Avx2>::try_from(matrix_data.to_vec().into_boxed_slice()).unwrap();
    {
        acc += rhs;
    }
    assert_eq!(acc_matrix_vec(acc), vec![7, 7, 7, 7, 7, 7]);

    let mut acc = AccMatrix::<i32, 2, 3, Avx2>::try_from(matrix_data.to_vec().into_boxed_slice()).unwrap();
    {
        acc *= 3;
    }
    assert_eq!(acc_matrix_vec(acc), vec![3, 6, 9, 12, 15, 18]);
}

#[test]
fn matrix_product_traits_with_autoselect_backend() {
    if !avx2_available() {
        return;
    }

    let matrix_data = [1_i32, 2, 3, 4, 5, 6];
    let vector_data = [7_i32, 8, 9];
    let col_major_data = [1_i32, 2, 3, 4, 5, 6];
    let matrix = Matrix::<i32, 2, 3, AutoSelect>::try_from(&matrix_data[..]).unwrap();
    let vector = Vector::<i32, 3, AutoSelect>::try_from(&vector_data[..]).unwrap();
    let col_major = ColumnMajorMatrix::<i32, 3, 2>::try_from(&col_major_data[..]).unwrap();

    let matvec: AccVector<i32,2,AutoSelect> = matrix.product(vector.clone());
    assert_eq!(acc_vector_array(matvec), [50, 122]);

    let vmat: AccVector<i32,2,AutoSelect> = vector.product(col_major.clone());
    assert_eq!(acc_vector_array(vmat), [50, 122]);

    let matmul: AccMatrix<i32, 2, 2, AutoSelect> = matrix.product(col_major);
    assert_eq!(acc_matrix_vec(matmul), vec![14, 32, 32, 77]);
}

#[test]
fn bind_and_bind_auto_chain_vector_and_matrix_results() {
    if !avx2_available() {
        return;
    }

    let a_owned = OwnedVector::from(Box::new([1_i32, 2, 3, 4, 5, 6, 7, 8, 9]));
    let b_owned = OwnedVector::from(Box::new([9_i32, 8, 7, 6, 5, 4, 3, 2, 1]));
    let c_owned = OwnedVector::from(Box::new([1_i32; 9]));

    let a = (&a_owned).bind::<Avx2>().unwrap();
    let b = (&b_owned).bind::<Avx2>().unwrap();
    let c = (&c_owned).bind::<Avx2>().unwrap();
    let added = &a + &b;
    let chained = &added - &c;
    assert_eq!(acc_vector_array(chained), [9; 9]);

    let multiplied = added * &c;
    let multiplied = (&multiplied).bind_auto().unwrap();
    let c_auto = (&c_owned).bind_auto().unwrap();
    assert_eq!(multiplied.dot(&c_auto), 90);

    let matrix_owned = OwnedMatrix::<i32, 2, 3>::from(vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    let vector_owned = OwnedVector::from(Box::new([7_i32, 8, 9]));
    let matrix = (&matrix_owned).bind::<Avx2>().unwrap();
    let vector = (&vector_owned).bind::<Avx2>().unwrap();
    let matvec: AccVector<i32, 2, Avx2> = matrix.product((&vector).into());
    let matvec = (&matvec).bind_auto().unwrap();
    let weights_owned = OwnedVector::from(Box::new([2_i32, 3]));
    let weights = (&weights_owned).bind_auto().unwrap();
    assert_eq!(unsafe { matvec.dot(&weights) }, 466);
}
