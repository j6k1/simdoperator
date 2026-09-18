#![cfg(target_arch = "x86_64")]

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

use simdoperator::backend::avx2::Avx2;
use simdoperator::backend::common::{Backend, Regs};
use simdoperator::traits::*;
use simdoperator::{
    ColumnMajorMatrix, Matrix, MatrixMut, OwnedMatrix, OwnedVector, Vector, VectorMut,
};

fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn vector<'a, T, BE, const N: usize>(data: &'a [T; N]) -> Vector<'a, T, N, BE>
where
    BE: Backend,
{
    Vector::try_from(&data[..]).unwrap()
}

fn matrix<'a, T, BE, const N: usize, const M: usize>(data: &'a [T]) -> Matrix<'a, T, N, M, BE>
where
    BE: Backend,
{
    Matrix::try_from(data).unwrap()
}

fn column_major<'a, T, BE, const N: usize, const M: usize>(
    data: &'a [T],
) -> ColumnMajorMatrix<'a, T, N, M, BE>
where
    BE: Backend,
{
    ColumnMajorMatrix::try_from(data).unwrap()
}

fn owned_vector_array<T, const N: usize>(v: OwnedVector<T, N>) -> [T; N] {
    *Box::<[T; N]>::from(v)
}

fn owned_matrix_vec<T, const N: usize, const M: usize>(m: OwnedMatrix<T, N, M>) -> Vec<T> {
    Box::<[T]>::from(m).into_vec()
}

fn assert_f32_bits_eq<const N: usize>(actual: [f32; N], expected: [f32; N]) {
    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
}

fn assert_f64_bits_eq<const N: usize>(actual: [f64; N], expected: [f64; N]) {
    assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
}

macro_rules! lane_cases_same {
    ($func:ident::<$be:ty, $ty:ty>; $less:expr, $equal:expr, $greater:expr, $div:expr, $nondiv:expr) => {{
        $func::<$be, $ty, $less>();
        $func::<$be, $ty, $equal>();
        $func::<$be, $ty, $greater>();
        $func::<$be, $ty, $div>();
        $func::<$be, $ty, $nondiv>();
    }};
}

macro_rules! lane_cases_mul {
    ($be:ty, $sl:ty, $sr:ty, $so:ty; $less:expr, $equal:expr, $greater:expr, $div:expr, $nondiv:expr) => {{
        check_mul_vector::<$be, $sl, $sr, $so, $less>();
        check_mul_vector::<$be, $sl, $sr, $so, $equal>();
        check_mul_vector::<$be, $sl, $sr, $so, $greater>();
        check_mul_vector::<$be, $sl, $sr, $so, $div>();
        check_mul_vector::<$be, $sl, $sr, $so, $nondiv>();
    }};
}

macro_rules! lane_cases_scalar {
    ($be:ty, $sl:ty, $sr:ty, $so:ty, $scalar:expr; $less:expr, $equal:expr, $greater:expr, $div:expr, $nondiv:expr) => {{
        check_scalarmul_vector::<$be, $sl, $sr, $so, $less>($scalar);
        check_scalarmul_vector::<$be, $sl, $sr, $so, $equal>($scalar);
        check_scalarmul_vector::<$be, $sl, $sr, $so, $greater>($scalar);
        check_scalarmul_vector::<$be, $sl, $sr, $so, $div>($scalar);
        check_scalarmul_vector::<$be, $sl, $sr, $so, $nondiv>($scalar);
    }};
}

macro_rules! lane_cases_scalar_assign {
    ($be:ty, $ty:ty, $scalar:expr; $less:expr, $equal:expr, $greater:expr, $div:expr, $nondiv:expr) => {{
        check_scalarmul_assign_vector::<$be, $ty, $less>($scalar);
        check_scalarmul_assign_vector::<$be, $ty, $equal>($scalar);
        check_scalarmul_assign_vector::<$be, $ty, $greater>($scalar);
        check_scalarmul_assign_vector::<$be, $ty, $div>($scalar);
        check_scalarmul_assign_vector::<$be, $ty, $nondiv>($scalar);
    }};
}

macro_rules! lane_cases_bit {
    ($be:ty, $ty:ty, $bits:ty; $less:expr, $equal:expr, $greater:expr, $div:expr, $nondiv:expr) => {{
        check_bit_vector::<$be, $ty, $bits, $less>();
        check_bit_vector::<$be, $ty, $bits, $equal>();
        check_bit_vector::<$be, $ty, $bits, $greater>();
        check_bit_vector::<$be, $ty, $bits, $div>();
        check_bit_vector::<$be, $ty, $bits, $nondiv>();
    }};
}

fn check_add_assign_vector<BE, T, const N: usize>()
where
    BE: Backend + SimdAddAssignVector<T, T, Backend = BE>,
    T: Copy + Default + Add<Output = T> + From<i8> + PartialEq + Debug,
{
    let mut l = OwnedVector::from(Box::new(std::array::from_fn(|i| T::from((i % 9) as i8))));
    let r_data: [T; N] = std::array::from_fn(|i| T::from(3 - (i as i8 % 7)));
    let expected = std::array::from_fn(|i| l[i] + r_data[i]);
    let r = vector::<T, BE, N>(&r_data);
    let be = BE::new().unwrap();

    {
        let mut l_mut = VectorMut::from(&mut l);
        be.add_assign_vector(&mut l_mut, &r);
    }

    assert_eq!(owned_vector_array(l), expected);
}

fn check_add_vector<BE, T, const N: usize>()
where
    BE: Backend + SimdAddVector<T, T, T, Backend = BE>,
    T: Copy + Default + Add<Output = T> + From<i8> + PartialEq + Debug,
{
    let l_data: [T; N] = std::array::from_fn(|i| T::from((i % 9) as i8));
    let r_data: [T; N] = std::array::from_fn(|i| T::from(3 - (i as i8 % 7)));
    let expected = std::array::from_fn(|i| l_data[i] + r_data[i]);
    let be = BE::new().unwrap();
    let actual = be.add_vector(&vector::<T, BE, N>(&l_data), &vector::<T, BE, N>(&r_data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_sub_assign_vector<BE, T, const N: usize>()
where
    BE: Backend + SimdSubAssignVector<T, T, Backend = BE>,
    T: Copy + Default + Sub<Output = T> + From<i8> + PartialEq + Debug,
{
    let mut l = OwnedVector::from(Box::new(std::array::from_fn(|i| T::from((i % 9) as i8))));
    let r_data: [T; N] = std::array::from_fn(|i| T::from(3 - (i as i8 % 7)));
    let expected = std::array::from_fn(|i| l[i] - r_data[i]);
    let r = vector::<T, BE, N>(&r_data);
    let be = BE::new().unwrap();

    {
        let mut l_mut = VectorMut::from(&mut l);
        be.sub_assign_vector(&mut l_mut, &r);
    }

    assert_eq!(owned_vector_array(l), expected);
}

fn check_sub_vector<BE, T, const N: usize>()
where
    BE: Backend + SimdSubVector<T, T, T, Backend = BE>,
    T: Copy + Default + Sub<Output = T> + From<i8> + PartialEq + Debug,
{
    let l_data: [T; N] = std::array::from_fn(|i| T::from((i % 9) as i8));
    let r_data: [T; N] = std::array::from_fn(|i| T::from(3 - (i as i8 % 7)));
    let expected = std::array::from_fn(|i| l_data[i] - r_data[i]);
    let be = BE::new().unwrap();
    let actual = be.sub_vector(&vector::<T, BE, N>(&l_data), &vector::<T, BE, N>(&r_data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_mul_vector<BE, SL, SR, SO, const N: usize>()
where
    BE: Backend + SimdMulVector<SL, SR, SO, Backend = BE>,
    SL: Copy + From<i8> + Mul<SR, Output = SO>,
    SR: Copy + From<i8>,
    SO: Copy + Default + PartialEq + Debug,
{
    let l_data: [SL; N] = std::array::from_fn(|i| SL::from(((i % 5) + 1) as i8));
    let r_data: [SR; N] = std::array::from_fn(|i| SR::from(3 - (i as i8 % 5)));
    let expected = std::array::from_fn(|i| l_data[i] * r_data[i]);
    let be = BE::new().unwrap();
    let actual = be.mul_vector(&vector::<SL, BE, N>(&l_data), &vector::<SR, BE, N>(&r_data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_mul_assign_vector<BE, T, const N: usize>()
where
    BE: Backend + SimdMulAssignVector<T, T, Backend = BE>,
    T: Copy + From<i8> + Mul<Output = T> + PartialEq + Debug,
{
    let mut l = OwnedVector::from(Box::new(std::array::from_fn(|i| {
        T::from(((i % 5) + 1) as i8)
    })));
    let r_data: [T; N] = std::array::from_fn(|i| T::from(3 - (i as i8 % 5)));
    let expected = std::array::from_fn(|i| l[i] * r_data[i]);
    let r = vector::<T, BE, N>(&r_data);
    let be = BE::new().unwrap();

    {
        let mut l_mut = VectorMut::from(&mut l);
        be.mul_assign_vector(&mut l_mut, &r);
    }

    assert_eq!(owned_vector_array(l), expected);
}

fn check_scalarmul_vector<BE, SL, SR, SO, const N: usize>(scalar: SL)
where
    BE: Backend + SimdScalarMulVector<SL, SR, SO, Backend = BE>,
    SL: Copy + Mul<SR, Output = SO>,
    SR: Copy + From<i8>,
    SO: Copy + Default + PartialEq + Debug,
{
    let r_data: [SR; N] = std::array::from_fn(|i| SR::from(3 - (i as i8 % 5)));
    let expected = std::array::from_fn(|i| scalar * r_data[i]);
    let be = BE::new().unwrap();
    let actual = be.scalarmul_vector(scalar, &vector::<SR, BE, N>(&r_data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_scalarmul_assign_vector<BE, T, const N: usize>(scalar: T)
where
    BE: Backend + SimdScalarMulAssignVector<T, T, Backend = BE>,
    T: Copy + From<i8> + Mul<Output = T> + PartialEq + Debug,
{
    let mut r: OwnedVector<T, N> = OwnedVector::from(Box::new(std::array::from_fn(|i| {
        T::from(3 - (i as i8 % 5))
    })));
    let expected: [T; N] = std::array::from_fn(|i| scalar * r[i]);
    let be = BE::new().unwrap();

    {
        let mut r_mut = VectorMut::from(&mut r);
        be.scalarmul_assign_vector(scalar, &mut r_mut);
    }

    assert_eq!(owned_vector_array(r), expected);
}

fn check_bit_vector<BE, T, B, const N: usize>()
where
    BE: Backend
        + SimdReg<T, Bits = B>
        + SimdBitAndVector<T, Backend = BE>
        + SimdBitOrVector<T, Backend = BE>
        + SimdBitXorVector<T, Backend = BE>,
    T: BitsBitAnd<Bits = B>
        + BitsBitOr<Bits = B>
        + BitsBitXor<Bits = B>
        + Copy
        + Default
        + PartialEq
        + Debug
        + From<i8>,
    B: Copy + From<i8>,
{
    let l_data: [T; N] = std::array::from_fn(|i| T::from((i as i8) ^ 0x55));
    let r_data: [B; N] = std::array::from_fn(|i| B::from((i as i8) | 0x0f));
    let expected_and = std::array::from_fn(|i| l_data[i].bits_bitand(r_data[i]));
    let expected_or = std::array::from_fn(|i| l_data[i].bits_bitor(r_data[i]));
    let expected_xor = std::array::from_fn(|i| l_data[i].bits_bitxor(r_data[i]));
    let l = vector::<T, BE, N>(&l_data);
    let r = vector::<B, BE, N>(&r_data);
    let be = BE::new().unwrap();

    assert_eq!(owned_vector_array(be.bitand_vector(&l, &r)), expected_and);
    assert_eq!(owned_vector_array(be.bitor_vector(&l, &r)), expected_or);
    assert_eq!(owned_vector_array(be.bitxor_vector(&l, &r)), expected_xor);
}

fn check_bitnot_vector<BE, T, const N: usize>()
where
    BE: Backend + SimdBitNotVector<T, Backend = BE>,
    T: BitsBitNot + Copy + Default + PartialEq + Debug + From<i8>,
{
    let data: [T; N] = std::array::from_fn(|i| T::from((i as i8) ^ 0x55));
    let expected = std::array::from_fn(|i| data[i].bits_bitnot());
    let be = BE::new().unwrap();
    let actual = be.bitnot_vector(&vector::<T, BE, N>(&data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_shift_vector<BE, T, const N: usize>()
where
    BE: Backend + SimdShlVector<T, Backend = BE> + SimdShrVector<T, Backend = BE>,
    T: BitsShl + BitsShr + Copy + Default + PartialEq + Debug + From<i8>,
{
    let data: [T; N] = std::array::from_fn(|i| T::from(((i % 5) + 1) as i8));
    let expected_shl = std::array::from_fn(|i| data[i].bits_shl(1));
    let expected_shr = std::array::from_fn(|i| data[i].bits_shr(1));
    let be = BE::new().unwrap();

    assert_eq!(
        owned_vector_array(be.shl_vector(&vector::<T, BE, N>(&data), 1)),
        expected_shl
    );
    assert_eq!(
        owned_vector_array(be.shr_vector(&vector::<T, BE, N>(&data), 1)),
        expected_shr
    );
}

fn check_promote_vector<BE, SS, SD, const N: usize>()
where
    BE: Backend + SimdPromoteVector<SS, SD, Backend = BE>,
    SS: Copy + From<i8>,
    SD: Copy + Default + From<SS> + PartialEq + Debug,
{
    let data: [SS; N] = std::array::from_fn(|i| SS::from((i as i8) - 4));
    let expected = std::array::from_fn(|i| SD::from(data[i]));
    let be = BE::new().unwrap();
    let actual = be.promotion_vector(&vector::<SS, BE, N>(&data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_demote_vector<BE, SS, SD, const N: usize>()
where
    BE: Backend + SimdDemoteVector<SS, SD, Backend = BE>,
    SS: Assume<SD> + Copy + From<i8>,
    SD: Copy + Default + PartialEq + Debug,
{
    let data: [SS; N] = std::array::from_fn(|i| SS::from((i as i8) - 4));
    let expected = std::array::from_fn(|i| data[i].assume());
    let be = BE::new().unwrap();
    let actual = be.demotion_vector(&vector::<SS, BE, N>(&data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_convert_vector<BE, SS, SD, const N: usize>()
where
    BE: Backend + SimdConvertVector<SS, SD, Backend = BE>,
    SS: Assume<SD> + Copy + From<i8>,
    SD: Copy + Default + PartialEq + Debug,
{
    let data: [SS; N] = std::array::from_fn(|i| SS::from((i as i8) - 4));
    let expected = std::array::from_fn(|i| data[i].assume());
    let be = BE::new().unwrap();
    let actual = be.convert_vector(&vector::<SS, BE, N>(&data));

    assert_eq!(owned_vector_array(actual), expected);
}

fn check_add_assign_matrix<BE, T, const N: usize, const M: usize>()
where
    BE: Backend + SimdAddAssignMatrix<T, T, Backend = BE>,
    T: Copy + Default + Add<Output = T> + From<i8> + PartialEq + Debug,
{
    let l_data: Vec<T> = (0..N * M).map(|i| T::from((i % 9) as i8)).collect();
    let r_data: Vec<T> = (0..N * M).map(|i| T::from(5 - (i as i8 % 11))).collect();
    let expected: Vec<T> = (0..N * M).map(|i| l_data[i] + r_data[i]).collect();
    let mut l = OwnedMatrix::<T, N, M>::from(l_data.into_boxed_slice());
    let r = matrix::<T, BE, N, M>(&r_data);
    let be = BE::new().unwrap();

    {
        let mut l_mut = MatrixMut::from(&mut l);
        be.add_assign_matrix(&mut l_mut, &r);
    }

    assert_eq!(owned_matrix_vec(l), expected);
}

fn check_scalar_mul_assign_matrix<BE, T, const N: usize, const M: usize>(scalar: T)
where
    BE: Backend + SimdScalarMulAssignMatrix<T, T, Backend = BE>,
    T: Copy + Default + From<i8> + Mul<Output = T> + PartialEq + Debug,
{
    let data: Vec<T> = (0..N * M).map(|i| T::from(((i % 5) + 1) as i8)).collect();
    let expected: Vec<T> = (0..N * M).map(|i| scalar * data[i]).collect();
    let mut m = OwnedMatrix::<T, N, M>::from(data.into_boxed_slice());
    let be = BE::new().unwrap();

    {
        let mut m_mut = MatrixMut::from(&mut m);
        be.scalar_mul_assign_matrix(scalar, &mut m_mut);
    }

    assert_eq!(owned_matrix_vec(m), expected);
}

fn check_scalar_mul_matrix<BE, T, const N: usize, const M: usize>(scalar: T)
where
    BE: Backend + SimdScalarMulMatrix<T, T, T, Backend = BE>,
    T: Copy + Default + From<i8> + Mul<Output = T> + PartialEq + Debug,
{
    let data: Vec<T> = (0..N * M).map(|i| T::from(((i % 5) + 1) as i8)).collect();
    let expected: Vec<T> = (0..N * M).map(|i| scalar * data[i]).collect();
    let be = BE::new().unwrap();
    let mut acc = OwnedMatrix::<T, N, M>::default();

    {
        let mut acc_mut = MatrixMut::from(&mut acc);
        be.scalar_mul_matrix(scalar, &matrix::<T, BE, N, M>(&data), &mut acc_mut);
    }

    assert_eq!(owned_matrix_vec(acc), expected);
}

fn check_convert_matrix<BE, SS, SD, const N: usize, const M: usize>()
where
    BE: Backend + SimdConvertMatrix<SS, SD, Backend = BE>,
    SS: Assume<SD> + Copy + From<i8>,
    SD: Copy + Default + PartialEq + Debug,
{
    let data: Vec<SS> = (0..N * M).map(|i| SS::from((i as i8) - 4)).collect();
    let expected: Vec<SD> = (0..N * M).map(|i| data[i].assume()).collect();
    let be = BE::new().unwrap();
    let mut acc = OwnedMatrix::<SD, N, M>::default();

    {
        let mut acc_mut = MatrixMut::from(&mut acc);
        be.convert_matrix(&matrix::<SS, BE, N, M>(&data), &mut acc_mut);
    }

    assert_eq!(owned_matrix_vec(acc), expected);
}

fn check_dot<BE, SL, SR, SO, const N: usize>()
where
    BE: Backend + SimdDot<SL, SR, SO, Backend = BE>,
    SL: Copy + From<i8> + Mul<SR, Output = SO>,
    SR: Copy + From<i8>,
    SO: Copy + Default + Add<Output = SO> + AddAssign + PartialEq + Debug,
{
    let l_data: [SL; N] = std::array::from_fn(|i| SL::from(((i % 5) + 1) as i8));
    let r_data: [SR; N] = std::array::from_fn(|i| SR::from(3 - (i as i8 % 5)));
    let expected = (0..N).fold(SO::default(), |acc, i| acc + l_data[i] * r_data[i]);
    let be = BE::new().unwrap();
    let actual = be.dot(&vector::<SL, BE, N>(&l_data), &vector::<SR, BE, N>(&r_data));

    assert_eq!(actual, expected);
}

fn check_matmul<BE, SL, SR, SO, const N: usize, const M: usize, const K: usize>()
where
    BE: Backend + SimdMatMul<SL, SR, SO, Backend = BE>,
    SL: Copy + From<i8>,
    SR: Copy + From<i8>,
    SO: Copy
        + Default
        + From<SL>
        + From<SR>
        + Add<Output = SO>
        + AddAssign
        + Mul<Output = SO>
        + PartialEq
        + Debug,
{
    let l_data: Vec<SL> = (0..N * K).map(|i| SL::from(((i % 5) + 1) as i8)).collect();
    let r_data: Vec<SR> = (0..K * M).map(|i| SR::from(3 - (i as i8 % 5))).collect();
    let mut expected = vec![SO::default(); N * M];

    for row in 0..N {
        for col in 0..M {
            let mut acc = SO::default();
            for k in 0..K {
                acc += SO::from(l_data[row * K + k]) * SO::from(r_data[col * K + k]);
            }
            expected[row * M + col] = acc;
        }
    }

    let be = BE::new().unwrap();
    let mut acc = OwnedMatrix::<SO, N, M>::default();

    {
        let mut acc_mut = MatrixMut::from(&mut acc);
        be.matmul(
            &matrix::<SL, BE, N, K>(&l_data),
            &column_major::<SR, BE, K, M>(&r_data),
            &mut acc_mut,
        );
    }

    assert_eq!(owned_matrix_vec(acc), expected);
}

#[test]
fn common_vector_add_sub_all_lane_boundaries() {
    if !avx2_available() {
        return;
    }

    lane_cases_same!(check_add_assign_vector::<Avx2, i8>; 31, 32, 33, 64, 65);
    lane_cases_same!(check_add_vector::<Avx2, i8>; 31, 32, 33, 64, 65);
    lane_cases_same!(check_sub_assign_vector::<Avx2, i8>; 31, 32, 33, 64, 65);
    lane_cases_same!(check_sub_vector::<Avx2, i8>; 31, 32, 33, 64, 65);
    lane_cases_same!(check_add_assign_vector::<Avx2, i16>; 15, 16, 17, 32, 33);
    lane_cases_same!(check_add_vector::<Avx2, i16>; 15, 16, 17, 32, 33);
    lane_cases_same!(check_sub_assign_vector::<Avx2, i16>; 15, 16, 17, 32, 33);
    lane_cases_same!(check_sub_vector::<Avx2, i16>; 15, 16, 17, 32, 33);
    lane_cases_same!(check_add_assign_vector::<Avx2, i32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_add_vector::<Avx2, i32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_sub_assign_vector::<Avx2, i32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_sub_vector::<Avx2, i32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_add_assign_vector::<Avx2, f32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_add_vector::<Avx2, f32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_sub_assign_vector::<Avx2, f32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_sub_vector::<Avx2, f32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_add_assign_vector::<Avx2, f64>; 3, 4, 5, 8, 9);
    lane_cases_same!(check_add_vector::<Avx2, f64>; 3, 4, 5, 8, 9);
    lane_cases_same!(check_sub_assign_vector::<Avx2, f64>; 3, 4, 5, 8, 9);
    lane_cases_same!(check_sub_vector::<Avx2, f64>; 3, 4, 5, 8, 9);
}

#[test]
fn common_vector_mul_all_type_combinations_and_lane_boundaries() {
    if !avx2_available() {
        return;
    }

    lane_cases_mul!(Avx2, i32, i32, i32; 7, 8, 9, 16, 17);
    lane_cases_mul!(Avx2, f32, f32, f32; 7, 8, 9, 16, 17);
    lane_cases_mul!(Avx2, f64, f64, f64; 3, 4, 5, 8, 9);

    lane_cases_same!(check_mul_assign_vector::<Avx2, i32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_mul_assign_vector::<Avx2, f32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_mul_assign_vector::<Avx2, f64>; 3, 4, 5, 8, 9);
}

#[test]
fn common_vector_scalar_mul_all_type_combinations_and_col_boundaries() {
    if !avx2_available() {
        return;
    }

    lane_cases_scalar!(Avx2, i32, i32, i32, 2_i32; 63, 64, 65, 128, 129);
    lane_cases_scalar!(Avx2, f32, f32, f32, 2.0_f32; 31, 32, 33, 64, 65);
    lane_cases_scalar!(Avx2, f64, f64, f64, 2.0_f64; 7, 8, 9, 16, 17);

    lane_cases_scalar_assign!(Avx2, i32, 2_i32; 63, 64, 65, 128, 129);
    lane_cases_scalar_assign!(Avx2, f32, 2.0_f32; 31, 32, 33, 64, 65);
    lane_cases_scalar_assign!(Avx2, f64, 2.0_f64; 7, 8, 9, 16, 17);
}

#[test]
fn common_bit_shift_and_conversion_vectors() {
    if !avx2_available() {
        return;
    }

    lane_cases_bit!(Avx2, i8, i8; 31, 32, 33, 64, 65);
    lane_cases_bit!(Avx2, i16, i16; 15, 16, 17, 32, 33);
    lane_cases_bit!(Avx2, i32, i32; 7, 8, 9, 16, 17);
    lane_cases_same!(check_bitnot_vector::<Avx2, i8>; 31, 32, 33, 64, 65);
    lane_cases_same!(check_bitnot_vector::<Avx2, i16>; 15, 16, 17, 32, 33);
    lane_cases_same!(check_bitnot_vector::<Avx2, i32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_bitnot_vector::<Avx2, f32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_bitnot_vector::<Avx2, f64>; 3, 4, 5, 8, 9);

    lane_cases_same!(check_shift_vector::<Avx2, i16>; 15, 16, 17, 32, 33);
    lane_cases_same!(check_shift_vector::<Avx2, i32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_shift_vector::<Avx2, f32>; 7, 8, 9, 16, 17);
    lane_cases_same!(check_shift_vector::<Avx2, f64>; 3, 4, 5, 8, 9);

    check_promote_vector::<Avx2, i8, i16, 31>();
    check_promote_vector::<Avx2, i8, i16, 32>();
    check_promote_vector::<Avx2, i8, i16, 33>();
    check_promote_vector::<Avx2, i16, i32, 15>();
    check_promote_vector::<Avx2, i16, i32, 16>();
    check_promote_vector::<Avx2, i16, i32, 17>();
    check_demote_vector::<Avx2, i32, i16, 7>();
    check_demote_vector::<Avx2, i32, i16, 8>();
    check_demote_vector::<Avx2, i32, i16, 9>();
    check_demote_vector::<Avx2, i16, i8, 15>();
    check_demote_vector::<Avx2, i16, i8, 16>();
    check_demote_vector::<Avx2, i16, i8, 17>();
    check_convert_vector::<Avx2, i16, f32, 15>();
    check_convert_vector::<Avx2, i16, f32, 16>();
    check_convert_vector::<Avx2, i16, f32, 17>();
}

#[test]
fn common_matrix_lane_row_col_boundaries() {
    if !avx2_available() {
        return;
    }

    check_add_assign_matrix::<Avx2, i8, 1, 31>();
    check_add_assign_matrix::<Avx2, i8, 1, 32>();
    check_add_assign_matrix::<Avx2, i8, 2, 33>();
    check_add_assign_matrix::<Avx2, i16, 1, 15>();
    check_add_assign_matrix::<Avx2, i16, 1, 16>();
    check_add_assign_matrix::<Avx2, i16, 2, 17>();
    check_add_assign_matrix::<Avx2, i32, 1, 7>();
    check_add_assign_matrix::<Avx2, i32, 2, 8>();
    check_add_assign_matrix::<Avx2, i32, 3, 9>();
    check_add_assign_matrix::<Avx2, f32, 1, 7>();
    check_add_assign_matrix::<Avx2, f32, 2, 8>();
    check_add_assign_matrix::<Avx2, f32, 3, 9>();
    check_add_assign_matrix::<Avx2, f64, 1, 3>();
    check_add_assign_matrix::<Avx2, f64, 1, 4>();
    check_add_assign_matrix::<Avx2, f64, 2, 5>();

    check_scalar_mul_assign_matrix::<Avx2, i32, 1, 7>(2);
    check_scalar_mul_assign_matrix::<Avx2, i32, 2, 8>(2);
    check_scalar_mul_assign_matrix::<Avx2, i32, 3, 9>(2);
    check_scalar_mul_assign_matrix::<Avx2, f32, 1, 7>(2.0);
    check_scalar_mul_assign_matrix::<Avx2, f32, 2, 8>(2.0);
    check_scalar_mul_assign_matrix::<Avx2, f32, 3, 9>(2.0);
    check_scalar_mul_assign_matrix::<Avx2, f64, 1, 3>(2.0);
    check_scalar_mul_assign_matrix::<Avx2, f64, 1, 4>(2.0);
    check_scalar_mul_assign_matrix::<Avx2, f64, 2, 5>(2.0);

    check_scalar_mul_matrix::<Avx2, i32, 1, 7>(2);
    check_scalar_mul_matrix::<Avx2, i32, 2, 8>(2);
    check_scalar_mul_matrix::<Avx2, i32, 3, 9>(2);
    check_scalar_mul_matrix::<Avx2, f32, 1, 7>(2.0);
    check_scalar_mul_matrix::<Avx2, f32, 2, 8>(2.0);
    check_scalar_mul_matrix::<Avx2, f32, 3, 9>(2.0);
    check_scalar_mul_matrix::<Avx2, f64, 1, 3>(2.0);
    check_scalar_mul_matrix::<Avx2, f64, 1, 4>(2.0);
    check_scalar_mul_matrix::<Avx2, f64, 2, 5>(2.0);

    check_convert_matrix::<Avx2, i16, f32, 1, 15>();
    check_convert_matrix::<Avx2, i16, f32, 1, 16>();
    check_convert_matrix::<Avx2, i16, f32, 2, 17>();
}

#[test]
fn avx2_dot_all_type_combinations_and_lane_boundaries() {
    if !avx2_available() {
        return;
    }

    check_dot::<Avx2, i32, i32, i32, 7>();
    check_dot::<Avx2, i32, i32, i32, 8>();
    check_dot::<Avx2, i32, i32, i32, 9>();
    check_dot::<Avx2, f32, f32, f32, 7>();
    check_dot::<Avx2, f32, f32, f32, 8>();
    check_dot::<Avx2, f32, f32, f32, 9>();
    check_dot::<Avx2, f64, f64, f64, 3>();
    check_dot::<Avx2, f64, f64, f64, 4>();
    check_dot::<Avx2, f64, f64, f64, 5>();
}

#[test]
fn avx2_matmul_type_combinations_rows_cols_and_lane_boundaries() {
    if !avx2_available() {
        return;
    }

    check_matmul::<Avx2, i8, i8, i32, 1, 7, 31>();
    check_matmul::<Avx2, i8, i8, i32, 1, 8, 32>();
    check_matmul::<Avx2, i8, i8, i32, 2, 9, 33>();
    check_matmul::<Avx2, i8, i16, i32, 1, 7, 31>();
    check_matmul::<Avx2, i8, i16, i32, 1, 8, 32>();
    check_matmul::<Avx2, i8, i16, i32, 2, 9, 33>();
    check_matmul::<Avx2, i16, i16, i32, 1, 7, 15>();
    check_matmul::<Avx2, i16, i16, i32, 1, 8, 16>();
    check_matmul::<Avx2, i16, i16, i32, 2, 9, 17>();
    check_matmul::<Avx2, f32, f32, f32, 1, 3, 7>();
    check_matmul::<Avx2, f32, f32, f32, 2, 4, 8>();
    check_matmul::<Avx2, f32, f32, f32, 3, 5, 9>();
    check_matmul::<Avx2, f64, f64, f64, 1, 1, 3>();
    check_matmul::<Avx2, f64, f64, f64, 1, 2, 4>();
    check_matmul::<Avx2, f64, f64, f64, 2, 3, 5>();
}

#[test]
fn avx2_backend_metadata() {
    if !avx2_available() {
        return;
    }

    Avx2::new().unwrap();
    assert_eq!(<Avx2 as SimdLanes<i8>>::LANES, 32);
    assert_eq!(<Avx2 as SimdLanes<i16>>::LANES, 16);
    assert_eq!(<Avx2 as SimdLanes<i32>>::LANES, 8);
    assert_eq!(<Avx2 as SimdLanes<i64>>::LANES, 4);
    assert_eq!(<Avx2 as SimdLanes<f32>>::LANES, 8);
    assert_eq!(<Avx2 as SimdLanes<f64>>::LANES, 4);
    assert_eq!(<Avx2 as SimdRows<i8>>::ROWS, 1);
    assert_eq!(<Avx2 as SimdRows<i16>>::ROWS, 1);
    assert_eq!(<Avx2 as SimdRows<i32>>::ROWS, 2);
    assert_eq!(<Avx2 as SimdRows<i64>>::ROWS, 1);
    assert_eq!(<Avx2 as SimdRows<f32>>::ROWS, 2);
    assert_eq!(<Avx2 as SimdRows<f64>>::ROWS, 1);
    assert_eq!(<Avx2 as SimdCols<i8>>::COLS, 8);
    assert_eq!(<Avx2 as SimdCols<i16>>::COLS, 8);
    assert_eq!(<Avx2 as SimdCols<i32>>::COLS, 8);
    assert_eq!(<Avx2 as SimdCols<f32>>::COLS, 4);
    assert_eq!(<Avx2 as SimdCols<f64>>::COLS, 2);
}

#[test]
fn avx2_register_level_traits() {
    if !avx2_available() {
        return;
    }

    let be = Avx2::new().unwrap();

    unsafe {
        let l8 = std::array::from_fn::<_, 32, _>(|i| i as i8);
        let r8 = std::array::from_fn::<_, 32, _>(|i| 3 - i as i8);
        let mut o8 = [0_i8; 32];
        be.store(
            o8.as_mut_ptr(),
            <Avx2 as SimdAdd<i8, i8, i8>>::add(
                &be,
                <Avx2 as SimdLoad<i8>>::load(&be, l8.as_ptr()),
                <Avx2 as SimdLoad<i8>>::load(&be, r8.as_ptr()),
            ),
        );
        assert_eq!(o8, std::array::from_fn(|i| l8[i].wrapping_add(r8[i])));

        let l16 = std::array::from_fn::<_, 16, _>(|i| i as i16);
        let r16 = std::array::from_fn::<_, 16, _>(|i| 3 - i as i16);
        let mut o16 = [0_i16; 16];
        be.store(
            o16.as_mut_ptr(),
            <Avx2 as SimdSub<i16, i16, i16>>::sub(
                &be,
                <Avx2 as SimdLoad<i16>>::load(&be, l16.as_ptr()),
                <Avx2 as SimdLoad<i16>>::load(&be, r16.as_ptr()),
            ),
        );
        assert_eq!(o16, std::array::from_fn(|i| l16[i] - r16[i]));

        let l32 = std::array::from_fn::<_, 8, _>(|i| i as i32);
        let r32 = std::array::from_fn::<_, 8, _>(|i| 3 - i as i32);
        let mut o32 = [0_i32; 8];
        be.store(
            o32.as_mut_ptr(),
            <Avx2 as SimdAdd<i32, i32, i32>>::add(
                &be,
                <Avx2 as SimdLoad<i32>>::load(&be, l32.as_ptr()),
                <Avx2 as SimdLoad<i32>>::load(&be, r32.as_ptr()),
            ),
        );
        assert_eq!(o32, std::array::from_fn(|i| l32[i] + r32[i]));

        let lf32 = std::array::from_fn::<_, 8, _>(|i| i as f32);
        let rf32 = std::array::from_fn::<_, 8, _>(|i| 3.0 - i as f32);
        let mut of32 = [0.0_f32; 8];
        let f32_prod = <Avx2 as SimdMul<f32, f32, f32>>::mul(
            &be,
            <Avx2 as SimdLoad<f32>>::load(&be, lf32.as_ptr()),
            <Avx2 as SimdLoad<f32>>::load(&be, rf32.as_ptr()),
        );
        let &[f32_prod] = f32_prod.as_ref();
        be.store(of32.as_mut_ptr(), f32_prod);
        assert_eq!(of32, std::array::from_fn(|i| lf32[i] * rf32[i]));

        let lf64 = std::array::from_fn::<_, 4, _>(|i| i as f64);
        let rf64 = std::array::from_fn::<_, 4, _>(|i| 3.0 - i as f64);
        let mut of64 = [0.0_f64; 4];
        let f64_prod = <Avx2 as SimdMul<f64, f64, f64>>::mul(
            &be,
            <Avx2 as SimdLoad<f64>>::load(&be, lf64.as_ptr()),
            <Avx2 as SimdLoad<f64>>::load(&be, rf64.as_ptr()),
        );
        let &[f64_prod] = f64_prod.as_ref();
        be.store(of64.as_mut_ptr(), f64_prod);
        assert_eq!(of64, std::array::from_fn(|i| lf64[i] * rf64[i]));

        assert_eq!(
            <Avx2 as SimdHSum<i32>>::hsum(
                &be,
                <Avx2 as SimdLoad<i32>>::load(&be, [1_i32, 2, 3, 4, 5, 6, 7, 8].as_ptr()),
            ),
            36
        );
        assert_eq!(
            <Avx2 as SimdHSum<f32>>::hsum(
                &be,
                <Avx2 as SimdLoad<f32>>::load(
                    &be,
                    [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].as_ptr(),
                ),
            ),
            36.0
        );
        assert_eq!(
            <Avx2 as SimdHSum<f64>>::hsum(
                &be,
                <Avx2 as SimdLoad<f64>>::load(&be, [1.0_f64, 2.0, 3.0, 4.0].as_ptr()),
            ),
            10.0
        );

        let i32_values = [1_i32, -2, 3, -4, 5, -6, 7, -8];
        let i32_converted = <Avx2 as SimdConvert<i32, f32>>::convert(
            &be,
            <Avx2 as SimdLoad<i32>>::load(&be, i32_values.as_ptr()),
        );
        let &[i32_converted] = i32_converted.as_ref();
        let mut converted = [0.0_f32; 8];
        be.store(converted.as_mut_ptr(), i32_converted);
        assert_eq!(converted, i32_values.map(|v| v as f32));

        let zero = <Avx2 as SimdZero<i32>>::zero();
        be.store(o32.as_mut_ptr(), zero);
        assert_eq!(o32, [0; 8]);

        let i8_l = std::array::from_fn::<_, 32, _>(|i| (i as i8 % 7) - 3);
        let i8_r = std::array::from_fn::<_, 32, _>(|i| 4 - (i as i8 % 7));
        let mut i8_prod = [0_i32; 32];
        be.store_seq(
            i8_prod.as_mut_ptr(),
            <Avx2 as SimdMul<i8, i8, i32>>::mul(
                &be,
                <Avx2 as SimdLoad<i8>>::load(&be, i8_l.as_ptr()),
                <Avx2 as SimdLoad<i8>>::load(&be, i8_r.as_ptr()),
            ),
        );
        assert_eq!(
            i8_prod,
            std::array::from_fn(|i| i8_l[i] as i32 * i8_r[i] as i32)
        );

        let i16_l = std::array::from_fn::<_, 16, _>(|i| (i as i16 % 7) - 3);
        let i16_r = std::array::from_fn::<_, 16, _>(|i| 4 - (i as i16 % 7));
        let mut i16_prod = [0_i32; 16];
        be.store_seq(
            i16_prod.as_mut_ptr(),
            <Avx2 as SimdMul<i16, i16, i32>>::mul(
                &be,
                <Avx2 as SimdLoad<i16>>::load(&be, i16_l.as_ptr()),
                <Avx2 as SimdLoad<i16>>::load(&be, i16_r.as_ptr()),
            ),
        );
        assert_eq!(
            i16_prod,
            std::array::from_fn(|i| i16_l[i] as i32 * i16_r[i] as i32)
        );

        let r1 = be.load([1_i32, 2, 3, 4, 5, 6, 7, 8].as_ptr());
        let r2 = be.load([10_i32, 20, 30, 40, 50, 60, 70, 80].as_ptr());
        let mut seq = [0_i32; 16];
        be.store_seq(seq.as_mut_ptr(), Regs::new([r1, r2]));
        assert_eq!(
            seq,
            [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80]
        );
    }
}

#[test]
fn avx2_bit_mask_reinterpret_shift_and_transpose_registers() {
    if !avx2_available() {
        return;
    }

    let be = Avx2::new().unwrap();

    unsafe {
        let l = [0b1010_i32; 8];
        let r = [0b1100_i32; 8];
        let lr = be.load(l.as_ptr());
        let rr = be.load(r.as_ptr());
        let mut actual = [0_i32; 8];

        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdBitAnd<i32>>::bitand(&be, lr, rr),
        );
        assert_eq!(actual, [0b1000; 8]);
        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdBitOr<i32>>::bitor(&be, lr, rr),
        );
        assert_eq!(actual, [0b1110; 8]);
        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdBitXor<i32>>::bitxor(&be, lr, rr),
        );
        assert_eq!(actual, [0b0110; 8]);
        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdBitNot<i32>>::bitnot(&be, lr),
        );
        assert_eq!(actual, [!0b1010; 8]);
        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdShl<i32>>::shl(
                &be,
                lr,
                <Avx2 as SimdShiftWidth<i32>>::shift_width(&be, 1),
            ),
        );
        assert_eq!(actual, [0b10100; 8]);
        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdShr<i32>>::shr(
                &be,
                lr,
                <Avx2 as SimdShiftWidth<i32>>::shift_width(&be, 1),
            ),
        );
        assert_eq!(actual, [0b0101; 8]);

        let gt = <Avx2 as SimdMask<i32>>::cmp_gt(
            &be,
            be.load([2_i32; 8].as_ptr()),
            be.load([1_i32; 8].as_ptr()),
        );
        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdMask<i32>>::select(
                &be,
                gt,
                be.load([7_i32; 8].as_ptr()),
                be.load([9_i32; 8].as_ptr()),
            ),
        );
        assert_eq!(actual, [7; 8]);
        be.store(
            actual.as_mut_ptr(),
            <Avx2 as SimdMask<i32>>::mask_zero(
                &be,
                <Avx2 as SimdMask<i32>>::tail_mask(&be, 0, 3),
                be.load([5_i32; 8].as_ptr()),
            ),
        );
        assert_eq!(&actual[..3], &[5; 3]);

        let f = [1.25_f32, -2.5, 3.5, -4.75, 5.25, -6.5, 7.5, -8.75];
        let fr = be.load(f.as_ptr());
        let bits = <Avx2 as SimdReinterpret<f32, i32>>::reinterpret(&be, fr);
        let fr2 = <Avx2 as SimdReinterpret<i32, f32>>::reinterpret(&be, bits);
        let mut f_actual = [0.0_f32; 8];
        be.store(f_actual.as_mut_ptr(), fr2);
        assert_f32_bits_eq(f_actual, f);

        let d = [1.25_f64, -2.5, 3.5, -4.75];
        let dr = be.load(d.as_ptr());
        let dbits = <Avx2 as SimdReinterpret<f64, i64>>::reinterpret(&be, dr);
        let dr2 = <Avx2 as SimdReinterpret<i64, f64>>::reinterpret(&be, dbits);
        let mut d_actual = [0.0_f64; 4];
        be.store(d_actual.as_mut_ptr(), dr2);
        assert_f64_bits_eq(d_actual, d);

        let rows = [
            be.load([1_i32, 2, 3, 4, 5, 6, 7, 8].as_ptr()),
            be.load([11_i32, 12, 13, 14, 15, 16, 17, 18].as_ptr()),
        ];
        let transposed = <Avx2 as SimdTranspose<i32, 2>>::transpose::<2, 8>(rows);
        let mut lo = [0_i32; 8];
        let mut hi = [0_i32; 8];
        be.store(lo.as_mut_ptr(), transposed[0]);
        be.store(hi.as_mut_ptr(), transposed[1]);
        assert_eq!(lo, [1, 11, 2, 12, 5, 15, 6, 16]);
        assert_eq!(hi, [3, 13, 4, 14, 7, 17, 8, 18]);
    }
}

#[test]
fn scalar_bits_and_assume_traits() {
    assert_eq!(5_i8.bits_bitand(3), 1);
    assert_eq!(5_i16.bits_bitor(2), 7);
    assert_eq!(5_i32.bits_bitxor(3), 6);
    assert_eq!(5_i64.bits_bitnot(), !5_i64);
    assert_f32_bits_eq([1.0_f32.bits_bitand(u32::MAX)], [1.0]);
    assert_f64_bits_eq([1.0_f64.bits_bitor(0)], [1.0]);
    assert_eq!(4_i16.bits_shl(1), 8);
    assert_eq!(4_i32.bits_shr(1), 2);
    assert_eq!(Assume::<i16>::assume(7_i32), 7_i16);
    assert_eq!(Assume::<f32>::assume(7_i16), 7.0_f32);
}

#[test]
fn compile_probe_common_widening_mul_vector_type_combinations() {
    if !avx2_available() {
        return;
    }

    check_mul_vector::<Avx2, i8, i8, i32, 31>();
    check_mul_vector::<Avx2, i8, i8, i32, 32>();
    check_mul_vector::<Avx2, i8, i8, i32, 33>();
    check_mul_vector::<Avx2, i8, i8, i32, 64>();
    check_mul_vector::<Avx2, i8, i8, i32, 65>();

    check_mul_vector::<Avx2, i8, i16, i32, 31>();
    check_mul_vector::<Avx2, i8, i16, i32, 32>();
    check_mul_vector::<Avx2, i8, i16, i32, 33>();
    check_mul_vector::<Avx2, i8, i16, i32, 64>();
    check_mul_vector::<Avx2, i8, i16, i32, 65>();

    check_mul_vector::<Avx2, i16, i16, i32, 15>();
    check_mul_vector::<Avx2, i16, i16, i32, 16>();
    check_mul_vector::<Avx2, i16, i16, i32, 17>();
    check_mul_vector::<Avx2, i16, i16, i32, 32>();
    check_mul_vector::<Avx2, i16, i16, i32, 33>();
}

#[test]
fn compile_probe_common_widening_scalar_mul_vector_type_combinations() {
    if !avx2_available() {
        return;
    }

    check_scalarmul_vector::<Avx2, i8, i8, i32, 255>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i8, i32, 256>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i8, i32, 257>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i8, i32, 512>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i8, i32, 513>(2_i8);

    check_scalarmul_vector::<Avx2, i8, i16, i32, 255>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i16, i32, 256>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i16, i32, 257>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i16, i32, 512>(2_i8);
    check_scalarmul_vector::<Avx2, i8, i16, i32, 513>(2_i8);

    check_scalarmul_vector::<Avx2, i16, i16, i32, 127>(2_i16);
    check_scalarmul_vector::<Avx2, i16, i16, i32, 128>(2_i16);
    check_scalarmul_vector::<Avx2, i16, i16, i32, 129>(2_i16);
    check_scalarmul_vector::<Avx2, i16, i16, i32, 256>(2_i16);
    check_scalarmul_vector::<Avx2, i16, i16, i32, 257>(2_i16);
}

#[test]
fn compile_probe_common_convert_i32_to_f32_vector_and_matrix() {
    if !avx2_available() {
        return;
    }

    check_convert_vector::<Avx2, i32, f32, 7>();
    check_convert_vector::<Avx2, i32, f32, 8>();
    check_convert_vector::<Avx2, i32, f32, 9>();
    check_convert_vector::<Avx2, i32, f32, 16>();
    check_convert_vector::<Avx2, i32, f32, 17>();

    check_convert_matrix::<Avx2, i32, f32, 1, 7>();
    check_convert_matrix::<Avx2, i32, f32, 2, 8>();
    check_convert_matrix::<Avx2, i32, f32, 3, 9>();
    check_convert_matrix::<Avx2, i32, f32, 4, 16>();
    check_convert_matrix::<Avx2, i32, f32, 5, 17>();
}

#[test]
fn compile_probe_avx2_widening_dot_type_combinations() {
    if !avx2_available() {
        return;
    }

    check_dot::<Avx2, i8, i8, i32, 31>();
    check_dot::<Avx2, i8, i8, i32, 32>();
    check_dot::<Avx2, i8, i8, i32, 33>();
    check_dot::<Avx2, i8, i16, i32, 31>();
    check_dot::<Avx2, i8, i16, i32, 32>();
    check_dot::<Avx2, i8, i16, i32, 33>();
    check_dot::<Avx2, i16, i16, i32, 15>();
    check_dot::<Avx2, i16, i16, i32, 16>();
    check_dot::<Avx2, i16, i16, i32, 17>();
    check_dot::<Avx2, f32, f64, f64, 3>();
    check_dot::<Avx2, f32, f64, f64, 4>();
    check_dot::<Avx2, f32, f64, f64, 5>();
}

#[test]
fn compile_probe_avx2_matmul_missing_type_combinations() {
    if !avx2_available() {
        return;
    }

    check_matmul::<Avx2, i32, i32, i32, 1, 7, 7>();
    check_matmul::<Avx2, i32, i32, i32, 2, 8, 8>();
    check_matmul::<Avx2, i32, i32, i32, 3, 9, 9>();
    check_matmul::<Avx2, f32, f64, f64, 1, 1, 3>();
    check_matmul::<Avx2, f32, f64, f64, 2, 2, 4>();
    check_matmul::<Avx2, f32, f64, f64, 3, 3, 5>();
}
