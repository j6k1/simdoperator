//! Trait and data type features for abstracting SIMD operations

use rayon::iter::ParallelIterator;
use std::ops::{Add, AddAssign, BitAnd, BitOr, BitXor, Index, IndexMut, Mul, MulAssign, Not, Shl, Shr, Sub, SubAssign};
use rayon::prelude::{ParallelSliceMut, IndexedParallelIterator};
use crate::backend::autoselect::{AutoSelect, SelectedBackend};
use crate::backend::avx2::Avx2;
use crate::error::{InstantiationError, TryFromSliceError};
use crate::backend::common::{Backend};
use crate::traits::{BindBackend, BitsBitAnd, BitsBitOr, BitsBitXor, Demote, Dims, Dot, IsScaler, NativeBackend, Product, Promote, SimdAddAssignMatrix, SimdAddAssignVector, SimdAddVector, SimdBitAndVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdConvertMatrix, SimdConvertVector, SimdDemoteVector, SimdDot, SimdMatMul, SimdMatVec, SimdMulAssignVector, SimdMulVector, SimdOuterProduct, SimdPromoteVector, SimdReg, SimdScalarMulAssignMatrix, SimdScalarMulAssignVector, SimdScalarMulMatrix, SimdScalarMulVector, SimdShlVector, SimdShrVector, SimdSubAssignVector, SimdSubVector, SimdVMat, ToColumnMajor, Transpose};
use crate::traits::private::BindBackendBase;

pub mod backend;
pub mod traits;
pub mod error;
/// Vectors associated with the backend
pub struct Vector<'a,T,const N: usize,BE: Backend = AutoSelect> {
    data: &'a [T; N],
    backend: BE
}
impl<'a,BE: Backend,T,const N: usize> Clone for Vector<'a,T,N,BE>
    where BE: Backend + Clone {
    fn clone(&self) -> Self {
        Vector {
            data: self.data,
            backend: self.backend.clone()
        }
    }
}
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a [T]> for Vector<'a,T,N,BE> {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(Vector {
                data: value.try_into()?,
                backend: BE::new()?
            })
        }
    }
}
impl<T,BE: Backend,const N: usize> Index<usize> for Vector<'_,T,N,BE> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,BE: Backend,const N: usize> AsRef<[T;N]> for Vector<'_,T,N,BE> {
    #[inline(always)]
    fn as_ref(&self) -> &[T;N] {
        &self.data
    }
}
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a OwnedVector<T,N>> for Vector<'a,T,N,BE> {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: &'a OwnedVector<T,N>) -> Result<Self,Self::Error> {
        Ok(Vector {
            data: &value.data,
            backend: BE::new()?
        })
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a AccVector<T,N,BE>> for Vector<'a,T,N,BE>
    where BE: Backend + Clone {
    #[inline(always)]
    fn from(value: &'a AccVector<T,N,BE>) -> Vector<'a,T,N,BE> {
        Vector {
            data: &value.data,
            backend: value.backend.clone()
        }
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a Vector<'a,T,N,BE>> for Box<[T;N]>
    where T: Clone + Copy {

    #[inline(always)]
    fn from(value: &'a Vector<'a,T,N,BE>) -> Self {
        Box::new(value.data.clone().into())
    }
}
impl<'a,BE: Backend,T,const N: usize> AsRef<[T;N]> for AccVector<T,N,BE> {
    fn as_ref(&self) -> &[T; N] {
        &self.data
    }
}
impl<'a,BE: Backend,T,const N: usize> Vector<'a,T,N,BE> {
    /// Returns a view of the vector as a vertical matrix.
    #[inline(always)]
    pub fn as_vertical(&self) -> Matrix<'a,T,N,1,BE> {
        Matrix {
            data: self.data,
            backend: BE::new().unwrap()
        }
    }

    /// Returns a view of the vector as a horizontal matrix.
    #[inline(always)]
    pub fn as_horizontal(&self) -> Matrix<'a,T,1,N,BE> {
        Matrix {
            data: self.data,
            backend: BE::new().unwrap()
        }
    }
}
/// A mutable vector associated with the backend
pub struct VectorMut<'a,T,const N: usize,BE: Backend = AutoSelect> {
    data: &'a mut [T; N],
    backend: BE
}
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a mut [T]> for VectorMut<'a,T,N,BE> {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a mut [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(VectorMut {
                data: value.try_into()?,
                backend: BE::new()?
            })
        }
    }
}
impl<T,BE: Backend,const N: usize> Index<usize> for VectorMut<'_,T,N,BE> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,BE: Backend,const N: usize> IndexMut<usize> for VectorMut<'_,T,N,BE> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
impl<T,BE: Backend,const N: usize> AsMut<[T;N]> for VectorMut<'_,T,N,BE> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T;N] {
        &mut self.data
    }
}
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a mut OwnedVector<T,N>> for VectorMut<'a,T,N,BE> {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: &'a mut OwnedVector<T,N>) -> Result<Self,Self::Error> {
        Ok(VectorMut {
            data: &mut value.data,
            backend: BE::new()?
        })
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a VectorMut<'a,T,N,BE>> for Box<[T;N]>
    where T: Clone + Copy {

    #[inline(always)]
    fn from(value: &'a VectorMut<'a,T,N,BE>) -> Self {
        Box::new(value.data.clone().into())
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a mut AccVector<T,N,BE>> for VectorMut<'a,T,N,BE>
    where BE: Backend + Clone {
    #[inline(always)]
    fn from(value: &'a mut AccVector<T,N,BE>) -> Self {
        VectorMut {
            data: &mut value.data,
            backend: value.backend.clone()
        }
    }
}

/// Backend-independent views that reference Vector
pub struct VectorView<'a,T,const N: usize> {
    data: &'a [T; N]
}
impl<'a,T,const N: usize> From<&'a [T;N]> for VectorView<'a,T,N> {
    #[inline(always)]
    fn from(value: &'a [T; N]) -> Self {
        VectorView {
            data: value
        }
    }
}
impl<'a,T,const N: usize> TryFrom<&'a [T]> for VectorView<'a,T,N> {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(VectorView {
                data: value.try_into()?
            })
        }
    }
}
impl<T,const N: usize> Index<usize> for VectorView<'_,T,N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,const N: usize> AsRef<[T;N]> for VectorView<'_,T,N> {
    #[inline(always)]
    fn as_ref(&self) -> &[T;N] {
        &self.data
    }
}
impl<'a,'b,T,BE,const N: usize> From<&'b Vector<'a,T,N,BE>> for VectorView<'b,T,N>
    where BE: Backend,
          'a: 'b {
    #[inline(always)]
    fn from(value: &'b Vector<'a,T,N,BE>) -> VectorView<'b,T,N> {
        VectorView {
            data: value.data
        }
    }
}
impl<'a,T,const N: usize> From<&'a OwnedVector<T,N>> for VectorView<'a,T,N> {
    #[inline(always)]
    fn from(value: &'a OwnedVector<T,N>) -> VectorView<'a,T,N> {
        VectorView {
            data: &value.data
        }
    }
}
impl<'a,T,const N: usize> From<&'a VectorView<'a,T,N>> for Box<[T;N]>
    where T: Clone + Copy {
    #[inline(always)]
    fn from(value: &'a VectorView<'a,T,N>) -> Box<[T;N]> {
        value.data.clone().into()
    }
}
impl<'a,T,const N: usize> Clone for VectorView<'a,T,N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        VectorView {
            data: self.data
        }
    }
}
impl<'a,T,const N: usize> VectorView<'a,T,N> {
    /// Returns a view of the vector as a vertical matrix.
    #[inline(always)]
    pub fn as_vertical(&self) -> MatrixView<'a,T,N,1> {
        MatrixView {
            data: self.data
        }
    }

    /// Returns a view of the vector as a horizontal matrix.
    #[inline(always)]
    pub fn as_horizontal(&self) -> MatrixView<'a,T,1,N> {
        MatrixView {
            data: self.data
        }
    }
}
/// A view type that represents a vector whose elements mutable
pub struct VectorMutView<'a,T,const N: usize> {
    data: &'a mut [T;N]
}
impl<'a,T,const N: usize> From<&'a mut OwnedVector<T,N>> for VectorMutView<'a,T,N> {
    #[inline(always)]
     fn from(value: &'a mut OwnedVector<T,N>) -> Self {
        VectorMutView {
            data: &mut value.data
        }
     }
}
impl<'a,'b,T,BE,const N: usize> From<&'b mut VectorMut<'a,T,N,BE>> for VectorMutView<'b,T,N>
    where BE: Backend,
          'a: 'b {
    #[inline(always)]
    fn from(value: &'b mut VectorMut<'a,T,N,BE>) -> Self {
        VectorMutView {
            data: value.data
        }
    }
}
impl<'a,T,const N: usize> Index<usize> for VectorMutView<'a,T,N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<'a,T,const N: usize> IndexMut<usize> for VectorMutView<'a,T,N> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}
impl<'a,T,const N: usize> AsRef<[T;N]> for VectorMutView<'a,T,N> {
    #[inline(always)]
    fn as_ref(&self) -> &[T; N] {
        &self.data
    }
}
impl<'a,T,const N: usize> AsMut<[T;N]> for VectorMutView<'a,T,N> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }
}
impl<'a,T,BE,const N: usize> From<&'a mut AccVector<T,N,BE>> for VectorMutView<'a,T,N>
    where BE: Backend {
    fn from(value: &'a mut AccVector<T, N, BE>) -> Self {
        VectorMutView {
            data: &mut value.data
        }
    }
}
impl<'a,T,const N: usize> TryFrom<&'a mut [T]> for VectorMutView<'a,T,N> {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: &'a mut [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(VectorMutView {
                data: value.try_into()?
            })
        }
    }
}
/// Vector Types with Ownership
pub struct OwnedVector<T,const N: usize> {
    data: Box<[T; N]>
}
impl<T,const N: usize> From<OwnedVector<T,N>> for Box<[T;N]> {
    #[inline(always)]
    fn from(value: OwnedVector<T,N>) -> Self {
        value.data
    }
}
impl<T,const N: usize> From<Box<[T;N]>> for OwnedVector<T,N> {
    #[inline(always)]
    fn from(value: Box<[T;N]>) -> Self {
        OwnedVector {
            data: value
        }
    }
}
impl<T,const N: usize> From<OwnedVector<T,N>> for Box<[T]> {
    #[inline(always)]
    fn from(value: OwnedVector<T,N>) -> Box<[T]> {
        value.data
    }
}
impl<T,const N: usize> AsMut<[T;N]> for OwnedVector<T,N> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T;N] {
        &mut self.data
    }
}
impl<T,const N: usize> Default for OwnedVector<T,N> where T: Default + Clone + Copy {
    #[inline(always)]
    fn default() -> Self {
        Self {
            data: [T::default();N].into()
        }
    }
}
impl<T,const N: usize> Index<usize> for OwnedVector<T,N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,const N: usize> IndexMut<usize> for OwnedVector<T,N> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
impl<T,const N: usize> BindBackendBase for OwnedVector<T,N> where T: 'static {
    type Output<BE: Backend> = AccVector<T,N,BE>;

    #[inline(always)]
    fn bind<BE: Backend>(self) -> Result<Self::Output<BE>, InstantiationError> {
        Ok(AccVector::try_from(self)?)
    }
}
impl<T,const N: usize> BindBackend for OwnedVector<T,N> where T: 'static {

}
/// Accumulator Vector
pub struct AccVector<T,const N: usize,BE = AutoSelect>
    where BE: Backend {
    data: Box<[T; N]>,
    backend: BE
}
impl<T,BE,const N: usize> From<AccVector<T,N,BE>> for Box<[T;N]>
    where BE: Backend {
    #[inline(always)]
    fn from(value: AccVector<T,N,BE>) -> Self {
        value.data
    }
}
impl<T,BE,const N: usize> TryFrom<OwnedVector<T,N>> for AccVector<T,N,BE>
    where BE: Backend {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: OwnedVector<T,N>) -> Result<AccVector<T,N,BE>,Self::Error> {
        Ok(AccVector {
            data: value.into(),
            backend: BE::new()?
        })
    }
}
impl<T,BE,const N: usize> TryFrom<Box<[T;N]>> for AccVector<T,N,BE>
    where BE: Backend {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: Box<[T;N]>) -> Result<AccVector<T,N,BE>,Self::Error> {
        Ok(AccVector {
            data: value,
            backend:BE::new()?
        })
    }
}
impl<T,BE,const N: usize> From<AccVector<T,N,BE>> for Box<[T]>
    where BE: Backend {
    #[inline(always)]
    fn from(value: AccVector<T,N,BE>) -> Box<[T]> {
        value.data
    }
}
impl<T,BE,const N: usize> AsMut<[T;N]> for AccVector<T,N,BE>
    where BE: Backend {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T;N] {
        &mut self.data
    }
}
impl<T,BE,const N: usize> Index<usize> for AccVector<T,N,BE>
    where BE: Backend {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,BE,const N: usize> IndexMut<usize> for AccVector<T,N,BE>
    where BE: Backend {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
/// A view type that represents a matrix whose elements are immutable
pub struct Matrix<'a,T,const N: usize,const M: usize,BE: Backend = AutoSelect> {
    data: &'a [T],
    backend: BE
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> Dims<N,M> for Matrix<'a,T,N,M,BE> {}
impl<'a,BE: Backend,T,const N: usize,const M: usize> Matrix<'a,T,N,M,BE> {
    /// Retrieve a specified row of a matrix as a vector
    ///
    /// # Arguments
    /// * `index` - The index of the row to retrieve
    #[inline(always)]
    pub fn row(&self,index:usize) -> Vector<'a,T,M,BE> {
        let view = &self.data[index * M..(index + 1) * M];

        Vector {
            data: view.try_into().unwrap(),
            backend: BE::new().unwrap()
        }
    }
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> TryFrom<&'a [T]> for Matrix<'a,T,N,M,BE> {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(Matrix {
                data: value,
                backend: BE::new()?
            })
        }
    }
}
impl<T,BE: Backend,const N: usize,const M: usize> Index<usize> for Matrix<'_,T,N,M,BE> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * M)..(index * M + M)]
    }
}
impl<'a,T,BE: Backend,const N: usize,const M: usize> Transpose<T,N,M> for Matrix<'a,T,N,M,BE>
    where T: Default + Clone + Copy {
    type Output = OwnedMatrix<T,M,N>;
    #[inline(always)]
    fn transpose(self) -> OwnedMatrix<T,M,N> {
        let mut r = OwnedMatrix::<T,M,N>::default();

        const BLOCK:usize = 64;

        for row in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
            for col in (0..((M + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
                for y in 0..BLOCK {
                    for x in 0..BLOCK {
                        if row + x >= N || col + y >= M {
                            continue;
                        }
                        r[(col + y,row + x)] = self.data[(row + x) * M + col + y];
                    }
                }
            }
        }

        r
    }
}
impl<'a,T,BE: Backend,const N: usize,const M: usize> ToColumnMajor<T,N,M> for Matrix<'a,T,N,M,BE>
    where T: Default + Clone + Copy + Send + Sync {
    type Output = OwnedColumnMajorMatrix<T,N,M>;
    #[inline(always)]
    fn to_column_major(self) -> OwnedColumnMajorMatrix<T,N,M> {
        let mut r = vec![T::default();M*N].into_boxed_slice();

        const BLOCK:usize = 32;

        r.par_chunks_exact_mut(N * BLOCK).zip(0..(M / BLOCK)).for_each(|(r,by)| {
            for col in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
                for x in 0..BLOCK {
                    for y in 0..BLOCK {
                        if col + x >= N {
                            continue;
                        }
                        r[y * N + (col + x)] = self.data[(col + x) * M + (by * BLOCK + y)];
                    }
                }
            }
        });

        let by = M / BLOCK;

        for col in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
            for x in 0..BLOCK {
                for y in 0..BLOCK {
                    if col + x >= N || by * BLOCK + y >= M {
                        continue;
                    }
                    r[(by * BLOCK + y) * N + (col + x)] = self.data[(col + x) * M + (by * BLOCK + y)];
                }
            }
        }
        OwnedColumnMajorMatrix { data: r }
    }
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> TryFrom<&'a OwnedMatrix<T,N,M>> for Matrix<'a,T,N,M,BE> {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: &'a OwnedMatrix<T,N,M>) -> Result<Self,Self::Error> {
        Ok(Matrix {
            data: &value.data,
            backend: BE::new()?
        })
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a Vector<'a,T,N,BE>> for Matrix<'a,T,N,1,BE> {
    #[inline(always)]
    fn from(value: &'a Vector<'a, T, N, BE>) -> Self {
        Matrix {
            data: value.data,
            backend: BE::new().unwrap()
        }
    }
}
impl<'a,BE: Backend,T,const N:usize,const M: usize> Clone for Matrix<'a,T,N,M,BE>
    where T: Clone,
          BE: Backend + Clone {
    fn clone(&self) -> Self {
        Matrix {
            data: self.data,
            backend: self.backend.clone()
        }
    }
}
/// A backend-independent view representing an invariant matrix
pub struct MatrixView<'a,T,const N: usize,const M: usize> {
    data: &'a [T]
}
impl<'a,T,const N: usize,const M: usize> Dims<N,M> for MatrixView<'a,T,N,M> {}
impl<'a,T,const N: usize,const M: usize> MatrixView<'a,T,N,M> {
    /// Retrieve a specified row of a matrix as a vector view.
    ///
    /// # Arguments
    /// * `index` - The index of the row to retrieve
    #[inline(always)]
    pub fn row(&self,index:usize) -> VectorView<'a,T,M> {
        let view = &self.data[index * M..(index + 1) * M];

        VectorView {
            data: view.try_into().unwrap()
        }
    }
}
impl<'a,T,BE: Backend,const N: usize,const M: usize> From<&'a Matrix<'a,T,N,M,BE>> for MatrixView<'a,T,N,M> {
    #[inline(always)]
    fn from(value: &'a Matrix<'a, T, N, M, BE>) -> Self {
        MatrixView {
            data: &value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a [T]> for MatrixView<'a,T,N,M> {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(MatrixView {
                data: value
            })
        }
    }
}
impl<T,const N: usize,const M: usize> Index<usize> for MatrixView<'_,T,N,M> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * M)..(index * M + M)]
    }
}
impl<'a,T,const N: usize,const M: usize> Transpose<T,N,M> for MatrixView<'a,T,N,M>
    where T: Default + Clone + Copy {
    type Output = OwnedMatrix<T,M,N>;
    #[inline(always)]
    fn transpose(self) -> OwnedMatrix<T,M,N> {
        let mut r = OwnedMatrix::<T,M,N>::default();

        const BLOCK:usize = 32;

        for row in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
            for col in (0..((M + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
                for y in 0..BLOCK {
                    for x in 0..BLOCK {
                        if row + x >= N || col + y >= M {
                            continue;
                        }
                        r[(col + y,row + x)] = self.data[(row + x) * M + col + y];
                    }
                }
            }
        }

        r
    }
}
impl<'a,T,const N: usize,const M: usize> ToColumnMajor<T,N,M> for MatrixView<'a,T,N,M>
    where T: Default + Clone + Copy + Send + Sync {
    type Output = OwnedColumnMajorMatrix<T,N,M>;
    #[inline(always)]
    fn to_column_major(self) -> OwnedColumnMajorMatrix<T,N,M> {
        let mut r = vec![T::default();M*N].into_boxed_slice();

        const BLOCK:usize = 64;

        r.par_chunks_exact_mut(N * BLOCK).zip(0..(M / BLOCK)).for_each(|(r,by)| {
            for col in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
                for x in 0..BLOCK {
                    for y in 0..BLOCK {
                        if col + x >= N {
                            continue;
                        }
                        r[y * N + (col + x)] = self.data[(col + x) * M + (by * BLOCK + y)];
                    }
                }
            }
        });

        let by = M / BLOCK;

        for col in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
            for x in 0..BLOCK {
                for y in 0..BLOCK {
                    if col + x >= N || by * BLOCK + y >= M {
                        continue;
                    }
                    r[(by * BLOCK + y) * N + (col + x)] = self.data[(col + x) * M + (by * BLOCK + y)];
                }
            }
        }

        OwnedColumnMajorMatrix { data: r }
    }
}
impl<'a,BE,T,const N: usize,const M: usize> From<&'a AccMatrix<T,N,M,BE>> for MatrixView<'a,T,N,M>
    where BE: Backend {
    fn from(value: &'a AccMatrix<T,N,M,BE>) -> MatrixView<'a,T,N,M> {
        MatrixView {
            data: &value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a OwnedMatrix<T,N,M>> for MatrixView<'a,T,N,M> {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: &'a OwnedMatrix<T,N,M>) -> Result<Self,Self::Error> {
        Ok(MatrixView {
            data: &value.data
        })
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a Vector<'a,T,N,BE>> for MatrixView<'a,T,N,1> {
    #[inline(always)]
    fn from(value: &'a Vector<'a, T, N, BE>) -> Self {
        MatrixView {
            data: value.data
        }
    }
}
impl<'a,T,const N: usize> From<&'a VectorView<'a,T,N>> for MatrixView<'a,T,N,1> {
    #[inline(always)]
    fn from(value: &'a VectorView<'a, T, N>) -> Self {
        MatrixView {
            data: value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> AsRef<[T]> for MatrixView<'a,T,N,M> {
    fn as_ref(&self) -> &'a [T] {
        self.data
    }
}
/// A mutable matrix associated with the backend
pub struct MatrixMut<'a,T,const N: usize,const M: usize,BE = AutoSelect> {
    data: &'a mut [T],
    backend: BE
}
impl<'a,T,BE,const N: usize,const M: usize> Dims<N,M> for MatrixMut<'a,T,N,M,BE> {}
impl<'a,T,BE,const N: usize,const M: usize> Index<(usize,usize)> for MatrixMut<'a,T,N,M,BE> {
    type Output = T;
    #[inline(always)]
    fn index(&self, (row,col): (usize,usize)) -> &Self::Output {
        &self.data[(row * M) + col]
    }
}
impl<'a,T,BE,const N: usize,const M: usize> IndexMut<(usize,usize)> for MatrixMut<'a,T,N,M,BE> {
    #[inline(always)]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut Self::Output {
        &mut self.data[(row * M) + col]
    }
}
impl<'a,T,BE,const N: usize,const M: usize> From<&'a mut AccMatrix<T,N,M,BE>> for MatrixMut<'a,T,N,M,BE>
    where BE: Backend {
    #[inline(always)]
    fn from(value: &'a mut AccMatrix<T, N, M, BE>) -> Self {
        MatrixMut {
            data: &mut value.data,
            backend: BE::new().unwrap(),
        }
    }
}
impl<'a,T,BE,const M: usize> From<&'a mut AccVector<T,M,BE>> for MatrixMut<'a,T,1,M,BE>
    where BE: Backend {
    #[inline(always)]
    fn from(value: &'a mut AccVector<T,M,BE>) -> Self {
        MatrixMut {
            data: &mut *value.data,
            backend: BE::new().unwrap(),
        }
    }
}
impl<'a,T,BE,const N: usize,const M: usize> TryFrom<&'a mut [T]> for MatrixMut<'a,T,N,M,BE>
    where BE: Backend {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a mut [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(MatrixMut {
                data: value,
                backend: BE::new()?
            })
        }
    }
}
impl<'a,T,BE,const N: usize,const M: usize> AsMut<[T]> for MatrixMut<'a,T,N,M,BE> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
/// A backend-independent representation of an immutable matrix
pub struct MatrixMutView<'a,T,const N: usize,const M: usize> {
    data: &'a mut [T]
}
impl<'a,T,const N: usize,const M: usize> Dims<N,M> for MatrixMutView<'a,T,N,M> {}
impl<'a,T,const N: usize,const M: usize> Index<(usize,usize)> for MatrixMutView<'a,T,N,M> {
    type Output = T;
    #[inline(always)]
    fn index(&self, (row,col): (usize,usize)) -> &Self::Output {
        &self.data[(row * M) + col]
    }
}
impl<'a,T,const N: usize,const M: usize> IndexMut<(usize,usize)> for MatrixMutView<'a,T,N,M> {
    #[inline(always)]
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut Self::Output {
        &mut self.data[(row * M) + col]
    }
}
impl<'a,T,const N: usize,const M: usize> From<&'a mut OwnedMatrix<T,N,M>> for MatrixMutView<'a,T,N,M> {
    #[inline(always)]
    fn from(value: &'a mut OwnedMatrix<T, N, M>) -> Self {
        MatrixMutView {
            data: &mut value.data
        }
    }
}
impl<'a,T,const M: usize> From<&'a mut OwnedVector<T,M>> for MatrixMutView<'a,T,1,M> {
    #[inline(always)]
    fn from(value: &'a mut OwnedVector<T,M>) -> Self {
        MatrixMutView {
            data: &mut *value.data
        }
    }
}
impl<'a,'b,T,BE,const N: usize,const M: usize> From<&'b mut MatrixMut<'a,T,N,M,BE>> for MatrixMutView<'b,T,N,M>
    where BE: Backend,
          'a: 'b {
    #[inline(always)]
    fn from(value: &'b mut MatrixMut<'a, T, N, M, BE>) -> Self {
        MatrixMutView {
            data: &mut value.data
        }
    }
}
impl<'a,T,BE,const N: usize,const M: usize> From<&'a mut AccMatrix<T,N,M,BE>> for MatrixMutView<'a,T,N,M>
    where BE: Backend {
    #[inline(always)]
    fn from(value: &'a mut AccMatrix<T, N, M, BE>) -> Self {
        MatrixMutView {
            data: &mut value.data
        }
    }
}
impl<'a,T,BE,const M: usize> From<&'a mut AccVector<T,M,BE>> for MatrixMutView<'a,T,1,M>
    where BE: Backend {
    #[inline(always)]
    fn from(value: &'a mut AccVector<T,M,BE>) -> Self {
        MatrixMutView {
            data: &mut *value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a mut [T]> for MatrixMutView<'a,T,N,M> {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a mut [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(MatrixMutView {
                data: value
            })
        }
    }
}
impl<'a,T,const N: usize,const M: usize> AsMut<[T]> for MatrixMutView<'a,T,N,M> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
/// A backend-independent representation of an owned matrix
pub struct OwnedMatrix<T,const N: usize,const M: usize> {
    data: Box<[T]>
}
impl<T,const N: usize,const M: usize> Default for OwnedMatrix<T,N,M> where T: Default + Clone {
    #[inline(always)]
    fn default() -> Self {
        Self {
            data: vec![T::default();N*M].into_boxed_slice()
        }
    }
}
impl<T,const N: usize,const M: usize> Index<(usize,usize)> for OwnedMatrix<T,N,M> {
    type Output = T;

    #[inline(always)]
    fn index(&self, (row,col): (usize, usize)) -> &Self::Output {
        &self.data[(row * M) + col]
    }
}
impl<T,const N: usize,const M: usize> IndexMut<(usize,usize)> for OwnedMatrix<T,N,M> {
    #[inline(always)]
    fn index_mut(&mut self, (row,col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[(row * M) + col]
    }
}
impl<T,const N: usize,const M: usize> From<OwnedMatrix<T,N,M>> for Box<[T]> {
    #[inline(always)]
    fn from(value: OwnedMatrix<T,N,M>) -> Self {
        value.data
    }
}
impl<T,const N: usize,const M: usize> From<Box<[T]>> for OwnedMatrix<T,N,M> {
    #[inline(always)]
    fn from(value: Box<[T]>) -> Self {
        OwnedMatrix { data: value }
    }
}
impl<T,const N: usize,const M: usize> AsMut<[T]> for OwnedMatrix<T,N,M> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
impl<T,const N: usize,const M:usize> Dims<N,M> for OwnedMatrix<T,N,M> {}
impl<'a,T,const N: usize,const M: usize> BindBackendBase for OwnedMatrix<T,N,M>
    where T: Copy + 'static {
    type Output<BE: Backend> = AccMatrix<T,N,M,BE>;

    #[inline(always)]
    fn bind<BE: Backend>(self) -> Result<Self::Output<BE>, InstantiationError> {
        Ok(AccMatrix::try_from(self)?)
    }
}
impl<'a,T,const N: usize,const M: usize> BindBackend for OwnedMatrix<T,N,M> where T: Copy + 'static {}
/// A backend-independent representation of an Accumulator matrix
pub struct AccMatrix<T,const N: usize,const M: usize,BE = AutoSelect>
    where BE: Backend {
    data: Box<[T]>,
    backend: BE
}
impl<T,BE,const N: usize,const M: usize> Index<(usize,usize)> for AccMatrix<T,N,M,BE>
    where BE: Backend {
    type Output = T;

    #[inline(always)]
    fn index(&self, (row,col): (usize, usize)) -> &Self::Output {
        &self.data[(row * M) + col]
    }
}
impl<T,BE,const N: usize,const M: usize> IndexMut<(usize,usize)> for AccMatrix<T,N,M,BE>
    where BE: Backend {
    #[inline(always)]
    fn index_mut(&mut self, (row,col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[(row * M) + col]
    }
}
impl<T,BE,const N: usize,const M: usize> From<AccMatrix<T,N,M,BE>> for Box<[T]>
    where BE: Backend {
    #[inline(always)]
    fn from(value: AccMatrix<T,N,M,BE>) -> Self {
        value.data
    }
}
impl<T,BE,const N: usize,const M: usize> TryFrom<Box<[T]>> for AccMatrix<T,N,M,BE>
    where BE: Backend {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: Box<[T]>) -> Result<AccMatrix<T,N,M,BE>,Self::Error> {
        Ok(AccMatrix { data: value, backend: BE::new()? })
    }
}
impl<T,BE,const N: usize,const M: usize> TryFrom<OwnedMatrix<T,N,M>> for AccMatrix<T,N,M,BE>
    where BE: Backend {
    type Error = InstantiationError;
    #[inline(always)]
    fn try_from(value: OwnedMatrix<T,N,M>) -> Result<AccMatrix<T,N,M,BE>,Self::Error> {
        Ok(AccMatrix { data: value.into(), backend: BE::new()? })
    }
}
impl<T,BE,const N: usize,const M: usize> AsMut<[T]> for AccMatrix<T,N,M,BE>
    where BE: Backend {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
impl<T,BE,const N: usize,const M:usize> Dims<N,M> for AccMatrix<T,N,M,BE> where BE: Backend {}
impl<'a,T,const N: usize,const M: usize> ColumnMajorMatrix<'a,T,N,M> {
    /// Retrieve a specified col of a matrix as a vector
    ///
    /// # Arguments
    /// * `index` - The index of the col to retrieve
    #[inline(always)]
    pub fn col(&self,index:usize) -> VectorView<'a,T,N> {
        let view = &self.data[index * N..(index + 1) * N];

        VectorView {
            data: view.try_into().unwrap()
        }
    }
}
/// A view representation of a matrix that stores data in column-major order
pub struct ColumnMajorMatrix<'a,T,const N: usize,const M: usize> {
    data: &'a [T]
}
impl<'a,T,const N: usize,const M: usize> Dims<N,M> for ColumnMajorMatrix<'a,T,N,M> {}
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a [T]> for ColumnMajorMatrix<'a,T,N,M> {
    type Error = InstantiationError;

    #[inline(always)]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(ColumnMajorMatrix {
                data: value
            })
        }
    }
}
impl<'a,BE: Backend,T,const M: usize> From<&'a Vector<'a,T,M,BE>> for ColumnMajorMatrix<'a,T,1,M> {
    #[inline(always)]
    fn from(value: &'a Vector<'a, T, M, BE>) -> Self {
        ColumnMajorMatrix {
            data: value.data
        }
    }
}
impl<'a,T,const M: usize> From<&'a VectorView<'a,T,M>> for ColumnMajorMatrix<'a,T,1,M> {
    #[inline(always)]
    fn from(value: &'a VectorView<'a, T, M>) -> Self {
        ColumnMajorMatrix {
            data: value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> AsRef<[T]> for ColumnMajorMatrix<'a,T,N,M> {
    fn as_ref(&self) -> &'a [T] {
        self.data
    }
}
impl<'a,T,const N: usize,const M: usize> Clone for ColumnMajorMatrix<'a,T,N,M> {
    fn clone(&self) -> Self {
        ColumnMajorMatrix {
            data: self.data
        }
    }
}
/// A type that retains ownership of data arranged in column-first order
pub struct OwnedColumnMajorMatrix<T,const M: usize,const N: usize> {
    data: Box<[T]>
}
impl<T,const N: usize,const M: usize> From<OwnedColumnMajorMatrix<T,N,M>> for Box<[T]> {
    #[inline(always)]
    fn from(value: OwnedColumnMajorMatrix<T,N,M>) -> Self {
        value.data
    }
}
impl<T,const N: usize,const M: usize> From<Box<[T]>> for OwnedColumnMajorMatrix<T,N,M> {
    #[inline(always)]
    fn from(value: Box<[T]>) -> Self {
        OwnedColumnMajorMatrix { data: value }
    }
}
impl<T,const N: usize,const M: usize> Dims<N,M> for OwnedColumnMajorMatrix<T,N,M> {}
impl<'a,T,const N: usize,const M: usize> From<&'a OwnedColumnMajorMatrix<T,N,M>> for ColumnMajorMatrix<'a,T,N,M> {
    #[inline(always)]
    fn from(value: &'a OwnedColumnMajorMatrix<T,N,M>) -> Self {
        ColumnMajorMatrix {
            data: &value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> From<&'a ColumnMajorMatrix<'a,T,N,M>>
    for OwnedColumnMajorMatrix<T,N,M>
    where T: Clone + Copy {
    #[inline(always)]
    fn from(value: &'a ColumnMajorMatrix<'a,T,N,M>) -> Self {
        OwnedColumnMajorMatrix {
            data: value.data.to_vec().into_boxed_slice()
        }
    }
}
impl<'a,T,const N: usize> Add<Vector<'a,T,N,AutoSelect>> for Vector<'a,T,N,AutoSelect>
    where T: 'static,
          AutoSelect: Backend,
          Avx2: SimdAddVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn add(self, rhs: Vector<'a,T,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.add_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            }
        }
    }
}
impl<'a,BE,T,const N: usize> Add<Vector<'a,T,N,BE>> for Vector<'a,T,N,BE>
    where T: 'static,
          BE: Backend + NativeBackend + SimdAddVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn add(self, rhs: Vector<'a,T,N,BE>) -> Self::Output {
        unsafe { self.backend.add_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,T,BE,const N: usize> Add<Vector<'a,T,N,BE>> for AccVector<T,N,BE>
    where BE: Backend,
          for<'b> Vector<'b,T,N,BE>: From<&'b AccVector<T,N,BE>>,
          for<'b> Vector<'b,T,N,BE>: Add<Vector<'a,T,N,BE>, Output = AccVector<T,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn add(self, rhs: Vector<'a,T,N,BE>) -> Self::Output {
        Vector::<T,N,BE>::from(&self) + rhs
    }
}
impl<'a,T,const N: usize> AddAssign<Vector<'a,T,N,AutoSelect>> for VectorMut<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdAddAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b VectorMut<'b,T,N,AutoSelect>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vector<'a,T,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.add_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,BE,T,const N: usize> AddAssign<Vector<'a,T,N,BE>> for VectorMut<'a,T,N,BE>
    where BE: Backend + NativeBackend + SimdAddAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b mut VectorMut<'a,T,N,BE>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vector<'a,T,N,BE>) {
        unsafe { rhs.backend.add_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,T,const N: usize> AddAssign<Vector<'a,T,N,AutoSelect>> for AccVector<T,N,AutoSelect>
    where Avx2: Backend + SimdAddAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b mut AccVector<T,N,AutoSelect>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vector<'a,T,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.add_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,BE,T,const N: usize> AddAssign<Vector<'a,T,N,BE>> for AccVector<T,N,BE>
    where BE: Backend + NativeBackend + SimdAddAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b mut AccVector<T,N,BE>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vector<'a,T,N,BE>) {
        unsafe { rhs.backend.add_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,T,const N: usize> SubAssign<Vector<'a,T,N,AutoSelect>> for VectorMut<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdSubAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b mut VectorMut<'a,T,N,AutoSelect>> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vector<'a,T,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.sub_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,BE,T,const N: usize> SubAssign<Vector<'a,T,N,BE>> for VectorMut<'a,T,N,BE>
    where BE: Backend + NativeBackend + SimdSubAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b mut VectorMut<'a,T,N,BE>> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vector<'a,T,N,BE>) {
        unsafe { rhs.backend.sub_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,T,const N: usize> SubAssign<Vector<'a,T,N,AutoSelect>> for AccVector<T,N,AutoSelect>
    where Avx2: Backend + SimdSubAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b mut AccVector<T,N,AutoSelect>> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vector<'a,T,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.sub_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,BE,T,const N: usize> SubAssign<Vector<'a,T,N,BE>> for AccVector<T,N,BE>
    where BE: Backend + NativeBackend + SimdSubAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
          for<'b> VectorMutView<'b,T,N>: From<&'b mut AccVector<T,N,BE>> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vector<'a,T,N,BE>) {
        unsafe { rhs.backend.sub_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,T,const N: usize> Sub<Vector<'a,T,N,AutoSelect>> for Vector<'a,T,N,AutoSelect>
    where T: 'static,
          Avx2: Backend + SimdSubVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn sub(self, rhs: Vector<'a,T,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.sub_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,T,const N: usize> Sub<Vector<'a,T,N,BE>> for Vector<'a,T,N,BE>
    where T: 'static,
          BE: Backend + NativeBackend + SimdSubVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn sub(self, rhs: Vector<'a,T,N,BE>) -> Self::Output {
        unsafe { self.backend.sub_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,T,BE,const N: usize> Sub<Vector<'a,T,N,BE>> for AccVector<T,N,BE>
    where BE: Backend,
          for<'b> Vector<'b,T,N,BE>: From<&'b AccVector<T,N,BE>>,
          for<'b> Vector<'b,T,N,BE>: Sub<Vector<'a,T,N,BE>, Output = AccVector<T,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn sub(self, rhs: Vector<'a,T,N,BE>) -> Self::Output {
        Vector::<T,N,BE>::from(&self) - rhs
    }
}
impl<'a,const N: usize> Mul<Vector<'a,i8,N,AutoSelect>> for Vector<'a,i8,N,AutoSelect>
    where Avx2: Backend + SimdMulVector<i8,i8,i32>,
                for<'b> VectorView<'b,i8,N>: From<&'b Vector<'a,i8,N,AutoSelect>> {
    type Output = AccVector<i32,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i8,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.mul_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,i8,N,BE>> for Vector<'a,i8,N,BE>
    where BE: Backend + SimdMulVector<i8,i8,i32>,
          for<'b> VectorView<'b,i8,N>: From<&'b Vector<'b,i8,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i8,N,BE>) -> Self::Output {
        unsafe { self.backend.mul_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,const N: usize> Mul<Vector<'a,i16,N,AutoSelect>> for Vector<'a,i16,N,AutoSelect>
    where Avx2: Backend + SimdMulVector<i16,i16,i32>,
      for<'b> VectorView<'b,i16,N>: From<&'b Vector<'b,i16,N,AutoSelect>> {
    type Output = AccVector<i32,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i16,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.mul_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,i16,N,BE>> for Vector<'a,i16,N,BE>
    where BE: Backend + SimdMulVector<i16,i16,i32>,
          for<'b> VectorView<'b,i16,N>: From<&'b Vector<'b,i16,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i16,N,BE>) -> Self::Output {
        unsafe { self.backend.mul_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,const N: usize> Mul<Vector<'a,i32,N,AutoSelect>> for Vector<'a,i32,N,AutoSelect>
    where Avx2: Backend + SimdMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'b,i32,N,AutoSelect>> {
    type Output = AccVector<i32,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i32,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.mul_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,i32,N,BE>> for Vector<'a,i32,N,BE>
    where BE: Backend + SimdMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'b,i32,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i32,N,BE>) -> Self::Output {
        unsafe { self.backend.mul_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,const N: usize> Mul<i8> for Vector<'a,i8,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulVector<i8,i8,i32>,
          for<'b> VectorView<'b,i8,N>: From<&'b Vector<'b,i8,N,AutoSelect>> {
    type Output = AccVector<i32,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: i8) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_vector(rhs,(&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,const N: usize> Mul<i8> for Vector<'a,i8,N,BE>
    where BE: Backend + SimdScalarMulVector<i8,i8,i32>,
          for<'b> VectorView<'b,i8,N>: From<&'b Vector<'b,i8,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: i8) -> Self::Output {
        unsafe { self.backend.scalarmul_vector(rhs,(&self).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,const N: usize> Mul<i16> for Vector<'a,i16,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulVector<i16,i16,i32>,
          for<'b> VectorView<'b,i16,N>: From<&'b Vector<'b,i16,N,AutoSelect>> {
    type Output = AccVector<i32,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: i16) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_vector(rhs, (&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,const N: usize> Mul<i32> for Vector<'a,i32,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'b,i32,N,AutoSelect>> {
    type Output = AccVector<i32,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: i32) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_vector(rhs, (&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,const N: usize> Mul<i32> for Vector<'a,i32,N,BE>
    where BE: Backend + SimdScalarMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'b,i32,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: i32) -> Self::Output {
        unsafe { self.backend.scalarmul_vector(rhs, (&self).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,const N: usize> Mul<f32> for Vector<'a,f32,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulVector<f32,f32,f32>,
          for<'b> VectorView<'b,f32,N>: From<&'b Vector<'b,f32,N,AutoSelect>> {
    type Output = AccVector<f32,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_vector(rhs, (&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,const N: usize> Mul<f32> for Vector<'a,f32,N,BE>
    where BE: Backend + SimdScalarMulVector<f32,f32,f32>,
          for<'b> VectorView<'b,f32,N>: From<&'b Vector<'b,f32,N,BE>> {
    type Output = AccVector<f32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        unsafe { self.backend.scalarmul_vector(rhs, (&self).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,const N: usize> Mul<f64> for Vector<'a,f64,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulVector<f64,f64,f64>,
          for<'b> VectorView<'b,f64,N>: From<&'b Vector<'b,f64,N,AutoSelect>> {
    type Output = AccVector<f64,N,AutoSelect>;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_vector(rhs, (&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,const N: usize> Mul<f64> for Vector<'a,f64,N,BE>
    where BE: Backend + SimdScalarMulVector<f64,f64,f64>,
          for<'b> VectorView<'b,f64,N>: From<&'b Vector<'b,f64,N,BE>> {
    type Output = AccVector<f64,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        unsafe { self.backend.scalarmul_vector(rhs,(&self).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,i8,N,BE>> for AccVector<i8,N,BE>
    where BE: Backend,
          for<'b> Vector<'b,i8,N,BE>: From<&'b AccVector<i8,N,BE>>,
          for<'b> Vector<'b,i8,N,BE>: Mul<Vector<'a,i8,N,BE>,Output = AccVector<i32,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i8,N,BE>) -> Self::Output {
        Vector::<i8,N,BE>::from(&self) * rhs
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,i16,N,BE>> for AccVector<i16,N,BE>
    where BE: Backend,
          for<'b> Vector<'b,i16,N,BE>: From<&'b AccVector<i16,N,BE>>,
          for<'b> Vector<'b,i16,N,BE>: Mul<Vector<'a,i16,N,BE>,Output = AccVector<i32,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i16,N,BE>) -> Self::Output {
        Vector::<i16,N,BE>::from(&self) * rhs
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,i32,N,BE>> for AccVector<i32,N,BE>
    where BE: Backend,
          for<'b> Vector<'b,i32,N,BE>: From<&'b AccVector<i32,N,BE>>,
          for<'b> Vector<'b,i32,N,BE>: Mul<Vector<'a,i32,N,BE>,Output = AccVector<i32,N,BE>> {
    type Output = AccVector<i32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,i32,N,BE>) -> Self::Output {
        Vector::<i32,N,BE>::from(&self) * rhs
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,f32,N,BE>> for AccVector<f32,N,BE>
    where BE: Backend,
          for<'b> Vector<'b,f32,N,BE>: From<&'b AccVector<f32,N,BE>>,
          for<'b> Vector<'b,f32,N,BE>: Mul<Vector<'a,f32,N,BE>,Output = AccVector<f32,N,BE>> {
    type Output = AccVector<f32,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,f32,N,BE>) -> Self::Output {
        Vector::<f32,N,BE>::from(&self) * rhs
    }
}
impl<'a,BE,const N: usize> Mul<Vector<'a,f64,N,BE>> for AccVector<f64,N,BE>
    where BE: Backend,
          for<'b> Vector<'b,f64,N,BE>: From<&'b AccVector<f64,N,BE>>,
          for<'b> Vector<'b,f64,N,BE>: Mul<Vector<'a,f64,N,BE>,Output = AccVector<f64,N,BE>> {
    type Output = AccVector<f64,N,BE>;

    #[inline(always)]
    fn mul(self, rhs: Vector<'a,f64,N,BE>) -> Self::Output {
        Vector::<f64,N,BE>::from(&self) * rhs
    }
}
impl<'a,const N: usize> MulAssign<Vector<'a,i32,N,AutoSelect>> for VectorMut<'a,i32,N,AutoSelect>
    where Avx2: Backend + SimdMulAssignVector<i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'b,i32,N,AutoSelect>>,
          for<'b> VectorMutView<'b,i32,N>: From<&'b mut VectorMut<'a,i32,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,i32,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.mul_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,BE,const N: usize> MulAssign<Vector<'a,i32,N,BE>> for VectorMut<'a,i32,N,BE>
    where BE: Backend + SimdMulAssignVector<i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'b,i32,N,BE>>,
          for<'b> VectorMutView<'b,i32,N>: From<&'b mut VectorMut<'a,i32,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,i32,N,BE>) {
        unsafe { rhs.backend.mul_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,const N: usize> MulAssign<i32> for VectorMut<'a,i32,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulAssignVector<i32,i32>,
          for<'b> VectorMutView<'b,i32,N>: From<&'b mut VectorMut<'a,i32,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: i32) {
        let selected = self.backend.backend();

        match selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_assign_vector(rhs, self.into())
            },
        }
    }
}
impl<'a,BE,const N: usize> MulAssign<i32> for VectorMut<'a,i32,N,BE>
    where BE: Backend + SimdScalarMulAssignVector<i32,i32> + Clone,
          for<'b> VectorMutView<'b,i32,N>: From<&'b mut VectorMut<'a,i32,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: i32) {
        let backend = self.backend.clone();
        unsafe { backend.scalarmul_assign_vector(rhs,self.into()) }
    }
}
impl<'a,const N: usize> MulAssign<Vector<'a,f32,N,AutoSelect>> for VectorMut<'a,f32,N,AutoSelect>
    where Avx2: Backend + SimdMulAssignVector<f32,f32>,
          for<'b> VectorView<'b,f32,N>: From<&'b Vector<'a,f32,N,AutoSelect>>,
          for<'b> VectorMutView<'b,f32,N>: From<&'b mut VectorMut<'a,f32,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,f32,N,AutoSelect>) {
        let selected = self.backend.backend();

        match selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.mul_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,BE,const N: usize> MulAssign<Vector<'a,f32,N,BE>> for VectorMut<'a,f32,N,BE>
    where BE: Backend + SimdMulAssignVector<f32,f32> + Clone,
          for<'b> VectorView<'b,f32,N>: From<&'b Vector<'a,f32,N,BE>>,
          for<'b> VectorMutView<'b,f32,N>: From<&'b mut VectorMut<'a,f32,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,f32,N,BE>) {
        let backend = self.backend.clone();

        unsafe { backend.mul_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,const N: usize> MulAssign<f32> for VectorMut<'a,f32,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulAssignVector<f32,f32>,
          for<'b> VectorMutView<'b,f32,N>: From<&'b mut VectorMut<'a,f32,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        let selected = self.backend.backend();

        match selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_assign_vector(rhs, self.into())
            },
        }
    }
}
impl<'a,BE,const N: usize> MulAssign<f32> for VectorMut<'a,f32,N,BE>
    where BE: Backend + SimdScalarMulAssignVector<f32,f32> + Clone,
          for<'b> VectorMutView<'b,f32,N>: From<&'b mut VectorMut<'a,f32,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        let backend = self.backend.clone();

        unsafe { backend.scalarmul_assign_vector(rhs,self.into()) }
    }
}
impl<'a,const N: usize> MulAssign<Vector<'a,f64,N,AutoSelect>> for VectorMut<'a,f64,N,AutoSelect>
    where Avx2: Backend + SimdMulAssignVector<f64,f64>,
          for<'b> VectorView<'b,f64,N>: From<&'b Vector<'b,f64,N,AutoSelect>>,
          for<'b> VectorMutView<'b,f64,N>: From<&'b mut VectorMut<'a,f64,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,f64,N,AutoSelect>) {
        let selected = self.backend.backend();

        match selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.mul_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,BE,const N: usize> MulAssign<Vector<'a,f64,N,BE>> for VectorMut<'a,f64,N,BE>
    where BE: Backend + SimdMulAssignVector<f64,f64> + Clone,
          for<'b> VectorView<'b,f64,N>: From<&'b Vector<'b,f64,N,BE>>,
          for<'b> VectorMutView<'b,f64,N>: From<&'b mut VectorMut<'a,f64,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,f64,N,BE>) {
        let backend = self.backend.clone();

        unsafe { backend.mul_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,const N: usize> MulAssign<f64> for VectorMut<'a,f64,N,AutoSelect>
    where Avx2: Backend + SimdScalarMulAssignVector<f64,f64>,
          for<'b> VectorMutView<'b,f64,N>: From<&'b mut VectorMut<'a,f64,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        let selected = self.backend.backend();

        match selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_assign_vector(rhs,self.into())
            },
        }
    }
}
impl<'a,BE,const N: usize> MulAssign<f64> for VectorMut<'a,f64,N,BE>
    where BE: Backend + SimdScalarMulAssignVector<f64,f64> + Clone,
          for<'b> VectorMutView<'b,f64,N>: From<&'b mut VectorMut<'a,f64,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        let backend = self.backend.clone();

        unsafe { backend.scalarmul_assign_vector(rhs,self.into()) }
    }
}
impl<'a,SL,SR,const N: usize> MulAssign<Vector<'a,SR,N,AutoSelect>> for AccVector<SL,N,AutoSelect>
    where Avx2: Backend + SimdMulAssignVector<SL,SR>,
          for<'b> VectorView<'b,SR,N>: From<&'b Vector<'b,SR,N,AutoSelect>>,
          for<'b> VectorMutView<'b,SL,N>: From<&'b mut AccVector<SL,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,SR,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.mul_assign_vector(self.into(), (&rhs).into())
            },
        }
    }
}
impl<'a,SL,SR,BE,const N: usize> MulAssign<Vector<'a,SR,N,BE>> for AccVector<SL,N,BE>
    where BE: Backend + NativeBackend + SimdMulAssignVector<SL,SR>,
          for<'b> VectorView<'b,SR,N>: From<&'b Vector<'b,SR,N,BE>>,
          for<'b> VectorMutView<'b,SL,N>: From<&'b mut AccVector<SL,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<'a,SR,N,BE>) {
        unsafe { rhs.backend.mul_assign_vector(self.into(), (&rhs).into()) }
    }
}
impl<'a,SL,SR,const N: usize> MulAssign<SR> for AccVector<SL,N,AutoSelect>
    where SR: IsScaler,
          Avx2: Backend + SimdScalarMulAssignVector<SR,SL>,
          for<'b> VectorMutView<'b,SL,N>: From<&'b mut AccVector<SL,N,AutoSelect>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: SR) {
        let selected = self.backend.backend();

        match selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.scalarmul_assign_vector(rhs, self.into())
            },
        }
    }
}
impl<'a,SL,SR,BE,const N: usize> MulAssign<SR> for AccVector<SL,N,BE>
    where BE: Backend + NativeBackend + SimdScalarMulAssignVector<SR,SL> + Clone,
          SR: IsScaler,
          for<'b> VectorMutView<'b,SL,N>: From<&'b mut AccVector<SL,N,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: SR) {
        let backend = self.backend.clone();
        unsafe { backend.scalarmul_assign_vector(rhs,self.into()) }
    }
}
impl<'a,T,const N: usize> BitXor<Vector<'a,<T as BitsBitXor>::Bits,N,AutoSelect>> for Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdBitXorVector<T>,
          T: BitsBitXor + 'static,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorView<'b,<T as BitsBitXor>::Bits,N>: From<&'b Vector<'b,<T as BitsBitXor>::Bits,N,AutoSelect>> {
    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn bitxor(self, rhs: Vector<'a,<T as BitsBitXor>::Bits,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.bitxor_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,T,const N: usize> BitXor<Vector<'a,<T as BitsBitXor>::Bits,N,BE>> for Vector<'a,T,N,BE>
    where BE: Backend + NativeBackend +
              SimdReg<T,Bits=<T as BitsBitXor>::Bits> +
              SimdReg<<T as BitsBitXor>::Bits> +
              SimdBitXorVector<T>,
              T: BitsBitXor + 'static,
              for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
              for<'b> VectorView<'b,<T as BitsBitXor>::Bits,N>: From<&'b Vector<'b,<T as BitsBitXor>::Bits,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn bitxor(self, rhs: Vector<'a,<T as BitsBitXor>::Bits,N,BE>) -> Self::Output {
        unsafe { self.backend.bitxor_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,T,const N: usize> BitOr<Vector<'a,<T as BitsBitOr>::Bits,N,AutoSelect>> for Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdBitOrVector<T>,
          T: BitsBitOr + 'static,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorView<'b,<T as BitsBitOr>::Bits,N>: From<&'b Vector<'b,<T as BitsBitOr>::Bits,N,AutoSelect>> {
    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn bitor(self, rhs: Vector<'a,<T as BitsBitOr>::Bits,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.bitor_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,T,const N: usize> BitOr<Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>> for Vector<'a,T,N,BE>
    where BE: Backend + NativeBackend +
              SimdReg<T,Bits=<T as BitsBitOr>::Bits> +
              SimdReg<<T as BitsBitOr>::Bits> +
              SimdBitOrVector<T>,
              T: BitsBitOr + 'static,
              for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
              for<'b> VectorView<'b,<T as BitsBitOr>::Bits,N>: From<&'b Vector<'b,<T as BitsBitOr>::Bits,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn bitor(self, rhs: Vector<'a,<T as BitsBitOr>::Bits,N,BE>) -> Self::Output {
        unsafe { self.backend.bitor_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,T,const N: usize> BitAnd<Vector<'a,<T as BitsBitAnd>::Bits,N,AutoSelect>> for Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdBitAndVector<T>,
      T: BitsBitAnd + 'static,
      for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
      for<'b> VectorView<'b,<T as BitsBitAnd>::Bits,N>: From<&'b Vector<'b,<T as BitsBitAnd>::Bits,N,AutoSelect>> {

    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn bitand(self, rhs: Vector<'a,<T as BitsBitAnd>::Bits,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.bitand_vector((&self).into(), (&rhs).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,T,const N: usize> BitAnd<Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>> for Vector<'a,T,N,BE>
    where BE: Backend + NativeBackend +
              SimdReg<T,Bits=<T as BitsBitAnd>::Bits> +
              SimdReg<<T as BitsBitAnd>::Bits> +
              SimdBitAndVector<T>,
              T: BitsBitAnd + 'static,
              for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
              for<'b> VectorView<'b,<T as BitsBitAnd>::Bits,N>: From<&'b Vector<'b,<T as BitsBitAnd>::Bits,N,BE>> {

    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn bitand(self, rhs: Vector<'a,<T as BitsBitAnd>::Bits,N,BE>) -> Self::Output {
        unsafe { self.backend.bitand_vector((&self).into(), (&rhs).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,T,const N: usize> Not for Vector<'a,T,N,AutoSelect>
    where T: 'static,
          Avx2: Backend + SimdBitNotVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {

    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn not(self) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.bitnot_vector((&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,T,const N: usize> Not for Vector<'a,T,N,BE>
    where T: 'static,
          BE: Backend + NativeBackend + SimdBitNotVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {

    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { self.backend.bitnot_vector((&self).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,T,const N: usize> Shl<usize> for Vector<'a,T,N,AutoSelect>
    where T: 'static,
          Avx2: Backend + SimdShlVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.shl_vector((&self).into(), rhs).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,T,const N: usize> Shl<usize> for Vector<'a,T,N,BE>
    where T: 'static,
          BE: Backend + NativeBackend + SimdShlVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        unsafe { self.backend.shl_vector((&self).into(), rhs).bind::<BE>().unwrap() }
    }
}
impl<'a,T,const N: usize> Shr<usize> for Vector<'a,T,N,AutoSelect>
    where T: 'static,
          Avx2: Backend + SimdShrVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = AccVector<T,N,AutoSelect>;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.shr_vector((&self).into(), rhs).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,T,const N: usize> Shr<usize> for Vector<'a,T,N,BE>
    where T: 'static,
          BE: Backend + NativeBackend + SimdShrVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = AccVector<T,N,BE>;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        unsafe { self.backend.shr_vector((&self).into(), rhs).bind::<BE>().unwrap() }
    }
}
impl<'a,SL,SR,const N: usize> Promote<AccVector<SR,N>> for Vector<'a,SL,N,AutoSelect>
    where SL: 'static,
          SR: 'static,
          Avx2: Backend + SimdPromoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>> {
    #[inline(always)]
    fn promotion(self) -> AccVector<SR,N,AutoSelect> {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.promotion_vector((&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,SL,SR,const N: usize> Promote<AccVector<SR,N,BE>> for Vector<'a,SL,N,BE>
    where SL: 'static,
          SR: 'static,
          BE: Backend + NativeBackend + SimdPromoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>> {
    #[inline(always)]
    fn promotion(self) -> AccVector<SR,N,BE> {
        unsafe { self.backend.promotion_vector((&self).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,SL,SR,const N: usize> Demote<AccVector<SR,N,AutoSelect>> for Vector<'a,SL,N,AutoSelect>
    where SL: 'static,
          SR: 'static,
          Avx2: Backend + SimdDemoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>> {
    #[inline(always)]
    fn demotion(self) -> AccVector<SR,N,AutoSelect> {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.demotion_vector((&self).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,SL,SR,const N: usize> Demote<AccVector<SR,N,BE>> for Vector<'a,SL,N,BE>
    where SL: 'static,
          SR: 'static,
          BE: Backend + NativeBackend + SimdDemoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>> {
    #[inline(always)]
    fn demotion(self) -> AccVector<SR,N,BE> {
        unsafe { self.backend.demotion_vector((&self).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,SL,SR,const N: usize> From<Vector<'a,SL,N,AutoSelect>> for AccVector<SR,N,AutoSelect>
    where SL: 'static,
          SR: 'static,
          Avx2: Backend + SimdConvertVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>> {
    #[inline(always)]
    fn from(s:Vector<'a,SL,N,AutoSelect>) -> AccVector<SR,N,AutoSelect> {
        match s.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.convert_vector((&s).into()).bind_auto().unwrap()
            },
        }
    }
}
impl<'a,BE,SL,SR,const N: usize> From<Vector<'a,SL,N,BE>> for AccVector<SR,N,BE>
    where SL: 'static,
          SR: 'static,
          BE: Backend + NativeBackend + SimdConvertVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>> {
    #[inline(always)]
    fn from(s:Vector<'a,SL,N,BE>) -> AccVector<SR,N,BE> {
        unsafe { s.backend.convert_vector((&s).into()).bind::<BE>().unwrap() }
    }
}
impl<'a,T,const N: usize,const M: usize> AddAssign<Matrix<'a,T,N,M,AutoSelect>> for MatrixMut<'a,T,N,M,AutoSelect>
    where AutoSelect: Backend,
          Avx2: SimdAddAssignMatrix<T,T>,
          for<'b> MatrixMutView<'b,T,N,M>: From<&'b MatrixMut<'a,T,N,M,AutoSelect>>,
          for<'b> MatrixView<'b,T,N,M>: From<&'b Matrix<'b,T,N,M,AutoSelect>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Matrix<'a,T,N,M,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.add_assign_matrix(self.into(), (&rhs).into())
            }
        }
    }
}
impl<'a,BE,T,const N: usize,const M: usize> AddAssign<Matrix<'a,T,N,M,BE>> for MatrixMut<'a,T,N,M,BE>
    where BE: Backend + NativeBackend + SimdAddAssignMatrix<T,T>,
          for<'b> MatrixMutView<'b,T,N,M>: From<&'b MatrixMut<'a,T,N,M,BE>>,
          for<'b> MatrixView<'b,T,N,M>: From<&'b Matrix<'b,T,N,M,BE>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Matrix<'a,T,N,M,BE>) {
        unsafe { rhs.backend.add_assign_matrix(self.into(), (&rhs).into()) }
    }
}
impl<'a,T,const N: usize,const M: usize> AddAssign<Matrix<'a,T,N,M,AutoSelect>> for AccMatrix<T,N,M,AutoSelect>
    where AutoSelect: Backend,
          Avx2: SimdAddAssignMatrix<T,T>,
          for<'b> MatrixMutView<'b,T,N,M>: From<&'b mut AccMatrix<T,N,M,AutoSelect>>,
          for<'b> MatrixView<'b,T,N,M>: From<&'b Matrix<'b,T,N,M,AutoSelect>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Matrix<'a,T,N,M,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.add_assign_matrix(self.into(), (&rhs).into())
            }
        }
    }
}
impl<'a,BE,T,const N: usize,const M: usize> AddAssign<Matrix<'a,T,N,M,BE>> for AccMatrix<T,N,M,BE>
    where BE: Backend + NativeBackend + SimdAddAssignMatrix<T,T>,
          for<'b> MatrixMutView<'b,T,N,M>: From<&'b mut AccMatrix<T,N,M,BE>>,
          for<'b> MatrixView<'b,T,N,M>: From<&'b Matrix<'b,T,N,M,BE>> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Matrix<'a,T,N,M,BE>) {
        unsafe { rhs.backend.add_assign_matrix(self.into(), (&rhs).into()) }
    }
}
impl<'a,BE,T,const N: usize,const M: usize> MulAssign<T> for MatrixMut<'a,T,N,M,BE>
    where BE: Backend + NativeBackend + SimdScalarMulAssignMatrix<T,T> + Clone,
          T: Copy + IsScaler,
          for<'b> MatrixMutView<'b,T,N,M>: From<&'b MatrixMut<'a,T,N,M,BE>>,
          for<'b> MatrixView<'b,T,N,M>: From<&'b Matrix<'b,T,N,M,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        let backend = self.backend.clone();

        unsafe { backend.scalar_mul_assign_matrix(rhs,self.into()) }
    }
}
impl<'a,BE,T,const N: usize,const M: usize> MulAssign<T> for AccMatrix<T,N,M,BE>
    where BE: Backend + NativeBackend + SimdScalarMulAssignMatrix<T,T> + Clone,
          T: Copy + IsScaler,
          for<'b> MatrixMutView<'b,T,N,M>: From<&'b mut AccMatrix<T,N,M,BE>>,
          for<'b> MatrixView<'b,T,N,M>: From<&'b Matrix<'b,T,N,M,BE>> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        let backend = self.backend.clone();

        unsafe { backend.scalar_mul_assign_matrix(rhs,self.into()) }
    }
}
impl<'a,BE,SL,SR,const N: usize,const M: usize> Mul<SR> for Matrix<'a,SL,N,M,BE>
    where BE: Backend + SimdScalarMulMatrix<SR,SL> + Clone,
          SL: Copy + IsScaler,
          SR: Copy + IsScaler,
          <BE as SimdScalarMulMatrix<SR,SL>>::OutputScalar: 'static,
          for<'b> MatrixView<'b,SL,N,M>: From<&'b Matrix<'b,SL,N,M,BE>> {
    type Output = AccMatrix<<BE as SimdScalarMulMatrix<SR,SL>>::OutputScalar,N,M,BE>;
    #[inline(always)]
    fn mul(self, rhs: SR) -> AccMatrix<<BE as SimdScalarMulMatrix<SR,SL>>::OutputScalar,N,M,BE> {
        let mut acc = OwnedMatrix::default();

        let backend = self.backend.clone();

        unsafe { backend.scalar_mul_matrix(rhs,(&self).into(),(&mut acc).into()) };

        acc.bind::<BE>().unwrap()
    }
}
impl<'a,BE,SL,SR,const N: usize,const M: usize> From<Matrix<'a,SL,N,M,BE>> for AccMatrix<SR,N,M,BE>
    where BE: Backend + SimdConvertMatrix<SL,SR>,
          SR: Default + Copy + 'static,
          for<'b> MatrixMutView<'b,SL,N,M>: From<&'b mut OwnedMatrix<SL,N,M>>,
          for<'b> MatrixView<'b,SL,N,M>: From<&'b Matrix<'b,SL,N,M,BE>> {
    #[inline(always)]
    fn from(s:Matrix<'a,SL,N,M,BE>) -> AccMatrix<SR,N,M,BE> {
        let mut acc = OwnedMatrix::default();

        unsafe { s.backend.convert_matrix((&s).into(),(&mut acc).into()) };

        acc.bind::<BE>().unwrap()
    }
}
impl<'a,SL,SR,SO,const N: usize> Dot<Vector<'a,SR,N,AutoSelect>,SO> for Vector<'a,SL,N,AutoSelect>
    where Avx2: Backend + SimdDot<SL,SR,SO>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>>,
          for<'b> VectorView<'b,SR,N>: From<&'b Vector<'b,SR,N,AutoSelect>> {
    #[inline(always)]
    fn dot(&self,r:Vector<'_,SR,N,AutoSelect>) -> SO {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => unsafe {
                backend.dot(&self.into(), (&r).into())
            },
        }
    }
}
impl<'a,BE,SL,SR,SO,const N: usize> Dot<Vector<'a,SR,N,BE>,SO> for Vector<'a,SL,N,BE>
    where BE: Backend + NativeBackend + SimdDot<SL,SR,SO>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>>,
          for<'b> VectorView<'b,SR,N>: From<&'b Vector<'b,SR,N,BE>> {
    #[inline(always)]
    fn dot(&self,r:Vector<'_,SR,N,BE>) -> SO {
        unsafe { self.backend.dot(&self.into(),(&r).into()) }
    }
}
impl<'a,SL,SR,SO,const N: usize,const M: usize> Product<Vector<'a,SR,M,AutoSelect>,AccMatrix<SO,N,M,AutoSelect>>
    for Vector<'a,SL,N,AutoSelect>
    where Avx2: Backend + SimdOuterProduct<SL,SR,SO>,
          SO: Default + Copy + Clone + 'static,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>>,
          for<'b> VectorView<'b,SR,M>: From<&'b Vector<'b,SR,M,AutoSelect>>,
          for<'b> MatrixMutView<'b,SO,N,M>: From<&'b mut OwnedMatrix<SO,N,M>> {
    #[inline(always)]
    fn product(&self,r:Vector<'a,SR,M,AutoSelect>) -> AccMatrix<SO,N,M,AutoSelect> {
        let mut o = OwnedMatrix::<SO,N,M>::default();

        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => {
                unsafe { backend.outer_product(self.into(),(&r).into(),(&mut o).into()) };
            }
        }

        o.bind_auto().unwrap()
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize> Product<Vector<'a,SR,M,BE>,AccMatrix<SO,N,M,BE>>
    for Vector<'a,SL,N,BE>
    where BE: Backend + NativeBackend + SimdOuterProduct<SL,SR,SO>,
          SO: Default + Copy + Clone + 'static,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>>,
          for<'b> VectorView<'b,SR,M>: From<&'b Vector<'b,SR,M,BE>>,
          for<'b> MatrixMutView<'b,SO,N,M>: From<&'b mut OwnedMatrix<SO,N,M>> {
    #[inline(always)]
    fn product(&self,r:Vector<'a,SR,M,BE>) -> AccMatrix<SO,N,M,BE> {
        let mut o = OwnedMatrix::<SO,N,M>::default();
        unsafe { self.backend.outer_product(self.into(),(&r).into(),(&mut o).into()) };

        o.bind::<BE>().unwrap()
    }
}
impl<'a,SL,SR,SO,const M: usize,const K: usize> Product<ColumnMajorMatrix<'a,SR,K,M>,AccVector<SO,M,AutoSelect>>
    for Vector<'a,SL,K,AutoSelect>
    where Avx2: Backend + SimdVMat<SL,SR,SO>,
          SO: Default + Copy + Clone + 'static,
          for<'b> VectorView<'b,SL,K>: From<&'b Vector<'b,SL,K,AutoSelect>> {
    #[inline(always)]
    fn product(&self,r:ColumnMajorMatrix<'a,SR,K,M>) -> AccVector<SO,M,AutoSelect> {
        let mut o = OwnedVector::<SO,M>::default();

        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => {
                unsafe { backend.vmat(self.into(),r,&mut o) };
            }
        }

        o.bind_auto().unwrap()
    }
}
impl<'a,BE,SL,SR,SO,const M: usize,const K: usize> Product<ColumnMajorMatrix<'a,SR,K,M>,AccVector<SO,M,BE>>
    for Vector<'a,SL,K,BE>
    where BE: Backend + NativeBackend + SimdVMat<SL,SR,SO>,
          SO: Default + Copy + Clone + 'static,
          for<'b> VectorView<'b,SL,K>: From<&'b Vector<'b,SL,K,BE>> {
    #[inline(always)]
    fn product(&self,r:ColumnMajorMatrix<'a,SR,K,M>) -> AccVector<SO,M,BE> {
        let mut o = OwnedVector::<SO,M>::default();

        unsafe { self.backend.vmat(self.into(),r,&mut o) };

        o.bind::<BE>().unwrap()
    }
}
impl<'a,SL,SR,SO,const N: usize,const K: usize> Product<Vector<'a,SR,K,AutoSelect>,AccVector<SO,N,AutoSelect>>
    for Matrix<'a,SL,N,K,AutoSelect>
    where Avx2: Backend + SimdMatVec<SL,SR,SO>,
          SO: Default + Copy + Clone + 'static,
          for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,AutoSelect>>,
          for<'b> VectorView<'b,SR,K>: From<&'b Vector<'b,SR,K,AutoSelect>> {
    #[inline(always)]
    fn product(&self, r: Vector<'a,SR,K,AutoSelect>) -> AccVector<SO,N,AutoSelect> {
        let mut o = OwnedVector::<SO,N>::default();

        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => {
                unsafe { backend.matvec(self.into(),(&r).into(),&mut o) };
            }
        }

        o.bind_auto().unwrap()
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const K: usize> Product<Vector<'a,SR,K,BE>,AccVector<SO,N,BE>>
    for Matrix<'a,SL,N,K,BE>
    where BE: Backend + NativeBackend + SimdMatVec<SL,SR,SO>,
        SO: Default + Copy + Clone + 'static,
        for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,BE>>,
        for<'b> VectorView<'b,SR,K>: From<&'b Vector<'b,SR,K,BE>> {
    #[inline(always)]
    fn product(&self, r: Vector<'a,SR,K,BE>) -> AccVector<SO,N,BE> {
        let mut o = OwnedVector::<SO,N>::default();

        unsafe { self.backend.matvec(self.into(), (&r).into(), &mut o) };

        o.bind::<BE>().unwrap()
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const K: usize> Product<Vector<'a,SR,K,BE>,AccVector<SO,N,BE>>
    for AccMatrix<SL,N,K,BE>
    where BE: Backend + NativeBackend + SimdMatVec<SL,SR,SO>,
          SO: Default + Copy + Clone + 'static,
          for<'b> MatrixView<'b,SL,N,K>: From<&'b AccMatrix<SL,N,K,BE>>,
          for<'b> VectorView<'b,SR,K>: From<&'b Vector<'a,SR,K,BE>> {
    #[inline(always)]
    fn product(&self, r: Vector<'a,SR,K,BE>) -> AccVector<SO,N,BE> {
        let mut o = OwnedVector::<SO,N>::default();

        unsafe { self.backend.matvec(self.into(), (&r).into(), &mut o) };

        o.bind::<BE>().unwrap()
    }
}
impl<'a,SL,SR,SO,const N: usize,const M: usize,const K: usize> Product<ColumnMajorMatrix<'a,SR,K,M>,AccMatrix<SO,N,M,AutoSelect>>
    for Matrix<'a,SL,N,K,AutoSelect>
    where Avx2: Backend + SimdMatMul<SL,SR,SO>,
          SO: Default + Copy + Clone + 'static,
          for<'b> MatrixMutView<'b,SL,N,K>: From<&'b mut OwnedMatrix<SL,N,K>>,
          for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,AutoSelect>> {
    #[inline(always)]
    fn product(&self, r: ColumnMajorMatrix<'a,SR,K,M>) -> AccMatrix<SO,N,M,AutoSelect> {
        let mut o = OwnedMatrix::<SO,N,M>::default();
        {
            let o = (&mut o).into();

            match self.backend.selected {
                SelectedBackend::Avx2(ref backend) => {
                    unsafe { backend.matmul(self.into(),r,o) };
                }
            }
        }

        o.bind_auto().unwrap()
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize,const K: usize> Product<ColumnMajorMatrix<'a,SR,K,M>,AccMatrix<SO,N,M,BE>>
    for Matrix<'a,SL,N,K,BE>
    where BE: Backend + NativeBackend + SimdMatMul<SL,SR,SO>,
          SL: Copy,
          SR: Copy,
          SO: Default + From<SL> + From<SR> + Mul<SO,Output=SO> + Copy + Clone + 'static,
          for<'b> MatrixMutView<'b,SL,N,K>: From<&'b mut OwnedMatrix<SL,N,K>>,
          for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,BE>> {
    #[inline(always)]
    fn product(&self, r: ColumnMajorMatrix<'a,SR,K,M>) -> AccMatrix<SO,N,M,BE> {
        let mut o = OwnedMatrix::<SO,N,M>::default();
        {
            let o = (&mut o).into();
            unsafe { self.backend.matmul(self.into(), r, o) };
        }

        o.bind::<BE>().unwrap()
    }
}