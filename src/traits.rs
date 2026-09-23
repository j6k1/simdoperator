//! Trait and data type features for abstracting SIMD operations

use crate::{ColumnMajorMatrix, MatrixMut, MatrixView, OwnedMatrix, OwnedVector, VectorMut, VectorView};
use crate::backend::autoselect::AutoSelect;
use crate::backend::common::{Backend};
use crate::error::InstantiationError;

/// Addition at the SIMD register level
pub trait SimdAdd<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the addition
    /// * `r` - Right hand side of the addition
    fn add(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
/// Subtract at the SIMD register level
pub trait SimdSub<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Subtraction
    /// * `r` - Right hand side of the Subtraction
    fn sub(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
/// Multiply at the SIMD register level
pub trait SimdMul<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdAdd<SO,SO,SO> + Sized {
    type Output: FoldRegs<SO,Self>;
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Multiply
    /// * `r` - Right hand side of the Multiply
    fn mul(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> Self::Output;
}
/// Multiply a scalar by a SIMD register
pub trait SimdScalarMul<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdMul<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Multiply
    /// * `r` - Right hand side of the Multiply
    fn scalarmul(&self,l:SL,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdMul<SL,SR,SO>>::Output;
}
/// Bitwise OR at the SIMD register level
pub trait SimdBitOr<S>: SimdReg<S>
    where S: BitsBitOr,
          Self: SimdReg<<S as BitsBitOr>::Bits> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Bitwise OR
    /// * `r` - Right hand side of the Bitwise OR
    fn bitor(&self,l:<Self as SimdReg<S>>::Reg,r:<Self as SimdReg<<S as BitsBitOr>::Bits>>::Reg) -> <Self as SimdReg<S>>::Reg;
}
/// Bitwise AND at the SIMD register level
pub  trait SimdBitAnd<S>: SimdReg<S>
    where S: BitsBitAnd,
          Self: SimdReg<<S as BitsBitAnd>::Bits>{
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Bitwise AND
    /// * `r` - Right hand side of the Bitwise AND
    fn bitand(&self,l:<Self as SimdReg<S>>::Reg,r:<Self as SimdReg<<S as BitsBitAnd>::Bits>>::Reg) -> <Self as SimdReg<S>>::Reg;
}
/// Bitwise XOR at the SIMD register level
pub trait SimdBitXor<S>: SimdReg<S>
    where S: BitsBitXor,
          Self: SimdReg<<S as BitsBitXor>::Bits> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Bitwise XOR
    /// * `r` - Right hand side of the Bitwise XOR
    fn bitxor(&self,l:<Self as SimdReg<S>>::Reg,r:<Self as SimdReg<<S as BitsBitXor>::Bits>>::Reg) -> <Self as SimdReg<S>>::Reg;
}
/// Bitwise NOT at the SIMD register level
pub trait SimdBitNot<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `v` - A register that stores a bit not
    fn bitnot(&self,v:Self::Reg) -> Self::Reg;
}
/// Generate register values for the shift width of each element of the SIMD register
pub trait SimdShiftWidth<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `w` - Scalar value of the shift width
    fn shift_width(&self,w:usize) -> Self::ShiftWidth;
}
/// Generate register values for bits left shift of each element of the SIMD register
pub trait SimdShl<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `w` - shift width
    fn shl(&self,v:Self::Reg,w:<Self as SimdReg<S>>::ShiftWidth) -> Self::Reg;
}
/// Generate register values for bits right shift of each element of the SIMD register
pub trait SimdShr<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `w` - shift width
    fn shr(&self,v:Self::Reg,w:<Self as SimdReg<S>>::ShiftWidth) -> Self::Reg;
}
/// Generate register values for splatting a scalar value into a SIMD register
pub trait SimdSplat<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `v` - A scalar value that will be splatted into a SIMD register
    fn splat(&self,v:S) -> Self::Reg;
}
/// Apply AddAssign to each element of a Vector
pub trait SimdAddAssignVector<SL,SR> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the AddAssign
    /// * `r` - Right hand side of the AddAssign`
    unsafe fn add_assign_vector<'a,const N: usize>(&self, l:&mut VectorMut<'a,SL,N>, r:&VectorView<'a,SR,N>);
}
/// Apply Add to each element of a Vector
pub trait SimdAddVector<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Add
    /// * `r` - Right hand side of the Add
    unsafe fn add_vector<'a,const N: usize>(&self, l:&VectorView<'a,SL,N>, r:&VectorView<'a,SR,N>) -> OwnedVector<SO,N>;
}
/// Apply SubAssign to each element of a Vector
pub trait SimdSubAssignVector<SL,SR> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the SubAssign
    /// * `r` - Right hand side of the SubAssign
    unsafe fn sub_assign_vector<'a,const N: usize>(&self, l:&mut VectorMut<'a,SL,N>, r:&VectorView<'a,SR,N>);
}
/// Apply Sub to each element of a Vector
pub trait SimdSubVector<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Sub
    /// * `r` - Right hand side of the Sub
    unsafe fn sub_vector<'a,const N: usize>(&self, l:&VectorView<'a,SL,N>, r:&VectorView<'a,SR,N>) -> OwnedVector<SO,N>;
}
/// Apply MulAssign to each element of a Vector
pub trait SimdMulAssignVector<SL,SR> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the MulAssign
    /// * `r` - Right hand side of the MulAssign
    unsafe fn mul_assign_vector<'a,const N: usize>(&self, l:&mut VectorMut<'a,SL,N>, r:&VectorView<'a,SR,N>);
}
/// Apply Mul to each element of a Vector
pub trait SimdMulVector<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Mul
    /// * `r` - Right hand side of the Mul
    unsafe fn mul_vector<'a,const N: usize>(&self, l:&VectorView<'a,SL,N>, r:&VectorView<'a,SR,N>) -> OwnedVector<SO,N>;
}
/// Multiply each element of the vector by a scalar value to update its value
pub trait SimdScalarMulAssignVector<SL,SR> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Mul
    /// * `r` - Right hand side of the Mul
    unsafe fn scalarmul_assign_vector<'a,const N: usize>(&self, l:SL, r:&mut VectorMut<'a,SR,N>);
}
/// Multiply each element of the vector by a scalar value
pub trait SimdScalarMulVector<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Mul
    /// * `r` - Right hand side of the Mul
    unsafe fn scalarmul_vector<'a,const N: usize>(&self, l:SL, r:&VectorView<'a,SR,N>) -> OwnedVector<SO,N>;
}
pub trait SimdScalarMulVectorInto<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Mul
    /// * `r` - Right hand side of the Mul
    unsafe fn scalarmul_vector_into<'a, const N: usize>(&self, l: SL, r: &VectorView<'a, SR, N>, o: &mut VectorMut<'a, SO, N>);
}
/// Apply a bitwise XOR to each element of a Vector
pub trait SimdBitXorVector<S>
    where S: BitsBitXor {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Bitwise XOR
    /// * `r` - Right hand side of the Bitwise XOR
    unsafe fn bitxor_vector<'a,const N: usize>(&self, l:&VectorView<'a,S,N>, r:&VectorView<'a,<S as BitsBitXor>::Bits,N>) -> OwnedVector<S,N>;
}
/// Apply a bitwise AND to each element of a Vector
pub trait SimdBitAndVector<S>
    where S: BitsBitAnd {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Bitwise AND
    /// * `r` - Right hand side of the Bitwise AND
    unsafe fn bitand_vector<'a,const N: usize>(&self, l:&VectorView<'a,S,N>, r:&VectorView<'a,<S as BitsBitAnd>::Bits,N>) -> OwnedVector<S,N>;
}
/// Apply a bitwise OR to each element of a Vector
pub trait SimdBitOrVector<S>
    where S: BitsBitOr {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Bitwise OR
    /// * `r` - Right hand side of the Bitwise OR
    unsafe fn bitor_vector<'a,const N: usize>(&self, l:&VectorView<'a,S,N>, r:&VectorView<'a,<S as BitsBitOr>::Bits,N>) -> OwnedVector<S,N>;
}
/// Apply a bitwise Bit Not to each element of a Vector
pub trait SimdBitNotVector<S> {
    ///
    /// # Arguments
    /// * `v` - Vector to apply Bit Not to
    unsafe fn bitnot_vector<'a,const N: usize>(&self, v:&VectorView<'a,S,N>) -> OwnedVector<S,N>;
}
/// Apply a bitwise Left Shift to each element of a Vector
pub trait SimdShlVector<S> {
    ///
    /// # Arguments
    /// * `v` - Vector to apply Left Shift to
    /// * `w` - Number of bits to shift by
    unsafe fn shl_vector<'a,const N: usize>(&self, v:&VectorView<'a,S,N>, w:usize) -> OwnedVector<S,N>;
}
/// Apply a bitwise Right Shift to each element of a Vector
pub trait SimdShrVector<S> {
    ///
    /// # Arguments
    /// * `v` - Vector to apply Right Shift to
    /// * `w` - Number of bits to shift by
    unsafe fn shr_vector<'a,const N: usize>(&self, v:&VectorView<'a,S,N>, w:usize) -> OwnedVector<S,N>;
}
/// Apply a bitwise Add Assign to each element of a Matrix
pub trait SimdAddAssignMatrix<SL,SR> {
    ///
    /// # Arguments
    /// * `l` - Left hand side of the Add Assign
    /// * `r` - Right hand side of the Add Assign
    unsafe fn add_assign_matrix<'a,const N: usize,const M: usize>(&self, l:&mut MatrixMut<'a,SL,N,M>, r:& MatrixView<'a,SR,N,M>);
}
/// Update each element of the Matrix with the result of multiplying it by a scalar value
pub trait SimdScalarMulAssignMatrix<SL,SR> {
    ///
    /// # Arguments
    /// * `l` - Scalar value to multiply each element of the Matrix by
    /// * `r` - Matrix to multiply each element of by the scalar value
    unsafe fn scalar_mul_assign_matrix<'a,const N: usize,const M: usize>(&self, l: SL, r:&mut MatrixMut<'a,SR,N,M>);
}
/// Writes the result of multiplying each element of the array by a scalar value to the argument `acc`
pub trait SimdScalarMulMatrix<SL,SR> {
    type OutputScalar: Default + Copy;
    ///
    /// # Arguments
    /// * `l` - Scalar value to multiply each element of the Matrix by
    /// * `r` - Matrix to multiply each element of by the scalar value
    /// * `acc` - Matrix to write the result of the multiplication to
    unsafe fn scalar_mul_matrix<'a,const N: usize,const M: usize>(&self, l: SL, r:&MatrixView<'a,SR,N,M>, acc:&'a mut MatrixMut<'a,Self::OutputScalar,N,M>);
}
/// Converting the data types of each element in a matrix
pub trait SimdConvertMatrix<SS,SD> {
    ///
    /// # Arguments
    /// * `s` - Matrix to convert the data types of each element of
    unsafe fn convert_matrix<'a,const N: usize,const M: usize>(&self, s:&MatrixView<'a,SS,N,M>, acc:&'a mut MatrixMut<'a,SD,N,M>);
}
/// Calculating the dot product of two vectors
pub trait SimdDot<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    unsafe fn dot<'a,const N: usize>(&self,l:&VectorView<'a,SL,N>,r:&VectorView<'a,SR,N>) -> SO;
}
/// Calculate the cross product of two vectors
pub trait SimdOuterProduct<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `o` - Output matrix
    unsafe fn outer_product<'a,const N: usize,const M: usize>(&self,l:&VectorView<'a,SL,N>,
                                                       r:&VectorView<'a,SR,M>,
                                                       o:&mut OwnedMatrix<SO,N,M>);
}
/// Vector * Matrix Product
pub trait SimdVMat<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `o` - Output vector
    unsafe fn vmat<'a,const M: usize,const K: usize>(&self,
                                              l:&VectorView<'a,SL,K>,
                                              r:&ColumnMajorMatrix<'a,SR,K,M>,
                                              o:&mut OwnedVector<SO,M>);
}
/// Matrix * Vector Product
pub trait SimdMatVec<SL,SR,SO> {
    ///
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `o` - Output vector
    unsafe fn matvec<'a,const N: usize,const K: usize>(&self,
                                                l:&MatrixView<'a,SL,N,K>,
                                                r:&VectorView<'a,SR,K>,
                                                o:&mut OwnedVector<SO,N>);
}
/// Matrix * Matrix Product
pub trait SimdMatMul<SL,SR,SO> {
    /// Matrix * Matrix Product
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `o` - Output matrix
    unsafe fn matmul<'a,const N: usize,const M: usize,const K: usize>(&self,
                                                               l:&MatrixView<'a,SL,N,K>,
                                                               r:&ColumnMajorMatrix<'a,SR,K,M>,
                                                               o:&mut MatrixMut<'a,SO,N,M>);
    /// Product of Submatrices by Tiles
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `i` - Tile index
    /// * `j` - Tile index
    /// * `acc` - Output matrix
    unsafe fn matmul_tile<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&MatrixView<'a,SL,N,K>,
        r:&ColumnMajorMatrix<'a,SR,K,M>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );
    /// Partial results of the matrix product at the tail of the rows
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `i` - Tile index
    /// * `j` - Tile index
    /// * `acc` - Output matrix
    unsafe fn matmul_tile_tail_rows<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&MatrixView<'a,SL,N,K>,
        r:&ColumnMajorMatrix<'a,SR,K,M>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );

    /// Partial results of the matrix product at the tail of the cols
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `i` - Tile index
    /// * `j` - Tile index
    /// * `acc` - Output matrix
    unsafe fn matmul_tile_tail_cols<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&MatrixView<'a,SL,N,K>,
        r:&ColumnMajorMatrix<'a,SR,K,M>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );
    /// Partial results of the matrix product at the tail of the rows and cols
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `i` - Tile index
    /// * `j` - Tile index
    /// * `acc` - Output matrix
    unsafe fn matmul_tile_tail_rows_cols<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&MatrixView<'a,SL,N,K>,
        r:&ColumnMajorMatrix<'a,SR,K,M>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );
}
/// Sums each element in the SIMD register as a single scalar value and returns the result
pub trait SimdHSum<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `v` - SIMD register
    fn hsum(&self,v:Self::Reg) -> S;
}
/// Max each element in the SIMD register as a single scalar value and returns the result
pub trait SimdHMax<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `v` - SIMD register
    fn hmax(&self,v:Self::Reg) -> S;
}
/// Min each element in the SIMD register as a single scalar value and returns the result
pub trait SimdHMin<S>: SimdReg<S> {
    ///
    /// # Arguments
    /// * `v` - SIMD register
    fn hmin(&self,v:Self::Reg) -> S;
}
/// Transpose the elements in a register within a group of registers and return them
pub trait SimdTranspose<S,const ROWS: usize> where Self: SimdReg<S> {
    /// # Arguments
    /// * `v` - SIMD registers
    fn transpose<'a,const N: usize,const M: usize>(v:[Self::Reg; ROWS]) -> [Self::Reg; ROWS];
}
/// Define the number of lanes in SIMD for each type
pub trait SimdLanes<S> {
    const LANES: usize;
}
/// For each type, this defines the value of `Rows` when performing parallel computations by rows or columns.
pub trait SimdRows<S> {
    const ROWS: usize;
}
/// For each type, this defines the value of `Cols` when performing parallel computations by rows or columns.
pub trait SimdCols<S> {
    const COLS: usize;
}
/// A trait that defines the SIMD register type for each type
pub trait SimdReg<S> {
    /// The type of the SIMD register
    type Reg: Copy;
    /// The type of the SIMD mask
    type Mask: Copy;
    /// The type of the bit sequence corresponding to `S`The type of the bit sequence corresponding to `S`
    type Bits: Copy;
    /// Register type representing the shift width
    type ShiftWidth: Copy;
}
/// Load a value from memory into a SIMD register
pub trait SimdLoad<S>: SimdReg<S> {
    /// # Arguments
    /// * `ptr`: Pointer to the memory location to load from
    unsafe fn load(&self,ptr: *const S) -> Self::Reg;
}
/// Store a value from a SIMD register into memory
pub trait SimdStore<S>: SimdReg<S> {
    /// # Arguments
    /// * `ptr`: Pointer to the memory location to store to
    /// * `reg`: Value to store
    unsafe fn store(&self, ptr: *mut S, reg: Self::Reg);
}
/// Load a value from memory into a multi-element SIMD register
pub trait SimdLoadSeq<S,R>
    where R: Copy {
    /// # Arguments
    /// * `ptr`: Pointer to the memory location to load from
    unsafe fn load_seq(&self, ptr: *const S) -> R;
}
/// Storing a multi-element register in memory
pub trait SimdStoreSeq<S,R>
    where R: Copy {
    /// # Arguments
    /// * `ptr` - Pointer to the memory location to store to
    /// * `reg` - Value to store
    unsafe fn store_seq(&self, ptr: *mut S, reg: R);
}
/// Generating Mask Registers
pub trait SimdMask<S>: SimdReg<S> {
    fn cmp_gt(&self,a:Self::Reg,b:Self::Reg) -> Self::Mask;
    fn cmp_eq(&self,a:Self::Reg,b:Self::Reg) -> Self::Mask;
    fn select(&self,mask:Self::Mask,a:Self::Reg,b:Self::Reg) -> Self::Reg;
    fn mask_zero(&self,m:Self::Mask,a:Self::Reg) -> Self::Reg;
    fn tail_mask(&self,index: usize, total: usize) -> Self::Mask;
}
/// Reinterpreting the Bit Sequence of SIMD Registers
pub trait SimdReinterpret<SS,SD>: SimdReg<SS> + SimdReg<SD> {
    /// # Arguments
    /// * `reg` - SIMD register to reinterpret
    fn reinterpret(&self,reg:<Self as SimdReg<SS>>::Reg) -> <Self as SimdReg<SD>>::Reg;
}
/// Upcasting the Type of a SIMD Register
pub trait SimdPromote<SS,SD>: SimdReg<SS> +
                              SimdReg<SD> +
                              SimdAdd<SD,SD,SD> where Self: Sized {
    type Output: FoldRegs<SD,Self>;
    /// # Arguments
    /// * `reg` - SIMD register to promote
    fn promotion(&self, reg:<Self as SimdReg<SS>>::Reg) -> Self::Output;
}
/// Downcasting the Type of a SIMD Register
pub trait SimdDemote<SS,SD>: SimdReg<SS> +
                             SimdReg<SD> +
                             SimdAdd<SS,SS,SS> where Self: Sized {
    type Input: FoldRegs<SS,Self>;
    /// # Arguments
    /// * `reg` - SIMD register to demote
    fn demotion(&self, reg:Self::Input) -> <Self as SimdReg<SD>>::Reg;
}
/// Converting the Type of SIMD Registers
pub trait SimdConvert<SS,SD>: SimdReg<SS> +
                              SimdReg<SD> +
                              SimdAdd<SD,SD,SD> where Self: Sized {
    type Output: FoldRegs<SD,Self>;
    /// # Arguments
    /// * `reg` - SIMD register to convert
    fn convert(&self,reg:<Self as SimdReg<SS>>::Reg) -> Self::Output;
}
/// Upcast the registers for each element of a vector
pub trait SimdPromoteVector<SS,SD> {
    /// # Arguments
    /// * `s` - SIMD vector to upcast
    unsafe fn promotion_vector<'a,const N: usize>(&self, s:&VectorView<'a,SS,N>) -> OwnedVector<SD,N>;
}
/// Downcast the registers for each element of a vector
pub trait SimdDemoteVector<SS,SD> {
    /// # Arguments
    /// * `s` - SIMD vector to downcast
    unsafe fn demotion_vector<'a,const N: usize>(&self, s:&VectorView<'a,SS,N>) -> OwnedVector<SD,N>;
}
/// Converting type the registers for each element of a vector
pub trait SimdConvertVector<SS,SD> {
    /// # Arguments
    /// * `s` - SIMD vector to convert
    unsafe fn convert_vector<'a,const N: usize>(&self, s:&VectorView<'a,SS,N>) -> OwnedVector<SD,N>;
}
/// Returns the SIMD registers with their signs inverted
pub trait SimdNeg<S>: SimdReg<S> {
    /// # Arguments
    /// * `v` - SIMD register
    fn neg(&self,v:Self::Reg) -> Self::Reg;
}
/// Returns the result of applying multiplication and addition simultaneously to the SIMD registers
pub trait SimdMulAdd<SL,SR,SO>:
    SimdReg<SL> +
    SimdReg<SR> +
    SimdReg<SO> + SimdMul<SL,SR,SO>
    where <Self as SimdReg<SO>>::Reg: Copy {
    /// Returns an accumulator initialized to zero
    fn zero_acc(&self) -> <Self as SimdMul<SL,SR,SO>>::Output;
    /// Returns the result of applying multiplication and addition simultaneously to the SIMD registers
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `acc` - Accumulator
    fn mul_add(&self,l:<Self as SimdReg<SL>>::Reg,
               r:<Self as SimdReg<SR>>::Reg,
               acc:<Self as SimdMul<SL,SR,SO>>::Output) -> <Self as SimdMul<SL,SR,SO>>::Output;
}
/// Returns a partial dot product using SIMD registers
pub trait SimdPartialDot<SL,SR,SO>: SimdReg<SL> +
                                    SimdReg<SR> +
                                    SimdReg<SO> +
                                    SimdAdd<SO,SO,SO> + Backend + Sized {
    type Output: FoldRegs<SO,Self>;
    /// Returns an accumulator initialized to zero
    fn zero_acc(&self) -> Self::Output;
    /// Returns a partial dot product using SIMD registers
    /// # Arguments
    /// * `l` - Left-hand side
    /// * `r` - Right-hand side
    /// * `acc` - Accumulator
    fn partial_dot(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg,acc:Self::Output) -> Self::Output;
}
/// Returns a zero-initialized SIMD register
pub trait SimdZero<S>: SimdReg<S> {
    fn zero() -> <Self as SimdReg<S>>::Reg;
}
/// A marker trait that indicates you are implementing backend functionality yourself
pub trait NativeBackend: Backend {}
/// A feature that converts and returns a backend-independent type into a type bound to a specific backend
pub trait BindBackend<'a>: Sized {
    /// Backend-bound types
    type Output<BE: Backend>: 'a;
    /// Binds a backend-independent type to a specific backend
    fn bind<BE: Backend>(self) -> Result<Self::Output<BE>,InstantiationError>;

    /// Binds a backend-independent type to the auto-selected backend
    fn bind_auto(self) -> Result<Self::Output<AutoSelect>,InstantiationError> {
        self.bind()
    }
}
/// Trait Implemented when multiplication is supported
pub trait SupportMul<K> {}
/// A trait that defines the calculation of Dot Product
pub trait Dot<R,O> {
    /// Returns the dot product of two vectors
    /// # Arguments
    /// * `r` - Right-hand side
    fn dot(&self,r:R) -> O;
}
/// A trait that defines the calculation of matrix multiplication
pub trait Product<T,O> {
    fn product(&self, r: T) -> O;
}
/// A characteristic that defines a horizontal sum
pub trait HSum<S> {
    fn hsum(&self) -> S;
}
/// A characteristic that defines a horizontal max
pub trait HMax<S> {
    fn hmax(&self) -> S;
}
/// A characteristic that defines a horizontal min
pub trait HMin<S> {
    fn hmin(&self) -> S;
}
/// A trait that transposes a matrix
pub trait Transpose<T,const N: usize, const M: usize> where Self: Dims<N,M> {
    type Output: Dims<M,N>;
    fn transpose(self) -> Self::Output;
}
/// A trait that converts a matrix to column major order
pub trait ToColumnMajor<T,const N: usize, const M: usize> where Self: Dims<N,M> {
    type Output: Dims<N,M>;
    fn to_column_major(self) -> Self::Output;
}
/// A trait that defines the dimension of a matrix
pub trait Dims<const N: usize,const M: usize> {
}
/// A trait that treats a type as a bit string and performs a bitwise AND operation
pub trait BitsBitAnd {
    /// Types for Representing Bits
    type Bits;

    /// # Arguments
    /// * `m` - Bit mask
    fn bits_bitand(self,m:Self::Bits) -> Self;
}
/// A trait that treats a type as a bit string and performs a bitwise OR operation
pub trait BitsBitOr {
    /// Types for Representing Bits
    type Bits;

    /// # Arguments
    /// * `m` - Bit mask
    fn bits_bitor(self,m:Self::Bits) -> Self;
}
/// A trait that treats a type as a bit string and performs a bitwise XOR operation
pub trait BitsBitXor {
    /// Types for Representing Bits
    type Bits;

    /// # Arguments
    /// * `m` - Bit mask
    fn bits_bitxor(self,m:Self::Bits) -> Self;
}
/// A trait that defines an operation to invert a bit sequence bit by bit
pub trait BitsBitNot {

    fn bits_bitnot(self) -> Self;
}
/// A trait that defines an operation that treats a type as a bit sequence and shifts it to the left
pub trait BitsShl {
    /// # Arguments
    /// * `w` - Shift width
    fn bits_shl(self,w:usize) -> Self;
}
/// A trait that defines an operation that treats a type as a bit sequence and shifts it to the right
pub trait BitsShr {
    /// # Arguments
    /// * `w` - Shift width
    fn bits_shr(self,w:usize) -> Self;
}
/// A trait that defines operations to force a type cast using `as`
pub trait Assume<T> {
    fn assume(self) -> T;
}
/// A trait that defines an operation to fold multiple registers into a single register
pub trait FoldRegs<S,BE: SimdReg<S> + SimdAdd<S,S,S>>
    where <BE as SimdReg<S>>::Reg: Copy {
    /// # Arguments
    /// * `backend` - Backend to use for folding
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg;
}
/// A trait that defines type upcasts
pub trait Promote<SD> {
    fn promotion(self) -> SD;
}
/// A trait that defines type downcasts
pub trait Demote<SD> {
    fn demotion(self) -> SD;
}