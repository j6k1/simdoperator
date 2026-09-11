#[macro_export]
macro_rules! matmul_tile {
    ($func_name:ident,$ROWS:expr,$COLS:expr,$SL:ty,$SR:ty,$SO:ty) => {
        fn $func_name <'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
            &self,
            l:&Matrix<'a,$SL,N,K,Self::Backend>,
            r:&ColumnMajorMatrix<'a,$SR,K,M,Self::Backend>,
            i:usize,
            j:usize,
            acc:&mut MatrixMut<'a,$SO,N,M>
        ) {
            let mut acc_tile = [[<Self as SimdZero<$SO>>::zero();COLS];ROWS];

            unsafe {
                for k in (0..K).step_by(<Self as SimdLanes<$SR>>::LANES) {
                    let mut rr: [<Self as SimdReg<$SL>>::Reg; ROWS] = std::mem::MaybeUninit::uninit().assume_init();
                    for r in 0..$ROWS {
                        rr[r] = <Self as SimdLoad<$SL>>::load(self,l.row(i + r).as_ref().as_ptr().add(k));
                    }

                    let mut cr:[<Self as SimdReg<$SR>>::Reg; COLS] = std::mem::MaybeUninit::uninit().assume_init();

                    for c in 0..$COLS {
                        cr[c] = <Self as SimdLoad<$SR>>::load(self,r.col(j + c).as_ref().as_ptr().add(k));
                    }

                    for r in 0..$ROWS {
                        for c in 0..$COLS {
                            acc_tile[r][c] = <Self as SimdPartialDot<$SL,$SR,$SO>>::partial_dot(self,rr[r],cr[c],acc_tile[r][c]);
                        }
                    }
                }

                for r in 0..$ROWS {
                    for c in 0..$COLS {
                        acc[(i+r,j+c)] = <Self as SimdHSum<$SO>>::hsum(self,acc_tile[r][c]);
                    }
                }
            }
        }
    };
}
#[macro_export]
macro_rules! derive_matmul {
    ($BE:ty,$SL:ty,$SR:ty,$SO:ty) => {
        impl SimdMatMul<$SL,$SR,$SO> for $BE
            where Self: SimdPartialDot<$SL,$SR,$SO> +
                        SimdRows<$SL> + SimdZero<$SL> + SimdReg<$SL> + SimdLoad<$SL> +
                        SimdCols<$SR> + SimdZero<$SR> + SimdReg<$SR> + SimdLoad<$SR> +
                        SimdZero<$SO> + SimdHSum<$SO> +
                        SimdLanes<$SR>,
                        <Self as SimdReg<$SL>>::Reg: Clone + Copy,
                        <Self as SimdReg<$SR>>::Reg: Clone + Copy,
                        <Self as SimdReg<$SO>>::Reg: Clone + Copy {
            type Backend = $BE;

            fn matmul<'a, const N: usize, const M: usize, const K: usize>(&self, l: &Matrix<'a, $SL, N, K, Self::Backend>, r: &ColumnMajorMatrix<'a, $SR, K, M, Self::Backend>, acc: &mut MatrixMut<'a,$SO, N, M>) {
                for i in (0..N).step_by(<Self as SimdRows<$SL>>::ROWS) {
                    for j in (0..M).step_by(<Self as SimdCols<$SR>>::COLS) {
                        self.matmul_tile::<N,M,K,{ <Self as SimdRows<$SL>>::ROWS }, { <Self as SimdCols<$SR>>::COLS }>(
                            l, r, i, j, acc
                        );
                    }
                }

                for i in (0..N).step_by(<Self as SimdRows<$SL>>::ROWS) {
                    self.matmul_tile_tail_rows::<N,M,K,{ <Self as SimdRows<$SL>>::ROWS }, { <Self as SimdCols<$SR>>::COLS }>(
                        l, r, i, M / <Self as SimdCols<$SR>>::COLS * <Self as SimdCols<$SR>>::COLS,
                        acc
                    );
                }

                for j in (0..M).step_by(<Self as SimdCols<$SR>>::COLS) {
                    self.matmul_tile_tail_cols::<N,M,K,{ <Self as SimdRows<$SL>>::ROWS },{ <Self as SimdCols<$SR>>::COLS }>(
                        l, r, N / <Self as SimdRows<$SL>>::ROWS * <Self as SimdRows<$SL>>::ROWS, j,
                        acc
                    );
                }

                self.matmul_tile_tail_rows_cols::<N,M,K,{ <Self as SimdRows<$SL>>::ROWS }, { <Self as SimdCols<$SR>>::COLS }>(
                    l,r,N / <Self as SimdRows<$SL>>::ROWS * <Self as SimdRows<$SL>>::ROWS,
                    M / <Self as SimdCols<$SR>>::COLS * <Self as SimdCols<$SR>>::COLS,
                    acc
                )
            }

            matmul_tile! (matmul_tile,<Self as SimdRows<$SL>>::ROWS,<Self as SimdCols<$SR>>::COLS,$SL,$SR,$SO);
            matmul_tile! (matmul_tile_tail_rows, N % <Self as SimdRows<$SL>>::ROWS,<Self as SimdCols<$SR>>::COLS,
                $SL,$SR,$SO
            );
            matmul_tile! (matmul_tile_tail_cols, <Self as SimdRows<$SL>>::ROWS,
                M % <Self as SimdCols<$SR>>::COLS,$SL,$SR,$SO
            );
            matmul_tile! (matmul_tile_tail_rows_cols,
                N % <Self as SimdRows<$SL>>::ROWS,
                M % <Self as SimdCols<$SR>>::COLS,$SL,$SR,$SO
            );
        }
    };
}
