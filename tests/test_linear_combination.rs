//! LinearCombination trait implementation tests for i16 vectors with i8 coefficients.
//! 
//! Tests verify that linear combination operations correctly add signed vector values
//! to accumulator vectors using the Sign trait (SignInt<i8>).

use simdoperator::{AccVector, OwnedVector, VectorView};
use simdoperator::traits::{LinearCombination, Sign};
use simdoperator::backend::common::SignInt;

/// Test basic linear combination with plus and minus signs for i16 AccVector.
/// Size N=8 matches AVX2 LANES for i32/i16 operations.
#[test]
fn test_linear_combination_i16_acc_plus_minus() {
    const N: usize = 8;
    
    // Initialize accumulator with zeros via OwnedVector
    let data: [i16; N] = [0; N];
    let owned: OwnedVector<i16, N> = OwnedVector::from(Box::new(data));
    let mut acc: AccVector<i16, N> = AccVector::try_from(owned).unwrap();
    
    // Prepare input vectors and signs
    let v1_data: [i8; N] = [1, 2, 3, 4, 5, 6, 7, 8];
    let v1_view: VectorView<i8, N> = VectorView::try_from(&v1_data).unwrap();
    
    let v2_data: [i8; N] = [1, 1, 1, 1, 1, 1, 1, 1];
    let v2_view: VectorView<i8, N> = VectorView::try_from(&v2_data).unwrap();
    
    let lines: [(VectorView<i8, N>, SignInt<i8>); 2] = [
        (v1_view, <SignInt<i8> as Sign>::plus()),
        (v2_view, <SignInt<i8> as Sign>::minus())
    ];
    
    // Execute linear combination
    acc.linear_combination(&lines);
    
    // Expected result: [0, 1, 2, 3, 4, 5, 6, 7]
    let expected: [i16; N] = [0, 1, 2, 3, 4, 5, 6, 7];
    
    for i in 0..N {
        assert_eq!(acc[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test linear combination with size not divisible by AVX2 LANES.
/// Size N=10 tests the remainder handling path.
#[test]
fn test_linear_combination_i16_acc_non_multiple() {
    const N: usize = 10; // Not a multiple of LANES (8)
    
    let data: [i16; N] = [0; N];
    let owned: OwnedVector<i16, N> = OwnedVector::from(Box::new(data));
    let mut acc: AccVector<i16, N> = AccVector::try_from(owned).unwrap();
    
    let v1_data: [i8; N] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let v1_view: VectorView<i8, N> = VectorView::try_from(&v1_data).unwrap();
    
    let v2_data: [i8; N] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let v2_view: VectorView<i8, N> = VectorView::try_from(&v2_data).unwrap();
    
    let lines: [(VectorView<i8, N>, SignInt<i8>); 2] = [
        (v1_view, <SignInt<i8> as Sign>::plus()),
        (v2_view, <SignInt<i8> as Sign>::minus())
    ];
    
    acc.linear_combination(&lines);
    
    let expected: [i16; N] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    
    for i in 0..N {
        assert_eq!(acc[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test linear combination with non-zero initial accumulator values.
#[test]
fn test_linear_combination_i16_acc_initial_values() {
    const N: usize = 8;
    
    // Initialize accumulator with ones via OwnedVector
    let data: [i16; N] = [1; N];
    let owned: OwnedVector<i16, N> = OwnedVector::from(Box::new(data));
    let mut acc: AccVector<i16, N> = AccVector::try_from(owned).unwrap();
    
    let v1_data: [i8; N] = [2, 3, 4, 5, 6, 7, 8, 9];
    let v1_view: VectorView<i8, N> = VectorView::try_from(&v1_data).unwrap();
    
    let lines: [(VectorView<i8, N>, SignInt<i8>); 1] = [
        (v1_view, <SignInt<i8> as Sign>::plus())
    ];
    
    acc.linear_combination(&lines);
    
    // Expected result: [3, 4, 5, 6, 7, 8, 9, 10]
    let expected: [i16; N] = [3, 4, 5, 6, 7, 8, 9, 10];
    
    for i in 0..N {
        assert_eq!(acc[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test linear combination with only negative signs.
#[test]
fn test_linear_combination_i16_acc_negative_only() {
    const N: usize = 8;
    
    let data: [i16; N] = [0; N];
    let owned: OwnedVector<i16, N> = OwnedVector::from(Box::new(data));
    let mut acc: AccVector<i16, N> = AccVector::try_from(owned).unwrap();
    
    let v1_data: [i8; N] = [1, 2, 3, 4, 5, 6, 7, 8];
    let v1_view: VectorView<i8, N> = VectorView::try_from(&v1_data).unwrap();
    
    let lines: [(VectorView<i8, N>, SignInt<i8>); 1] = [
        (v1_view, <SignInt<i8> as Sign>::minus())
    ];
    
    acc.linear_combination(&lines);
    
    // Expected result: [-1, -2, -3, -4, -5, -6, -7, -8]
    let expected: [i16; N] = [-1, -2, -3, -4, -5, -6, -7, -8];
    
    for i in 0..N {
        assert_eq!(acc[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test linear combination with multiple lines including mixed signs.
#[test]
fn test_linear_combination_i16_acc_multiple_lines() {
    const N: usize = 8;
    
    let data: [i16; N] = [0; N];
    let owned: OwnedVector<i16, N> = OwnedVector::from(Box::new(data));
    let mut acc: AccVector<i16, N> = AccVector::try_from(owned).unwrap();
    
    let v1_data: [i8; N] = [5, 5, 5, 5, 5, 5, 5, 5];
    let v1_view: VectorView<i8, N> = VectorView::try_from(&v1_data).unwrap();
    
    let v2_data: [i8; N] = [3, 3, 3, 3, 3, 3, 3, 3];
    let v2_view: VectorView<i8, N> = VectorView::try_from(&v2_data).unwrap();
    
    let v3_data: [i8; N] = [1, 1, 1, 1, 1, 1, 1, 1];
    let v3_view: VectorView<i8, N> = VectorView::try_from(&v3_data).unwrap();
    
    let lines: [(VectorView<i8, N>, SignInt<i8>); 3] = [
        (v1_view, <SignInt<i8> as Sign>::plus()),   // +5
        (v2_view, <SignInt<i8> as Sign>::minus()),   // -3
        (v3_view, <SignInt<i8> as Sign>::plus())     // +1
    ];
    
    acc.linear_combination(&lines);
    
    // Expected result: [3, 3, 3, 3, 3, 3, 3, 3]
    let expected: [i16; N] = [3, 3, 3, 3, 3, 3, 3, 3];
    
    for i in 0..N {
        assert_eq!(acc[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test linear combination with size N=1 (smaller than LANES).
#[test]
fn test_linear_combination_i16_acc_small_size() {
    const N: usize = 1;
    
    let data: [i16; N] = [0; N];
    let owned: OwnedVector<i16, N> = OwnedVector::from(Box::new(data));
    let mut acc: AccVector<i16, N> = AccVector::try_from(owned).unwrap();
    
    let v1_data: [i8; N] = [42];
    let v1_view: VectorView<i8, N> = VectorView::try_from(&v1_data).unwrap();
    
    let lines: [(VectorView<i8, N>, SignInt<i8>); 1] = [
        (v1_view, <SignInt<i8> as Sign>::plus())
    ];
    
    acc.linear_combination(&lines);
    
    let expected: [i16; N] = [42];
    
    for i in 0..N {
        assert_eq!(acc[i], expected[i], "Mismatch at index {}", i);
    }
}

/// Test linear combination with size N=16 (multiple of LANES).
#[test]
fn test_linear_combination_i16_acc_large_size() {
    const N: usize = 16;
    
    let data: [i16; N] = [0; N];
    let owned: OwnedVector<i16, N> = OwnedVector::from(Box::new(data));
    let mut acc: AccVector<i16, N> = AccVector::try_from(owned).unwrap();
    
    let v1_data: [i8; N] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let v1_view: VectorView<i8, N> = VectorView::try_from(&v1_data).unwrap();
    
    let v2_data: [i8; N] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let v2_view: VectorView<i8, N> = VectorView::try_from(&v2_data).unwrap();
    
    let lines: [(VectorView<i8, N>, SignInt<i8>); 2] = [
        (v1_view, <SignInt<i8> as Sign>::plus()),
        (v2_view, <SignInt<i8> as Sign>::minus())
    ];
    
    acc.linear_combination(&lines);
    
    let expected: [i16; N] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    
    for i in 0..N {
        assert_eq!(acc[i], expected[i], "Mismatch at index {}", i);
    }
}
