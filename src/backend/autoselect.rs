//! Automatically select the best backend for the current CPU.

use crate::backend::avx2::Avx2;
use crate::backend::common::Backend;
use crate::error::InstantiationError;

/// The selected backend.
#[derive(Clone)]
pub enum SelectedBackend {
    Avx2(Avx2)
}
/// The auto-selected backend.
#[derive(Clone)]
pub struct AutoSelect {
    /// The selected backend.
    pub(crate) selected: SelectedBackend
}
impl Backend for AutoSelect {
    fn new() -> Result<Self, InstantiationError> {
        if is_x86_feature_detected!("avx2") {
            let backend = Avx2::new()?;

            Ok(AutoSelect {
                selected: SelectedBackend::Avx2(backend)
            })
        } else {
            Err(InstantiationError::NotSupportingError)
        }
    }
}
impl AutoSelect {
    pub fn backend(&self) -> SelectedBackend {
        self.selected.clone()
    }
}
#[cfg(test)]
impl From<Avx2> for AutoSelect {
    fn from(value: Avx2) -> Self {
        AutoSelect {
            selected: SelectedBackend::Avx2(value)
        }
    }
}