use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub enum CacheState<T> {
    Value(T),
    Empty,
}

/// A cache for a tensor.
pub struct TensorCache<B: Backend, const D: usize> {
    pub state: CacheState<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> TensorCache<B, D> {
    /// Creates a new empty cache.
    ///
    /// # Returns
    ///
    /// The empty cache.
    pub fn empty() -> Self {
        Self {
            state: CacheState::Empty,
        }
    }
}
