use burn::tensor::{
    activation::relu, backend::Backend, BasicOps, Bool, Element, Float, Int, Numeric, Tensor,
    TensorKind,
};

use num_traits::ToPrimitive;

pub fn tensor_max_scalar<B: Backend, const D: usize>(x: Tensor<B, D>, max: f64) -> Tensor<B, D> {
    relu(x.sub_scalar(max)).add_scalar(max)
}

pub fn tensor_min_scalar<B: Backend, const D: usize>(x: Tensor<B, D>, min: f64) -> Tensor<B, D> {
    -tensor_max_scalar(-x, -min)
}

pub fn tensor_max<B: Backend, const D: usize>(x: Tensor<B, D>, max: Tensor<B, D>) -> Tensor<B, D> {
    relu(x - max.clone()) + max
}

pub fn tensor_min<B: Backend, const D: usize>(x: Tensor<B, D>, min: Tensor<B, D>) -> Tensor<B, D> {
    -tensor_max(-x, -min)
}

pub fn tensor_log10<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let ln10 = (10.0f64).ln();
    x.log() / ln10
}

pub fn tensor_max_element<B: Backend, const D: usize>(x: Tensor<B, D>) -> f64 {
    let flat: Tensor<B, 1> = x.flatten(0, D - 1);
    let max_index = flat.clone().argmax(0);

    flat.select(0, max_index).into_scalar().to_f64().unwrap()
}

pub fn all_zeros<B: Backend, const D: usize>(x: Tensor<B, D>) -> bool {
    x.powf(2.0).sum().into_scalar().to_f64().unwrap() == 0.0
}

pub fn max_dim<B: Backend>(x: Tensor<B, 2>, dim: usize) -> Tensor<B, 2> {
    let indices = x.clone().argmax(dim).flatten(0, 1);
    x.select(dim, indices)
}

pub fn _10pow<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let log10 = (10.0f64).ln();
    (x * log10).exp()
}

pub fn to_float<B: Backend, const D: usize>(x: Tensor<B, D, Int>) -> Tensor<B, D, Float> {
    let device = x.device();
    Tensor::from_data(x.into_data().convert()).to_device(&device)
}

pub fn to_float_bool<B: Backend, const D: usize>(x: Tensor<B, D, Bool>) -> Tensor<B, D, Float> {
    let device = x.device();
    Tensor::from_data(x.into_data().convert()).to_device(&device)
}

pub fn reverse<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B> + Numeric<B>>(
    x: Tensor<B, D, K>,
    dim: usize,
) -> Tensor<B, D, K>
where
    <K as BasicOps<B>>::Elem: Element,
{
    let len = x.dims()[dim];
    let indices = -Tensor::arange_device(0..len, &x.device()) + (len - 1) as i64;
    x.select(dim, indices)
}
