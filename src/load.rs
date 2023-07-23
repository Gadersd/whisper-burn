

use burn::{
    module::Module, 
    nn::{self, conv::{Conv1d, Conv1dConfig, Conv1dPaddingConfig, Conv1dRecord}}, 
    tensor::{
        backend::Backend,
        activation::relu, 
        Tensor,
        Int, 
        Bool, 
    },
};

use crate::model::*;

