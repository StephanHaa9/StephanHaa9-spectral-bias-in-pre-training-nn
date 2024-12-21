from flax import linen as nn
from typing import Tuple
import optax
import znnl as nl

class CNN_Architecture(nn.Module):
    """
    Simple CNN module with configurable parameters.
    """
    # Konfigurationsparameter als Felder
    features_conv_0: int = 32
    kernel_size_conv_0: Tuple[int, int] = (3, 3)
    max_pool_window_shape_0: Tuple[int, int] = (2, 2)
    max_pool_strides_0: Tuple[int, int] = (2, 2)

    features_conv_1: int = 32
    kernel_size_conv_1: Tuple[int, int] = (2, 2)
    max_pool_window_shape_1: Tuple[int, int] = (2, 2)
    max_pool_strides_1: Tuple[int, int] = (2, 2)

    width_dense_0: int = 128
    width_dense_1: int = 10

    @nn.compact
    def __call__(self, x):
        # Conv 0 -> ReLU -> MaxPool 
        x = nn.Conv(features=self.features_conv_0, kernel_size=self.kernel_size_conv_0)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=self.max_pool_window_shape_0, strides=self.max_pool_strides_0)

        # Conv 1 -> ReLU -> MaxPool
        x = nn.Conv(features=self.features_conv_1, kernel_size=self.kernel_size_conv_1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=self.max_pool_window_shape_1, strides=self.max_pool_strides_1)

        # Dense 0 -> ReLU -> Dense 1 (Output)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=self.width_dense_0)(x)
        x = nn.relu(x)
        x = nn.Dense(self.width_dense_1)(x)  # Output layer
        return x


def create_n_cnn_models(
        number_of_models: int,
        learning_rate_models: float,
        seed_models: int,
        input_shape: Tuple[int, int, int, int] = (1, 28, 28, 1),  # A tuple specifying the shape of the input
        features_conv_0: int = 32,
        kernel_size_conv_0: Tuple[int, int] = (3, 3),
        max_pool_window_shape_0: Tuple[int, int] = (2, 2),
        max_pool_strides_0: Tuple[int, int] = (2, 2),
        features_conv_1: int = 32,
        kernel_size_conv_1: Tuple[int, int] = (2, 2),
        max_pool_window_shape_1: Tuple[int, int] = (2, 2),
        max_pool_strides_1: Tuple[int, int] = (2, 2),
        width_dense_0: int = 128,
        width_dense_1: int = 10
    ) -> list:
    """
    Function to create a list of CNN models with specified parameters.
    """
    print('----------------CNN MODELS---------------------')
    print('PARAMS:')
    print('--------')
    print(f'number_of_models : {number_of_models}')
    print(f'learning_rate : {learning_rate_models}')
    print(f'seed_models : {seed_models}')
    print(f'Conv_0 : feature_maps:{features_conv_0}, kernel: {kernel_size_conv_0}')
    print(f'MaxPool_0 : window_shape:{max_pool_window_shape_0}, strides: {max_pool_strides_0}')
    print(f'Conv_1 : feature_maps:{features_conv_1}, kernel: {kernel_size_conv_1}')
    print(f'MaxPool_1 : window_shape:{max_pool_window_shape_1}, strides: {max_pool_strides_1}')
    print(f'Dense_0 : width:{width_dense_0}')
    print(f'Dense_1 (Output) : width:{width_dense_1}')

    optimizer = optax.adam(learning_rate=learning_rate_models)

    models = []
    for _ in range(number_of_models):
        model_ = nl.models.FlaxModel(
            flax_module=CNN_Architecture(
                features_conv_0=features_conv_0,
                kernel_size_conv_0=kernel_size_conv_0,
                max_pool_window_shape_0=max_pool_window_shape_0,
                max_pool_strides_0=max_pool_strides_0,
                features_conv_1=features_conv_1,
                kernel_size_conv_1=kernel_size_conv_1,
                max_pool_window_shape_1=max_pool_window_shape_1,
                max_pool_strides_1=max_pool_strides_1,
                width_dense_0=width_dense_0,
                width_dense_1=width_dense_1,
            ),
            optimizer=optimizer,
            seed=seed_models,
            input_shape=input_shape
        )
        models.append(model_)
    print(f'------------- {number_of_models} MODELS ARE CREATED -------------')

    return models
