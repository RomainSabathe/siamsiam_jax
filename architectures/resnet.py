from typing import Optional, Dict

import jax
import haiku as hk
import jax.numpy as jnp


class ResNet18(hk.Module):
    def __init__(self, num_classes: int, name: Optional[str] = None):
        super().__init__(name=name)

        self.num_classes = num_classes

        self.bn_config = {}
        self.bn_config.setdefault("create_scale", True),
        self.bn_config.setdefault("create_offset", True),
        self.bn_config.setdefault("decay_rate", 0.9),
        self.bn_config.setdefault("eps", 1e-5),
        self.bn_config.setdefault("scale_init", jnp.ones),
        self.bn_config.setdefault("offset_init", jnp.zeros),

    def __call__(self, x: jnp.array, is_training: bool) -> jnp.array:
        x = hk.Conv2D(output_channels=64, kernel_shape=7, stride=2)(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = jax.nn.relu(x)

        x = hk.MaxPool(window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")(
            x
        )

        x = ResBlock(
            output_channels=64, stride=1, bn_config=self.bn_config, name="block2"
        )(x, is_training)
        x = ResBlock(
            output_channels=128, stride=2, bn_config=self.bn_config, name="block3"
        )(x, is_training)
        x = ResBlock(
            output_channels=256, stride=2, bn_config=self.bn_config, name="block4"
        )(x, is_training)
        x = ResBlock(
            output_channels=512, stride=2, bn_config=self.bn_config, name="block5"
        )(x, is_training)

        x = jnp.mean(x, axis=(1, 2))
        return hk.Linear(
            output_size=self.num_classes, with_bias=True, w_init=jnp.zeros
        )(x)


class ResBlock(hk.Module):
    def __init__(
        self,
        output_channels: int,
        bn_config: Dict,
        stride: int = 1,
        name: Optional[str] = None,
    ):
        self.output_channels = output_channels
        self.stride = stride
        self.bn_config = bn_config
        super().__init__(name=name)

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        x_main = hk.Conv2D(
            output_channels=self.output_channels,
            kernel_shape=3,
            stride=self.stride,
        )(x)
        x_main = hk.BatchNorm(**self.bn_config)(x_main, is_training)
        x_main = jax.nn.relu(x_main)
        x_main = hk.Conv2D(output_channels=self.output_channels, kernel_shape=3)(x_main)
        x_main = hk.BatchNorm(**self.bn_config)(x_main, is_training)

        if x.shape[-1] != self.output_channels or self.stride > 1:
            x = hk.Conv2D(
                output_channels=self.output_channels, kernel_shape=1, stride=self.stride
            )(x)

        x = x_main + x
        x = jax.nn.relu(x)
        return x
