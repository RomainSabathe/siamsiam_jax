from typing import Optional

import jax
import haiku as hk
import jax.numpy as jnp


class ResNet18Sim(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

        self.bn_config = {}
        self.bn_config.setdefault("create_scale", True),
        self.bn_config.setdefault("create_offset", True),
        self.bn_config.setdefault("decay_rate", 0.9),
        self.bn_config.setdefault("eps", 1e-5),
        self.bn_config.setdefault("scale_init", jnp.ones),
        self.bn_config.setdefault("offset_init", jnp.zeros),

    def __call__(self, x: jnp.array, is_training: bool) -> jnp.array:
        x = hk.Conv2D(
            output_channels=64,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=False,
        )(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = jax.nn.relu(x)

        x = ResStack0(filters=64, nb_blocks=2, stride=1, bn_config=self.bn_config)(
            x, is_training
        )
        x = ResStack0(filters=128, nb_blocks=2, stride=2, bn_config=self.bn_config)(
            x, is_training
        )
        x = ResStack0(filters=256, nb_blocks=2, stride=2, bn_config=self.bn_config)(
            x, is_training
        )
        x = ResStack0(filters=512, nb_blocks=2, stride=2, bn_config=self.bn_config)(
            x, is_training
        )

        return x


class ResStack0(hk.Module):
    def __init__(self, filters, nb_blocks, stride, bn_config, name: str = None):
        super().__init__(name=name)

        self.filters = filters
        self.nb_blocks = nb_blocks
        self.stride = stride
        self.bn_config = bn_config

    def __call__(self, x: jnp.array, is_training: bool) -> jnp.array:
        x = ResBlock0(
            filters=self.filters,
            kernel_shape=3,
            stride=self.stride,
            bn_config=self.bn_config,
        )(x, is_training)

        for _ in range(self.nb_blocks - 1):
            x = ResBlock0(
                filters=self.filters,
                kernel_shape=3,
                stride=1,
                bn_config=self.bn_config,
                conv_shortcut=False,
            )(x, is_training)

        return x


class ResBlock0(hk.Module):
    def __init__(
        self,
        filters: int,
        kernel_shape,
        stride,
        bn_config,
        conv_shortcut=True,
        name: str = None,
    ):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.conv_shortcut = conv_shortcut
        self.bn_config = bn_config

    def __call__(self, x: jnp.array, is_training: bool) -> jnp.array:
        x_main = hk.Conv2D(
            output_channels=self.filters,
            kernel_shape=self.kernel_shape,
            stride=self.stride,
            padding="SAME",
            with_bias=False,
            # In the TF code, they use a LeCun Uniform initialization here.
        )(x)
        x_main = hk.BatchNorm(**self.bn_config)(x_main, is_training)
        x_main = jax.nn.relu(x_main)

        x_main = hk.Conv2D(
            output_channels=self.filters,
            kernel_shape=self.kernel_shape,
            stride=1,
            padding="SAME",
            with_bias=False,
            # In the TF code, they use a LeCun Uniform initialization here.
        )(x_main)
        x_main = hk.BatchNorm(**self.bn_config)(x_main, is_training)

        if self.conv_shortcut:
            x_shortcut = hk.Conv2D(
                output_channels=self.filters,
                kernel_shape=1,
                stride=self.stride,
                padding="SAME",
                with_bias=False,
            )(x)
            x_shortcut = hk.BatchNorm(**self.bn_config)(x_shortcut, is_training)
        else:
            x_shortcut = x

        return jax.nn.relu(x_main + x_shortcut)
