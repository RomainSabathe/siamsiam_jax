import jax.numpy as jnp


def imagenet_preprocessing(x: jnp.array) -> jnp.array:
    mean = jnp.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3])
    std = jnp.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3])

    x = jnp.float32(x)
    x = x / 255.0
    x = (x - mean) / std

    return x
