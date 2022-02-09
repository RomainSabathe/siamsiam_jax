import jax
import haiku as hk
import jax.numpy as jnp


class Projector(hk.Module):
    def __init__(self, nb_dims: int, nb_layers: int, name: str = None):
        super().__init__(name=name)
        self.nb_dims = nb_dims
        self.nb_layers = nb_layers

        self.bn_config = {}
        self.bn_config.setdefault("create_scale", True),
        self.bn_config.setdefault("create_offset", True),
        self.bn_config.setdefault("decay_rate", 0.9),
        self.bn_config.setdefault("eps", 1e-5),
        self.bn_config.setdefault("scale_init", jnp.ones),
        self.bn_config.setdefault("offset_init", jnp.zeros),

    def __call__(self, x: jnp.array, is_training: bool) -> jnp.array:
        for i in range(self.nb_layers):
            bn_config = self.bn_config
            if i == self.nb_layers - 1:
                # For the final layer, we remove the affine transforms of batch norm,
                # as indicated in SiamSiam paper.
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
                bn_config["scale_init"] = None
                bn_config["offset_init"] = None

            x = hk.Linear(output_size=self.nb_dims, with_bias=False)(x)
            x = hk.BatchNorm(**bn_config)(x, is_training)

            # We skip the ReLU for the final layer.
            if i < self.nb_layers - 1:
                x = jax.nn.relu(x)

        # Also returning the std of the output. This is used as monitoring metric.
        # If the value goes to 0, then we face a degenerate solution.
        monitor = x / jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        # We return statistics on the batch.
        monitor = jnp.std(monitor, axis=0)

        return x, monitor


class Predictor(hk.Module):
    def __init__(self, hidden_dims: int, name: str = None):
        super().__init__(name=name)
        self.hidden_dims = hidden_dims

        self.bn_config = {}
        self.bn_config.setdefault("create_scale", True),
        self.bn_config.setdefault("create_offset", True),
        self.bn_config.setdefault("decay_rate", 0.9),
        self.bn_config.setdefault("eps", 1e-5),
        self.bn_config.setdefault("scale_init", jnp.ones),
        self.bn_config.setdefault("offset_init", jnp.zeros),

    def __call__(self, x: jnp.array, is_training: bool) -> jnp.array:
        output_dims = x.shape[-1]

        x = hk.Linear(output_size=self.hidden_dims, with_bias=False)(x)
        x = hk.BatchNorm(**self.bn_config)(x, is_training)
        x = jax.nn.relu(x)
        x = hk.Linear(output_size=output_dims, with_bias=True)(x)

        # Also returning the std of the output. This is used as monitoring metric.
        # If the value goes to 0, then we face a degenerate solution.
        monitor = x / jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        # We return statistics on the batch.
        monitor = jnp.std(monitor, axis=0)

        return x, monitor
