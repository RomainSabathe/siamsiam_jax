import os
from functools import partial
from typing import NamedTuple
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import jax
import optax
import haiku as hk
import jax.numpy as jnp
from tqdm import tqdm

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
gpus = tf.config.get_visible_devices("GPU")

from architectures import ResNet18Sim
from data.utils import Batch
from data.cifar10 import load_cifar10_dataset
from data.preprocessing import imagenet_preprocessing

logdir = os.getenv("LOGDIR") or "logs/baseline/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
global step
step = 0


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


def _forward(batch: Batch, is_training: bool) -> jnp.array:
    x = batch["image"]
    x = imagenet_preprocessing(x)
    network = ResNet18Sim()
    feature_maps = network(x, is_training=is_training)
    embeddings = jnp.mean(feature_maps, axis=(1, 2))
    logits = hk.Linear(output_size=10, with_bias=True)(embeddings)
    return logits


forward = hk.transform_with_state(_forward)
jit_forward_eval = jax.jit(partial(forward.apply, is_training=False))


def eval_step(train_state, dataset_iterator):
    preds_are_correct = []

    params, state, _ = train_state
    for batch in tqdm(dataset_iterator):
        logits, _ = jit_forward_eval(params, state, None, batch)
        preds = jnp.argmax(logits, axis=-1)
        preds_are_correct.append(preds == batch["label"])

    return jnp.mean(jnp.concatenate(preds_are_correct, axis=0))


def loss_fn(params, state, batch: Batch) -> jnp.array:
    logits, state = forward.apply(params, state, None, batch, is_training=True)
    labels_one_hot = jax.nn.one_hot(batch["label"], num_classes=10)
    loss = optax.softmax_cross_entropy(logits, labels_one_hot).mean()
    return loss, state


def train_step(train_state, batch: Batch):
    params, state, opt_state = train_state

    (loss, state), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
        params, state, batch
    )

    batch_size = batch["image"].shape[0]
    updates, new_opt_state = get_optimizer(batch_size).update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return TrainState(new_params, state, new_opt_state), loss


def get_optimizer(batch_size: int):
    base_learning_rate = 1e-2 / 128.0
    learning_rate = base_learning_rate * batch_size
    # schedule = optax.cosine_delay_schedule(init_value=learning_rate, decay_steps=1_000)
    return optax.adam(learning_rate=learning_rate)


def main():
    rng = jax.random.PRNGKey(42)
    batch_size = 128
    validate_every = int(352 * 128 / batch_size)

    params, state = forward.init(
        rng, {"image": jnp.ones((1, 32, 32, 3), dtype=jnp.uint8)}, is_training=True
    )
    opt_state = get_optimizer(batch_size).init(params)
    train_state = TrainState(params, state, opt_state)

    fast_train_step = jax.jit(train_step)

    global step
    for batch in tqdm(
        load_cifar10_dataset(
            split="train",
            shuffle=True,
            apply_augmentations=True,
            batch_size=batch_size,
            rng=rng,
        )
    ):
        train_state, loss = fast_train_step(train_state, batch)
        step += 1
        tf.summary.scalar(f"train_loss", data=float(loss), step=step)

        if step % validate_every == 0:
            accuracy = eval_step(
                train_state,
                load_cifar10_dataset(
                    split="val",
                    shuffle=False,
                    apply_augmentations=False,
                    batch_size=batch_size,
                    rng=rng,
                ),
            )
            tf.summary.scalar(f"val_accuracy", data=float(accuracy), step=step)


if __name__ == "__main__":
    main()
