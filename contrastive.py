import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import argparse
from pathlib import Path
from functools import partial
from datetime import datetime

import pickle
from typing import (
    Callable,
    Optional,
    Dict,
    NamedTuple,
    Tuple,
    Any,
    Union
)

import jax
import optax
import haiku as hk
from tqdm import tqdm
import tensorflow as tf
import jax.numpy as jnp
from jax.tree_util import PyTreeDef
from optax._src.alias import ScalarOrSchedule

tf.config.set_visible_devices([], "GPU")
gpus = tf.config.get_visible_devices("GPU")

from data.utils import Batch, Scalars
from data.cifar10 import load_cifar10_dataset
from data.contrastive import make_augmented_dataset
from data.preprocessing import imagenet_preprocessing
from architectures import ResNet18Sim, Projector, Predictor

print("jax version {}".format(jax.__version__))
from jax.lib import xla_bridge

print("jax backend {}".format(xla_bridge.get_backend().platform))

logdir = os.getenv("LOGDIR") or "logs/scalars/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
global step
step = 0

def save_pytree(data: PyTreeDef, path: Union[str, Path], overwrite: bool = False):
    suffix = ".pickle"

    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pytree(path: Union[str, Path]) -> PyTreeDef:
    suffix = ".pickle"

    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

p = argparse.ArgumentParser()
p.add_argument("--lr-ratio", type=float, default=3e-2 / 256)
p.add_argument("--pre-train-epochs", type=int, default=800)
p.add_argument("--random-seed", type=int, default=42)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--weight-decay", type=float, default=5.0e-4)
p.add_argument("--debug", action="store_true", default=False)
FLAGS = p.parse_args()


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


class Index(NamedTuple):
    zs: jnp.array
    ps: jnp.array
    labels: jnp.array


def _forward(batch: Batch, is_training: bool) -> Dict[str, jnp.array]:
    images = batch["image"]
    images = imagenet_preprocessing(images)

    backbone = ResNet18Sim()
    projector = Projector(nb_dims=2_048, nb_layers=2)
    predictor = Predictor(hidden_dims=512)

    feature_maps = backbone(images, is_training=is_training)
    h = jnp.mean(feature_maps, axis=(1, 2))
    z, std_projector = projector(h, is_training=is_training)
    p, std_predictor = predictor(z, is_training=is_training)

    return {
        "z": z,
        "p": p,
        "std_projector": std_projector.mean(),
        "std_predictor": std_predictor.mean(),
    }


forward = hk.transform_with_state(_forward)
fast_forward_eval = jax.jit(partial(forward.apply, is_training=False))


def cosine_distance_solo(lhs, rhs, eps=1e-8):
    lhs = lhs / (jnp.linalg.norm(lhs, ord=2, axis=-1, keepdims=True) + eps)
    rhs = rhs / (jnp.linalg.norm(rhs, ord=2, axis=-1, keepdims=True) + eps)

    return jnp.sum(lhs * rhs)


dist_fn = jax.vmap(
    jax.vmap(cosine_distance_solo, in_axes=(None, 0), out_axes=0),
    in_axes=(0, None),
    out_axes=0,
)
fast_dist_fn = jax.jit(dist_fn)


def loss_fn(
    params: hk.Params, state: hk.State, rng: jnp.ndarray, batch1: Batch, batch2: Batch
) -> jnp.float32:
    rng1, rng2 = jax.random.split(rng)

    out1, state = forward.apply(params, state, rng1, batch1, is_training=True)
    out2, state = forward.apply(params, state, rng2, batch2, is_training=True)

    p1, z1 = out1["p"], out1["z"]
    p2, z2 = out2["p"], out2["z"]

    z1 = jax.lax.stop_gradient(z1)
    z2 = jax.lax.stop_gradient(z2)

    extra_metrics = {
        "std_projector": (out1["std_projector"] * 0.5 + out2["std_projector"]),
        "std_predictor": (out1["std_predictor"] * 0.5 + out2["std_predictor"]),
    }

    # I do 1 - cosine_dist to mimick the behaviour of TF Similarity.
    loss = 1 - ((cosine_distance(p1, z2) / 2.0) + (cosine_distance(p2, z1) / 2.0))
    return loss, {
        "state": state,
        "extra_metrics": extra_metrics,
    }


def train_step(
    train_state: TrainState, rng: jnp.ndarray, batch1: Batch, batch2: Batch
) -> Tuple[TrainState, Scalars]:
    params, state, opt_state = train_state
    (loss, extras), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
        params, state, rng, batch1, batch2
    )
    new_state = extras["state"]
    extra_metrics = extras["extra_metrics"]

    batch_size = batch1["image"].shape[0]
    # Passing the `params` is necessary to apply weight decay.
    updates, new_opt_state = get_optimizer(batch_size).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return TrainState(new_params, new_state, new_opt_state), {
        "loss": loss,
        **extra_metrics,
    }


def cosine_distance(lhs: jnp.array, rhs: jnp.array, eps: float = 1e-8) -> jnp.array:
    # lhs and rhs are batched.
    lhs = lhs / (jnp.linalg.norm(lhs, ord=2, axis=-1, keepdims=True) + eps)
    rhs = rhs / (jnp.linalg.norm(rhs, ord=2, axis=-1, keepdims=True) + eps)

    return jnp.sum(lhs * rhs, axis=1).mean()


def get_optimizer(batch_size: int) -> optax.GradientTransformation:
    init_learning_rate = FLAGS.lr_ratio * batch_size
    PRE_TRAIN_EPOCHS = FLAGS.pre_train_epochs
    len_x_train = 45_000
    PRE_TRAIN_STEPS_PER_EPOCH = len_x_train // batch_size

    schedule = optax.cosine_decay_schedule(
        init_value=init_learning_rate,
        decay_steps=PRE_TRAIN_EPOCHS * PRE_TRAIN_STEPS_PER_EPOCH,
    )
    return sgdw(learning_rate=schedule, weight_decay=FLAGS.weight_decay, momentum=0.9)


def sgdw(
    learning_rate: ScalarOrSchedule,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
):
    return optax.chain(
        optax.sgd(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            accumulator_dtype=accumulator_dtype,
        ),
        optax.add_decayed_weights(weight_decay=-weight_decay, mask=weight_decay_mask),
    )


def build_index(params, state, rng, dataset):
    zs, ps, labels = [], [], []
    for batch in tqdm(dataset):
        out, _ = fast_forward_eval(params, state, rng, batch)
        zs.append(out["z"])
        ps.append(out["p"])
        labels.append(batch["label"])

        if FLAGS.debug:
            break

    return Index(
        jnp.concatenate(zs, 0), jnp.concatenate(ps, 0), jnp.concatenate(labels, 0)
    )


def evaluate(train_state, rng, dataset_factory) -> float:
    params, state, _ = train_state

    rng_dataset, rng = jax.random.split(rng, 2)  # Not stricly necessary
    index = build_index(params, state, rng, dataset_factory(split="index", rng=rng_dataset))
    query = build_index(params, state, rng, dataset_factory(split="query", rng=rng_dataset))

    dists = fast_dist_fn(query.zs, index.zs)
    matches = jnp.argmin(dists, axis=1)
    accuracy_projector = jnp.mean(query.labels == index.labels[matches])

    dists = fast_dist_fn(query.zs, index.zs)
    matches = jnp.argmin(dists, axis=1)
    accuracy_predictor = jnp.mean(query.labels == index.labels[matches])

    return {
        "accuracy_projector": accuracy_projector,
        "accuracy_predictor": accuracy_predictor,
    }


def main():
    rng = jax.random.PRNGKey(FLAGS.random_seed)

    img_size = 32
    batch_size = FLAGS.batch_size

    batch1 = {
        "image": jnp.array(
            jax.random.uniform(
                rng, [batch_size, img_size, img_size, 3], jnp.float32, 0, 255
            )
        )
    }

    # TODO: have a function that returns an initial state.
    # is_training=True is necessary to initialize the batch norm stats.
    params, state = forward.init(rng, batch1, is_training=True)
    opt_state = get_optimizer(batch_size).init(params)
    train_state = TrainState(params, state, opt_state)

    fast_train_step = jax.jit(train_step)

    global step
    for epoch in tqdm(range(FLAGS.pre_train_epochs if not FLAGS.debug else 1)):
        rng_dataset, rng = jax.random.split(rng, 2)
        for batch1, batch2 in tqdm(
            make_augmented_dataset(
                load_cifar10_dataset,
                rng=rng_dataset,
                split="train",
                batch_size=FLAGS.batch_size,
                shuffle=True,
                apply_augmentations=True,
            )
        ):
            step += 1

            # rng1, rng2, rng = jax.random.split(rng, 3)
            # batch1 = {
            #    "image": jnp.array(
            #        jax.random.uniform(
            #            rng1, [batch_size, img_size, img_size, 3], jnp.float32, 0, 255
            #        )
            #    )
            # }
            # batch2 = {
            #    "image": jnp.array(
            #        jax.random.uniform(
            #            rng2, [batch_size, img_size, img_size, 3], jnp.float32, 0, 255
            #        )
            #    )
            # }

            rng_step, rng = jax.random.split(rng, 2)
            train_state, metrics = fast_train_step(
                train_state, rng_step, batch1, batch2
            )
            for metric_name, value in metrics.items():
                tf.summary.scalar(f"train_{metric_name}", data=float(value), step=step)

            if step == 10 and FLAGS.debug:
                break

        val_metrics = evaluate(
            train_state,
            rng,
            dataset_factory=partial(
                load_cifar10_dataset,
                shuffle=False,
                apply_augmentations=False,
                batch_size=128 if not FLAGS.debug else 16,
            ),
        )
        for metric_name, value in val_metrics.items():
            tf.summary.scalar(f"val_{metric_name}", data=float(value), step=step)
        save_pytree(train_state, path=Path(logdir) / "train_state.pikle", overwrite=True)


if __name__ == "__main__":
    # dataset_checks()
    main()
