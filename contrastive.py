import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64/"
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

from collections import defaultdict
from optparse import Option
from typing import Mapping, Generator, Callable, Optional, Dict

import jax
import numpy as np
import haiku as hk
from tqdm import tqdm

# from PIL import Image
import jax.numpy as jnp

import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([], "GPU")
gpus = tf.config.get_visible_devices("GPU")
# from einops import rearrange
# from tabulate import tabulate
# import imgaug.augmenters as iaa

# tf.config.set_visible_devices([], device_type="GPU")  # make sure TF doesn't use GPU.
# import tensorflow_datasets as tfds

print("jax version {}".format(jax.__version__))
from jax.lib import xla_bridge

print("jax backend {}".format(xla_bridge.get_backend().platform))

Batch = Mapping[str, np.array]


def _forward(batch: Batch, is_training: bool) -> Dict[str, jnp.array]:
    images = batch["image"]
    images = imagenet_preprocessing(images)

    backbone = ResNet18Sim()
    projector = Projector(nb_dims=2_048, nb_layers=2)
    predictor = Predictor(hidden_dims=512)

    feature_maps = backbone(images, is_training=is_training)
    embeddings = jnp.mean(feature_maps, axis=(1, 2))
    representations, projector_monitor = projector(
        feature_maps, is_training=is_training
    )
    predictions, predictor_monitor = predictor(representations, is_training=is_training)

    return {
        "z": representations,
        "p": predictions,
        "z_monitor": projector_monitor,
        "p_monitor": predictor_monitor,
    }

forward = hk.transform_with_state(_forward)


class Cifar10WrappedBatch:
    def __init__(self, batch: Batch):
        self.batch = batch

    @property
    def x(self) -> jnp.array:
        return jnp.array(self.batch["image"])

    @property
    def y(self) -> jnp.array:
        return jnp.array(self.batch["label"])

    @property
    def y_one_hot(self):
        return jax.nn.one_hot(self.y, num_classes=10)


def load_cifar10_dataset(
    split: str,
    shuffle: bool,
    apply_augmentations: bool,
    batch_size: int,
    rng: Optional[jnp.array] = None,
    data_order_rng: Optional[jnp.array] = None,
    augmentation_rng: Optional[jnp.array] = None,
    as_numpy: bool = True,
) -> Generator[Cifar10WrappedBatch, None, None]:
    if rng is None and (data_order_rng is None and augmentation_rng is None):
        raise Exception(
            "Must provide either `rng` or `data_order_rng`+`augmentation_rng`."
        )
    if rng is not None:
        data_order_rng, augmentation_rng = jax.random.split(rng)

    split = {
        "val": "train[:1000]",
        "query": "train[1000:3000]",
        "index": "train[3000:5000]",
        "train": "train[5000:]",
        "test": "test",
    }[split]

    dataset = tfds.load("cifar10", split=split)
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10, seed=int(data_order_rng[1]))

    dataset = dataset.batch(batch_size)

    if apply_augmentations:
        (
            aug_order_seed,
            crop_seed,
            flip_seed,
            sometimes8_seed,
            sometimes2_seed,
            brightness_seed,
            contrast_seed,
            saturation_seed,
            hue_seed,
            grayscale_seed,
            blur_seed,
        ) = jax.random.split(augmentation_rng, 11)

        _convert_seed = lambda jax_seed: int(jax_seed[1])
        sometimes8 = lambda aug: iaa.Sometimes(
            0.8, aug, seed=_convert_seed(sometimes8_seed)
        )
        sometimes2 = lambda aug: iaa.Sometimes(
            0.2, aug, seed=_convert_seed(sometimes2_seed)
        )

        augmentation_fn = iaa.Sequential(
            [
                iaa.Crop(
                    percent=(0.0, 0.2),
                    sample_independently=True,
                    keep_size=True,
                    seed=_convert_seed(crop_seed),
                ),
                iaa.Fliplr(0.5, seed=_convert_seed(flip_seed)),
                # Disable because imgaug has bug whereby the seed has no effect
                # ==> we can't reproduce the augmentations.
                # sometimes8(
                #   iaa.MultiplyBrightness(
                #       (0.6, 1.4), seed=_convert_seed(brightness_seed)
                #   ),
                # ),
                sometimes8(
                    iaa.LinearContrast((0.6, 1.4), seed=_convert_seed(contrast_seed))
                ),
                sometimes8(
                    iaa.MultiplySaturation(
                        (0.6, 1.4), seed=_convert_seed(saturation_seed)
                    )
                ),
                sometimes8(iaa.MultiplyHue((-0.1, 0.1), seed=_convert_seed(hue_seed))),
                sometimes2(iaa.Grayscale((0, 1), seed=_convert_seed(grayscale_seed))),
                sometimes2(
                    iaa.GaussianBlur(sigma=(0.1, 0.5), seed=_convert_seed(blur_seed))
                ),
            ],
            random_order=True,
            seed=_convert_seed(aug_order_seed),
        )

        def transform_batch_fn(batch: Batch):
            images = batch["image"]
            dtype = images.dtype
            shape = tf.shape(images)

            out = tf.numpy_function(
                func=lambda x: augmentation_fn(images=x), inp=[images], Tout=[dtype]
            )
            out = out[0]  # Only 1 output.
            out = tf.reshape(out, shape=shape)
            batch["image"] = out
            return batch

        dataset = dataset.map(transform_batch_fn)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if as_numpy:
        dataset = tfds.as_numpy(dataset)
    return dataset


def dataset_sanity_check(rng: jnp.array, dataset_loading_fn, num_batches: int = -1):
    splits = ["train", "val", "query", "index", "test"]
    statistics = {split: defaultdict(lambda: 0) for split in splits}

    for split in tqdm(splits):
        ds = dataset_loading_fn(rng, split=split, batch_size=1_024)
        for i, batch in enumerate(ds):
            if num_batches > 0 and i >= num_batches:
                break
            batch = Cifar10WrappedBatch(batch)
            for label in batch.y:
                label = int(label)
                statistics[split][label] += 1

    sizes = [
        (
            split,
            *[sub_dict[label] for label in range(10)],
            np.sum([v for v in sub_dict.values()]),
        )
        for (split, sub_dict) in statistics.items()
    ]
    print(tabulate(sizes, headers=[v for v in range(10)] + ["Total size"]))


def save_image_batch(img_batch: jnp.array, save_path: str):
    batch_size = img_batch.shape[0]
    size = np.sqrt(batch_size)
    if size != int(size):
        raise Exception(
            "This function does not support non-perfect square batch sizes."
        )
    size = int(size)

    imgs = rearrange(img_batch, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=size, b2=size)
    imgs = np.array(imgs)
    Image.fromarray(imgs).save(save_path)
    return


def compare_augmentations(
    dataset_loading_fn: Callable, save_path: str, split: str = "train", batch_size=8 * 8
):
    rng = jax.random.PRNGKey(0)

    size = np.sqrt(batch_size)
    if size != int(size):
        raise Exception(
            "This function does not support non-perfect square batch sizes."
        )
    size = int(size)
    common_kwargs = dict(rng=rng, split=split, shuffle=True, batch_size=batch_size)

    for i, batch in enumerate(
        dataset_loading_fn(apply_augmentations=False, **common_kwargs)
    ):
        batch = Cifar10WrappedBatch(batch)
        imgs1 = batch.x
        break

    for i, batch in enumerate(
        load_cifar10_dataset(apply_augmentations=True, **common_kwargs)
    ):
        batch = Cifar10WrappedBatch(batch)
        imgs2 = batch.x
        break

    imgs = np.stack([imgs1, imgs2], axis=0)
    imgs = rearrange(imgs, "n (b1 b2) h w c -> (b1 h) (b2 n w) c", b1=size, b2=size)
    imgs = np.array(imgs)
    Image.fromarray(imgs).save(save_path)


def make_augmented_dataset(
    dataset_loading_fn: Callable, rng: jnp.array, *args, **kwargs
):
    data_order_rng, rng = jax.random.split(rng)
    augmentation_rng_1, augmentation_rng_2 = jax.random.split(rng)

    for key in [
        "rng",
        "data_order_rng",
        "augmentation_rng",
        "apply_augmentations",
        "as_numpy",
    ]:
        if key in kwargs:
            kwargs.pop(key)
    kwargs["apply_augmentations"] = True
    kwargs["data_order_rng"] = data_order_rng
    kwargs["as_numpy"] = False

    dataset1 = dataset_loading_fn(augmentation_rng=augmentation_rng_1, *args, **kwargs)
    dataset2 = dataset_loading_fn(augmentation_rng=augmentation_rng_2, *args, **kwargs)

    def sanity_check(batch1, batch2):
        tf.assert_equal(
            tf.math.reduce_all(tf.equal(batch1["label"], batch2["label"])), True
        )
        tf.assert_equal(tf.math.reduce_all(tf.equal(batch1["id"], batch2["id"])), True)
        return {
            "id": batch1["id"],
            "label": batch1["label"],
            "image1": batch1["image"],
            "image2": batch2["image"],
        }

    return tf.data.Dataset.zip((dataset1, dataset2)).map(sanity_check)


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


def imagenet_preprocessing(x: jnp.array) -> jnp.array:
    mean = jnp.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3])
    std = jnp.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3])

    x = jnp.float32(x)
    x = x / 255.0
    x = (x - mean) / std

    return x


def cosine_distance(lhs: jnp.array, rhs: jnp.array) -> jnp.array:
    # lhs and rhs are batched.
    lhs = lhs / jnp.linalg.norm(lhs, ord=2, axis=-1, keepdims=True)
    rhs = rhs / jnp.linalg.norm(lhs, ord=2, axis=-1, keepdims=True)

    return -jnp.dot(lhs, rhs)


def main():
    rng = jax.random.PRNGKey(42)

    def _forward(batch):
        images = batch["image"]
        images = imagenet_preprocessing(images)

        # return ResNet18(num_classes=10)(images, is_training)
        network = ResNet18Sim()
        feature_maps = network(images, is_training=True)
        return jnp.mean(feature_maps, axis=(1, 2))

    def _project(embeddings: jnp.array) -> jnp.array:
        return Projector(nb_dims=2_048, nb_layers=2)(embeddings, is_training=True)

    def _predict(representations: jnp.array) -> jnp.array:
        return Predictor(hidden_dims=512)(representations, is_training=True)

    # for batch in load_cifar10_dataset(
    #    split="train", shuffle=False, apply_augmentations=False, batch_size=8, rng=rng
    # ):
    #    break

    img_size = 32
    batch_size = 8
    rng1, rng2, rng = jax.random.split(rng, 3)
    batch1 = {
        "image": jnp.array(
            jax.random.uniform(
                rng1, [batch_size, img_size, img_size, 3], jnp.float32, 0, 255
            )
        )
    }
    batch2 = {
        "image": jnp.array(
            jax.random.uniform(
                rng2, [batch_size, img_size, img_size, 3], jnp.float32, 0, 255
            )
        )
    }

    forward = hk.transform_with_state(_forward)
    params_backbone, state_backbone = forward.init(rng, batch)
    embeddings, _ = forward.apply(params_backbone, state_backbone, rng, batch)
    print(
        f"Backbone parameters: {sum(x.size for x in jax.tree_leaves(params_backbone)):,}"
    )
    print(f"Backbone output shape: {embeddings.shape}")

    project = hk.transform_with_state(_project)
    params_projector, state_projector = project.init(rng, embeddings)
    (representation, monitor), _ = project.apply(
        params_projector, state_projector, rng, embeddings
    )
    print(
        f"Projector parameters: {sum(x.size for x in jax.tree_leaves(params_projector)):,}"
    )
    print(f"Projector output shape: {representation.shape}")
    print(f"Projector monitor output shape: {monitor.shape}")
    print(f"Projector initial std monitoring value: {monitor.mean()}")

    predict = hk.transform_with_state(_predict)
    params_predictor, state_predictor = predict.init(rng, representation)
    (preds, monitor), _ = predict.apply(
        params_predictor, state_predictor, rng, representation
    )
    print(
        f"Predictor parameters: {sum(x.size for x in jax.tree_leaves(params_predictor)):,}"
    )
    print(f"Predictor output shape: {preds.shape}")
    print(f"Predictor monitor output shape: {monitor.shape}")
    print(f"Predictor initial std monitoring value: {monitor.mean()}")
    import ipdb

    ipdb.set_trace()
    pass

    # fast_forward = jax.jit(forward.apply)

    is_training = True
    y, state = forward.apply(params, state, rng, batch)
    print(y.shape)
    import ipdb

    ipdb.set_trace()
    pass

    # dataset_sanity_check(rng, load_cifar10_dataset)
    # compare_augmentations(
    #     load_cifar10_dataset, save_path="/mnt/c/Users/Romain/Desktop/img.png"
    # )
    for batch in tqdm(
        make_augmented_dataset(
            load_cifar10_dataset,
            rng=rng,
            split="train",
            batch_size=2 ** 8,
            shuffle=True,
        )
    ):
        continue
        save_path = "/mnt/c/Users/Romain/Desktop/haha.png"
        imgs1 = batch["image1"]
        imgs2 = batch["image2"]
        size = 8
        imgs = np.stack([imgs1, imgs2], axis=0)
        imgs = rearrange(imgs, "n (b1 b2) h w c -> (b1 h) (b2 n w) c", b1=size, b2=size)
        imgs = np.array(imgs)
        Image.fromarray(imgs).save(save_path)

        import ipdb

        ipdb.set_trace()
        pass


if __name__ == "__main__":
    main()
