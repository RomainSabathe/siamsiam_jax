from typing import Mapping, Union
from einops import asnumpy

import numpy as np
import jax.numpy as jnp


Batch = Mapping[str, Union[np.array, jnp.array]]
Scalars = Mapping[str, jnp.ndarray]
