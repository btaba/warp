from functools import partial

import jax
import jax.numpy as jnp

import warp as wp
from warp.jax_experimental import jax_callable, GraphMode


GRAPH_MODE = GraphMode.WARP_STAGED_INCLUSIVE


@wp.kernel
def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_inplace_kernel(a: wp.array(dtype=float), s: float):
    tid = wp.tid()
    a[tid] *= s


@wp.kernel
def scale_inplace_vec_kernel(a: wp.array(dtype=wp.vec2), s: float):
    tid = wp.tid()
    a[tid] *= s


# The Python function to call.
# Note the argument annotations, just like Warp kernels.
def scale_func(
    # inputs
    a: wp.array(dtype=float),
    b: wp.array(dtype=wp.vec2),
    # in-out args
    c: wp.array(dtype=float),
    d: wp.array(dtype=wp.vec2),
    s: float,
    # outputs
    e: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec2),
):
    wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[e])
    wp.launch(scale_vec_kernel, dim=b.shape, inputs=[b, s], outputs=[f])
    wp.launch(scale_inplace_kernel, dim=a.shape, inputs=[c, s])
    wp.launch(scale_inplace_vec_kernel, dim=b.shape, inputs=[d, s])


def example1():
    jax_func = jax_callable(scale_func, num_outputs=4, in_out_argnames=["c", "d"], graph_mode=GRAPH_MODE)

    @jax.jit
    def fun(a, b, c, d):
        s = 2.0

        # output shapes
        output_dims = {"e": a.shape, "f": b.shape}
        return jax_func(a, b, c, d, s, output_dims=output_dims)

    a = jnp.arange(10, dtype=jnp.float32)
    b = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
    c = jnp.arange(10, dtype=jnp.float32)
    d = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
    e, f, g, h = fun(a, b, c, d)
    print(e)
    print(f)
    print(g)
    print(h)

    print("------")

    a = 10 + jnp.arange(10, dtype=jnp.float32)
    b = 10 + jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
    c = 10 + jnp.arange(10, dtype=jnp.float32)
    d = 10 + jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
    e, f, g, h = fun(a, b, c, d)
    print(e)
    print(f)
    print(g)
    print(h)

    print("------")

    a = 20 + jnp.arange(10, dtype=jnp.float32)
    b = 20 + jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
    c = 20 + jnp.arange(10, dtype=jnp.float32)
    d = 20 + jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
    e, f, g, h = fun(a, b, c, d)
    print(e)
    print(f)
    print(g)
    print(h)


def bench1(graph_mode=GRAPH_MODE, num_elements=10_000, num_iters=1000, reuse_arrays=False, use_nvtx=False, verbose=True):
    jax_func = jax_callable(scale_func, num_outputs=4, in_out_argnames=["c", "d"], graph_mode=graph_mode)

    @jax.jit
    def fun(a, b, c, d):
        s = 2.0

        # output shapes
        output_dims = {"e": a.shape, "f": b.shape}
        return jax_func(a, b, c, d, s, output_dims=output_dims)

    times = []

    # retain arrays to force cache misses
    retained_arrays = [] 

    a = jnp.arange(num_elements, dtype=jnp.float32)
    b = jnp.arange(num_elements, dtype=jnp.float32).reshape((num_elements // 2, 2))  # wp.vec2
    c = jnp.arange(num_elements, dtype=jnp.float32)
    d = jnp.arange(num_elements, dtype=jnp.float32).reshape((num_elements // 2, 2))  # wp.vec2

    for iter in range(num_iters):
        wp.synchronize()
        with wp.ScopedTimer(f"iter_{iter}", synchronize=True, print=False, use_nvtx=use_nvtx) as timer:
            e, f, g, h = fun(a, b, c, d)
        
        times.append(timer.elapsed)

        if not reuse_arrays:
            retained_arrays.extend([a, b, c, d, e, f, g, h])
            a = a.copy()
            b = b.copy()
            c = c.copy()
            d = d.copy()

    trim = int(0.5 * num_iters) // 2
    if trim > 0:
        times = sorted(times)[trim:-trim]

    avg_time = sum(times) / len(times)

    if verbose:
        print(f"{avg_time} ms")

    return avg_time


# example1()
# bench1(use_nvtx=True)

time_1 = bench1(GraphMode.WARP, reuse_arrays=True, verbose=False)
time_2 = bench1(GraphMode.WARP, reuse_arrays=False, verbose=False)
time_3 = bench1(GraphMode.WARP_STAGED, verbose=False)
time_4 = bench1(GraphMode.WARP_STAGED_INCLUSIVE, verbose=False)
print(f"{time_1 :.4f} ms (WARP, reuse arrays)")
print(f"{time_2 :.4f} ms (WARP, recapture)")
print(f"{time_3 :.4f} ms (WARP_STAGED)")
print(f"{time_4 :.4f} ms (WARP_STAGED_INCLUSIVE)")
