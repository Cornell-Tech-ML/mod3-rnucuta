# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32 # 256? maximize occupancy (warps used) vs resource usage
# warps always 32 threads
# occupancy is more important than raw speed

# grid
# thread
# block


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # shared_out_shape = cuda.shared.array(MAX_DIMS, numba.int32)
        # local_i = cuda.threadIdx.x
        # try to read in like conv example
        # TODO: Implement for Task 3.3.
        if i < out_size: # gaurding against out of bounds access
            # out size is local since it is primitive
            # out_shape.size is not local tho
            # Copy to local memory once
            # this is only worthy optimization
            # because it lowers global reads from 2 to 1
            # for out shape
            # out_shape_size = out_shape.size
            # elements_per_thread = (out_shape_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            # for j in range(elements_per_thread):
            #     idx = local_i * elements_per_thread + j
            #     if idx < out_shape_size:
            #         shared_out_shape[idx] = out_shape[idx]
            # if local_i < len(out_shape):
            #     shared_out_shape[local_i] = out_shape[local_i]
            # elif 32 < local_i < len(out_shape):
            #     shared_out_shape[local_i] = 0.0
            # cuda.syncthreads()

            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # ordinals
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)

            # 5 reads and 1 write to global memory
            out[out_position] = fn(in_storage[in_position])
            # TOTAL: 

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # shared_out_shape = cuda.shared.array(MAX_DIMS, numba.int32)
        # local_i = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            # Copy to local memory once
            # this is only worthy optimization
            # because it lowers global reads from 2 to 1
            # for out shape
            # if local_i < out_shape.size:
            #     shared_out_shape[local_i] = out_shape[local_i]
            # elif local_i >= out_shape.size:
            #     shared_out_shape[local_i] = 0.0
            # cuda.syncthreads()
            # convert cuda block/thread index to multidimensional index
            to_index(i, out_shape, out_index)

            # broadcast index to match shape
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Calculate ordinal positions for a, b, and out
            in_position_a = index_to_position(a_index, a_strides)
            in_position_b = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)

            # 7 reads 1 write to global memory
            out[out_position] = fn(
                a_storage[in_position_a], b_storage[in_position_b]
            )

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    if i < size: # guarding against out of bounds access
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0   
    cuda.syncthreads()
    
    if pos % 2 == 0:
        cache[pos] += cache[pos + 1]
    cuda.syncthreads()
    if pos % 4 == 0:
        cache[pos] += cache[pos + 2]
    cuda.syncthreads()
    if pos % 8 == 0:
        cache[pos] += cache[pos + 4]
    cuda.syncthreads()
    if pos % 16 == 0:
        cache[pos] += cache[pos + 8]
    cuda.syncthreads()
    if pos % 32 == 0:
        cache[pos] += cache[pos + 16]
    
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice reduction sum by a single block"""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        # a_pos = cuda.blockIdx.x * cuda.blockDim.x + pos
        # reduce_size = cuda.shared.array(1, numba.int32)
        # if pos == 0:
        #     reduce_size[0] = a_shape[reduce_dim]
        # cuda.syncthreads()
        reduce_size = a_shape[reduce_dim]
        reduce_stride = a_strides[reduce_dim]
        # if pos < len(out_shape):
        #     local_out_shape[pos] = out_shape[pos]
        # else:
        #     local_out_shape[pos] = 0.0
        # TODO: Implement for Task 3.3.
        # memory contention for out shape and a_strides
        # does not help with performance
        # to use, need to change to_index method to support shape buffers
        # of fixed size
        # shared_out_shape = cuda.shared.array(MAX_DIMS, numba.int32)
        # shared_a_strides = cuda.shared.array(MAX_DIMS, numba.int32)
        # reduce contention for out shape and a_strides
        # if pos < len(out_shape):
        #     shared_out_shape[pos] = out_shape[pos]
        # elif pos < MAX_DIMS:
        #     shared_out_shape[pos] = 0.0
        # cuda.syncthreads()
        # if pos < len(a_strides):
        #     shared_a_strides[pos] = a_strides[pos]
        # elif pos < MAX_DIMS:
        #     shared_a_strides[pos] = 0.0
        # cuda.syncthreads()
        if out_pos < out_size: # this guard should be unnecessary 
            # given the way outsize is set in higher order tensor reduce
            # copy out shape to local memory
            to_index(out_pos, out_shape, out_index)
            base_idx = index_to_position(out_index, a_strides)
            if pos < reduce_size:
                cache[pos] = fn(reduce_value, a_storage[base_idx + pos * reduce_stride])
            else:
                cache[pos] = reduce_value
            cuda.syncthreads()

            tree_level = 1
            while tree_level < BLOCK_DIM:
                if pos % (2 * tree_level) == 0:
                    cache[pos] = fn(cache[pos], cache[pos + tree_level])
                tree_level = tree_level * 2
                cuda.syncthreads()

            if pos == 0:
                out[out_pos] = cache[0]
        # numba implementation for reference
        # for i in prange(len(out)):
        #     out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
        #     reduce_size: int = a_shape[reduce_dim]
        #     to_index(i, out_shape, out_index)
        #     output_position = index_to_position(out_index, out_strides)

        #     reduce_stride = a_strides[reduce_dim]
        #     baseIdx = index_to_position(out_index, a_strides)
        #     acc = out[output_position]
        #     for j in range(reduce_size):
        #         a_position = baseIdx + j * reduce_stride
        #         acc = fn(acc, a_storage[a_position])

        #     out[output_position] = acc

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # basic structure of matrix multiply
    # for s in range(0, K, TPB):
    #     sharedA[local_i, local_j] = a[i, s + local_j]
    #     sharedB[local_i, local_j] = b[s + local_i, j]
    #     ...
    #     for k in range(TPB):
    #         t += sharedA[local_i, k] * sharedB[k, local_j]
    # out[i, j] = t
    # only one block per grid
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    sharedA = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    sharedB = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    if i < size and j < size:
        sharedA[i, j] = a[i * size + j]
        sharedB[i, j] = b[i * size + j]
        cuda.syncthreads()

        acc = 0.0
        for k in range(size):
            acc += sharedA[i, k] * sharedB[k, j]
        out[i * size + j] = acc



jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    raise NotImplementedError("Need to implement for Task 3.4")

    # blockspergrid = (
    #         (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
    #         (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
    #         out.shape[0],
    #     )
    #     threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
