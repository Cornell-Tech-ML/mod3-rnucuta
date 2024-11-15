# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Diagnostics from `python parallel_check.py` for `map`, `zip`, `reduce`, `matmul`

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py (163)
-------------------------------------------------------------------------------|loop #ID
    def _map(                                                                  |
        out: Storage,                                                          |
        out_shape: Shape,                                                      |
        out_strides: Strides,                                                  |
        in_storage: Storage,                                                   |
        in_shape: Shape,                                                       |
        in_strides: Strides,                                                   |
    ) -> None:                                                                 |
        # TODO: Implement for Task 3.1.                                        |
        # fn = njit()(fn)                                                      |
        # Main loop in parallel                                                |
        # All indices use numpy buffers                                        |
        if len(out_shape) > MAX_DIMS or len(in_shape) > MAX_DIMS:              |
            raise ValueError(f"Tensor dimensions cannot exceed {MAX_DIMS}")    |
                                                                               |
        # When out and in are stride-aligned, avoid indexing                   |
        if (                                                                   |
            len(out_shape) == len(in_shape)                                    |
            and len(out_strides) == len(in_strides)                            |
            and (out_shape == in_shape).all()----------------------------------| #0
            and (out_strides == in_strides).all()------------------------------| #1
        ):                                                                     |
            for i in prange(len(out)):-----------------------------------------| #2
                out[i] = fn(in_storage[i])                                     |
                                                                               |
        else:                                                                  |
            for i in prange(len(out)):-----------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)          |
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)           |
                # Convert flat index to multidimensional index                 |
                # to_index function is useful just for getting                 |
                # continous set of indices                                     |
                to_index(i, out_shape, out_index)                              |
                                                                               |
                broadcast_index(out_index, out_shape, in_shape, in_index)      |
                                                                               |
                # ordinals                                                     |
                in_position = index_to_position(in_index, in_strides)          |
                out_position = index_to_position(out_index, out_strides)       |
                                                                               |
                out[out_position] = fn(in_storage[in_position])                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(190) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(191) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(231)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py (231)
-------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                  |
        out: Storage,                                                          |
        out_shape: Shape,                                                      |
        out_strides: Strides,                                                  |
        a_storage: Storage,                                                    |
        a_shape: Shape,                                                        |
        a_strides: Strides,                                                    |
        b_storage: Storage,                                                    |
        b_shape: Shape,                                                        |
        b_strides: Strides,                                                    |
    ) -> None:                                                                 |
        # TODO: Implement for Task 3.1.                                        |
        if (                                                                   |
            len(out_shape) > MAX_DIMS                                          |
            or len(a_shape) > MAX_DIMS                                         |
            or len(b_shape) > MAX_DIMS                                         |
        ):                                                                     |
            raise ValueError(f"Tensor dimensions cannot exceed {MAX_DIMS}")    |
                                                                               |
        n = len(out)                                                           |
        if (                                                                   |
            len(out_strides) == len(a_strides)                                 |
            and len(out_shape) == len(a_shape)                                 |
            and len(a_strides) == len(b_strides)                               |
            and len(b_shape) == len(a_shape)                                   |
            and (out_strides == a_strides).all()-------------------------------| #4
            and (a_strides == b_strides).all()---------------------------------| #5
            and (out_shape == a_shape).all()-----------------------------------| #6
            and (a_shape == b_shape).all()-------------------------------------| #7
        ):                                                                     |
            for i in prange(n):------------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                        |
        else:                                                                  |
            # global broacasting case                                          |
            for i in prange(n):------------------------------------------------| #9
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                 |
                in_index_a = np.empty(MAX_DIMS, dtype=np.int32)                |
                in_index_b = np.empty(MAX_DIMS, dtype=np.int32)                |
                to_index(i, out_shape, out_index)                              |
                                                                               |
                broadcast_index(out_index, out_shape, a_shape, in_index_a)     |
                broadcast_index(out_index, out_shape, b_shape, in_index_b)     |
                                                                               |
                # Calculate ordinals                                           |
                in_position_a = index_to_position(in_index_a, a_strides)       |
                in_position_b = index_to_position(in_index_b, b_strides)       |
                out_position = index_to_position(out_index, out_strides)       |
                                                                               |
                out[out_position] = fn(                                        |
                    a_storage[in_position_a], b_storage[in_position_b]         |
                )                                                              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(266) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(267) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index_a = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(268) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index_b = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(307)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py (307)
-------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                               |
        out: Storage,                                                          |
        out_shape: Shape,                                                      |
        out_strides: Strides,                                                  |
        a_storage: Storage,                                                    |
        a_shape: Shape,                                                        |
        a_strides: Strides,                                                    |
        reduce_dim: int,                                                       |
    ) -> None:                                                                 |
        # TODO: Implement for Task 3.1.                                        |
        if len(out_shape) > MAX_DIMS or len(a_shape) > MAX_DIMS:               |
            raise ValueError(f"Tensor dimensions cannot exceed {MAX_DIMS}")    |
                                                                               |
        for i in prange(len(out)):---------------------------------------------| #10
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)              |
            reduce_size: int = a_shape[reduce_dim]                             |
            to_index(i, out_shape, out_index)                                  |
            output_position = index_to_position(out_index, out_strides)        |
                                                                               |
            reduce_stride = a_strides[reduce_dim]                              |
            baseIdx = index_to_position(out_index, a_strides)                  |
            acc = out[output_position]                                         |
            for j in range(reduce_size):                                       |
                a_position = baseIdx + j * reduce_stride                       |
                acc = fn(acc, a_storage[a_position])                           |
                                                                               |
            out[output_position] = acc                                         |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(321) is hoisted out of the parallel loop labelled #10 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py
(338)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /mnt/c/Users/raymo/Documents/CTS1/CS5781/mod3-rnucuta/minitorch/fast_ops.py (338)
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                |
    out: Storage,                                                                           |
    out_shape: Shape,                                                                       |
    out_strides: Strides,                                                                   |
    a_storage: Storage,                                                                     |
    a_shape: Shape,                                                                         |
    a_strides: Strides,                                                                     |
    b_storage: Storage,                                                                     |
    b_shape: Shape,                                                                         |
    b_strides: Strides,                                                                     |
) -> None:                                                                                  |
    """NUMBA tensor matrix multiply function.                                               |
                                                                                            |
    Should work for any tensor shapes that broadcast as long as                             |
                                                                                            |
    ```                                                                                     |
    assert a_shape[-1] == b_shape[-2]                                                       |
    ```                                                                                     |
                                                                                            |
    Optimizations:                                                                          |
                                                                                            |
    * Outer loop in parallel                                                                |
    * No index buffers or function calls                                                    |
    * Inner loop should have no global writes, 1 multiply.                                  |
                                                                                            |
                                                                                            |
    Args:                                                                                   |
    ----                                                                                    |
        out (Storage): storage for `out` tensor                                             |
        out_shape (Shape): shape for `out` tensor                                           |
        out_strides (Strides): strides for `out` tensor                                     |
        a_storage (Storage): storage for `a` tensor                                         |
        a_shape (Shape): shape for `a` tensor                                               |
        a_strides (Strides): strides for `a` tensor                                         |
        b_storage (Storage): storage for `b` tensor                                         |
        b_shape (Shape): shape for `b` tensor                                               |
        b_strides (Strides): strides for `b` tensor                                         |
                                                                                            |
    Returns:                                                                                |
    -------                                                                                 |
        None : Fills in `out`                                                               |
                                                                                            |
    """                                                                                     |
    # TODO: Implement for Task 3.2.                                                         |
    assert a_shape[-1] == b_shape[-2], "a_shape[-1] != b_shape[-2]"                         |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  |
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0                            |
    for batch in prange(out_shape[0]):------------------------------------------------------| #11
        batch_idx_a = batch * a_batch_stride                                                |
        batch_idx_b = batch * b_batch_stride                                                |
        out_batch_idx = batch * out_batch_stride                                            |
        for r in range(out_shape[-2]):                                                      |
            for c in range(out_shape[-1]):                                                  |
                acc = 0.0                                                                   |
                for i in range(a_shape[-1]):  # want to do the inner loop before col        |
                    # like BLAS implementation to be more cache friendly,                   |
                    # but this causes global writes                                         |
                    acc += (                                                                |
                        a_storage[batch_idx_a + r * a_strides[-2] + i * a_strides[-1]]      |
                        * b_storage[batch_idx_b + i * b_strides[-2] + c * b_strides[-1]]    |
                    )                                                                       |
                out[out_batch_idx + r * out_strides[-2] + c * out_strides[-1]] = acc        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Training logs
## Simple Training Log
#### CPU
`python ./project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05`
```
Epoch  0  loss  5.227103929845736 correct 45
Epoch  10  loss  1.3389582763514563 correct 48
Epoch  20  loss  1.3519684713592408 correct 49
Epoch  30  loss  0.9245312218812626 correct 49
Epoch  40  loss  1.4164051163174942 correct 50
Epoch  50  loss  0.45788323772728445 correct 50
Epoch  60  loss  0.38155450284024284 correct 50
Epoch  70  loss  0.014909589581081328 correct 50
Epoch  80  loss  0.3008786030572473 correct 50
Epoch  90  loss  0.6494520648799507 correct 49
Epoch  100  loss  0.08704423913167461 correct 49
Epoch  110  loss  0.12610651339803905 correct 50
Epoch  120  loss  0.5958331058939254 correct 50
Epoch  130  loss  0.0392377047443921 correct 50
Epoch  140  loss  0.3262092598167953 correct 50
Epoch  150  loss  0.39027393984991376 correct 50
Epoch  160  loss  0.961315530705107 correct 50
Epoch  170  loss  0.6004209261400778 correct 50
Epoch  180  loss  0.7371469959981003 correct 50
Epoch  190  loss  0.2048617802773216 correct 50
Epoch  200  loss  0.012391585265649579 correct 49
Epoch  210  loss  1.2266988284180493 correct 49
Epoch  220  loss  0.033514582161942946 correct 50
Epoch  230  loss  0.011431578740936107 correct 50
Epoch  240  loss  0.037224722776147634 correct 50
Epoch  250  loss  0.31704622874708055 correct 50
Epoch  260  loss  0.6449834704023654 correct 50
Epoch  270  loss  0.18896344801140144 correct 50
Epoch  280  loss  0.004314658497240619 correct 50
Epoch  290  loss  0.7112669813016048 correct 50
Epoch  300  loss  0.20182973625959677 correct 50
Epoch  310  loss  0.8845883326648963 correct 50
Epoch  320  loss  0.1485518341509886 correct 49
Epoch  330  loss  0.22741454838964129 correct 50
Epoch  340  loss  0.4180044900303508 correct 50
Epoch  350  loss  1.3460912142147405 correct 49
Epoch  360  loss  0.3435388876210477 correct 50
Epoch  370  loss  0.41499008083489186 correct 50
Epoch  380  loss  0.7431559339326753 correct 50
Epoch  390  loss  0.9085135781416908 correct 49
Epoch  400  loss  0.6912849092684514 correct 49
Epoch  410  loss  0.007903194041654849 correct 50
Epoch  420  loss  0.0017670371716705926 correct 50
Epoch  430  loss  0.035595874356683786 correct 50
Epoch  440  loss  0.8341673688668696 correct 49
Epoch  450  loss  0.017196812873795177 correct 50
Epoch  460  loss  0.6581918110753417 correct 50
Epoch  470  loss  0.22930500948373622 correct 50
Epoch  480  loss  0.49264039078558475 correct 50
Epoch  490  loss  0.37152586893728445 correct 50
Time per epoch: 0.1437527356147766 seconds
```

### CUDA
`python ./project/run_fast_tensor.py --BACKEND cuda --HIDDEN 100 --DATASET simple --RATE 0.05`
```
Epoch  0  loss  7.243783161323494 correct 43
Epoch  10  loss  1.459041656205871 correct 49
Epoch  20  loss  1.1800416379399232 correct 50
Epoch  30  loss  1.7837028344755796 correct 50
Epoch  40  loss  0.7006525683647677 correct 50
Epoch  50  loss  1.4677851019811776 correct 50
Epoch  60  loss  1.3519005991140178 correct 50
Epoch  70  loss  0.5542759405325405 correct 48
Epoch  80  loss  0.5348039614396258 correct 50
Epoch  90  loss  0.9459047281357238 correct 49
Epoch  100  loss  0.9720697259731854 correct 49
Epoch  110  loss  0.44455083392332845 correct 50
Epoch  120  loss  0.34462954428759557 correct 50
Epoch  130  loss  0.08765366614974934 correct 50
Epoch  140  loss  0.35113807712155215 correct 50
Epoch  150  loss  0.42266550501234607 correct 50
Epoch  160  loss  0.19998018942102663 correct 50
Epoch  170  loss  0.23063383120479453 correct 50
Epoch  180  loss  0.35269643155860225 correct 50
Epoch  190  loss  0.26653880875002595 correct 50
Epoch  200  loss  0.9040063266252222 correct 50
Epoch  210  loss  0.3190180076178658 correct 50
Epoch  220  loss  0.17625484788733517 correct 50
Epoch  230  loss  0.6450723685502265 correct 50
Epoch  240  loss  0.7586974454894064 correct 50
Epoch  250  loss  0.008499650472842388 correct 50
Epoch  260  loss  0.0012104217175565818 correct 50
Epoch  270  loss  0.5281917521267143 correct 50
Epoch  280  loss  0.32822978392749813 correct 50
Epoch  290  loss  0.0039021707466255617 correct 50
Epoch  300  loss  0.3646075238620817 correct 50
Epoch  310  loss  0.0006407326641070009 correct 50
Epoch  320  loss  0.39232116010146734 correct 50
Epoch  330  loss  0.42985850911304446 correct 50
Epoch  340  loss  0.13857376289066947 correct 50
Epoch  350  loss  0.297020648707508 correct 50
Epoch  360  loss  0.05242830157060363 correct 50
Epoch  370  loss  0.005810698018719904 correct 50
Epoch  380  loss  0.00978284434778215 correct 50
Epoch  390  loss  0.009578578979131135 correct 50
Epoch  400  loss  0.0809979019959614 correct 50
Epoch  410  loss  0.06912387763247914 correct 50
Epoch  420  loss  0.29573581763664136 correct 50
Epoch  430  loss  0.34499080354621053 correct 50
Epoch  440  loss  0.30606274286219143 correct 50
Epoch  450  loss  0.11156501444730793 correct 50
Epoch  460  loss  0.3509864910327962 correct 50
Epoch  470  loss  0.030280180319778242 correct 50
Epoch  480  loss  0.0022034721127655163 correct 50
Epoch  490  loss  0.031823963548858324 correct 50
Time per epoch: 1.6751077361106872 seconds
```

## XOR Training Log
### Smaller run (Hidden = 100)
#### CPU
`python ./project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
Epoch  0  loss  5.693083966107005 correct 26
Epoch  10  loss  4.6981559356365885 correct 43
Epoch  20  loss  3.8398042889490545 correct 44
Epoch  30  loss  2.716533249421442 correct 45
Epoch  40  loss  2.4609250651834422 correct 46
Epoch  50  loss  2.311874223155685 correct 45
Epoch  60  loss  1.3838175952702099 correct 47
Epoch  70  loss  0.6421032191854584 correct 46
Epoch  80  loss  1.7031654732113597 correct 48
Epoch  90  loss  1.7104532223884332 correct 48
Epoch  100  loss  2.022498885024041 correct 46
Epoch  110  loss  1.8792580692954641 correct 48
Epoch  120  loss  2.2619918436862427 correct 48
Epoch  130  loss  2.0841160542626196 correct 48
Epoch  140  loss  1.5464155146431375 correct 48
Epoch  150  loss  0.7175522310718367 correct 48
Epoch  160  loss  2.146027202778324 correct 49
Epoch  170  loss  0.5749554208096838 correct 49
Epoch  180  loss  1.9322636336183023 correct 49
Epoch  190  loss  0.6781198612614056 correct 49
Epoch  200  loss  0.4274509013326555 correct 49
Epoch  210  loss  0.3315664731474562 correct 50
Epoch  220  loss  0.8343325906048453 correct 49
Epoch  230  loss  1.389942307702451 correct 50
Epoch  240  loss  0.07916166652160868 correct 49
Epoch  250  loss  0.41676205297782976 correct 50
Epoch  260  loss  0.9813501807734972 correct 50
Epoch  270  loss  0.20431343388523976 correct 50
Epoch  280  loss  0.9487243443681118 correct 50
Epoch  290  loss  0.9816856438317626 correct 50
Epoch  300  loss  0.39203526523902865 correct 50
Epoch  310  loss  0.3905050415624437 correct 50
Epoch  320  loss  0.5318805026007691 correct 50
Epoch  330  loss  0.38538670149922244 correct 50
Epoch  340  loss  0.6275811717638677 correct 50
Epoch  350  loss  0.6248886301673465 correct 50
Epoch  360  loss  0.3575169611816596 correct 50
Epoch  370  loss  0.0592047349292933 correct 50
Epoch  380  loss  0.2426519367183988 correct 50
Epoch  390  loss  0.34948539067802303 correct 50
Epoch  400  loss  0.6981152840768168 correct 50
Epoch  410  loss  0.15692047278831175 correct 50
Epoch  420  loss  0.12633037341137474 correct 50
Epoch  430  loss  0.14032567593205486 correct 50
Epoch  440  loss  0.04403672250369732 correct 50
Epoch  450  loss  0.5148632781779042 correct 50
Epoch  460  loss  0.338838263493079 correct 50
Epoch  470  loss  0.060327089495696866 correct 50
Epoch  480  loss  0.5609132867345588 correct 50
Epoch  490  loss  0.5887413414368559 correct 50
Time per epoch: 0.13361714506149291 seconds
```

#### CUDA
`python ./project/run_fast_tensor.py --BACKEND cuda --HIDDEN 100 --DATASET xor --RATE 0.05`
```

```

### Larger run (Hidden = 200)
#### CPU
`python ./project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05`
```
Epoch  0  loss  7.672171427477381 correct 35
Epoch  10  loss  4.333986012482991 correct 41
Epoch  20  loss  2.6862749632426057 correct 43
Epoch  30  loss  2.293044224756492 correct 38
Epoch  40  loss  2.979163962404327 correct 45
Epoch  50  loss  1.8321084832218875 correct 45
Epoch  60  loss  2.6938576016687277 correct 46
Epoch  70  loss  0.6869476133508531 correct 45
Epoch  80  loss  2.6933087264226225 correct 49
Epoch  90  loss  0.6587601011391733 correct 45
Epoch  100  loss  1.4776809869985572 correct 48
Epoch  110  loss  1.410563102650878 correct 49
Epoch  120  loss  0.8188366316312666 correct 49
Epoch  130  loss  0.8141791100830479 correct 45
Epoch  140  loss  2.145965403183965 correct 49
Epoch  150  loss  2.025716487022357 correct 49
Epoch  160  loss  1.003832118820757 correct 49
Epoch  170  loss  0.9547441877526733 correct 47
Epoch  180  loss  0.4848056536491146 correct 49
Epoch  190  loss  1.5127988687853864 correct 48
Epoch  200  loss  2.096439305036917 correct 49
Epoch  210  loss  1.0255439028124198 correct 48
Epoch  220  loss  0.1739344953942815 correct 46
Epoch  230  loss  3.1174089246816568 correct 46
Epoch  240  loss  0.3036927335977648 correct 49
Epoch  250  loss  0.6866706140276412 correct 49
Epoch  260  loss  1.0980179032603274 correct 49
Epoch  270  loss  1.2952889913115988 correct 47
Epoch  280  loss  1.061301718752033 correct 49
Epoch  290  loss  0.5590598213544224 correct 49
Epoch  300  loss  0.38577058100571127 correct 49
Epoch  310  loss  2.70993793817709 correct 49
Epoch  320  loss  1.5649789169969919 correct 49
Epoch  330  loss  0.8316279452842207 correct 49
Epoch  340  loss  0.32972708959580166 correct 48
Epoch  350  loss  0.7087358844451501 correct 49
Epoch  360  loss  1.908574335726159 correct 49
Epoch  370  loss  0.47655008298153867 correct 49
Epoch  380  loss  0.8185006412506841 correct 49
Epoch  390  loss  1.000724454986849 correct 49
Epoch  400  loss  2.5656705969938107 correct 50
Epoch  410  loss  2.581946219963387 correct 47
Epoch  420  loss  0.2874336607711841 correct 49
Epoch  430  loss  0.892782237713158 correct 49
Epoch  440  loss  1.0267405603886863 correct 49
Epoch  450  loss  0.11037744710388973 correct 49
Epoch  460  loss  0.20722206782588762 correct 49
Epoch  470  loss  0.5905865290233788 correct 49
Epoch  480  loss  0.13308747892148345 correct 49
Epoch  490  loss  2.384239489326367 correct 49
Time per epoch: 0.41191515111923216 seconds
```

#### CUDA
`python ./project/run_fast_tensor.py --BACKEND cuda --HIDDEN 200 --DATASET xor --RATE 0.05`
```

```

## Split Training Log
#### CPU
`python ./project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
Epoch  0  loss  11.747783258721734 correct 32
Epoch  10  loss  5.135393568454575 correct 39
Epoch  20  loss  4.634162001302658 correct 48
Epoch  30  loss  3.08084712590211 correct 39
Epoch  40  loss  3.331248623046255 correct 47
Epoch  50  loss  1.8518944607161754 correct 45
Epoch  60  loss  2.5207694245440595 correct 49
Epoch  70  loss  2.0402874459092875 correct 47
Epoch  80  loss  1.9949417216351775 correct 49
Epoch  90  loss  1.1106489738795322 correct 49
Epoch  100  loss  0.797498991621453 correct 49
Epoch  110  loss  1.061668225254795 correct 49
Epoch  120  loss  2.323119574627956 correct 48
Epoch  130  loss  1.3607030379902565 correct 50
Epoch  140  loss  1.1223965642674691 correct 49
Epoch  150  loss  1.0604912600728194 correct 49
Epoch  160  loss  1.8375626605500937 correct 50
Epoch  170  loss  0.4494887070413367 correct 50
Epoch  180  loss  1.132044054728375 correct 48
Epoch  190  loss  1.2677803398992356 correct 50
Epoch  200  loss  1.273861618210789 correct 50
Epoch  210  loss  0.47052593586899893 correct 49
Epoch  220  loss  0.30575753173612785 correct 50
Epoch  230  loss  0.7033229074646383 correct 50
Epoch  240  loss  0.45307813695732285 correct 50
Epoch  250  loss  0.9906702958711222 correct 50
Epoch  260  loss  0.788026593078978 correct 50
Epoch  270  loss  0.6975040344915274 correct 50
Epoch  280  loss  0.9474257066802163 correct 50
Epoch  290  loss  0.4402530164757002 correct 50
Epoch  300  loss  0.27131667517590063 correct 50
Epoch  310  loss  0.6334286662594925 correct 50
Epoch  320  loss  0.7295904350059753 correct 50
Epoch  330  loss  1.0286530106186698 correct 50
Epoch  340  loss  0.28569846766933904 correct 50
Epoch  350  loss  0.1773983388553765 correct 50
Epoch  360  loss  0.35207519727396885 correct 50
Epoch  370  loss  0.28301553660608775 correct 50
Epoch  380  loss  0.44257620401774106 correct 50
Epoch  390  loss  0.5670454377879482 correct 50
Epoch  400  loss  0.07206658545736053 correct 50
Epoch  410  loss  0.6033277589772934 correct 50
Epoch  420  loss  0.6052768638051578 correct 50
Epoch  430  loss  0.418298058648908 correct 50
Epoch  440  loss  0.3404205492254168 correct 50
Epoch  450  loss  0.6810195613941384 correct 50
Epoch  460  loss  0.12201924822191491 correct 50
Epoch  470  loss  0.11128989619925227 correct 50
Epoch  480  loss  0.4809244478581089 correct 50
Epoch  490  loss  0.3272802593550309 correct 50
Time per epoch: 0.15243664693832398 seconds
```

#### CUDA
`python ./project/run_fast_tensor.py --BACKEND cuda --HIDDEN 100 --DATASET split --RATE 0.05`
```
Epoch  0  loss  7.102552448813833 correct 27
Epoch  10  loss  5.6352677827082225 correct 37
Epoch  20  loss  7.0354511537289985 correct 44
Epoch  30  loss  5.01690548372525 correct 44
Epoch  40  loss  3.9422232389887713 correct 42
Epoch  50  loss  2.2857964714861 correct 46
Epoch  60  loss  4.0549409546862 correct 48
Epoch  70  loss  1.7156695358778071 correct 45
Epoch  80  loss  1.4683235990640193 correct 48
Epoch  90  loss  1.2886726927369154 correct 49
Epoch  100  loss  2.6884497210620015 correct 44
Epoch  110  loss  1.7133466158018296 correct 49
Epoch  120  loss  1.1391856337802095 correct 48
Epoch  130  loss  1.8964720828884787 correct 45
Epoch  140  loss  3.5238894901002054 correct 49
Epoch  150  loss  0.9633914565166215 correct 49
Epoch  160  loss  0.4041153797688939 correct 48
Epoch  170  loss  0.5585132869971287 correct 49
Epoch  180  loss  0.5532659329840499 correct 49
Epoch  190  loss  1.723197426124809 correct 48
Epoch  200  loss  0.7209808578618647 correct 50
Epoch  210  loss  1.0368780245121483 correct 49
Epoch  220  loss  0.20931722642383327 correct 47
Epoch  230  loss  0.6835388563265614 correct 50
Epoch  240  loss  0.5527669555593506 correct 49
Epoch  250  loss  0.8469765595417391 correct 49
Epoch  260  loss  2.1907573333369874 correct 50
Epoch  270  loss  1.3024480144243318 correct 49
Epoch  280  loss  2.1055174194163637 correct 50
Epoch  290  loss  0.45855351869484795 correct 50
Epoch  300  loss  0.5242971962192966 correct 49
Epoch  310  loss  0.7414755948600911 correct 50
Epoch  320  loss  1.2162708381265839 correct 50
Epoch  330  loss  0.8186147064029199 correct 50
Epoch  340  loss  1.1807333625200567 correct 50
Epoch  350  loss  0.6784319361833122 correct 49
Epoch  360  loss  0.5513131156272417 correct 50
Epoch  370  loss  0.1349103902660922 correct 50
Epoch  380  loss  0.579783657118596 correct 50
Epoch  390  loss  0.5544605817989194 correct 50
Epoch  400  loss  2.298263695132476 correct 44
Epoch  410  loss  0.7786645181358797 correct 50
Epoch  420  loss  0.5714615581193812 correct 50
Epoch  430  loss  0.4391702622503952 correct 50
Epoch  440  loss  0.40466780101557226 correct 50
Epoch  450  loss  1.1560017565061154 correct 50
Epoch  460  loss  0.1613069891703906 correct 50
Epoch  470  loss  0.634226745782767 correct 50
Epoch  480  loss  0.49735942324974775 correct 50
Epoch  490  loss  0.17711782461539932 correct 50
Time per epoch: 1.6429388070106505 seconds
```

