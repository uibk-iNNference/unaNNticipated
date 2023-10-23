# Consumer GTX 970

This is a shortened trace for the minimal model
TODO: fix or discuss the display format of the arguments in the trace

## Initialization

The first 11k lines or so are initialization

## Setting Variables and Allocating Memory

Then, after some more validation and a few layers of checker functions and polymorphism, we send the first task to GPU:

```
_Send input 1 from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0
MemcpyH2D{'context_id': '$$1', 'correlation_id': '3', 'device_id': '0', 'memcpy_details': 'kind:unknown size:8 dest:0 async:1'}
MemcpyH2D{'annotation': '#edge_name=copy#', 'context_id': '$$1', 'correlation_id': '3', 'edge_name': 'copy', 'is_eager': '0', 'memcpy_details': 'kind:pageable size:8 dest:0 async:1', 'tf_op': ''}
EagerKernelExecute
AssignVariableOp{'is_eager': '1', 'long_name': 'AssignVariableOp:AssignVariableOp'}
```

This happens 2 or 3 more times, until interesting things start to happen.

## Currently Unexplained Behaviour

Next, a few kernels we currently can't explain are executed:

- `RandomUniform` (`tensorflow::functor::FillPhiloxRandomKernelLaunch`)
- `Sub`
- `Mul`
- `Add`

Whether this is in preparation for one of the following operations or for benchmarking, we are currently unsure.
The RandomUniform is especially confounding, and further indicates the benchmarking hypothesis.

After this we have mor variable handle getting and assignment, some more fills, and a `Cast` operation.
A `Fill` also happens, which copies data from the host (CPU) to the device (GPU), possibly among other operations.
This is all reasonably expected, as inputs come as numpy arrays and need to be converted to tensors.

## Convolution

At line 40k we get to the actual Convolution

```
Conv2D{'is_eager': '1', 'long_name': 'Conv2D:Conv2D'}
```

This is followed by a _lot_ of memory management, which we will skip.

### Implicit General Matrix Multiplication (GEMM)

The first call to an actual convolution function looks like the following:

```cpp
void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '40', 'is_eager': '1', 'occupancy_min_grid_size': '13', 'occupancy_suggested_block_size': '1024', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '50'}
```

There are multiple calls to this function in the trace, and based on later invocations I assume it's actually called twice.
This function implicitly converts inputs and kernels into a form that allows for computation of the convolution via general matrix multiplication (GEMM).
It also seems to be called twice.

### Explicit GEMM

```cpp
void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensorStruct, float const*, float*){'context_id': '$$1', 'correlation_id': '65', 'device_id': '0'}
void explicit_convolve_sgemm<float, int, 1024, 5, 5, 3, 3, 3, 0, false>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, unsigned long long, int, unsigned long long, int, float, float, int, float const*, float const*){'context_id': '$$1', 'correlation_id': '66', 'device_id': '0'}
```

This explicitly transforms the inputs and then computes the multiplication, already converting the output.
In the previous call this happened implicitly, hence the names.

### Fourier Transformation

The third variant that is executed goes as follows:

```cpp
void flip_filter<float, float>(float*, float const*, int, int, int, int){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '83', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '14.0625'}
void fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int){'context_id': '$$1', 'correlation_id': '84', 'device_id': '0'}
void fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int){'context_id': '$$1', 'correlation_id': '85', 'device_id': '0'}
void gemv2T_kernel_val<int, int, float2, float2, float2, float2, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>, float2, float2){'context_id': '$$1', 'correlation_id': '86', 'device_id': '0'}
void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*){'context_id': '$$1', 'correlation_id': '87', 'device_id': '0'}
```

And is a convolution in the frequency domain, done via Fast Fourier Transformation (FFT) of the input and kernel, computing the convolution in the frequency domain as pointwise multiplication, and transforming the output back into the spatial domain.

### Winograd

Next, the Winograd convolution algorithm is run.
In CUDA, this requires generating the kernels on the fly:

```cpp
void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>){'context_id': '$$1', 'correlation_id': '104', 'device_id': '0'}
maxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0{'context_id': '$$1', 'correlation_id': '105', 'device_id': '0'}
```

### Another FFT variant

Next follows another frequency domain solution, but using a different kernel:

```cpp
void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int){'context_id': '$$1', 'correlation_id': '122', 'device_id': '0'}
void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int){'context_id': '$$1', 'correlation_id': '123', 'device_id': '0'}
void gemv2N_kernel<int, int, float2, float2, float2, float2, 128, 8, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>){'context_id': '$$1', 'correlation_id': '124', 'device_id': '0'}
void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int){'context_id': '$$1', 'correlation_id': '125', 'device_id': '0'}
```

### Explicit Winograd

And the _non-fused_ version of the Winograd algorithm, that explicitly transforms inputs and outputs, and computes the result using a similar kernel than the above call:

```cpp
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>){'context_id': '$$1', 'correlation_id': '142', 'device_id': '0'}
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>){'context_id': '$$1', 'correlation_id': '143', 'device_id': '0'}
void gemv2N_kernel<int, int, float, float, float, float, 128, 8, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>){'context_id': '$$1', 'correlation_id': '144', 'device_id': '0'}
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>){'context_id': '$$1', 'correlation_id': '145', 'device_id': '0'}
```

### Teardown

All of these functions are called twice, possibly as a middle ground between better benchmarking measurements and lower initialization time.
The calls are again followed by a lot of memory information, some frees, and some allacations.
The frees could indicate that the benchmarks happen on synthetic data, which would explain the `RandomUniform`.

In this there is no copying from host to device that we found, so the actual data must already reside on the device.

## We Have a Winner

After these tries, one of the convolution methods is called again:

```cpp
void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '165', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '576', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '56.25'}
maxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0{'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '166', 'is_eager': '1', 'occupancy_min_grid_size': '13', 'occupancy_suggested_block_size': '512', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '25'}
void tensorflow::functor::ShuffleInTensor3Simple<float, 0, 2, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '167', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '100'}
```

Afterwards the Bias is added:

```
BiasAdd{'is_eager': '1', 'long_name': 'BiasAdd:BiasAdd'}
void tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int){'annotation': 'BiasAdd:BiasAdd', 'context_id': '$$1', 'correlation_id': '168', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'BiasAdd:BiasAdd', 'theoretical_occupancy_pct': '100'}
BiasAdd{'long_name': 'BiasAdd:BiasAdd'}
```

And after a lot of checks and more polymorphism we get the data from the device:

```
MemcpyD2H{'context_id': '$$1', 'correlation_id': '170', 'device_id': '0', 'memcpy_details': 'kind:unknown size:16 dest:0 async:1'}
MemcpyD2H{'annotation': '#edge_name=copy#', 'context_id': '$$1', 'correlation_id': '170', 'edge_name': 'copy', 'is_eager': '0', 'memcpy_details': 'kind:pinned size:16 dest:0 async:1', 'tf_op': ''}
```

## The Second Sample

Execution for the second sample is much more streamlined.
First, we store the data on the device and cast it to the desired datatype and format:

```cpp
_Send input 0 from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0
MemcpyH2D{'context_id': '$$1', 'correlation_id': '171', 'device_id': '0', 'memcpy_details': 'kind:unknown size:64 dest:0 async:1'}
MemcpyH2D{'annotation': '#edge_name=copy#', 'context_id': '$$1', 'correlation_id': '171', 'edge_name': 'copy', 'is_eager': '0', 'memcpy_details': 'kind:pageable size:64 dest:0 async:1', 'tf_op': ''}
EagerKernelExecute
Cast{'is_eager': '1', 'long_name': 'Cast:Cast'}
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long){'context_id': '$$1', 'correlation_id': '172', 'device_id': '0'}
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long){'annotation': 'Cast:Cast', 'context_id': '$$1', 'correlation_id': '172', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'Cast:Cast', 'theoretical_occupancy_pct': '100'}
```

Afterwards a few pointer assignments happen to get the logical organization of data in order.
Then, the desired convolution happens with the winning algorithm:

```cpp
Conv2D{'is_eager': '1', 'long_name': 'Conv2D:Conv2D'}
void tensorflow::functor::ShuffleInTensor3Simple<float, 0, 2, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*){'context_id': '$$1', 'correlation_id': '173', 'device_id': '0'}
void tensorflow::functor::ShuffleInTensor3Simple<float, 0, 2, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '173', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '100'}
Conv2D{'long_name': 'Conv2D:Conv2D'}
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*){'context_id': '$$1', 'correlation_id': '174', 'device_id': '0'}
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '174', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '100'}
void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>){'context_id': '$$1', 'correlation_id': '175', 'device_id': '0'}
maxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0{'context_id': '$$1', 'correlation_id': '176', 'device_id': '0'}
void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '175', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '576', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '56.25'}
maxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0{'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '176', 'is_eager': '1', 'occupancy_min_grid_size': '13', 'occupancy_suggested_block_size': '512', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '25'}
void tensorflow::functor::ShuffleInTensor3Simple<float, 0, 2, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*){'context_id': '$$1', 'correlation_id': '177', 'device_id': '0'}
void tensorflow::functor::ShuffleInTensor3Simple<float, 0, 2, 1, false>(int, float const*, tensorflow::functor::Dimension<3>, float*){'annotation': 'Conv2D:Conv2D', 'context_id': '$$1', 'correlation_id': '177', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'Conv2D:Conv2D', 'theoretical_occupancy_pct': '100'}
```

Again, adding the bias:

```cpp
BiasAdd{'is_eager': '1', 'long_name': 'BiasAdd:BiasAdd'}
void tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int){'context_id': '$$1', 'correlation_id': '178', 'device_id': '0'}
void tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int){'annotation': 'BiasAdd:BiasAdd', 'context_id': '$$1', 'correlation_id': '178', 'is_eager': '1', 'occupancy_min_grid_size': '26', 'occupancy_suggested_block_size': '1024', 'tf_op': 'BiasAdd:BiasAdd', 'theoretical_occupancy_pct': '100'}
```

And getting the data from the device:

```
cuStreamSynchronize{'context_id': '$$1', 'correlation_id': '179', 'device_id': '0'}
MemcpyD2H{'context_id': '$$1', 'correlation_id': '180', 'device_id': '0', 'memcpy_details': 'kind:unknown size:16 dest:0 async:1'}
MemcpyD2H{'annotation': '#edge_name=copy#', 'context_id': '$$1', 'correlation_id': '180', 'edge_name': 'copy', 'is_eager': '0', 'memcpy_details': 'kind:pinned size:16 dest:0 async:1', 'tf_op': ''}
```

Followed by the final teardown.

## Other traces

For the RTX 2070 and GTX 1650 trace all functions are called twice.
The `implicit_convolve_sgemm` function is called 4 times, and twice in the 970 trace.
This could be an indicator that this function suffers from higher volatility and is run more often during benchmarking (it might also just have less overhead so it's cheaper to run multiple times).
We have observed that the other traces run all algorithms twice as often as in the 970 trace, so the initialization might be adaptive to the hardware speed.

However, our observed general structure holds, and indicates benchmarking happens before the actual convolution is run.

## Next steps

The logical next step is to rerun our observed GPU nondeterminism results (especially the RTX 2070), and test whether each observed equivalence class on that device belongs to a different winning algorithm.
