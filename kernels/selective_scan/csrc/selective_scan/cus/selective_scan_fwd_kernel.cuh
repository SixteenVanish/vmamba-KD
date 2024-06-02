/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "static_switch.h"

// 定义选择性扫描前向核函数的特征结构体模板
template<int kNThreads_, int kNItems_, bool kIsEvenLen_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);  // 确保每次处理的项目数是4的倍数
    using input_t = input_t_;  // 定义输入数据类型
    using weight_t = weight_t_;  // 定义权重数据类型
    static constexpr int kNThreads = kNThreads_;  // 定义线程数量
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;  // 根据线程数量设置最小块数
    static constexpr int kNItems = kNItems_;  // 定义每次处理的项目数量
    static constexpr int MaxDState = MAX_DSTATE;  // 定义最大状态数
    static constexpr int kNBytes = sizeof(input_t);  // 计算输入类型的字节数
    static_assert(kNBytes == 2 || kNBytes == 4);  // 确保输入类型为2字节或4字节
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);  // 计算每次加载的元素数量
    static_assert(kNItems % kNElts == 0);  // 确保每次处理的项目数量是元素数量的倍数
    static constexpr int kNLoads = kNItems / kNElts;  // 计算每次加载的次数
    static constexpr bool kIsEvenLen = kIsEvenLen_;  // 定义是否为偶长度

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;  // 定义是否为直接IO

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;  // 定义向量类型
    using scan_t = float2;  // 定义扫描类型
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;  // 定义块加载类型
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;  // 定义向量块加载类型
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;  // 定义权重块加载类型
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;  // 定义向量权重块加载类型
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;  // 定义块存储类型
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;  // 定义向量块存储类型
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;  // 定义块扫描类型
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});  // 计算共享内存IO大小
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);  // 计算共享内存大小
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;  // 定义线程数量
    constexpr int kNItems = Ktraits::kNItems;  // 定义每次处理的项目数量    kNItems的值决定了每个线程在一次循环迭代中要处理的输入数据数量。
    constexpr bool kDirectIO = Ktraits::kDirectIO;  // 定义是否为直接IO
    using input_t = typename Ktraits::input_t;  // 使用输入数据类型
    using weight_t = typename Ktraits::weight_t;  // 使用权重数据类型
    using scan_t = typename Ktraits::scan_t;  // 使用扫描数据类型

    // Shared memory.   声明共享内存
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);  // 加载块的共享内存
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);  // 加载权重块的共享内存
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));  // 加载权重块的共享内存1
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);  // 存储块的共享内存
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);  // 扫描块的共享内存
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);  // 运行前缀的共享内存

    const int batch_id = blockIdx.x;  // 批次ID     0~bsz-1   0~9
    const int dim_id = blockIdx.y;  // 维度ID       0~(K*D-1) 0~767
    const int group_id = dim_id / (params.dim_ngroups_ratio);  // 组ID = dim_id / D     dim_id/192
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;  // 输入数据指针
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;  // delta数据指针
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;  // 权重A指针
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;  // 变量B指针
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;  // 变量C指针
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.n_chunks * params.dstate;

    float D_val = 0; // attention!
    if (params.D_ptr != nullptr) {
        D_val = reinterpret_cast<float *>(params.D_ptr)[dim_id];
    }
    float delta_bias = 0;
    if (params.delta_bias_ptr != nullptr) {
        delta_bias = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];
    }

    int dstate = params.dstate; // additional
    constexpr int kChunkSize = kNThreads * kNItems; // 定义块大小 代表每个线程块在一次迭代中处理的数据项总数。=线程块中的线程数量*每个线程处理的数据项数量
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNItems], delta_vals_load[kNItems];  // 声明输入值和加载的delta值数组
        __syncthreads();  // 同步线程
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);  // 加载输入数据 smem_load 的具体作用是作为共享内存的临时存储，用于从全局内存加载数据到共享内存，然后再由共享内存分发给每个线程
        if constexpr (!kDirectIO) { __syncthreads(); }  // 如果不是直接IO，同步线程
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);  // 加载delta数据
        u += kChunkSize;  // 更新输入数据指针
        delta += kChunkSize;  // 更新delta数据指针

        float delta_vals[kNItems], delta_u_vals[kNItems], out_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            float u_val = float(u_vals[i]);  // 转换输入值为浮点型
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;  // 计算delta值
            if (params.delta_softplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];  // 计算softplus值
            }
            delta_u_vals[i] = delta_vals[i] * u_val;  // 计算delta_u值
            out_vals[i] = D_val * u_val;  // 计算输出值
        }

        __syncthreads();  // 同步线程
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            constexpr float kLog2e = M_LOG2E;  // 定义log2(e)
            weight_t A_val = A[state_idx * params.A_dstate_stride];  // 获取A值
            A_val *= kLog2e;  // 计算A值
            weight_t B_vals[kNItems], C_vals[kNItems];  // 声明B值和C值数组
            load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));  // 加载权重B
            load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight1, (params.seqlen - chunk * kChunkSize));  // 加载权重C
            __syncthreads();
            scan_t thread_data[kNItems];  // 声明线程数据数组
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                thread_data[i] = make_float2(exp2f(delta_vals[i] * A_val), B_vals[i] * delta_u_vals[i]);    // 计算 Ahat Bhat
                if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct  // 如果不是偶长度
                    if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                        thread_data[i] = make_float2(1.f, 0.f);  // 设置线程数据
                    }
                }
            }
            // Initialize running total
            scan_t running_prefix;
            // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
            running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
            // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
            );
            // There's a syncthreads in the scan op, so we don't need to sync here.
            // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
            if (threadIdx.x == 0) {
                smem_running_prefix[state_idx] = prefix_op.running_prefix;
                x[chunk * params.dstate + state_idx] = prefix_op.running_prefix;
            }
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                out_vals[i] += thread_data[i].y * C_vals[i];
            }

            // additional
            float thread_data_y[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                thread_data_y[i] = thread_data[i].y;
            }
            input_t *h = reinterpret_cast<input_t *>(params.h_ptr) + batch_id * params.h_batch_stride
                + dim_id * params.h_d_stride + chunk * kChunkSize + state_idx * params.h_dstate_stride;
            __syncthreads();
            store_output<Ktraits>(h, thread_data_y, smem_store, params.seqlen - chunk * kChunkSize); 
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * params.out_d_stride + chunk * kChunkSize;     // 获取输出数据指针
        __syncthreads();
        store_output<Ktraits>(out, out_vals, smem_store, params.seqlen - chunk * kChunkSize);
        Bvar += kChunkSize;
        Cvar += kChunkSize;
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, input_t, weight_t>;    // 定义特征类型
        constexpr int kSmemSize = Ktraits::kSmemSize + Ktraits::MaxDState * sizeof(typename Ktraits::scan_t);   // 计算共享内存大小 TODO
        dim3 grid(params.batch, params.dim);    // 定义网格大小
        auto kernel = &selective_scan_fwd_kernel<Ktraits>;  // 获取核函数指针
        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);    // 启动核函数
        C10_CUDA_KERNEL_LAUNCH_CHECK(); // 检查核函数启动是否成功
    });
}

template<int knrows, typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    } else {
        selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
}
