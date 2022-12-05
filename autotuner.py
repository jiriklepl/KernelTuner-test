#!/usr/bin/env python3
import numpy as np
import kernel_tuner as kt

kernel_string = """
#include <cstdlib>
#include <cstdint>
#include <sys/time.h>
#include <math.h>

#define N 1260

float mmm_kernel(std::int32_t *c, std::int32_t *a, std::int32_t *b)
{
    int n = block_size_x * (N / block_size_x);
    long start, end;
    timeval timecheck;

    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    volatile int _init = 0;
    int init = _init;


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            a[i * N +j] = b[i * N + j] = _init;

        a[i * N + i] = b[i * N + i] = 1;
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            c[i * N + j] = _init;

    int sum = 0;

    for (int j1 = 0; j1 < n; j1 += block_size_x)
    {
        for (int k1 = 0; k1 < n; k1 += block_size_x)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = j1; j < j1 + block_size_x; j++)
                {
                    sum = c[i * N + j];

                    for (int k = k1; k < k1 + block_size_x; k++)
                    {
                        sum += a[i * N + k] * b[k * N + j];
                    }

                    c[i * N + j] = sum;
                }
            }
        }
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            init += c[i * N + j];
    
    _init = init;

    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    
    if (_init == N)
        return (float)(end - start);
    else
        return INFINITY;
}
"""

size = 1260 ** 2

a = np.random.randn(size).astype(np.int32)
b = np.random.randn(size).astype(np.int32)
c = np.zeros_like(b, dtype=np.int32)
n = np.int32(size)

args = [c, a, b]

tune_params = dict()
tune_params["block_size_x"] = range(8, 128)

kt.tune_kernel("mmm_kernel", kernel_string, size, args, tune_params, strategy="dual_annealing")
