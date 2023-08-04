#include <stdio.h>
#include <immintrin.h>
#include "net_virtualvoid_llama2_VectMult.h"

// reference implementation, not too bad with -O3 -march=native
JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_matMul2
  (JNIEnv *env, jclass, jobject A, jfloatArray v, jfloatArray dest) {
    jint dim1 = (*env)->GetArrayLength(env, dest);
    jint dim2 = (*env)->GetArrayLength(env, v);
    float *va = (*env)->GetPrimitiveArrayCritical(env, v, 0);
    float *desta = (*env)->GetPrimitiveArrayCritical(env, dest, 0);
    float *fb = (*env)->GetDirectBufferAddress(env, A);

    for (int i = 0; i < dim1; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim2; j++) {
            sum += fb[i * dim2 + j] * va[j];
        }
        desta[i] = sum;
    }

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, v, va, 0);
}

// https://stackoverflow.com/a/23190168/7647
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}


// manual try to evaluate the effect of loop unrolling
JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_matMul
  (JNIEnv *env, jclass, jobject A, jfloatArray v, jfloatArray dest) {
    jint dim1 = (*env)->GetArrayLength(env, dest);
    jint dim2 = (*env)->GetArrayLength(env, v);
    float *va = (*env)->GetPrimitiveArrayCritical(env, v, 0);
    float *desta = (*env)->GetPrimitiveArrayCritical(env, dest, 0);
    float *fb = (*env)->GetDirectBufferAddress(env, A);

    for (int i = 0; i < dim1; i++) {
        float *x = fb + i * dim2;
        float *y = va;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        for (int j = 0; j < dim2; j += 32) {
            __m256 a0 = _mm256_loadu_ps(x + j);
            __m256 a1 = _mm256_loadu_ps(x + j + 8);
            __m256 a2 = _mm256_loadu_ps(x + j + 16);
            __m256 a3 = _mm256_loadu_ps(x + j + 24);
            __m256 b0 = _mm256_loadu_ps(y + j );
            __m256 b1 = _mm256_loadu_ps(y + j + 8);
            __m256 b2 = _mm256_loadu_ps(y + j + 16);
            __m256 b3 = _mm256_loadu_ps(y + j + 24);
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        }
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);
        float sumf = _mm256_reduce_add_ps(sum0);

        desta[i] = sumf;
    }

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, v, va, 0);
}
