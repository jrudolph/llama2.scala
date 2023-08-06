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

float vecdot(float *x, float *y, int dim2) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    //__m256 sum2 = _mm256_setzero_ps();
    //__m256 sum3 = _mm256_setzero_ps();
    for (int j = 0; j < dim2; j += 16) {
        //printf("j=%d\n", j);
        __m256 a0 = _mm256_loadu_ps(x + j);
        __m256 a1 = _mm256_loadu_ps(x + j + 8);
        //__m256 a2 = _mm256_loadu_ps(x + j + 16);
        //__m256 a3 = _mm256_loadu_ps(x + j + 24);
        __m256 b0 = _mm256_loadu_ps(y + j );
        __m256 b1 = _mm256_loadu_ps(y + j + 8);
        //__m256 b2 = _mm256_loadu_ps(y + j + 16);
        //__m256 b3 = _mm256_loadu_ps(y + j + 24);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
        //sum2 = _mm256_fmadd_ps(a2, b2, sum2);
        //sum3 = _mm256_fmadd_ps(a3, b3, sum3);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    //sum2 = _mm256_add_ps(sum2, sum3);
    //sum0 = _mm256_add_ps(sum0, sum2);
    return _mm256_reduce_add_ps(sum0);
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

        desta[i] = vecdot(x, y, dim2);
    }

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, v, va, 0);
}

JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_matMul4
  (JNIEnv *env, jclass, jobject A, jfloatArray v, jfloatArray dest) {
    jint dim1 = (*env)->GetArrayLength(env, dest);
    jint dim2 = (*env)->GetArrayLength(env, v);
    float *va = (*env)->GetPrimitiveArrayCritical(env, v, 0);
    float *desta = (*env)->GetPrimitiveArrayCritical(env, dest, 0);
    float *fb = (*env)->GetDirectBufferAddress(env, A);

    for (int i = 0; i < dim1; i+=2) {
        float *x = fb + i * dim2;
        float *y = va;

        __m256 sum00 = _mm256_setzero_ps();
        __m256 sum01 = _mm256_setzero_ps();
        __m256 sum10 = _mm256_setzero_ps();
        __m256 sum11 = _mm256_setzero_ps();
        //__m256 sum2 = _mm256_setzero_ps();
        //__m256 sum3 = _mm256_setzero_ps();
        for (int j = 0; j < dim2; j += 16) {
            //printf("j=%d\n", j);
            __m256 a00 = _mm256_loadu_ps(x + j);
            __m256 a01 = _mm256_loadu_ps(x + j + 8);
            __m256 a10 = _mm256_loadu_ps(x + j + dim2);
            __m256 a11 = _mm256_loadu_ps(x + j + dim2 + 8);
            //__m256 a2 = _mm256_loadu_ps(x + j + 16);
            //__m256 a3 = _mm256_loadu_ps(x + j + 24);
            __m256 b0 = _mm256_loadu_ps(y + j );
            __m256 b1 = _mm256_loadu_ps(y + j + 8);
            //__m256 b1 = _mm256_loadu_ps(y + j + 8);
            //__m256 b2 = _mm256_loadu_ps(y + j + 16);
            //__m256 b3 = _mm256_loadu_ps(y + j + 24);
            sum00 = _mm256_fmadd_ps(a00, b0, sum00);
            sum01 = _mm256_fmadd_ps(a01, b1, sum01);
            sum10 = _mm256_fmadd_ps(a10, b0, sum10);
            sum11 = _mm256_fmadd_ps(a11, b1, sum11);
            //sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            //sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            //sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        }
        sum00 = _mm256_add_ps(sum00, sum01);
        sum10 = _mm256_add_ps(sum10, sum11);
        //sum2 = _mm256_add_ps(sum2, sum3);
        //sum0 = _mm256_add_ps(sum0, sum2);

        desta[i] = _mm256_reduce_add_ps(sum00);
        desta[i + 1] = _mm256_reduce_add_ps(sum10);
    }

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, v, va, 0);
}

// manual try to evaluate the effect of loop unrolling
JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_matMul3
  (JNIEnv *env, jclass, jobject A, jfloatArray v, jfloatArray dest) {
    jint dim1 = (*env)->GetArrayLength(env, dest);
    jint dim2 = (*env)->GetArrayLength(env, v);
    float *va = (*env)->GetPrimitiveArrayCritical(env, v, 0);
    float *desta = (*env)->GetPrimitiveArrayCritical(env, dest, 0);
    float *fb = (*env)->GetDirectBufferAddress(env, A);

    for (int i = 0; i < dim1; i+=4) {
        float *x = fb + i * dim2;
        float *y = va;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (int j = 0; j < dim2; j += 8) {
            //printf("j=%d\n", j);
            __m256 a0 = _mm256_loadu_ps(x + j);
            __m256 a1 = _mm256_loadu_ps(x + j + dim2);
            __m256 a2 = _mm256_loadu_ps(x + j + dim2 * 2);
            __m256 a3 = _mm256_loadu_ps(x + j + dim2 * 3);
            //__m256 a2 = _mm256_loadu_ps(x + j + 16);
            //__m256 a3 = _mm256_loadu_ps(x + j + 24);
            __m256 b0 = _mm256_loadu_ps(y + j );
            //__m256 b1 = _mm256_loadu_ps(y + j + 8);
            //__m256 b2 = _mm256_loadu_ps(y + j + 16);
            //__m256 b3 = _mm256_loadu_ps(y + j + 24);
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b0, sum1);
            sum2 = _mm256_fmadd_ps(a2, b0, sum2);
            sum3 = _mm256_fmadd_ps(a3, b0, sum3);

            //sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            //sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            //sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        }
        //sum00 = _mm256_add_ps(sum00, sum01);
        //sum10 = _mm256_add_ps(sum10, sum11);
        //sum2 = _mm256_add_ps(sum2, sum3);
        //sum0 = _mm256_add_ps(sum0, sum2);

        desta[i] = _mm256_reduce_add_ps(sum0);
        desta[i + 1] = _mm256_reduce_add_ps(sum1);
        desta[i + 2] = _mm256_reduce_add_ps(sum2);
        desta[i + 3] = _mm256_reduce_add_ps(sum3);
    }

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, v, va, 0);
}

void matmulQ8(const int8_t *restrict qa, const float *restrict qaf, const int8_t *restrict qv, const float *restrict qvf, float *restrict desta, const int dim1, const int dim2) {
    int K = 32;
    int numBlocksV = dim2 / K;
    for (int i = 0; i < dim1; i++) {
        int8_t * a = qa + i * dim2;
        float *af = qaf + i * dim2 / K;

        float sum = 0.0f;
        for (int j = 0; j < numBlocksV; j++) {
            int sumb = 0;
            for (int k = 0; k < K; k++) {
                sumb += a[j * K + k] * qv[j * K + k];
                //if(i == 0 && j == 0)
                //    printf("i=%d, j=%d, k=%d, sumb=%d qa=%d qv=%d\n", i, j, k, sumb, i * dim2 + j * K + k, qv[j * K + k]);
            }

            sum += ((float)sumb) * af[j] * qvf[j];
        }
        desta[i] = sum;
    }
}
void matmulQ8_avx2(int8_t *qa, float *qaf, int8_t *qv, float *qvf, float *desta, int dim1, int dim2) {
    int K = 32;
    int numBlocksV = dim2 / K;
    for (int i = 0; i < dim1; i++) {
        // keep block sums vectorized
        __m256 sv = _mm256_setzero_ps();
        for (int j = 0; j < numBlocksV; j++) { // assumes K = 32 = number of lanes in m256i
            __m256 f = _mm256_set1_ps(qaf[i * dim2 / K + j] * qvf[j]);
            __m256i a = _mm256_loadu_si256(qa + i * dim2 + j * K);
            __m256i v = _mm256_loadu_si256(qv + j * K);

            // take absolute value of a and propagate result sign to v2 (maddubs works with unsigned * signed)
            __m256i a2 = _mm256_sign_epi8(a, a);
            __m256i v2 = _mm256_sign_epi8(v, a);

            // sum with saturation to i16
            __m256i s = _mm256_maddubs_epi16(a2, v2);

            // sum the 16-bit integers to 32-bit integers
            __m256i s2 = _mm256_madd_epi16(s, _mm256_set1_epi16(1));
            __m256 s3 = _mm256_cvtepi32_ps(s2);
            // scale with the factor
            sv = _mm256_fmadd_ps(f, s3, sv);
        }
        // reduce at the end
        desta[i] = _mm256_reduce_add_ps(sv);
    }
}

JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_matMulQ8
  (JNIEnv * env, jclass, jbyteArray quantizedA, jfloatArray quantizedFactor, jbyteArray quantizedV, jfloatArray quantizedFactorV, jfloatArray dest) {
    jint dim1 = (*env)->GetArrayLength(env, dest);
    jint dim2 = (*env)->GetArrayLength(env, quantizedV);
    int8_t *qa = (*env)->GetPrimitiveArrayCritical(env, quantizedA, 0);
    float *qaf = (*env)->GetPrimitiveArrayCritical(env, quantizedFactor, 0);
    int8_t *qv = (*env)->GetPrimitiveArrayCritical(env, quantizedV, 0);
    float *qvf = (*env)->GetPrimitiveArrayCritical(env, quantizedFactorV, 0);
    float *desta = (*env)->GetPrimitiveArrayCritical(env, dest, 0);

    matmulQ8_avx2(qa, qaf, qv, qvf, desta, dim1, dim2);

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedFactorV, qvf, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedV, qv, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedFactor, qaf, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedA, qa, 0);
}