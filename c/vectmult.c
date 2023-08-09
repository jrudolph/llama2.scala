#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include "net_virtualvoid_llama2_VectMult.h"

int global_parallelism = 1;

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

float vecdot(float *restrict x, float *restrict y, int dim2) {
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

    int i;
    #pragma omp parallel for private(i) num_threads(global_parallelism)
    for (i = 0; i < dim1; i++) {
        float *x = fb + i * dim2;
        float *y = va;

        desta[i] = vecdot(x, y, dim2);
    }

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, v, va, 0);
}

void matmulQ8(const int8_t *restrict qa, const float *restrict qaf, const int8_t *restrict qv, const float *restrict qvf, float *restrict desta, const int dim1, const int dim2) {
    int K = 32;
    int numBlocksV = dim2 / K;
    int i;
    #pragma omp parallel for private(i) num_threads(global_parallelism)
    for (i = 0; i < dim1; i++) {
        int8_t * a = qa + i * dim2;
        float *af = qaf + i * dim2 / K;

        float sum = 0.0f;
        for (int j = 0; j < numBlocksV; j++) {
            int sumb = 0;
            for (int k = 0; k < K; k++) {
                sumb += a[j * K + k] * qv[j * K + k];
            }

            sum += ((float)sumb) * af[j] * qvf[j];
        }
        desta[i] = sum;
    }
}
void matmulQ8_avx2(int8_t *restrict qa, float *restrict qaf, int8_t *restrict qv, float *restrict qvf, float *restrict desta, int dim1, int dim2) {
    int K = 32;
    int numBlocksV = dim2 / K;
    int i;
    #pragma omp parallel for private(i) num_threads(global_parallelism)
    for (i = 0; i < dim1; i++) {
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

void matmulQ4(const int8_t *restrict qa, const float *restrict qaf, const int8_t *restrict qv, const float *restrict qvf, float *restrict desta, const int dim1, const int dim2)
{
    int K = 32;
    int numBlocksV = dim2 / K;
    int i;
    #pragma omp parallel for private(i) num_threads(global_parallelism)
    for (i = 0; i < dim1; i++) {
        int8_t * a = qa + i * dim2 / 2;
        float *af = qaf + i * dim2 / K;

        float sum = 0.0f;
        for (int j = 0; j < numBlocksV; j++) {
            int sumb = 0;
            for (int k = 0; k < K / 2; k++) {
                int8_t e = a[j * K / 2 + k];
                int8_t x0 = (e & 0x0f) - 8;
                int8_t x1 = ((e & 0xf0) >> 4) - 8;

                int8_t v0 = qv[j * K + k];
                int8_t v1 = qv[j * K + k + K / 2];

                sumb += (x0 * v0) + (x1 * v1);
            }

            sum += ((float)sumb) * af[j] * qvf[j];
        }
        desta[i] = sum;
    }
}

// some infra from ggml (MIT license)
static inline float fp32_from_bits(uint32_t w) {
    return *(float*)&w;
}

static inline uint32_t fp32_to_bits(float f) {
    return *(uint32_t*)&f;
}

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

void matmulQ4_buffer(const int8_t *restrict buffer, const int8_t *restrict qv, const float *restrict qvf, float *restrict desta, const int dim1, const int dim2)
{
    int K = 32;
    int numBlocksV = dim2 / K;
    int i;
    #pragma omp parallel for private(i) num_threads(global_parallelism)
    for (i = 0; i < dim1; i++) {
        int blockOff = i * dim2 / K;

        float sum = 0.0f;
        for (int j = 0; j < numBlocksV; j++) {
            int block = blockOff + j;
            int8_t *bp = buffer + block * 18;
            int8_t *a = bp + 2;
            uint16_t rawFactor = *((uint16_t*)(bp));
            float af = fp16_to_fp32(rawFactor);
            float qf = qvf[j];

            int sumb = 0;
            for (int k = 0; k < K / 2; k++) {
                int8_t e = a[k];
                int8_t x0 = (e & 0x0f) - 8;
                int8_t x1 = ((e & 0xf0) >> 4) - 8;

                int8_t v0 = qv[j * K + k];
                int8_t v1 = qv[j * K + k + K / 2];

                sumb += (x0 * v0) + (x1 * v1);
            }

            sum += ((float)sumb) * af * qf;
        }
        desta[i] = sum;
    }
}

// copied verbatim from ggml (MIT license)
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
static inline __m256i bytes_from_nibbles_32(const int8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

void matmulQ4_avx2(const int8_t *restrict qa, const float *restrict qaf, const int8_t *restrict qv, const float *restrict qvf, float *restrict desta, const int dim1, const int dim2)
{
    int K = 32;
    int numBlocksV = dim2 / K;
    int i;
    #pragma omp parallel for private(i) num_threads(global_parallelism)
    for (i = 0; i < dim1; i++) {
        int8_t * a = qa + i * dim2 / 2;
        float *af = qaf + i * dim2 / K;

        // keep block sums vectorized
        __m256 sv = _mm256_setzero_ps();
        for (int j = 0; j < numBlocksV; j++) { // assumes K = 32 = number of lanes in m256i
            __m256 f = _mm256_set1_ps(qaf[i * dim2 / K + j] * qvf[j]);

            //__m256i a = _mm256_loadu_si256(qa + i * dim2 + j * K);
            __m256i a = bytes_from_nibbles_32(qa + (i * dim2 + j * K)/2);

            // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
            const __m256i off = _mm256_set1_epi8( 8 );
            a = _mm256_sub_epi8( a, off );

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

void matmulQ4_avx2_buffer(const void *restrict buffer, const int8_t *restrict qv, const float *restrict qvf, float *restrict desta, const int dim1, const int dim2)
{
    int K = 32;
    int numBlocksV = dim2 / K;
    int i;
    #pragma omp parallel for private(i) num_threads(global_parallelism)
    for (i = 0; i < dim1; i++) {
        int blockOff = i * dim2 / K;

        // keep block sums vectorized
        __m256 sv = _mm256_setzero_ps();
        for (int j = 0; j < numBlocksV; j++) { // assumes K = 32 = number of lanes in m256i
            int block = blockOff + j;
            int8_t *bp = buffer + block * 18;
            int8_t *qa = bp + 2;
            uint16_t rawFactor = *((uint16_t*)(bp));
            float af = fp16_to_fp32(rawFactor);
            float qf = qvf[j];

            __m256 f = _mm256_set1_ps(af * qf);

            __m256i a = bytes_from_nibbles_32(qa);

            // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
            const __m256i off = _mm256_set1_epi8( 8 );
            a = _mm256_sub_epi8( a, off );

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

JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_matMulQ4_1buffer
  (JNIEnv *env, jclass, jobject qBuffer, jbyteArray quantizedV, jfloatArray quantizedFactorV, jfloatArray dest)
{
    jint dim1 = (*env)->GetArrayLength(env, dest);
    jint dim2 = (*env)->GetArrayLength(env, quantizedV);
    void *q = (*env)->GetDirectBufferAddress(env, qBuffer);
    int8_t *qv = (*env)->GetPrimitiveArrayCritical(env, quantizedV, 0);
    float *qvf = (*env)->GetPrimitiveArrayCritical(env, quantizedFactorV, 0);
    float *desta = (*env)->GetPrimitiveArrayCritical(env, dest, 0);

    matmulQ4_avx2_buffer(q, qv, qvf, desta, dim1, dim2);

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedFactorV, qvf, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedV, qv, 0);
}


JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_matMulQ4
  (JNIEnv * env, jclass, jbyteArray quantizedA, jfloatArray quantizedFactor, jbyteArray quantizedV, jfloatArray quantizedFactorV, jfloatArray dest)
{
    jint dim1 = (*env)->GetArrayLength(env, dest);
    jint dim2 = (*env)->GetArrayLength(env, quantizedV);
    int8_t *qa = (*env)->GetPrimitiveArrayCritical(env, quantizedA, 0);
    float *qaf = (*env)->GetPrimitiveArrayCritical(env, quantizedFactor, 0);
    int8_t *qv = (*env)->GetPrimitiveArrayCritical(env, quantizedV, 0);
    float *qvf = (*env)->GetPrimitiveArrayCritical(env, quantizedFactorV, 0);
    float *desta = (*env)->GetPrimitiveArrayCritical(env, dest, 0);

    matmulQ4_avx2(qa, qaf, qv, qvf, desta, dim1, dim2);

    (*env)->ReleasePrimitiveArrayCritical(env, dest, desta, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedFactorV, qvf, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedV, qv, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedFactor, qaf, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, quantizedA, qa, 0);
}

JNIEXPORT void JNICALL Java_net_virtualvoid_llama2_VectMult_setParallelism
  (JNIEnv *, jclass, jint parallelism)
{
    global_parallelism = parallelism;
}