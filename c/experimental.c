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