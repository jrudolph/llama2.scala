# A Scala 2 port of Andrej Karpathy's llama2.c

This is a Scala port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), a bare bones implementation
to run inference of models with a [Llama](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.

The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) and [`stories15M.bin`](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) in the current directory.

This started as a port of the original code in pure Scala. Later, more high-level abstractions were
added and low-level C kernels with AVX2 intrinsics to speed up matrix multiplication.

[![asciicast](https://asciinema.org/a/h7dJq7SOkmlCHmgI3DLRQBp58.svg)](https://asciinema.org/a/h7dJq7SOkmlCHmgI3DLRQBp58)

## Features:
 * Two implementations of the model architecture are available:
   * `Llama2SimpleTransformer` which is a direct port of the original C code
   * `Llama2TensorTransformer` which uses a `Tensor` abstraction to make the code more readable
 * Different matrix multiplication kernels:
   * `ScalaMathImplementation` (direct port of llama2.c in pure Scala)
   * `AVX2MathImplementation` (using JNI to call kernels written with C SIMD intrinsics)
 * Models are mapped into memory to avoid loading them into the JVM heap
 * Quantization modes:
   * use the weights as given in the model
   * Q8: quantize weights after loading to 8 bits (all but rmsnorm)
   * Q4: quantize weights after loading to 4 bits (all but rmsnorm)
 * Multi-threading:
   * `AVX2MathImplementation` uses OpenMP
 * Support for loading ggml models
   * only weights in Q4_0 and FP32 are supported
 * scala-native support (mostly broken right now)

## Performance

Current numbers run with version [08c65d04](https://github.com/jrudolph/llama2.scala/tree/08c65d04c0a3a4345510db289779e3243bcf7ff9) on my AMD Ryzen 7 4800H laptop with GraalVM JDK 17.

Implementations:
 * Scala = `Llama2TensorTransformer` with `ScalaMathImplementation`
 * native-avx2 = `Llama2TensorTransformer` with `AVX2MathImplementation` (using JNI to call kernels written with C SIMD intrinsics)
 * llama2.c = as of [94a3a5e0](https://github.com/karpathy/llama2.c/tree/94a3a5e0a5f63f06ffbfa7ec5452553eedafc215)
 * llama.cpp = as of [d783f798](https://github.com/ggerganov/llama.cpp/tree/d783f7982e0e823a2626a9956359c0d36c1a7e21)
 * scala-native = Using scala-native 0.4.14 with the `Llama2SimpleTransformer` implementation

Notes:
* Approximate speedups are:
    * pure Scala -> AVX2: > 10x
    * FP32 -> Q8/Q4 (in Scala): same speed
    * FP32 -> Q8 (AVX2): ~ 2x
    * Q8 -> Q4 (AVX2) on one thread: same speed
    * Q4 1 thread -> 6 threads on small models: ~ 2x
    * Q4 1 thread -> 6 threads on large models: ~ 3x
* The pure Scala mode GraalVM JDK 17 is only competitive with a llama2.c version compiled with `-O3`.
  Using `-Ofast` on C already makes a huge difference. Would be interesting to see the exact differences
  between JIT compiled code and gcc output with `-Ofast`. Not sure if something like `-Ofast` (using less strict
  FP math) is possible on the JVM.
* Using (i.e. mostly adapting from llama.cpp) kernels in C with SIMD intrinsics and calling them with JNI
  from Scala makes a huge difference. It is easy to do locally, but, of course, much harder to do in a
  portable way.
* As expected, quantization gives another boost. Interesting that it is more pronounced when multi-threading
  is enabled.
* OMP-based multithreading is simple to use from C and helps a lot. Scaling is not perfect, with benefits diminishing
  sharply after using more than 6 (of 8) threads.
* Multithreading is interesting, as the task units are quite small (one matrix multiplication) and overheads can be
  significant.
* Quantization only helps with SIMD optimization. Only SIMD will give access to byte-wise (int8) operations and
  *decreasing* the data type size will *increase* the number of lanes per vector with the same factor. It is unclear
  why going from 32-bit to 8-bit gives only a 2x speedup while being able to run 4x more operations in parallel. One
  explanation could be that you need more instructions because of the added complexity of quantization.


| Model                      | Quantization | Implementation              | Threads | tok / s |
|----------------------------|--------------|-----------------------------|---------|---------|
| stories15M.bin             | Q4           | native-avx2                 | 1       | 494     |
| stories15M.bin             | Q4           | native-avx2                 | 6       | 931     |
| stories15M.bin             | Q4           | Scala                       | 1       | 65      |
| stories15M.bin             | Q8           | native-avx2                 | 1       | 533     |
| stories15M.bin             | Q8           | native-avx2                 | 6       | 800     |
| stories15M.bin             | Q8           | Scala                       | 1       | 57      |
| stories15M.bin             | none         | native-avx2                 | 1       | 374     |
| stories15M.bin             | none         | native-avx2                 | 6       | 677     |
| stories15M.bin             | none         | Scala                       | 1       | 66      |
| stories15M.bin             | none         | scala-native vanilla        | 1       | 14      |
| stories15M.bin             | none         | scala-native (native mmaps) | 1       | 50      |
| stories42M.bin             | Q4           | native-avx2                 | 1       | 223     |
| stories42M.bin             | Q4           | native-avx2                 | 6       | 497     |
| stories42M.bin             | Q4           | Scala                       | 1       | 24      |
| stories42M.bin             | Q8           | native-avx2                 | 1       | 229     |
| stories42M.bin             | Q8           | native-avx2                 | 6       | 407     |
| stories42M.bin             | Q8           | Scala                       | 1       | 22      |
| stories42M.bin             | none         | native-avx2                 | 1       | 137     |
| stories42M.bin             | none         | native-avx2                 | 6       | 243     |
| stories42M.bin             | none         | Scala                       | 1       | 24      |
| stories42M.bin             | none         | llama2.c / run              | 1       | 21      |
| stories42M.bin             | none         | llama2.c / runfast          | 1       | 69      |
| stories42M.bin             | none         | llama2.c / runomp           | 1       | 98      |
| stories42M.bin             | none         | llama2.c / runomp           | 6       | 195     |
| stories110M.bin            | Q4           | native-avx2                 | 1       | 95      |
| stories110M.bin            | Q4           | native-avx2                 | 6       | 239     |
| stories110M.bin            | Q4           | Scala                       | 1       | 9.6     |
| stories110M.bin            | Q8           | native-avx2                 | 1       | 99      |
| stories110M.bin            | Q8           | native-avx2                 | 6       | 183     |
| stories110M.bin            | Q8           | Scala                       | 1       | 8.4     |
| stories110M.bin            | none         | native-avx2                 | 1       | 50      |
| stories110M.bin            | none         | native-avx2                 | 6       | 85      |
| stories110M.bin            | none         | Scala                       | 1       | 8.9     |
| stories110M.bin            | none         | llama2.c / runomp           | 6       | 77      |
| llama2_7b.bin              | Q4           | native-avx2                 | 1       | 2.0     |
| llama2_7b.bin              | Q4           | native-avx2                 | 6       | 6.5     |
| llama2_7b.bin              | Q4           | Scala                       | 1       | 0.16    |
| llama2_7b.bin              | Q8           | native-avx2                 | 1       | 1.9     |
| llama2_7b.bin              | Q8           | native-avx2                 | 6       | 4.46    |
| llama2_7b.bin              | Q8           | Scala                       | 1       | 0.14    |
| llama-2-7b.ggmlv3.q4_0.bin | as provided  | native-avx2                 | 1       | 1.66    |
| llama-2-7b.ggmlv3.q4_0.bin | as provided  | native-avx2                 | 6       | 6.71    |
| llama-2-7b.ggmlv3.q4_0.bin | as provided  | Scala                       | 1       | 0.13    |
| llama-2-7b.ggmlv3.q4_0.bin | as provided  | llama.cpp                   | 1       | 2.0     |
| llama-2-7b.ggmlv3.q4_0.bin | as provided  | llama.cpp                   | 6       | 8.1     |


## License

MIT
