# A Scala 2 port of Andrej Karpathy's llama2.c

This is a Scala port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), a bare bones implementation
to run inference of models with a [Llama](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.

The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) and [`stories15M.bin`](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) in the current directory.

So far, there's nothing original to see here, just a port of the original code. I'm planning to do some experiments,
to raise the abstraction layer to a more reasonable level (e.g. remove all of the while-loops) but I wanted to start
with an almost verbatim port to get some intial performance numbers to see if it even makes sense to continue on the
JVM.

[![asciicast](https://asciinema.org/a/h7dJq7SOkmlCHmgI3DLRQBp58.svg)](https://asciinema.org/a/h7dJq7SOkmlCHmgI3DLRQBp58)

## Implementation details

There are currently two implementations:

 * `Llama2SimpleTransformer` which is a direct port of the original C code
 * `Llama2TensorTransformer` which uses a `Tensor` abstraction to make the code more readable

## Performance

Current numbers on my AMD Ryzen 7 4800H laptop:

| Implementation                                                | Tokens per second | Speedup |
|---------------------------------------------------------------|-------------------|---------|
| llama2.c single-thread                                        | 65                | 1.0x    |
| llama2.c multi-thread (OMP)                                   | ~350              | 5.5x    |
| Llama2SimpleTransformer on scala-native 0.4.14 vanilla        | 14                | 0.22x   |
| Llama2SimpleTransformer on scala-native 0.4.14 (native mmaps) | 50                | 0.77x   |
| Llama2SimpleTransformer on OpenJDK 11 single-thread           | 50                | 0.77x   |
| Llama2SimpleTransformer on GraalVM JDK 17 single-thread       | 61                | 0.94x   |

So, with a fast JVM, there's little slowdown compared to the original C code, even if the Scala code has not been optimized at all yet (aside from being written in a hardly bearable C-like fashion...).


## License

MIT
