package net.virtualvoid.llama2;

import java.io.File;
import java.nio.ByteBuffer;

public class VectMult {
    static {
        System.load(new File("../c/a.out").getAbsolutePath());
    }

    public static native void matMul(java.nio.FloatBuffer A, float[] v, float[] dest);

    public static native void matMulQ8(byte[] quantizedA, float[] quantizedFactorA, byte[] quantizedV, float[] quantizedFactorV, float[] dest);

    public static native void matMulQ8New(
            java.nio.Buffer quantizedA,
            java.nio.Buffer quantizedFactorA,
            java.nio.Buffer quantizedV,
            java.nio.Buffer quantizedFactorV,
            float[] dest,
            int dim1, int dim2
    );

    public static native ByteBuffer allocateAligned(int size, int alignment);/* {
        return ByteBuffer.allocateDirect(size);
    }*/
    public static native void freeAligned(java.nio.Buffer buffer);
}