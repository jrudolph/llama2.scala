package net.virtualvoid.llama2;

import java.io.File;

public class VectMult {
    static {
        System.load(new File("../c/a.out").getAbsolutePath());
    }

    public static native void matMul(java.nio.FloatBuffer A, float[] v, float[] dest);

    public static native void matMulQ8(byte[] quantizedA, float[] quantizedFactorA, byte[] quantizedV, float[] quantizedFactorV, float[] dest);
    public static native void matMulQ4(byte[] quantizedA, float[] quantizedFactorA, byte[] quantizedV, float[] quantizedFactorV, float[] dest);
    public static native void matMulQ4_buffer(java.nio.Buffer qBuffer, byte[] quantizedV, float[] quantizedFactorV, float[] dest);
}