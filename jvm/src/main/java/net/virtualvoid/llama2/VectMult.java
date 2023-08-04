package net.virtualvoid.llama2;

import java.io.File;

public class VectMult {
    static {
        System.load(new File("../c/a.out").getAbsolutePath());
    }

    public static native void matMul(java.nio.FloatBuffer A, float[] v, float[] dest);
}