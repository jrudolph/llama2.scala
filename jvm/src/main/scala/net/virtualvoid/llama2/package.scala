package net.virtualvoid

package object llama2 {
  type FloatBuffer = java.nio.FloatBuffer

  object FloatBuffer {
    def fromNioBuffer(buffer: java.nio.FloatBuffer): FloatBuffer = buffer
  }
}