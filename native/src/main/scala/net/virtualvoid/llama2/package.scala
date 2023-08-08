package net.virtualvoid

package object llama2 {
  trait FloatBuffer {
    def get(i: Int): Float
    def get(dest: Array[Float], offset: Int, size: Int): Unit

    def duplicate(): FloatBuffer = this
    def slice(): FloatBuffer = this
    def position(i: Int): FloatBuffer

    def limit: Int
  }

  object FloatBuffer {

    import scalanative.unsafe

    def apply(ptr: unsafe.Ptr[Float], offset: Long, size: Int): FloatBuffer = new FloatBuffer {
      def get(i: Int): Float = ptr(offset + i)
      def get(dest: Array[Float], offset: Int, size: Int): Unit = {
        // FIXME: is more direct access to array possible?

        val source = duplicate().position(offset * size).slice()
        var i = 0
        while (i < size) {
          dest(i) = source.get(i)
          i += 1
        }
      }

      def position(i: Int): FloatBuffer = apply(ptr, offset + i, size - i)

      def limit: Int = size
    }

    def fromNioBuffer(buffer: java.nio.FloatBuffer): FloatBuffer = new FloatBuffer {
      def get(i: Int): Float = buffer.get(i)
      def get(dest: Array[Float], offset: Int, size: Int): Unit = buffer.get(dest, offset, size)
      def position(i: Int): FloatBuffer = {
        buffer.position(i)
        this
      }
      def limit: Int = buffer.limit()
    }
  }
}