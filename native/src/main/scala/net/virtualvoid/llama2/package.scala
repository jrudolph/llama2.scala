package net.virtualvoid

package object llama2 {
  trait FloatBuffer {
    def get(i: Int): Float
    def get(dest: Array[Float], offset: Int, size: Int): Unit

    def duplicate(): FloatBuffer = this
    def slice(): FloatBuffer = this
    def position(i: Int): FloatBuffer
  }

  object FloatBuffer {

    import scalanative.unsafe

    def apply(ptr: unsafe.Ptr[Float], offset: Long): FloatBuffer = new FloatBuffer {
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

      def position(i: Int): FloatBuffer = apply(ptr, offset + i)
    }
  }
}