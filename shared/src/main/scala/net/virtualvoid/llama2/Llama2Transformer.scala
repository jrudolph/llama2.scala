package net.virtualvoid.llama2

trait Llama2Transformer {
  def step(token: Int, pos: Int, kv: KV): Array[Float]
}
