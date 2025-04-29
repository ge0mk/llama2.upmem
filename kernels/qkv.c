#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "math.h"
#include "model_config.h"

float __mram_noinit wq[DIM * NR_TASKLETS * 2];
float __mram_noinit wk[DIM * NR_TASKLETS * 2];
float __mram_noinit wv[DIM * NR_TASKLETS * 2];
float __mram_noinit x[DIM];

float __mram_noinit q[NR_TASKLETS * 2];
float __mram_noinit k[NR_TASKLETS * 2];
float __mram_noinit v[NR_TASKLETS * 2];

__mram_noinit struct {
  uint32_t dpu;
  uint32_t pos;
} data;

BARRIER_INIT(barrier, NR_TASKLETS);

int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

  float *wram_w = mem_alloc(DIM * sizeof(float));
  float *wram_x = mem_alloc(DIM * sizeof(float));
  float *wram_q = mem_alloc(2 * sizeof(float));
  float *wram_k = mem_alloc(2 * sizeof(float));
  float *wram_v = mem_alloc(2 * sizeof(float));

  mram_read(x, wram_x, DIM * sizeof(float));

  // qkv matmuls
  for (size_t i = 0; i < 2; i++) {
    const size_t offset = tasklet_id * 2 + i;
    mram_read(wq + offset * DIM, wram_w, DIM * sizeof(float));
    wram_q[i] = dot(wram_w, wram_x, DIM);
    mram_read(wk + offset * DIM, wram_w, DIM * sizeof(float));
    wram_k[i] = dot(wram_w, wram_x, DIM);
    mram_read(wv + offset * DIM, wram_w, DIM * sizeof(float));
    wram_v[i] = dot(wram_w, wram_x, DIM);
  }

  // RoPE relative positional encoding: complex-valued rotate q and k in
  // each head
  const size_t i = (data.dpu * NR_TASKLETS + tasklet_id) * 2;
  const size_t head_dim = i % HEAD_SIZE;
  const float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)HEAD_SIZE);
  const float val = data.pos * freq;
  const float fcr = cosf(val);
  const float fci = sinf(val);
  float v0, v1;

  v0 = wram_q[0];
  v1 = wram_q[1];
  wram_q[0] = v0 * fcr - v1 * fci;
  wram_q[1] = v0 * fci + v1 * fcr;

  v0 = wram_k[0];
  v1 = wram_k[1];
  wram_k[0] = v0 * fcr - v1 * fci;
  wram_k[1] = v0 * fci + v1 * fcr;

  mram_write(wram_q, q + tasklet_id * 2, 2 * sizeof(float));
  mram_write(wram_k, k + tasklet_id * 2, 2 * sizeof(float));
  mram_write(wram_v, v + tasklet_id * 2, 2 * sizeof(float));

  return 0;
}
