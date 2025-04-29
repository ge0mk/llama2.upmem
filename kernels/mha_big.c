#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "math.h"

#define SEQ_LEN 256
#define HEAD_SIZE 4096

float __mram_noinit att[SEQ_LEN];
float __mram_noinit q[HEAD_SIZE];
float __mram_noinit kc[SEQ_LEN * HEAD_SIZE];
float __mram_noinit vc[HEAD_SIZE * SEQ_LEN];
float __mram_noinit x[HEAD_SIZE];

__mram_noinit struct {
  float scale;
  uint32_t pos;
} data;

BARRIER_INIT(barrier, NR_TASKLETS);
BARRIER_INIT(softmax_barrier, NR_TASKLETS);
BARRIER_INIT(softmax_done_barrier, NR_TASKLETS);

void softmax() {
  float *wram_att = mem_alloc(SEQ_LEN * sizeof(float));
  mram_read(att, wram_att, SEQ_LEN * sizeof(float));

  float max_val = wram_att[0];
  for (int i = 1; i < SEQ_LEN; i++) {
    if (wram_att[i] > max_val) {
      max_val = wram_att[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < SEQ_LEN; i++) {
    wram_att[i] = expf(wram_att[i] - max_val);
    sum += wram_att[i];
  }

  for (int i = 0; i < SEQ_LEN; i++) {
    wram_att[i] /= sum;
  }

  mram_write(wram_att, att, SEQ_LEN * sizeof(float));
}

size_t chunk_size_for_tasklet(size_t data_size, size_t tasklet_id) {
  return (data_size / NR_TASKLETS) +
         (tasklet_id < (data_size % NR_TASKLETS) / 2 ? 2 : 0);
}

size_t chunk_offset_for_tasklet(size_t data_size, size_t tasklet_id) {
  size_t chunk_offset = 0;
  for (size_t t = 0; t < tasklet_id; t++) {
    chunk_offset += chunk_size_for_tasklet(data_size, t);
  }
  return chunk_offset;
}

int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

  {
    const size_t chunk_size = chunk_size_for_tasklet(SEQ_LEN, tasklet_id);
    const size_t chunk_offset = chunk_offset_for_tasklet(SEQ_LEN, tasklet_id);

    const size_t segment_size = 128;
    const size_t segment_count = HEAD_SIZE / segment_size;

    float *wram_q = mem_alloc(segment_size * sizeof(float));
    float *wram_kc = mem_alloc(segment_size * sizeof(float));
    float *wram_att = mem_alloc(chunk_size * sizeof(float));

    for (size_t i = 0; i < chunk_size; i++) {
      const size_t t = chunk_offset + i;
      wram_att[i] = t <= data.pos ? 0.0f : -INFINITY;
    }

    for (size_t s = 0; s < segment_count; s++) {
      const size_t segment_offset = s * segment_size;
      mram_read(q + segment_offset, wram_q, segment_size * sizeof(float));

      for (size_t i = 0, t = chunk_offset; i < chunk_size && t <= data.pos;
           i++, t++) {
        mram_read(kc + t * HEAD_SIZE + segment_offset, wram_kc,
                  segment_size * sizeof(float));
        wram_att[i] += dot(wram_q, wram_kc, segment_size);
      }
    }

    for (size_t i = 0; i < chunk_size; i++) {
      wram_att[i] /= data.scale;
    }

    mram_write(wram_att, att + chunk_offset, chunk_size * sizeof(float));
  }

  barrier_wait(&softmax_barrier);

  if (tasklet_id == 0) {
    mem_reset();
    softmax();
    mem_reset();
  }

  barrier_wait(&softmax_done_barrier);

  {
    const size_t chunk_size = chunk_size_for_tasklet(HEAD_SIZE, tasklet_id);
    const size_t chunk_offset = chunk_offset_for_tasklet(HEAD_SIZE, tasklet_id);

    const size_t segment_size = 64;
    const size_t segment_count =
        chunk_size / segment_size + (chunk_size % segment_size != 0);

    const size_t vc_segment_size = 64;
    const size_t vc_segment_count = SEQ_LEN / vc_segment_size;

    // printf("[%i] %i %i - %i %i\n", tasklet_id, chunk_size, chunk_offset,
    //        segment_size, segment_count);

    float *wram_vc = mem_alloc(vc_segment_size * sizeof(float));
    float *wram_x = mem_alloc(segment_size * sizeof(float));
    float *wram_att = mem_alloc(SEQ_LEN * sizeof(float));

    mram_read(att, wram_att, SEQ_LEN * sizeof(float));

    for (size_t s = 0; s < segment_count; s++) {
      const size_t segment_offset = s * segment_size;
      const size_t current_segment_size =
          (segment_offset + segment_size > chunk_size)
              ? (chunk_size % segment_size)
              : segment_size;

      for (size_t i = 0; i < current_segment_size; i++) {
        wram_x[i] = 0.0f;
        for (size_t k = 0; k < vc_segment_count; k++) {
          const size_t vc_segment_offset = k * vc_segment_size;
          mram_read(vc + (chunk_offset + segment_offset + i) * SEQ_LEN +
                        vc_segment_offset,
                    wram_vc, vc_segment_size * sizeof(float));
          wram_x[i] +=
              dot(wram_att + vc_segment_offset, wram_vc, vc_segment_size);
        }
      }

      mram_write(wram_x, x + chunk_offset + segment_offset,
                 current_segment_size * sizeof(float));
    }
  }

  return 0;
}
