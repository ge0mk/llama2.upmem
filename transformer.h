#pragma once

#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "kernels/model_config.h"

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  uint32_t dim;        // transformer dimension
  uint32_t hidden_dim; // for ffn layers
  uint32_t n_layers;   // number of layers
  uint32_t n_heads;    // number of query heads
  uint32_t n_kv_heads; // number of key/value heads (can be < query heads
                       // because of multiquery)
  int32_t vocab_size;  // vocabulary size, usually 256 (byte-level)
  uint32_t seq_len;    // max sequence length
} Config;

typedef struct {
  // token embedding table
  float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float *wq; // (layer, dim, n_heads * head_size)
  float *wk; // (layer, dim, n_kv_heads * head_size)
  float *wv; // (layer, dim, n_kv_heads * head_size)
  float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float *w1; // (layer, hidden_dim, dim)
  float *w2; // (layer, dim, hidden_dim)
  float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim,)
  float *xb;     // same, but inside a residual branch (dim,)
  float *xb2;    // an additional buffer just for convenience (dim,)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;      // query (dim,)
  float *k;      // key (dim,)
  float *v;      // value (dim,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;           // file descriptor for memory mapping
  float *data;      // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
  bool use_upmem;
} Transformer;

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size);

void softmax(float *x, int size);

void matmul(float *xout, float *x, float *w, int n, int d);

float *forward_cpu(Transformer *transformer, int token, int pos);

float *forward_upmem(Transformer *transformer, int token, int pos);

float *forward(Transformer *transformer, int token, int pos);

void print_vector(float *vec, int size);

bool compare_vector(const char *name, float *a, float *b, size_t size);

void mha_big_test(int pos);
