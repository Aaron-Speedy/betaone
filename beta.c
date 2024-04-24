#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DS_IMPL
#include "ds.h"

#define MATH_IMPL
#include "math.h"

typedef enum {
  ACT_RELU,
  ACT_TANH,
  ACT_SIGMOID,
} Activation;

typedef struct {
  List(Matrix) ws;
  List(Matrix) bs;
  List(int) layer_sizes;
  Activation act;
} NN;

float activation(Activation act, float x) {
  switch (act) {
  case ACT_RELU: return max(0.0, x);
  case ACT_TANH: return tanh(x);
  case ACT_SIGMOID: return 1.0 / (1.0 + expf(-x));
  default: assert(!"Invalid activation");
  }
}

float activation_deriv(Activation act, float x) {
  float a;
  switch (act) {
  case ACT_RELU: return x > 0;
  case ACT_TANH: { a = tanh(x); return 1 - a * a; }
  case ACT_SIGMOID: { a = activation(ACT_SIGMOID, x); return a * (1 - a); };
  default: assert(!"Invalid activation");
  }
}

float cost(float x, float e) {
  float error = x - e;
  return error * error;
}

float cost_deriv(float x, float e) {
  return 2 * (x - e);
}

void nn_init_rand(Arena *al, NN *nn, float wmin, float wmax, float bmin, float bmax) {
  nn->ws.cap = 256;
  da_init_ar(al, &nn->ws);

  nn->bs.cap = 256;
  da_init_ar(al, &nn->bs);

  for (int i = 1; i < nn->layer_sizes.count; i++) {
    Matrix w = {
      .w = nn->layer_sizes.items[i - 1],
      .h = nn->layer_sizes.items[i],
    };
    matrix_init(al, &w);
    matrix_randomize(&w, wmin, wmax);

    Matrix b = {
      .w = 1,
      .h = nn->layer_sizes.items[i - 1],
    };
    matrix_init(al, &b);
    matrix_randomize(&b, bmin, bmax);

    da_push(&nn->ws, w);
    da_push(&nn->bs, b);
  }
}

Matrix nn_run(Arena *al, NN *nn, Matrix in) {
  assert(nn->ws.items[0].w == in.h);
  assert(in.w == 1);

  Matrix l = in;

  for (int i = 0; i < nn->ws.count; i++) {
    if (i != 0) al->pos -= al->last_alloc_len;
    l = matrix_multiply(al, nn->ws.items[i], l);

    Matrix b = nn->bs.items[i];
    for (int i = 0; i < b.h; i++) {
      m_at(&l, 0, i) += m_at(&b, 0, i);
      m_at(&l, 0, i) = activation(nn->act, m_at(&l, 0, i));
    }
  }

  return l;
}

int main() {
  srand(time(0));

  Arena al = { .len = 100000, };
  arena_init(&al);

  NN nn = { .act = ACT_RELU, };

  nn.layer_sizes.cap = 256;
  da_init_ar(&al, &nn.layer_sizes);
  da_push_ar(&al, &nn.layer_sizes, 784);
  da_push_ar(&al, &nn.layer_sizes, 16);
  da_push_ar(&al, &nn.layer_sizes, 16);
  da_push_ar(&al, &nn.layer_sizes, 10);

  nn_init_rand(&al, &nn, -1.0, 1.0, -1.0, 1.0);

  Matrix in = { .w = 1, .h = nn.layer_sizes.items[0], };
  matrix_init(&al, &in);
  matrix_randomize(&in, 0.0, 1.0);

  // matrix_print(in);

  matrix_print(nn_run(&al, &nn, in));

  return 0;
}
