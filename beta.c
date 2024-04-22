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
  ACT_
} Activation;

typedef struct {
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

int main() {
  srand(time(0));

  Arena al = { .len = 10000, };
  arena_init(&al);

  Matrix a = { .w = 2, .h = 2, };
  matrix_init(&al, &a);

  for (int i = 0; i < a.w; i++) {
    for (int j = 0; j < a.h; j++) {
      m_at(&a, i, j) = randi(0, 10);
    }
  }

  Matrix b = { .w = 2, .h = 2, };
  matrix_init(&al, &b);

  for (int i = 0; i < b.w; i++) {
    for (int j = 0; j < b.h; j++) {
      m_at(&b, i, j) = randi(0, 10);
    }
  }

  Matrix out = matrix_multiply(&al, &a, &b);

  matrix_print(&a);
  printf("\n");
  matrix_print(&b);
  printf("\n");
  matrix_print(&out);

  return 0;
}
