#pragma once

#define MASTER 0
#define PAD 2

#define strmatch(a, b) (strcmp(a, b) == 0)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

typedef struct
{
  int local_nx;
  int local_ny;
  int global_nx;
  int global_ny;
  int niters;
  int rank;
  int nranks;

} Mesh;

typedef struct
{
  double* Ap;
  double* b;
  double* r;
  double* x;
  double* p;

} State;

static inline void initialise_mesh(Mesh* mesh);
static inline void initialise_state(const int nx, const int ny, State* state);

