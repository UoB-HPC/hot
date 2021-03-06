#include "../../cuda/reduction.k"

// https://inldigitallibrary.inl.gov/sti/3952796.pdf
// Take the average of the coefficients at the cells surrounding
// each face
__global__ void calc_s_x(const int nx, const int ny, const int pad,
                         const double dt, const double heat_capacity,
                         const double conductivity, double* s_x,
                         const double* density, const double* edgedx) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));
  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= ny - pad)
    return;
  s_x[(ii) * (nx + 1) + (jj)] =
      (dt * conductivity *
       (density[(ii)*nx + (jj)] + density[(ii)*nx + (jj - 1)])) /
      (2.0 * density[(ii)*nx + (jj)] * density[(ii)*nx + (jj - 1)] *
       edgedx[jj] * edgedx[jj] * heat_capacity);
}

__global__ void calc_s_y(const int nx, const int ny, const int pad,
                         const double dt, const double heat_capacity,
                         const double conductivity, double* s_y,
                         const double* density, const double* edgedy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  if (ii < pad || jj < pad || jj >= nx - pad || ii >= (ny + 1) - pad)
    return;
  s_y[(ii)*nx + (jj)] =
      (dt * conductivity *
       (density[(ii)*nx + (jj)] + density[(ii - 1) * nx + (jj)])) /
      (2.0 * density[(ii)*nx + (jj)] * density[(ii - 1) * nx + (jj)] *
       edgedy[ii] * edgedy[ii] * heat_capacity);
}

__global__ void calc_initial_r2(const int nx, const int ny, const int pad,
                                double* s_x, double* s_y,
                                const double* temperature, double* p, double* r,
                                double* reduce_array) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if (ii >= pad && jj >= pad && jj < nx - pad && ii < ny - pad) {
    r[(ii)*nx + (jj)] =
        temperature[(ii)*nx + (jj)] -
        ((s_y[(ii)*nx + (jj)] + s_x[(ii) * (nx + 1) + (jj)] + 1.0 +
          s_x[(ii) * (nx + 1) + (jj + 1)] + s_y[(ii + 1) * nx + (jj)]) *
             temperature[(ii)*nx + (jj)] -
         s_y[(ii)*nx + (jj)] * temperature[(ii - 1) * nx + (jj)] -
         s_x[(ii) * (nx + 1) + (jj)] * temperature[(ii)*nx + (jj - 1)] -
         s_x[(ii) * (nx + 1) + (jj + 1)] * temperature[(ii)*nx + (jj + 1)] -
         s_y[(ii + 1) * nx + (jj)] * temperature[(ii + 1) * nx + (jj)]);
    p[(ii)*nx + (jj)] = r[(ii)*nx + (jj)];
    sdata[tid] = r[(ii)*nx + (jj)] * r[(ii)*nx + (jj)];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(threadIdx.x, sdata);
  if (tid == 0)
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void calc_pAp(const int nx, const int ny, const int pad,
                         const double* s_x, const double* s_y, double* p,
                         double* Ap, double* reduce_array) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  // You don't need to use a matrix as the band matrix is fully predictable
  // from the 5pt stencil
  if (ii >= pad & jj >= pad && jj < nx - pad && ii < ny - pad) {
    Ap[(ii)*nx + (jj)] =
        (s_y[(ii)*nx + (jj)] + s_x[(ii) * (nx + 1) + (jj)] + 1.0 +
         s_x[(ii) * (nx + 1) + (jj + 1)] + s_y[(ii + 1) * nx + (jj)]) *
            p[(ii)*nx + (jj)] -
        s_y[(ii)*nx + (jj)] * p[(ii - 1) * nx + (jj)] -
        s_x[(ii) * (nx + 1) + (jj)] * p[(ii)*nx + (jj - 1)] -
        s_x[(ii) * (nx + 1) + (jj + 1)] * p[(ii)*nx + (jj + 1)] -
        s_y[(ii + 1) * nx + (jj)] * p[(ii + 1) * nx + (jj)];
    sdata[threadIdx.x] = p[(ii)*nx + (jj)] * Ap[(ii)*nx + (jj)];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(threadIdx.x, sdata);
  if (tid == 0)
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void calc_new_r2(const int nx, const int ny, const int pad,
                            double alpha, double* temperature, double* p,
                            double* r, double* Ap, double* reduce_array) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if (ii >= pad & jj >= pad && jj < nx - pad && ii < ny - pad) {
    temperature[(ii)*nx + (jj)] += alpha * p[(ii)*nx + (jj)];
    r[(ii)*nx + (jj)] -= alpha * Ap[(ii)*nx + (jj)];
    sdata[threadIdx.x] = r[(ii)*nx + (jj)] * r[(ii)*nx + (jj)];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(threadIdx.x, sdata);
  if (tid == 0)
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void update_p(const int nx, const int ny, const int pad,
                         const double beta, const double* r, double* p) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  p[(ii)*nx + (jj)] = r[(ii)*nx + (jj)] + beta * p[(ii)*nx + (jj)];
}
