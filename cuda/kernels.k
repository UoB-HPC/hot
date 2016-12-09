#include "../../cuda/reduction.k"

// https://inldigitallibrary.inl.gov/sti/3952796.pdf
// Take the average of the coefficients at the cells surrounding 
// each face
__global__ void calc_s_x(
    const int nx, const int ny, const double dt, double* s_x, 
    const double* rho, const double* edgedx) 
{
  set_cuda_indices(1);
  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) return;
  s_x[ind1] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-1]))/
    (2.0*rho[ind0]*rho[ind0-1]*edgedx[jj]*edgedx[jj]*HEAT_CAPACITY);
}

__global__ void calc_s_y(
    const int nx, const int ny, const double dt, double* s_y, 
    const double* rho, const double* edgedy) 
{
  set_cuda_indices(0);
  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) return;
  s_y[ind0] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-nx]))/
    (2.0*rho[ind0]*rho[ind0-nx]*edgedy[ii]*edgedy[ii]*HEAT_CAPACITY);
}

__global__ void calc_initial_r2(
    const int nx, const int ny, double* s_x, double* s_y, 
    const double* x, double* p, double* r, double* reduce_array) 
{
  set_cuda_indices(0);
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(ii >= PAD && jj >= PAD && jj < nx-PAD && ii < ny-PAD) {
    r[ind0] = x[ind0] -
      ((s_y[ind0]+s_x[ind1]+1.0+s_x[ind1+1]+s_y[ind0+nx])*x[ind0]
       - s_y[ind0]*x[ind0-nx]
       - s_x[ind1]*x[ind0-1] 
       - s_x[ind1+1]*x[ind0+1]
       - s_y[ind0+nx]*x[ind0+nx]);
    p[ind0] = r[ind0];
    sdata[tid] = r[ind0]*r[ind0];
  }
  __syncthreads();

  sum_reduce_in_shared<NTHREADS>(threadIdx.x, sdata);
  if(tid == 0) 
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void calc_pAp(
    const int nx, const int ny, const double* s_x, 
    const double* s_y, double* p, double* Ap, double* reduce_array) 
{
  set_cuda_indices(0);
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  // You don't need to use a matrix as the band matrix is fully predictable
  // from the 5pt stencil
  if(ii >= PAD & jj >= PAD && jj < nx-PAD && ii < ny-PAD) {
    Ap[ind0] = 
      (s_y[ind0]+s_x[ind1]+1.0+s_x[ind1+1]+s_y[ind0+nx])*p[ind0]
      - s_y[ind0]*p[ind0-nx]
      - s_x[ind1]*p[ind0-1] 
      - s_x[ind1+1]*p[ind0+1]
      - s_y[ind0+nx]*p[ind0+nx];
    sdata[threadIdx.x] = p[ind0]*Ap[ind0];
  }
  __syncthreads();

  sum_reduce_in_shared<NTHREADS>(threadIdx.x, sdata);
  if (tid == 0) 
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void calc_new_r2(
    const int nx, const int ny, double alpha, 
    double* x, double* p, double* r, double* Ap, double* reduce_array)
{
  set_cuda_indices(0);
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(ii >= PAD & jj >= PAD && jj < nx-PAD && ii < ny-PAD) {
    x[ind0] += alpha*p[ind0];
    r[ind0] -= alpha*Ap[ind0];
    sdata[threadIdx.x] = r[ind0]*r[ind0];
  }
  __syncthreads();

  sum_reduce_in_shared<NTHREADS>(threadIdx.x, sdata);
  if (tid == 0) 
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void update_p(
    const int nx, const int ny, const double beta, const double* r, double* p)
{
  set_cuda_indices(0);
  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) return;

  p[ind0] = r[ind0] + beta*p[ind0];
}

