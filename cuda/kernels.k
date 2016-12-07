
// https://inldigitallibrary.inl.gov/sti/3952796.pdf
// Take the average of the coefficients at the cells surrounding 
// each face
__global__ void calc_s_x(
    const int nx, const int ny, const double dt, double* s_x, double* rho, double* edgedx) 
{
  set_cuda_indices(1);
  if(ii < PAD || jj < PAD || ii >= ny-PAD || jj >= (nx+1)-PAD) return;
  s_x[ind1] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-1]))/
    (2.0*rho[ind0]*rho[ind0-1]*edgedx[jj]*edgedx[jj]*HEAT_CAPACITY);
}

__global__ void calc_s_y(
    const int nx, const int ny, const double dt, double* s_y, double* rho, double* edgedy) 
{
  set_cuda_indices(0);
  if(ii < PAD || jj < PAD || ii >= (ny+1)-PAD || jj >= nx-PAD) return;
  s_y[ind0] = (dt*CONDUCTIVITY*(rho[ind0]+rho[ind0-nx]))/
    (2.0*rho[ind0]*rho[ind0-nx]*edgedy[ii]*edgedy[ii]*HEAT_CAPACITY);
}

__global__ void calc_initial_r2(
    const int nx, const int ny, const double dt, double* s_x, double* s_y, 
    double* x, double* p, double* r, double* edgedy) 
{
  set_cuda_indices(0);
  if(ii < PAD || jj < PAD || ii >= ny-PAD || jj >= nx-PAD) return;

  r[ind0] = x[ind0] -
    ((s_y[ind0]+s_x[ind1]+1.0+s_x[ind1+1]+s_y[ind0+nx])*x[ind0]
     - s_y[ind0]*x[ind0-nx]
     - s_x[ind1]*x[ind0-1] 
     - s_x[ind1+1]*x[ind0+1]
     - s_y[ind0+nx]*x[ind0+nx]);
  p[ind0] = r[ind0];
  initial_r2 += r[ind0]*r[ind0];


  // TODO: Needs reduction here
}

__global__ void calc_pAp(
    const int nx, const int ny, const double dt, double* s_x, double* s_y, 
    double* p, double* Ap, double* edgedy) 
{
  set_cuda_indices(0);
  if(ii < PAD || jj < PAD || ii >= ny-PAD || jj >= nx-PAD) return;

  // You don't need to use a matrix as the band matrix is fully predictable
  // from the 5pt stencil
  Ap[ind0] = 
    (s_y[ind0]+s_x[ind1]+1.0+s_x[ind1+1]+s_y[ind0+nx])*p[ind0]
    - s_y[ind0]*p[ind0-nx]
    - s_x[ind1]*p[ind0-1] 
    - s_x[ind1+1]*p[ind0+1]
    - s_y[ind0+nx]*p[ind0+nx];

  pAp += p[ind0]*Ap[ind0];
}

