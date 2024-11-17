#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// creates random values from 0.45 to 0.55
void init(cufftDoubleComplex *A, int Nx, int Ny)
{
    int idx;

    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            idx = i*Ny + j;
            // access real part
            A[idx].x = 0.45 + 0.1 *((double)rand()/(double)RAND_MAX);
            // access imaginary part
            A[idx].y = 0.0;
        }
    }
}
__global__
void normalise(cufftDoubleComplex *A, int Nx, int Ny)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    int idx = i*Ny + j;

    if (i < Nx && j < Ny && idx < Nx*Ny)
    {
        A[idx].x /= (double)(Nx*Ny);
        A[idx].y /= (double)(Nx*Ny);
    }
}

// to calculate dfdc = 2*A*C*(1-C)*(1-2C)
__global__
void kernel1(cufftDoubleComplex *c, cufftDoubleComplex *g, int Nx, int Ny, double A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i + j*Nx;
    if (i < Nx && j < Ny && idx < Nx*Ny)
    {
        g[idx].x = 2.0 * A * c[idx].x * (1-c[idx].x) * (1.0 - 2.0 * c[idx].x);
        g[idx].y = 2.0 * A * c[idx].y * (1-c[idx].y) * (1.0 - 2.0 * c[idx].y);
    }
}


__global__
void kernel2(cufftDoubleComplex *c_new, cufftDoubleComplex *c, cufftDoubleComplex *g,int Nx, int Ny,double *dk2,
                double M, double dt, double kappa)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i+Nx*j;
    if (i < Nx && j < Ny && idx < Nx*Ny)
    {
        c_new[idx].x = (c[idx].x - M*dt*dk2[idx]*g[idx].x)/(1+2*M*kappa*(dk2[idx]*dk2[idx])*dt);
        c_new[idx].y = (c[idx].y - M*dt*dk2[idx]*g[idx].y)/(1+2*M*kappa*(dk2[idx]*dk2[idx])*dt);
    }

}

void writeVTK(cufftDoubleComplex *hc, char file[100],int Nx,int Ny) {
	FILE *fp;
	fp = fopen(file, "w");
	if (fp == NULL) {
		printf("Failed to open %s for writing\n", file);
		exit(1);
	}
	// Write header information for .vtk file
	fprintf(fp, "# vtk DataFile Version 3.0\n");
	fprintf(fp, "vtk output\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "DATASET STRUCTURED_POINTS\n");
	fprintf(fp, "DIMENSIONS %d %d %d\n", Nx, Ny, 1);
	fprintf(fp, "SPACING 1.0 1.0 1.0\n");
	fprintf(fp, "ORIGIN 0 0 0\n");
	fprintf(fp, "POINT_DATA %d\n", Nx * Ny);
	fprintf(fp, "SCALARS input double\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	// Write values
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			fprintf(fp, "%lf\n", hc[i + Nx * j].x);
		}
	}
	fclose(fp);

}

int main(int argc, char **argv)

{
    double dx= 1.0;
    double dy= 1.0;
    double dt= 0.5;
    double A=1.0;
    double M =1.0;
    double kappa=1.0;
    int nt =10000;
    int Nx = 256;
    int Ny = 256; 
    // device pointers
    cufftDoubleComplex *dc, *dg, *dc_new;
    // host pointers
    cufftDoubleComplex *hc, *hc_new;

    // allocate on device
    cudaMalloc((void**)&dc, sizeof(cufftDoubleComplex)*Nx*Ny);
    cudaMalloc((void**)&dg, sizeof(cufftDoubleComplex)*Nx*Ny);
    cudaMalloc((void**)&dc_new, sizeof(cufftDoubleComplex)*Nx*Ny);

    // allocate on host
    hc = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*Nx*Ny);
    hc_new = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*Nx*Ny);

    double dkx; double dky;
    double *hk2; double *dk2;

    // defining periodic boundary conditions
    hk2 = (double*)malloc(sizeof(double)*Nx*Ny);
    cudaMalloc((void**)&dk2, sizeof(double)*Nx*Ny);

   
    dkx = 2.0 * M_PI /((double) Nx *dx);
    dky = 2.0 * M_PI /((double) Ny *dy);
    
    for (unsigned int i =0; i<Nx; i++){    
        for (unsigned int j =0; j<Ny ; j++){
            if (j<= Ny/2)
            hk2[i+Nx*j] += (j*dky) *(j*dky);
            else
            hk2[i+Nx*j] += (j-(double)Ny)*dky * (j-(double)Ny)*dky;
            if (i<=Nx/2)
            hk2[i+Nx *j]+= (i*dkx)*(i*dkx);
            else
            hk2[i+Nx*j] += (i-(double)Nx)*dkx * (i-(double)Nx)*dkx;
        }
    }
    // copying pb's to device
    cudaMemcpy(dk2, hk2, sizeof(double)*Nx*Ny, cudaMemcpyHostToDevice);

    // initialise randomness on host
    init(hc, Nx, Ny);
    // copy initialised to device
    cudaMemcpy(dc, hc, sizeof(cufftDoubleComplex)*Nx*Ny, cudaMemcpyHostToDevice);
                    
    dim3 blocksize(Nx,1,1);
    dim3 gridsize(Nx/blocksize.x, Ny/blocksize.y, 1);

    cufftHandle plan =0;
    cufftPlan2d(&plan, Nx, Ny, CUFFT_Z2Z);
    
    char file[100];
    for (int n=0;n<nt;n++){
        
        // calculating the dfdc i.e dg in this program
        kernel1<<<gridsize,blocksize>>>(dc,dg,Nx,Ny,A);
        
        // taking dg &dc to fourier space
        cufftExecZ2Z(plan, dg, dg, CUFFT_FORWARD);
        cufftExecZ2Z(plan, dc, dc, CUFFT_FORWARD);
        // calculating dc_new using dc & dg in fourier space
        kernel2<<<gridsize,blocksize>>>(dc_new, dc, dg,Nx,Ny,dk2,M,dt,kappa);
        // copying the same to dc
        cudaMemcpy(dc, dc_new, sizeof(cufftDoubleComplex)*Nx*Ny, cudaMemcpyDeviceToDevice);
        // taking back to real space
        cufftExecZ2Z(plan, dc, dc, CUFFT_INVERSE);
        normalise<<<gridsize, blocksize>>>(dc, Nx, Ny);

        // wrting the output file

        if(n%50 == 0){
            cudaMemcpy(hc_new, dc, sizeof(cufftDoubleComplex)*Nx*Ny, cudaMemcpyDeviceToHost);
            for (int i = 0; i < sizeof(file); i++)
				file[i] = '\0';
			sprintf(file, "Data/Output=%d.vtk", n);
			writeVTK(hc_new, file,Nx,Ny);
        }

    }

    cufftDestroy(plan);
	free(hc);
	free(hc_new);
	free(hk2);
	cudaFree(dc);
    cudaFree(dc_new);
    cudaFree(dk2);
    cudaFree(dg);

}
