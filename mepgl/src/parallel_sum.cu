/************************************************************
 * Parallel sum code
 ************************************************************/
 
#include "parallel_sum.cuh"
#include "cuda_errors.cuh"

/***************************************** Sum over a collection of real array ******************************************/
namespace parallel_sum{

    __global__ 
    void sumMultiBlockF(const real *array, const int arraySize, real *out) {

        // Starting index of the sum relative to the frame
        const int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        // Separation between elements to be added
        const int delta = blockDim.x*gridDim.x;

        // Beginning and end of the frame data
        const int frameStartIdx =  blockIdx.y*arraySize;
        const int frameEndIdx   = (blockIdx.y+1)*arraySize;
        
        // Exectue the partial sum over the array
        real sum = 0;
        for (int i = frameStartIdx + idx; i < frameEndIdx; i += delta)
            sum += array[i];

        // Copy the partial sums in shared array
        __shared__ real sharedArray[blockSizeX];
        sharedArray[threadIdx.x] = sum;
        __syncthreads();
        
        // Sum the partial sums 
        for (int size = blockDim.x/2; size>0; size/=2) {
            if (threadIdx.x<size)
                sharedArray[threadIdx.x] += sharedArray[threadIdx.x + size];
            __syncthreads();
        }

        // Write the result
        if (threadIdx.x == 0)
            out[blockIdx.y*gridSizeX + blockIdx.x] = sharedArray[0];
    }
    


    __host__ 
    void sumArrayF(real* dev_array, real* dev_results, int F, const int arraySize){  
        // Create a buffer of dimension F*gridSizeX
        real* dev_out;
        checkCudaErrors(cudaMalloc(&dev_out, F*gridSizeX*sizeof(real)));
            
        sumMultiBlockF<<<dim3(gridSizeX, F), dim3(blockSizeX, 1)>>>(dev_array, arraySize, dev_out);    // dev_out now holds the partial results
        checkCudaErrors(cudaDeviceSynchronize());
        
        sumMultiBlockF<<< dim3(1, F), dim3(blockSizeX, 1)>>>(dev_out, gridSizeX, dev_out);   // dev_out[n*F] now hold the final results
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Copy the results
        for(int n = 0; n<F; n++){
            cudaMemcpy(&dev_results[n], &dev_out[n*gridSizeX], sizeof(real), cudaMemcpyDeviceToDevice);
        }

        // Free memory
        checkCudaErrors(cudaFree(dev_out));
    }

    /***************************************** Sum over a collection of real2 array ******************************************/

    __global__ 
    void sumMultiBlockF2(const real2 *array, const int arraySize, real2 *out) {

        // Starting index of the sum relative to the frame
        const int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        // Separation between elements to be added
        const int delta = blockDim.x*gridDim.x;

        // Beginning and end of the frame data
        const int frameStartIdx =  blockIdx.y*arraySize;
        const int frameEndIdx   = (blockIdx.y+1)*arraySize;
        
        // Exectue the partial sum over the array
        real2 sum;
        sum.x = 0;
        sum.y = 0;

        for (int i = frameStartIdx + idx; i < frameEndIdx; i += delta){
            sum.x += array[i].x;
            sum.y += array[i].y;
        }
            

        // Copy the partial sums in shared array
        __shared__ real2 sharedArray[blockSizeX];
        sharedArray[threadIdx.x].x = sum.x;
        sharedArray[threadIdx.x].y = sum.y;
        __syncthreads();
        
        // Sum the partial sums 
        for (int size = blockDim.x/2; size>0; size/=2) {
            if (threadIdx.x<size){
                sharedArray[threadIdx.x].x += sharedArray[threadIdx.x + size].x;
                sharedArray[threadIdx.x].y += sharedArray[threadIdx.x + size].y;
            }
            __syncthreads();
        }

        // Write the result
        if (threadIdx.x == 0){
            out[blockIdx.y*gridSizeX + blockIdx.x].x = sharedArray[0].x;
            out[blockIdx.y*gridSizeX + blockIdx.x].y = sharedArray[0].y;
        }
    }
    


    __host__ 
    void sumArrayF2(real2* dev_array, real2* dev_results, int F, const int arraySize){  
        // Create a buffer of dimension F*gridSizeX
        real2* dev_out;
        checkCudaErrors(cudaMalloc(&dev_out, F*gridSizeX*sizeof(real2)));
            
        sumMultiBlockF2<<<dim3(gridSizeX, F), dim3(blockSizeX, 1)>>>(dev_array, arraySize, dev_out); // dev_out now holds the partial results
        checkCudaErrors(cudaDeviceSynchronize());
        
        sumMultiBlockF2<<< dim3(1, F), dim3(blockSizeX, 1)>>>(dev_out, gridSizeX, dev_out);    // dev_out[n*F] now hold the final results
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Copy the results
        for(int n = 0; n<F; n++){
            cudaMemcpy(&(dev_results[n].x), &(dev_out[n*gridSizeX].x), sizeof(real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&(dev_results[n].y), &(dev_out[n*gridSizeX].y), sizeof(real), cudaMemcpyDeviceToDevice);
        }

        // Free memory
        checkCudaErrors(cudaFree(dev_out));
    }

    /***************************************** Sum over a collection of real2 array ******************************************/

    __global__ 
    void sumMultiBlockF4(const real4 *array, const int arraySize, real4 *out) {

        // Starting index of the sum relative to the frame
        const int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        // Separation between elements to be added
        const int delta = blockDim.x*gridDim.x;

        // Beginning and end of the frame data
        const int frameStartIdx =  blockIdx.y*arraySize;
        const int frameEndIdx   = (blockIdx.y+1)*arraySize;
        
        // Exectue the partial sum over the array
        real4 sum;
        sum.x = 0;
        sum.y = 0;
        sum.z = 0;
        sum.w = 0;

        for (int i = frameStartIdx + idx; i < frameEndIdx; i += delta){
            sum.x += array[i].x;
            sum.y += array[i].y;
            sum.z += array[i].z;
            sum.w += array[i].w;
        }
            

        // Copy the partial sums in shared array
        __shared__ real4 sharedArray[blockSizeX];
        sharedArray[threadIdx.x].x = sum.x;
        sharedArray[threadIdx.x].y = sum.y;
        sharedArray[threadIdx.x].z = sum.z;
        sharedArray[threadIdx.x].w = sum.w;
        __syncthreads();
        
        // Sum the partial sums 
        for (int size = blockDim.x/2; size>0; size/=2) {
            if (threadIdx.x<size){
                sharedArray[threadIdx.x].x += sharedArray[threadIdx.x + size].x;
                sharedArray[threadIdx.x].y += sharedArray[threadIdx.x + size].y;
                sharedArray[threadIdx.x].z += sharedArray[threadIdx.x + size].z;
                sharedArray[threadIdx.x].w += sharedArray[threadIdx.x + size].w;
            }
            __syncthreads();
        }

        // Write the result
        if (threadIdx.x == 0){
            out[blockIdx.y*gridSizeX + blockIdx.x].x = sharedArray[0].x;
            out[blockIdx.y*gridSizeX + blockIdx.x].y = sharedArray[0].y;
            out[blockIdx.y*gridSizeX + blockIdx.x].z = sharedArray[0].z;
            out[blockIdx.y*gridSizeX + blockIdx.x].w = sharedArray[0].w;
        }
    }


    __host__ 
    void sumArrayF4(real4* dev_array, real4* dev_results, int F, const int arraySize){  
        // Create a buffer of dimension F*gridSizeX
        real4* dev_out;
        checkCudaErrors(cudaMalloc(&dev_out, F*gridSizeX*sizeof(real4)));
            
        sumMultiBlockF4<<<dim3(gridSizeX, F), dim3(blockSizeX, 1)>>>(dev_array, arraySize, dev_out);   // dev_out now holds the partial results
        checkCudaErrors(cudaDeviceSynchronize());

        sumMultiBlockF4<<< dim3(1, F), dim3(blockSizeX, 1)>>>(dev_out, gridSizeX, dev_out);   // dev_out[n*F] now hold the final results
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Copy the results
        for(int n = 0; n<F; n++){
            cudaMemcpy(&(dev_results[n].x), &(dev_out[n*gridSizeX].x), sizeof(real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&(dev_results[n].y), &(dev_out[n*gridSizeX].y), sizeof(real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&(dev_results[n].z), &(dev_out[n*gridSizeX].z), sizeof(real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&(dev_results[n].w), &(dev_out[n*gridSizeX].w), sizeof(real), cudaMemcpyDeviceToDevice);
        }

        // Free memory
        checkCudaErrors(cudaFree(dev_out));
    }
}

/***************************************************************************
 * TEST CODE
 ***************************************************************************/

// using namespace std;
// using namespace parallel_sum;
// const int arraySize = int(1e6);
// const int F = 4;

// int main(){
//     real* v = new real[F*arraySize];
 
//     for(int j=0; j<F; j++)
//         for(int i=0; i<arraySize; i++){
//         v[i+arraySize*j]=j;
//     }
 
//     real *dev_v = nullptr;
//     cudaMalloc(&dev_v, F*arraySize*sizeof(real));
//     cudaMemcpy(dev_v, v, F*arraySize*sizeof(real), cudaMemcpyHostToDevice);
 
//     real* results = nullptr;
//     cudaMallocManaged(&results, F*arraySize*sizeof(real));
    
//     sumArrayF(dev_v, results, F, arraySize);
 
//     printf("%e\n", results[0]);
//     printf("%e\n", results[1]);
//     printf("%e\n", results[2]);
//     printf("%e\n", results[3]);
 
    
//     cudaFree(dev_v);
//     cudaFree(results);
//     delete[] v;

//     return 0;
// }

// int main(){
//     real2* v = new real2[F*arraySize];
 
//     for(int j=0; j<F; j++)
//         for(int i=0; i<arraySize; i++){
//         v[i+arraySize*j].x=j;
//         v[i+arraySize*j].y=j*5;
//     }
 
//     real2 *dev_v = nullptr;
//     cudaMalloc(&dev_v, F*arraySize*sizeof(real2));
//     cudaMemcpy(dev_v, v, F*arraySize*sizeof(real2), cudaMemcpyHostToDevice);
 
//     real2* results = nullptr;
//     cudaMallocManaged(&results, F*arraySize*sizeof(real2));
    
//     sumArrayF2(dev_v, results, F, arraySize);
 
//     printf("%e\n", results[0].x);
//     printf("%e\n", results[1].x);
//     printf("%e\n", results[2].x);
//     printf("%e\n", results[3].x);

//     printf("%e\n", results[0].y);
//     printf("%e\n", results[1].y);
//     printf("%e\n", results[2].y);
//     printf("%e\n", results[3].y);
    
//     cudaFree(dev_v);
//     cudaFree(results);
//     delete[] v;

//     return 0;
// }