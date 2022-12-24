// Last update: 16/12/2020
#include <stdio.h>
#include <stdint.h>

__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

__global__ void extractBitsKernel(const uint32_t * src, int n, int * bits, int bitIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        bits[i] = (src[i] >> bitIdx) & 1;
    }
}

__global__ void scanKernel(int * in, int n, int * out, volatile int * blkSums)
{
    __shared__ int bi;
    if (threadIdx.x == 0)
        bi = atomicAdd(&bCount, 1);
    __syncthreads();

    extern __shared__ int s_data[];
	int i1 = bi * 2 * blockDim.x + threadIdx.x;
	int i2 = i1 + blockDim.x;
	if (i1 < n)
		s_data[threadIdx.x] = in[i1];
	if (i2 < n)
		s_data[threadIdx.x + blockDim.x] = in[i2];
	__syncthreads();

	// 2. Each block does scan with data on SMEM
	// 2.1. Reduction phase
	for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // To avoid warp divergence
		if (s_dataIdx < 2 * blockDim.x)
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
		__syncthreads();
	}
	// 2.2. Post-reduction phase
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride; // Wow
		if (s_dataIdx < 2 * blockDim.x)
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
		__syncthreads();
	}

	if (blkSums != NULL && threadIdx.x == 0)
		blkSums[bi] = s_data[2 * blockDim.x - 1];

    if (threadIdx.x == 0)
    {
        if (bi > 0)
        {
            while (bCount1 < bi) {} // Chờ block bi-1
            blkSums[bi] += blkSums[bi - 1]; // Tính tổng của bi+1 block (0→bi)
            __threadfence(); // Đảm bảo blkSums được cập nhật trước bCount1
        }
        bCount1 += 1; // Bật tín hiệu để block bi+1 biết
    }
    __syncthreads();

    if (bi > 0)
    {
        s_data[threadIdx.x] += blkSums[bi - 1];
        s_data[threadIdx.x + blockDim.x] += blkSums[bi - 1];
    }

    // 3. Each block writes results from SMEM to GMEM
    if (i1 + 1 < n)
        out[i1 + 1] = s_data[threadIdx.x];
    if (i2 + 1 < n)
        out[i2 + 1] = s_data[threadIdx.x + blockDim.x];
}

__global__ void computeRankKernel(int * bits, uint32_t * src, uint32_t * dst, int n, int * nOnesBefore, int nZeros)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int rank;
        if (bits[i] == 0)
            rank = i - nOnesBefore[i];
        else
            rank = nZeros + nOnesBefore[i];
        dst[rank] = src[i];
    }
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO

    int * d_bits, * d_Before, * d_block;
    int s= n * sizeof(uint32_t);
    uint32_t * d_src, * d_dst;

    uint32_t * src = (uint32_t *)malloc(s);
    memcpy(src, in,s);
     
    CHECK(cudaMalloc(&d_bits, n * sizeof(int)));
    CHECK(cudaMalloc(&d_Before, n * sizeof(int)));
    CHECK(cudaMalloc(&d_src, s));
    CHECK(cudaMalloc(&d_dst, s));

    int block_data_s = 2 * blockSize;
    int grid_s = (n - 1) / block_data_s + 1;

    size_t nBytes = n * sizeof(int);
    int const_0 = 0;

    int * bits = (int *)malloc(n * sizeof(int));
    int * Before = (int *)malloc(n * sizeof(int));
    int grid_s2 = (n - 1) / blockSize + 1;

    for (int i = 0; i < sizeof(uint32_t) * 8 ; i++){

        CHECK(cudaMemcpy(d_src, src, s , cudaMemcpyHostToDevice));
        extractBitsKernel<<<grid_s2, blockSize>>>(d_src, n, d_bits,i);
		    CHECK(cudaMemcpy(bits, d_bits, n * sizeof(int), cudaMemcpyDeviceToHost));

        CHECK(cudaMemcpyToSymbol(bCount1, &const_0, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount, &const_0, sizeof(int)));

        if (grid_s > 1){
            CHECK(cudaMalloc(&d_block, grid_s * sizeof(int)));
        }
        else{
            d_block = NULL;
        }

        size_t smem = block_data_s * sizeof(int);
        scanKernel<<<grid_s, blockSize, smem>>>(d_bits, n, d_Before, d_block);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(Before, d_Before, nBytes, cudaMemcpyDeviceToHost));

        int nZeros = n - Before[n - 1] - bits[n - 1];
        computeRankKernel<<<grid_s2, blockSize>>>(d_bits, d_src, d_dst, n, d_Before, nZeros);
        
        uint32_t * dst = out;
        CHECK(cudaMemcpy(dst, d_dst, s, cudaMemcpyDeviceToHost));

        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    memcpy(out, src, s);

    free(bits);
    free(Before);

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_block));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    // int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        // in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    // printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    // printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
