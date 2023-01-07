#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
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
		cudaEventRecord(start,0);
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

void printArray(int * a, int n)
{
    // for (int i = 0; i < 512; i++)
    //     printf("%i ", a[n-1-512-i]);
    // printf("\n");

    for (int r = 470; r < 480; r++)
    {
        for (int c = 0; c < 755; c++)
        {
            printf("%i ", a[r * 755 + c]);
        }
        printf("\n");
    }
}

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

void convertRgb2GrayByHost(uchar3 * inPixels, int width, int height,
		uint8_t * outPixels)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            uint8_t red = inPixels[i].x;
            uint8_t green = inPixels[i].y;
            uint8_t blue = inPixels[i].z;
            outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
        }
    }
}

__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, 
		uint8_t * outPixels)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    { 
		int i = r * width + c;
		uint8_t red = inPixels[i].x;
		uint8_t green = inPixels[i].y;
		uint8_t blue = inPixels[i].z;
		outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
    }
}

void calcPixelGravityByHost(uint8_t * inPixels, int width, int height,
        int * outPixels)
{
    int xSobel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int ySobel[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
    {
        for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
        {
            int xOutPixel = 0;
            int yOutPixel = 0;
            for (int filterR = 0; filterR < 3; filterR++)
            {
                for (int filterC = 0; filterC < 3; filterC++)
                {
                    int xVal = xSobel[filterR * 3 + filterC];
                    int yVal = ySobel[filterR * 3 + filterC];

                    int inPixelsR = outPixelsR - 3 / 2 + filterR;
                    int inPixelsC = outPixelsC - 3 / 2 + filterC;
                    inPixelsR = min(max(0, inPixelsR), height - 1);
                    inPixelsC = min(max(0, inPixelsC), width - 1);
                    uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];
                    
                    xOutPixel += xVal * inPixel;
                    yOutPixel += yVal * inPixel;
                }
            }
            outPixels[outPixelsR * width + outPixelsC] = abs(xOutPixel) + abs(yOutPixel);
        }
    }
}

__global__ void calcPixelGravityKernel(uint8_t * inPixels, int width, int height,
        int * outPixels)
{
	int outPixelR = blockIdx.y * blockDim.y + threadIdx.y;
    int outPixelC = blockIdx.x * blockDim.x + threadIdx.x;

    int xSobel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int ySobel[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    if (outPixelR < height && outPixelC < width)
	{
        int xOutPixel = 0;
        int yOutPixel = 0;
		for (int filterR = 0; filterR < 3; filterR++)
		{
			for (int filterC = 0; filterC < 3; filterC++)
			{
                int xVal = xSobel[filterR * 3 + filterC];
                int yVal = ySobel[filterR * 3 + filterC];

				int inPixelR = outPixelR - 3 / 2 + filterR;
				int inPixelC = outPixelC - 3 / 2 + filterC;
				inPixelR = min(max(0, inPixelR), height - 1);
				inPixelC = min(max(0, inPixelC), width - 1);
                uint8_t inPixel = inPixels[inPixelR * width + inPixelC];

				xOutPixel += xVal * inPixel;
                yOutPixel += yVal * inPixel;
			}
		}
		outPixels[outPixelR * width + outPixelC] = abs(xOutPixel) + abs(yOutPixel);
	}
}

void calcSeamGravityByHost(int * inPixels, int width, int height,
        int * outPixels, int8_t * trace)
{
    for (int i = width * (height - 1); i < width * height; i++)
    {
        outPixels[i] = inPixels[i];
        trace[i] = 0;
    }
    for (int i = width * (height - 1) - 1; i >= 0 ; i--)
    {
        outPixels[i] = inPixels[i];

        int down[] = {0, 0, 0};
        down[0] = (i % width != 0) ? outPixels[i + width - 1] : 4000000;
        down[1] = outPixels[i + width];
        down[2] = ((i + 1) % width != 0) ? outPixels[i + width + 1] : 4000000;

        int minValue = down[0];
        int index = 0;
        for (int j = 1; j < 3; j++)
        {
            if (down[j] < minValue)
            {
                minValue = down[j];
                index = j;
            }
        }

        outPixels[i] += minValue;
        trace[i] = index - 1;
    }
}

__global__ void calcSeamGravityKernel(int * inPixels, int width, int height,
        int * outPixels, int8_t * trace, int row)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // for (int row = height - 1; row >= 0; row--)
    // {
        if (col < width)
        {
            int i = row * width + col;
            outPixels[i] = inPixels[i];

            if (row == height - 1)
            {
                trace[i] = 0;
            }
            else
            {
                int down[] = {0, 0, 0};
                down[0] = (i % width != 0) ? outPixels[i + width - 1] : 4000000;
                down[1] = outPixels[i + width];
                down[2] = ((i + 1) % width != 0) ? outPixels[i + width + 1] : 4000000;

                int minValue = down[0];
                int index = 0;
                for (int j = 1; j < 3; j++)
                {
                    if (down[j] < minValue)
                    {
                        minValue = down[j];
                        index = j;
                    }
                }

                outPixels[i] += minValue;
                trace[i] = index - 1;
            }
        }
    // }
}


void findSeamIndexByHost(int * seamsImportance, int width, int height,
        int8_t * trace, uint32_t * seamIndex)
{
    int minValue = seamsImportance[0];
    uint32_t index = 0;

    for (int i = 1; i < width; i++)
    {
        if (seamsImportance[i] < minValue)
        {
            minValue = seamsImportance[i];
            index = i;
        }
    }

    seamIndex[0] = index;
    for (int i = 1; i < height; i++)
    {
        seamIndex[i] = seamIndex[i - 1] + width + trace[seamIndex[i - 1]];
    }
}


void deleteSeamByHost(uchar3 * pixels, int width, int height,
        uint32_t * seam)
{
    uchar3 * origin = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(origin, pixels, width * height * sizeof(uchar3));

    int j = 0;
    int k = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (i == seam[j])
        {
            j++;
            continue;
        }
        pixels[k] = origin[i];
        k++;
    }

    free(origin);
}

// __global__ void deleteSeamByHost(uchar3 * pixels, int width, int height,
//         uint32_t * seam,int row)
// {
//     int col = blockIdx.y*blockDim.y+threadIdx.y;
    
//     if (col<height)
//     {
//       uchar3 * origin = (uchar3 *)malloc(width * height * sizeof(uchar3));
//       memcpy(origin, pixels, width * height * sizeof(uchar3));

//       int k = 0;
//       for (int i = 0; i < width-1; i++)
//       {
//           if (i == seam[row])
//           {
//               row++;
//               continue;
//           }
//           pixels[k] = origin[i];
//           k++;
//       }

//       free(origin);
//     }
    
// }

__global__ void deleteSeamKernel(uchar3 * pixels,int originWidth, int width,int height,uint32_t * seam){
    int row = blockIdx.x; 
    int j=row * originWidth;

    uchar3 * origin = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(origin, pixels, width * height * sizeof(uchar3));  

    for (int i=0;i<seam[row];++i)
    {
        pixels[j+i]=origin[j+i];
    }
    for (int i=seam[row];i<width-1;++i)
    {
        pixels[j+i]=origin[j+i+1];
    }
    free(origin);
}

// __global__ void deleteSeamKernel(uchar3 * pixels, int width, int height,uint32_t * seam, int row)
// {
//     int c = blockIdx.x* blockDim.x+threadIdx.x;    
//     if (c< width-1)
//     {
//         if(c>=seam[row])
//         {
//           int i= row*width+c; 
//           pixels[i]=pixels[i+1];
//         }
//     }
// }

// Sequential Seam Carving
void seamCarvingByHost(uchar3 * inPixels, int inWidth, int height,
        uchar3 * outPixels, int outWidth)
{
    memcpy(outPixels, inPixels, inWidth * height * sizeof(uchar3));

    uint8_t * grayPixels = (uint8_t *)malloc(inWidth * height * sizeof(uint8_t));
    int * detectedPixels = (int *)malloc(inWidth * height * sizeof(int));
    int * seamsImportance = (int *)malloc(inWidth * height * sizeof(int));
    int8_t * trace = (int8_t *)malloc(inWidth * height * sizeof(int8_t));
    uint32_t * seamIndex = (uint32_t *)malloc(height * sizeof(uint32_t));
    while (inWidth != outWidth)
    {
        convertRgb2GrayByHost(outPixels, inWidth, height, grayPixels);
        calcPixelGravityByHost(grayPixels, inWidth, height, detectedPixels);
        calcSeamGravityByHost(detectedPixels, inWidth, height, seamsImportance, trace);
        findSeamIndexByHost(seamsImportance, inWidth, height, trace, seamIndex);
        deleteSeamByHost(outPixels, inWidth, height, seamIndex);

        // printArray(seamsImportance, inWidth * height); ///////////////// debug here

        inWidth--;
    }

    free(grayPixels);
    free(detectedPixels);
    free(seamsImportance);
    free(trace);
    free(seamIndex);
}

// Parallel Seam Carving
void seamCarvingByDevice(uchar3 * inPixels, int inWidth, int height,
        uchar3 * outPixels, int outWidth, dim3 blockSize)
{
    int size = inWidth * height;
    memcpy(outPixels, inPixels, size * sizeof(uchar3));
    const  int originWidth=inWidth;

    uchar3 * d_outPixels;
    CHECK(cudaMalloc(&d_outPixels, size * sizeof(uchar3)));
    CHECK(cudaMemcpy(d_outPixels, outPixels, size * sizeof(uchar3), cudaMemcpyHostToDevice));

    uint8_t * d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, size * sizeof(uint8_t)));

    int * d_detectedPixels;
    CHECK(cudaMalloc(&d_detectedPixels, size * sizeof(int)));

    int * d_seamsImportance;
    CHECK(cudaMalloc(&d_seamsImportance, size * sizeof(int)));

    int8_t * d_trace;
    CHECK(cudaMalloc(&d_trace, size * sizeof(int8_t)));

    uint32_t* d_seamIndex;
    CHECK(cudaMalloc(&d_seamIndex,size*sizeof(uint32_t)));

    int * seamsImportance = (int *)malloc(inWidth * height * sizeof(int));
    int8_t * trace = (int8_t *)malloc(inWidth * height * sizeof(int8_t));
    uint32_t * seamIndex = (uint32_t *)malloc(height * sizeof(uint32_t));

    while (inWidth != outWidth)
    {
        dim3 gridSize((inWidth - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
        convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_outPixels, inWidth, height, d_grayPixels);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        calcPixelGravityKernel<<<gridSize, blockSize>>>(d_grayPixels, inWidth, height, d_detectedPixels);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        for (int i = height - 1; i >= 0; i--)
        {
            calcSeamGravityKernel<<<gridSize.x, blockSize.x>>>(d_detectedPixels, inWidth, height, d_seamsImportance, d_trace, i);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
        }

        CHECK(cudaMemcpy(seamsImportance, d_seamsImportance, size * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(trace, d_trace, size * sizeof(int8_t), cudaMemcpyDeviceToHost));

        // printArray(seamsImportance, size); ///////////////// debug here
        // for (int i=0;i<height;i++)
        // {
        //   findSeamIndexKernel<<<gridSize.x,blockSize.x>>>(d_seamsImportance,inWidth, height, d_trace, d_seamIndex, i);
        //   cudaDeviceSynchronize();
        //   CHECK(cudaGetLastError());
        // }

        // findSeamIndexKernel<<<gridSize,blockSize>>>(d_seamsImportance,inWidth, height, d_trace, d_seamIndex);
        
        // cudaDeviceSynchronize();
        // CHECK(cudaGetLastError());
        // CHECK(cudaMemcpy(seamIndex, d_seamIndex, size * sizeof(int), cudaMemcpyDeviceToHost));

        findSeamIndexByHost(seamsImportance, inWidth, height, trace, seamIndex);
        // deleteSeamByHost(outPixels, inWidth, height, seamIndex);
        CHECK(cudaMemcpy(d_seamIndex,seamIndex,height*sizeof(int),cudaMemcpyHostToDevice));

        deleteSeamKernel<<<height,1>>>(d_outPixels,originWidth,inWidth,height, d_seamIndex);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        inWidth--;
        size = inWidth * height;
        CHECK(cudaMemcpy(d_outPixels, outPixels, size * sizeof(uchar3), cudaMemcpyHostToDevice));
    }

    CHECK(cudaMemcpy(outPixels, d_outPixels, size * sizeof(uchar3), cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_detectedPixels));
    CHECK(cudaFree(d_seamsImportance));
    CHECK(cudaFree(d_trace));
    free(seamsImportance);
    free(trace);
    free(seamIndex);
}

// Resize by seam carving
void seamCarving(uchar3 * inPixels, int inWidth, int height,
        uchar3 * outPixels, int outWidth,
        bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nSeam Carving by host\n");
        seamCarvingByHost(inPixels, inWidth, height, outPixels, outWidth);
    }
    else
    {
        printf("\nSeam Carving by device\n");
        seamCarvingByDevice(inPixels, inWidth, height, outPixels, outWidth, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
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

int main(int argc, char ** argv)
{
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

	// Read input image file
	int inWidth, height;
	uchar3 * inPixels;
	readPnm(argv[1], inWidth, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", inWidth, height);

    // Calculate output image size
    int outWidth = atoi(argv[3]);
    if (outWidth <= 0 || outWidth >= inWidth)
    {
        printf("The target width is invalid\n");
        return EXIT_FAILURE;
    }
    printf("Number of removed seams: %d\n\n", inWidth - outWidth);

    // Seam Carving using host
	uchar3 * correctOutPixels = (uchar3 *)malloc(inWidth * height * sizeof(uchar3));
	seamCarving(inPixels, inWidth, height, correctOutPixels, outWidth);

    // Seam Carving using device
    dim3 blockSize(32, 32); // Default
    if (argc == 6)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    }

	uchar3 * outPixels = (uchar3 *)malloc(inWidth * height * sizeof(uchar3));
	seamCarving(inPixels, inWidth, height, outPixels, outWidth, true, blockSize);
	printError(outPixels, correctOutPixels, outWidth, height);

    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(correctOutPixels, outWidth, height, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(outPixels, outWidth, height, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(inPixels);
    free(correctOutPixels);
    free(outPixels);
    
    return EXIT_SUCCESS;
}