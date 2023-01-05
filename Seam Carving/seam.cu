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

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
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

	pixels = (uint8_t *)malloc(width * height * numChannels)*sizeof(uint8_t);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	int r = blockIdx.y*blockDim.y+threadIdx.y;
	int c = blockIdx.x*blockDim.x+threadIdx.x;

	if (r<height && c<width){
		int i=r*width+c;
		outPixels[i]=0.299f*inPixels[3 * i] + 0.587f*inPixels[3 * i + 1] + 0.114f*inPixels[3 * i + 2];
	}
}

void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, 
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                uint8_t red = inPixels[3 * i];
                uint8_t green = inPixels[3 * i + 1];
                uint8_t blue = inPixels[3 * i + 2];
                outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
            }
        }
	}
	else // use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);
		// TODO: Allocate device memories

		uint8_t *d_inPixels,*d_outPixels;

		cudaMalloc( &d_inPixels,width*height*3*sizeof(uint8_t));
		cudaMalloc( &d_outPixels,width*height*sizeof(uint8_t));
		// TODO: Copy data to device memories
		cudaMemcpy(d_inPixels ,inPixels ,width*height*3*sizeof(uint8_t) , cudaMemcpyHostToDevice);
		cudaMemcpy(d_outPixels ,outPixels ,width*height*sizeof(uint8_t) , cudaMemcpyHostToDevice);
		// TODO: Set grid size and call kernel (remember to check kernel error)
		dim3 gridSize((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);
		convertRgb2GrayKernel<<<gridSize,blockSize>>>(d_inPixels,width,height,d_outPixels);
		cudaError_t errSyn= cudaGetLastError();
		cudaError_t errAsyn=cudaDeviceSynchronize();

		if (errSyn != cudaSuccess) 
		printf("Sync Error: %s\n", cudaGetErrorString(errSyn));
		if (errAsyn != cudaSuccess)
		printf("Async Error: %s\n", cudaGetErrorString(errAsyn));

		// TODO: Copy result from device memories
		CHECK(cudaMemcpy( outPixels,d_outPixels ,width*height*sizeof(uint8_t) , cudaMemcpyDeviceToHost));
		// TODO: Free device memories
		CHECK(cudaFree( d_inPixels));
		CHECK(cudaFree( d_outPixels));

	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

void edgeDetection(uint8_t * inPixels, int width, int height, int * filter,
        int * outPixels)
{
  int filterWidth=3;
  for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
		{
			for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
			{
				uint8_t outPixel = 0;
				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						float filterVal = filter[filterR*filterWidth + filterC];
						int inPixelsR = outPixelsR - filterWidth/2 + filterR;
						int inPixelsC = outPixelsC - filterWidth/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);
						uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];
						outPixel += filterVal * inPixel;
					}
				}
				outPixels[outPixelsR*width + outPixelsC] = outPixel; 
			}
		}
}


int findMin(int a, int b, int c){
  int min=a;
  if (min < b){
    min = b;
  }
  if(min < c ){
    min=c;
  }
  return  min;

}

int findMinIndx(int a, int b, int c){
  int min=a;
  int indx=-1;
  if (min < b){
    min = b;
    indx=0;
  }
  if(min < c ){
    min=c;
    indx=1;
  }
  return  indx;

}
void findPathByHost(uint8_t* inPixels,int height,int width,int* mask, int* path){
  
	int index =0;
	for (int i= height ;i>0;i--){
  
	  for(int j= width ; j>0;j--){
		
		index = i*width + j;
  
		if (i==height){
		  mask[index] = inPixels[index];
  
		}else{
		  
		  int min= mask[index - width];
		  int k=0;
  
		  if (j== width){
			if (mask[index - width-1]<min){
			  min = mask[index - width-1];
			  k=-1;
			}
		  }
		  if (j == 0){
			if (mask[index - width+1]<min){
			  min = mask[index - width+1];
			  k=1;
			}
		  }
		  else{
			if (mask[index - width-1]<min){
			  min = mask[index - width-1];
			  k=-1;
			}
			if (mask[index - width+1]<min){
			  min = mask[index - width+1];
			  k=1;
			}
		  }
		  mask[index]=inPixels[index]+min;
		  path[index]=k;
		}
	  }
	}
  }
    

void addVecByHost(int n, int * in1, int * in2, int * out)
{
    for (int i = 0; i < n; i++)
        out[i] = in1[i] + in2[i];
}


void printArray(int * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

void calcPixelsImportanceByHost(uint8_t * inPixels, int width, int height,
        int * outPixels)
{
    int * x_DetectPixels= (int *)malloc(width * height)*sizeof(int);
    int * y_DetectPixels= (int *)malloc(width * height)*sizeof(int);
    int n=width*height;

      // Set up a x_sobel filter 
    int x_filter[9]={1,0,-1,2,0,-2,1,0,-1};

    // Set up a y_sobel filter
    int y_filter[9]={1,2,1,0,0,0,-1,-2,-1};

    // // Edge Detection 
    edgeDetection(inPixels,width,height,y_filter,y_DetectPixels);
    edgeDetection(inPixels,width,height,x_filter,x_DetectPixels);

    // CALCULATE IMPORTANCE PIXELS
    addVecByHost(n,x_DetectPixels,y_DetectPixels,outPixels);
    printArray(outPixels,n);  

    free(x_DetectPixels);
    free(y_DetectPixels);
}

void findSeamByHost(uint32_t * lowestImportance, int width, int height,
        int * directory, int * seam)
{
    int minValue = lowestImportance[0];
    int index = 0;

    for (int i = 1; i < width; i++)
    {
        if (lowestImportance[i] < minValue)
        {
            minValue = lowestImportance[i];
            index = i;
        }
    }

    seam[0] = index;
    for (int i = 1; i < height; i++)
    {
        seam[i] = seam[i - 1] + width - 1 + directory[i - 1];
    }
}

void deleteSeamByHost(uint8_t * inPixels, int width, int height, int * seam)
{

    int index=0;
    int col=0;
    uint8_t *  result= (uint8_t *)malloc((width-1) * height);
    printf("HEIGT: %i", height);
    int i=0;
    while(i<height)
    {   
        col = 0;
        printf("\nSEAM: %i",seam[i]);
        index = i * width + col;

        while(index!= seam[i]){
          result[index]=inPixels[index];
          col++;
          index = i * width + col ;
        }
        printf("\nEND %i \n",col);
        col++;
        while(col<width){
          index = i*width+col;
          result[i*(width-1)+col]=inPixels[index];
          col++;
        }

        printf("i %i \n",i);
        printf("Height %i \n",height);
        printf("WIDTH: %i", width);
        printf("\n");
        i++;
    }
    
    for (int i=0;i< (width-1)*height;i++){
      inPixels[i]=result[i];
    }
    free(result);
    
    // for (int i=0;i<height;i++){
    //   for (int j=seam[i];j<width;j++){

    //     inPixels[i*width+j]=inPixels[i*width+(j+1)];
    //   }
    // }
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += abs((int)a1[i] - (int)a2[i]);
	err /= n;
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}


int main(int argc, char ** argv)
{	
	if (argc != 3 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}
  	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension

  
	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale not using device
	uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
 
	convertRgb2Gray(inPixels, width, height, correctOutPixels);
  writePnm(correctOutPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));

  int * im_Pixels= (int *)malloc(width * height)*sizeof(int);
  calcPixelsImportanceByHost(correctOutPixels,width,height,im_Pixels);
  findPathByHost(im_Pixels,height,width,path);
 
	// // Convert RGB to grayscale using device
	// uint8_t * outPixels= (uint8_t *)malloc(width * height);
	// dim3 blockSize(32, 32); // Default
	// if (argc == 5)
	// {
	// 	blockSize.x = atoi(argv[3]);
	// 	blockSize.y = atoi(argv[4]);
	// } 
	// convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize); 

	// // // Compute mean absolute error between host result and device result
	// float err = computeError(outPixels, correctOutPixels, width * height);
	// printf("Error between device result and host result: %f\n", err);

	// Write results to files

	// writePnm(outPixels, 1, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
  free(im_Pixels);
	// free(outPixels);
}
