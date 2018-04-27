#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "bitmap.hpp"
#include <sstream>
#include <algorithm>
#include <omp.h>
#include <time.h>

#define MNIST_DATA_DIR "../mnist_data"
int NUM_OF_THREADS = 4;
//this is P(C=c)
double simple_probability_array[10];
//counts how many of each label there is in the training
double train_label_counts [10];
//this is P(Fj=1|C=c)
double conditional_probability_vector[10][784];
//number of white pixels in images of digit c
double white_pixel_in_c[10][784];

std::vector<std::vector<unsigned char>> trainImages;
    // get training labels
std::vector<unsigned char> trainLabels;

double accur;

int predicted_classes[10000];

int output_matrix[10][10];

void print_bitmap(){
	int numLabels =10;
	int numFeatures = 784;
	for(int c =0; c<numLabels; c++){
		std::vector<unsigned char> classFs(numFeatures);
		for(int f=0; f<numFeatures; f++){
			double p = conditional_probability_vector[c][f];
			uint8_t v = 255*p;
			classFs[f] = (unsigned char)v;

		}
		std::stringstream ss;
		ss << "../output/digit" << c << ".bmp";
		Bitmap::writeBitmap(classFs, 28, 28, ss.str(),false);
	}
}

void initialization(){
	for(int i= 0; i< 10; i++){
		train_label_counts[i] = 0.0;
	}

	for(int i= 0; i < 10; i++){
		#pragma omp parallel
		{
		int total = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int start = tid * (784)/total ;
		int end = (tid + 1) * (784)/total ;	
		for(int j = start; j < end; j++){
			white_pixel_in_c[i][j] = 0.0;
		}
		}
	}
}

void simple_probability(std::vector<unsigned char> &trainLabels, int start_index, int end_index){


	for(int j = start_index; j < end_index; j++){
		int label = static_cast<int>(trainLabels[j]);
		#pragma omp critical
		{
		train_label_counts[label] = ((double)train_label_counts[label]+1);
		}
	}

	
}

void calculate_simple_probability(){
for(int i= 0; i< 10; i++){
		simple_probability_array[i] = ((double)train_label_counts[i]/60000);
	}
	double sum =0;
	for(int i= 0; i< 10; i++){
		sum += simple_probability_array[i];
	}
}



void conditional_probability(std::vector<unsigned char> &trainLabels, std::vector<std::vector<unsigned char>> &trainImages, int start_index, int end_index){
	for(int i = start_index; i < end_index; i++){
		for(int j = 0; j < 784; j++){
			int pixelValue = static_cast<int>(trainImages[i][j]);
			if(pixelValue == 1){
				int label = static_cast<int>(trainLabels[i]);
	#pragma omp critical
				{
				white_pixel_in_c[label][j] = (white_pixel_in_c[label][j]+1);
				}
			}

		}	
	}

	

}

void calculate_conditional_probability(){
	for(int i= 0; i < 10; i++){
		#pragma omp parallel
		{
		int total = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int start = tid * (784)/total ;
		int end = (tid + 1) * (784)/total ;	
		for(int j = start; j < end; j++){
			conditional_probability_vector[i][j] = ( (white_pixel_in_c[i][j] + 1 )/(train_label_counts[i]+2) );
		}
		}
	}
}

void parallel_function(int start_index, int end_index){
		simple_probability(trainLabels, start_index, end_index);
		conditional_probability(trainLabels, trainImages, start_index, end_index);
}

int find_label_for_test(int image_number,  std::vector<std::vector<unsigned char>> &testImages){
	int maxclass = 0;
	double sum [10];
	for(int i = 0; i < 10; i++){
		sum[i] = 0.0;
	}
	for(int c = 0 ; c < 10; c++){
		for(int j = 0 ; j < 784; j++){
			int pixelValue = static_cast<int>(testImages[image_number][j]);
			if(pixelValue == 1){
				sum[c] += (log(conditional_probability_vector[c][j]));
			}
			else{
				double negate = (1-conditional_probability_vector[c][j]);
				sum[c] += (log(negate));
			}
		}
		sum[c] = (sum[c] + log(simple_probability_array[c]));
	}
	maxclass = std::max_element(sum, sum+10)-sum;
	/*int max = sum[0];
	maxclass = 0;
	for(int i=0; i <10; i++){
		std::cout << "the sume values are " << sum[i] << std::endl;
		if(sum[i] > max){
			max = sum[i];
			maxclass = i;
		}
	}*/
	return maxclass;
}
void print_result(){
	std::ofstream file2;
	file2.open("network.txt");
	for(int j=0; j <2; j++){
		for(int i=0; i< 784; i++){
			file2 << conditional_probability_vector[j][i] << std::endl;
		}
	}

	for(int i =0 ; i<10; i++){
		file2 << simple_probability_array[i] << std::endl;
	}
}

void test_on_all_images(std::vector<std::vector<unsigned char>> &testImages, std::vector<unsigned char> &testLabels){
	std::ofstream file;
	file.open("classification-summary.txt");
	int counter = 0;
	#pragma omp parallel
		{
		int total = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int start = tid * (10000)/total ;
		int end = (tid + 1) * (10000)/total ;
		for(int i= start ; i < end; i++){
			int class1 = find_label_for_test(i, testImages);
			// if(i<20)
			// 	std::cout<<class1<<std::endl;
			predicted_classes[i] = class1;
			int orig_label = static_cast<int>(testLabels[i]);
			if(predicted_classes[i] == orig_label){
				#pragma omp critical
				{
				counter++;
				}
			}
			#pragma omp critical
				{
				output_matrix[orig_label][class1]++;
				}

		}
	}
	accur = (((double)counter/10000)*100);
	for(int i=0 ;i<10; i++){
		for(int j=0 ; j <10; j++){
			file << output_matrix[i][j] << "	";
		}
		file << std::endl;
	}
	file << accur;
	file.close();
}

int main(int argc, char* argv[]) {
    //Read in the data set from the files
        struct timespec start, stop; 
		double time;
	NUM_OF_THREADS = atoi(argv[1]);
	printf("Number of threads : %d\n", NUM_OF_THREADS);
    omp_set_dynamic(0);
	omp_set_num_threads(NUM_OF_THREADS);

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_DIR);
    //Binarize the data set (so that pixels have values of either 0 or 1)
    mnist::binarize_dataset(dataset);
    //There are ten possible digits 0-9 (classes)
    int numLabels = 10;
    //There are 784 features (one per pixel in a 28x28 image)
    int numFeatures = 784;
    //Each pixel value can take on the value 0 or 1
    int numFeatureValues = 2;
    //image width
    int width = 28;
    //image height
    int height = 28;
    //image to print (these two images were randomly selected by me with no particular preference)
    int trainImageToPrint = 50;
    int testImageToPrint = 5434;
    // get training images
    trainImages = dataset.training_images;
    // get training labels
    trainLabels = dataset.training_labels;
    // get test images
    std::vector<std::vector<unsigned char>> testImages = dataset.test_images;
    // get test labels
    std::vector<unsigned char> testLabels = dataset.test_labels;
    //print out one of the training images
    for (int f=0; f<numFeatures; f++) {
        // get value of pixel f (0 or 1) from training image trainImageToPrint
        int pixelIntValue = static_cast<int>(trainImages[trainImageToPrint][f]);
        if (f % width == 0) {
            //std::cout<<std::endl;
        }
        //std::cout<<pixelIntValue<<" ";
    }
    //std::cout<<std::endl;
    // print the associated label (correct digit) for training image trainImageToPrint
    //std::cout<<"Label: "<<static_cast<int>(trainLabels[trainImageToPrint])<<std::endl;
    //print out one of the test images
    for (int f=0; f<numFeatures; f++) {
        // get value of pixel f (0 or 1) from training image trainImageToPrint
        int pixelIntValue = static_cast<int>(testImages[testImageToPrint][f]);
        if (f % width == 0) {
            //std::cout<<std::endl;
        }
        //std::cout<<pixelIntValue<<" ";
    }
    //std::cout<<std::endl;
    // print the associated label (correct digit) for test image testImageToPrint
    //std::cout<<"Label: "<<static_cast<int>(testLabels[testImageToPrint])<<std::endl;
    std::vector<unsigned char> trainI(numFeatures);
    std::vector<unsigned char> testI(numFeatures);
    for (int f=0; f<numFeatures; f++) {
        int trainV = 255*(static_cast<int>(trainImages[trainImageToPrint][f]));
        int testV = 255*(static_cast<int>(testImages[testImageToPrint][f]));
        trainI[f] = static_cast<unsigned char>(trainV);
        testI[f] = static_cast<unsigned char>(testV);
    }
    std::stringstream ssTrain;
    std::stringstream ssTest;
    ssTrain << "../output/train" <<trainImageToPrint<<"Label"<<static_cast<int>(trainLabels[trainImageToPrint])<<".bmp";
    ssTest << "../output/test" <<testImageToPrint<<"Label"<<static_cast<int>(testLabels[testImageToPrint])<<".bmp";
    Bitmap::writeBitmap(trainI, 28, 28, ssTrain.str(), false);
    Bitmap::writeBitmap(testI, 28, 28, ssTest.str(), false);
    for(int i=0; i<10; i++){
    	for(int j=0; j <10; j++){
    		output_matrix[i][j] = 0;
    	}
    }
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    initialization();
    #pragma omp parallel
    
    {
    	int total = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int start = tid * (60000)/total ;
		int end = (tid + 1) * (60000)/total ;
    	parallel_function(start, end);
	}
    calculate_simple_probability();	
    calculate_conditional_probability();
    test_on_all_images(testImages, testLabels);
    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf(" Execution time = %f sec\n",time);
    print_bitmap();
    print_result();
    return 0;
}

