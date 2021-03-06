#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "bitmap.hpp"
#include <sstream>
#include <algorithm>
#include <pthread.h>
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

std::vector<std::vector<unsigned char>> testImages;

std::vector<unsigned char> testLabels;

double accur;

int counter;

int predicted_classes[10000];

int output_matrix[10][10];

pthread_mutex_t lock1, lock2;

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
		for(int j = 0; j < 784; j++){
			white_pixel_in_c[i][j] = 0.0;
		}
		
	}
}

void simple_probability(std::vector<unsigned char> &trainLabels, int start_index, int end_index){


	for(int j = start_index; j < end_index; j++){
		int label = static_cast<int>(trainLabels[j]);
		pthread_mutex_lock(&lock1);
		train_label_counts[label] = ((double)train_label_counts[label]+1);
		pthread_mutex_unlock(&lock1);

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
				pthread_mutex_lock(&lock2);
				white_pixel_in_c[label][j] = (white_pixel_in_c[label][j]+1);
				pthread_mutex_unlock(&lock2);
			}

		}	
	}

	

}

void calculate_conditional_probability(){
	for(int i= 0; i < 10; i++){
		for(int j = 0; j < 784; j++){
			conditional_probability_vector[i][j] = ( (white_pixel_in_c[i][j] + 1 )/(train_label_counts[i]+2) );
		}
	}
}

void *parallel_function(void *start){
		int start_index;
		start_index = *(int *)start;
		free(start);
		//printf("%d\n",start_index);
		simple_probability(trainLabels, start_index, start_index+(60000/NUM_OF_THREADS));
		conditional_probability(trainLabels, trainImages, start_index, start_index+(60000/NUM_OF_THREADS));
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


void test_on_all_images(std::vector<std::vector<unsigned char>> &testImages, std::vector<unsigned char> &testLabels, int start_index, int end_index){
	
	counter = 0;
		for(int i = start_index ; i < end_index; i++){
			int class1 = find_label_for_test(i, testImages);
			// if(i<20)
			// 	std::cout<<class1<<std::endl;
			predicted_classes[i] = class1;
			int orig_label = static_cast<int>(testLabels[i]);
			if(predicted_classes[i] == orig_label){
				pthread_mutex_lock(&lock1);	
				counter++;
				pthread_mutex_unlock(&lock1);
			}
				pthread_mutex_lock(&lock2);
				output_matrix[orig_label][class1]++;
				pthread_mutex_unlock(&lock2);
		}

	
}

void *parallel_test_function(void *start){
	int start_index;
		start_index = *(int *)start;
		free(start);
	test_on_all_images(testImages, testLabels, start_index, start_index + (10000/NUM_OF_THREADS));
}

void dump_results(){
	std::ofstream file;
	file.open("classification-summary.txt");
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
	pthread_t thread[NUM_OF_THREADS];

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
    testImages = dataset.test_images;
    // get test labels
    testLabels = dataset.test_labels;
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
    for(int i = 0; i < NUM_OF_THREADS; i++){
    	int start_th = i * 60000/NUM_OF_THREADS;
    	int *p_start_th = (int *)malloc(sizeof(*p_start_th));
    	*p_start_th = start_th;
    	pthread_create( &thread[i], NULL, parallel_function, (void*) p_start_th);
	}
    for(int i = 0; i < NUM_OF_THREADS; i++){
    	pthread_join(thread[i], NULL);
	}
    calculate_simple_probability();	
    calculate_conditional_probability();
    for(int i = 0; i < NUM_OF_THREADS; i++){
    	int start_th = i * 10000/NUM_OF_THREADS;
    	int *p_start_th = (int *)malloc(sizeof(*p_start_th));
    	*p_start_th = start_th;
    	pthread_create( &thread[i], NULL, parallel_test_function, (void*) p_start_th);
	}
    for(int i = 0; i < NUM_OF_THREADS; i++){
    	pthread_join(thread[i], NULL);
	}
    dump_results();
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf(" Execution time = %f sec\n",time);
    print_bitmap();
    print_result();
    return 0;
}

