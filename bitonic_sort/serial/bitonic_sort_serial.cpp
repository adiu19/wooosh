#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
void merge(int arr[], int count, int lower, int direction);
void sort(int arr[], int count, int lower, int direction);

void sort(int arr[], int lower, int count, int direction)
{
	if (count > 1)
	{
		int k = count / 2;
		sort(arr, lower, k, 1);
		sort(arr, lower+k, k, 0);
		merge(arr, count, lower, direction);
	}
}

void merge(int arr[], int count, int lower, int direction)
{
	if (count > 1)
	{
		int k = count / 2;
		for (int i = lower; i < lower+k; i++) {
			if (direction == (arr[i] > arr[i+k]))
				swap(arr[i],arr[i+k]);
		}
		merge(arr, k, lower, direction);
		merge(arr, k, lower+k, direction);
	}
}

int main(int argc, char** argv)
{
	if (argc != 2) {
        cout << "Not the correct number of arguments, retry!" << endl;
        exit(1);
    }
    int input = atoi(argv[1]);

    int NUM_VALS = input*input;
    int *mat = (int*)malloc(sizeof(int) * NUM_VALS);
	string input_file_name = "../input_data/input_array_" + to_string(input) +".txt";
    ifstream in(input_file_name);
    int number;
    int k = 0;
    while (in >> number) {
        mat[k++] = number;
	}
    in.close();

    printf(" k = %d\n", k);
    clock_t start = clock();
	int increasing_order = 1; // means sort in ascending order
	sort(mat, 0, NUM_VALS, increasing_order);

	cout << "Time: " << ((double) (clock() - start)) / CLOCKS_PER_SEC << " seconds." << endl;

    for (int i = 0; i < NUM_VALS - 1; ++i) {
        if (mat[i] > mat[i + 1]) {
            printf("Array is not sorted. The order is incorrect %d %d\n", mat[i], mat[i + 1]);
            return 1;
        }
    }
    printf("Successfully Sorted!!\n");
	
	return 0;
}
