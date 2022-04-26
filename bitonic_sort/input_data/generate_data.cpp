#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>


using namespace std;

void array_fill(int *arr, int length)
{
	for (int i = 0; i < length; ++i) 
	{
		arr[i] = rand()%500 + 1;
	}
}

int main(int argc, char** argv) {

    // Step1 : Read the input from cmd line
    if (argc != 2) {
        cout << "Not the correct number of arguments, retry!" << endl;
        exit(1);
    }
    int input = atoi(argv[1]);
    int NUM_VALS  = input*input;
 
    int *values = (int*) malloc(NUM_VALS * sizeof(int));	
	array_fill(values, NUM_VALS);
    //Code to write to file
    string out_file_name = "input_array_" + to_string(input) +".txt";
    ofstream fout(out_file_name);
    if(fout.is_open())
        {
            for(int i = 0; values[i] != '\0'; i++)
                {
                    fout << values[i] << " "; 
                }
                fout << endl;
        }
    fout.close();

}
