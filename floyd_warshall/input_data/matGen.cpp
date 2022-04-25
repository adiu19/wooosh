#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

#define INF 99999
using namespace std;


void mat_fill(int *arr, int length)
{
	for (int i = 0; i < length; ++i) 
	{
		arr[i] = rand()%500 + 1;
	}

}

int main(int argc, char** argv) {

    if (argc != 2) {
        cout << "Not the correct number of arguments, retry!" << endl;
        exit(1);
    }

    int DATA_SIZE = atoi(argv[1]);
    int NUM_VALS = DATA_SIZE*DATA_SIZE;
    int *path_dis_mat = (int*) malloc(NUM_VALS * sizeof(int));	
    mat_fill(path_dis_mat, NUM_VALS);


    //Code to write to file
    string out_file_name = "./input_mat_" + to_string(DATA_SIZE) +".txt";
    ofstream fout(out_file_name);
    if(fout.is_open())
        {
            for(int i = 0; path_dis_mat[i] != '\0'; i++)
                {
                    fout << path_dis_mat[i] << " "; 
                }
                fout << endl;
        }
    fout.close();
}