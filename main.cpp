#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

// -----------
#include <climits>
#include <bits/stdc++.h>
// -----------

using namespace std;

bool distanceComparison(tuple<int, double> v1, tuple<int, double> v2)
{
    return get<1>(v1) < get<1>(v2);
}

int* KNN(ArffData* dataset)
{
    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    
    // The following two lines show the syntax to retrieve the attribute values and the class value for a given instance in the dataset
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement the KNN here, fill the predictions array

    int k = 5; // number of neighbors to use for prediction

    for(int i = 0; i < dataset->num_instances(); i++)
    {

        // getNeighbors()
        int neighbors[5];
        tuple<int, double>* distances = (tuple<int, double>*)malloc(dataset->num_instances() * sizeof(tuple<int, double>));
        for(int j = 0; j < dataset->num_instances(); j++)
        {

            // map(dataset, (train) => (train, distance(train)))
            if(j == i)
            {
                distances[j] = tuple<int, double>(j, INT_MAX);
                continue;
            }

            distances[j] = tuple<int, double>(
                j, 
                sqrt(
                    pow(dataset->get_instance(i)->get(0)->operator float() - dataset->get_instance(j)->get(0)->operator float(),  2) +
                    pow(dataset->get_instance(i)->get(1)->operator float() - dataset->get_instance(j)->get(1)->operator float(),  2) +
                    pow(dataset->get_instance(i)->get(2)->operator float() - dataset->get_instance(j)->get(2)->operator float(),  2) +
                    pow(dataset->get_instance(i)->get(3)->operator float() - dataset->get_instance(j)->get(3)->operator float(),  2) +
                    pow(dataset->get_instance(i)->get(4)->operator float() - dataset->get_instance(j)->get(4)->operator float(),  2) +
                    pow(dataset->get_instance(i)->get(5)->operator float() - dataset->get_instance(j)->get(5)->operator float(),  2) +
                    pow(dataset->get_instance(i)->get(6)->operator float() - dataset->get_instance(j)->get(6)->operator float(),  2) 
                )
            );
        }

        // distances.sort()
        std::sort(distances, distances + dataset->num_instances(), distanceComparison);

        // distances.take(5)
        for(int x = 0; x < k; x++)
        {
            neighbors[x] = get<0>(distances[x]);
        }

        // map(neighbors, (x) => neighbors.class)
        int outputValues[k];
        for(int j = 0; j < k; j++)
        {
            outputValues[j] = dataset->get_instance(neighbors[j])->get(dataset->num_attributes() - 1)->operator int32();
        }

        // maxCount()
        int freqFalse = 0;
        int freqTrue = 0;
        for (int i = 0; i < k; i++) 
        { 
            if (outputValues[i] == 0) 
            {
                freqFalse++;
            }
            else if (outputValues[i] == 1) 
            {
                freqTrue++;
            }
        }

        int result = (freqFalse > freqTrue ? 0 : 1);

        predictions[i] = result;

    }
    
    return predictions;
}

/*
int* OpenMP_KNN(ArffData* dataset)
{
    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    
    // The following two lines show the syntax to retrieve the attribute values and the class value for a given instance in the dataset
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement the KNN here, fill the predictions array

    
    
    return predictions;
}
*/

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }
    
    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    // Get the class predictions
    int* predictions = KNN(dataset);
    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
