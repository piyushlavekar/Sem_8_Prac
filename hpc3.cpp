#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Serial Min
int serialMin(vector<int>& vec) {
    int min_val = vec[0];
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] < min_val) {
            min_val = vec[i];
        }
    }
    return min_val;
}

// Serial Max
int serialMax(vector<int>& vec) {
    int max_val = vec[0];
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }
    return max_val;
}

// Serial Sum
int serialSum(vector<int>& vec) {
    int sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}
    
// Serial Average
float serialAverage(vector<int>& vec) {
    int sum = serialSum(vec);
    return float(sum) / vec.size();
}

// Parallel Min
int parallelMin(vector<int>& vec) {
    int min_val = vec[0];

    #pragma omp parallel for reduction(min:min_val)
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] < min_val) {
            min_val = vec[i];
        }
    }

    return min_val;
}

// Parallel Max
int parallelMax(vector<int>& vec) {
    int max_val = vec[0];

    #pragma omp parallel for reduction(max:max_val)
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }

    return max_val;
}

// Parallel Sum
int parallelSum(vector<int>& vec) {
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }

    return sum;
}

// Parallel Average
float parallelAverage(vector<int>& vec) {
    int sum = parallelSum(vec);
    return float(sum) / vec.size();
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> vec(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; ++i) {
        cin >> vec[i];
    }

    // Measure time for Serial calculations
    auto serialStart = high_resolution_clock::now();
    int serialMinVal = serialMin(vec);
    int serialMaxVal = serialMax(vec);
    int serialSumVal = serialSum(vec);
    float serialAvgVal = serialAverage(vec);
    auto serialEnd = high_resolution_clock::now();

    // Measure time for Parallel calculations
    auto parallelStart = high_resolution_clock::now();
    int parallelMinVal = parallelMin(vec);
    int parallelMaxVal = parallelMax(vec);
    int parallelSumVal = parallelSum(vec);
    float parallelAvgVal = parallelAverage(vec);
    auto parallelEnd = high_resolution_clock::now();

    // Print the results
    cout << "Serial results:" << endl;
    cout << "Minimum value: " << serialMinVal << endl;
    cout << "Maximum value: " << serialMaxVal << endl;
    cout << "Sum of values: " << serialSumVal << endl;
    cout << "Average of values: " << serialAvgVal << endl;

    cout << "\nParallel results:" << endl;
    cout << "Minimum value: " << parallelMinVal << endl;
    cout << "Maximum value: " << parallelMaxVal << endl;
    cout << "Sum of values: " << parallelSumVal << endl;
    cout << "Average of values: " << parallelAvgVal << endl;

    // Calculate time taken for serial and parallel executions
    auto serialDuration = duration_cast<microseconds>(serialEnd - serialStart).count() / 1000000.0;
    auto parallelDuration = duration_cast<microseconds>(parallelEnd - parallelStart).count() / 1000000.0;

    cout << "\nTime for Serial execution (seconds): " << serialDuration << endl;
    cout << "Time for Parallel execution (seconds): " << parallelDuration << endl;

    // Time difference between Serial and Parallel execution
    cout << "Time difference (Serial - Parallel): " << serialDuration - parallelDuration << " seconds" << endl;

    return 0;
}
