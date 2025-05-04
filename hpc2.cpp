#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Serial Bubble Sort
void serialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped;

    for (int pass = 0; pass < n - 1; pass++) {
        swapped = false;
        for (int i = 0; i < n - 1 - pass; i++) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
        if (!swapped)  // If no swaps, array is already sorted
            break;
    }
}

// Parallel Bubble Sort using OpenMP
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped;

    for (int pass = 0; pass < n - 1; pass++) {
        swapped = false;

        #pragma omp parallel for shared(arr, swapped)
        for (int i = 0; i < n - 1 - pass; i++) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }

        if (!swapped)  // If no swaps, array is already sorted
            break;
    }
}

// Merge function for Merge Sort
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp;
    int left = l, right = m + 1;

    while (left <= m && right <= r) {
        if (arr[left] <= arr[right])
            temp.push_back(arr[left++]);
        else
            temp.push_back(arr[right++]);
    }

    while (left <= m) temp.push_back(arr[left++]);
    while (right <= r) temp.push_back(arr[right++]);


    for (int i = l; i <= r; i++)
        arr[i] = temp[i - l];
}

// Serial Merge Sort
void serialMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        serialMergeSort(arr, l, m);
        serialMergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);
            
            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> originalArray(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; i++)
        cin >> originalArray[i];

    // Copy the array for both sorting algorithms
    vector<int> bubbleArray = originalArray;
    vector<int> mergeArray = originalArray;

    // Measure time for Serial Bubble Sort
    auto bubbleStart = high_resolution_clock::now();
    serialBubbleSort(bubbleArray);
    auto bubbleEnd = high_resolution_clock::now();

    cout << "Sorted array using Serial Bubble Sort: ";
    for (int num : bubbleArray)
        cout << num << " ";
    cout << endl;

    // Measure time for Parallel Bubble Sort
    auto parallelBubbleStart = high_resolution_clock::now();
    parallelBubbleSort(bubbleArray);
    auto parallelBubbleEnd = high_resolution_clock::now();

    cout << "Sorted array using Parallel Bubble Sort: ";
    for (int num : bubbleArray)
        cout << num << " ";
    cout << endl;

    // Measure time for Serial Merge Sort
    auto mergeStart = high_resolution_clock::now();
    serialMergeSort(mergeArray, 0, n - 1);
    auto mergeEnd = high_resolution_clock::now();

    cout << "Sorted array using Serial Merge Sort: ";
    for (int num : mergeArray)
        cout << num << " ";
    cout << endl;

    // Measure time for Parallel Merge Sort
    auto parallelMergeStart = high_resolution_clock::now();
    parallelMergeSort(mergeArray, 0, n - 1);
    auto parallelMergeEnd = high_resolution_clock::now();

    cout << "Sorted array using Parallel Merge Sort: ";
    for (int num : mergeArray)
        cout << num << " ";
    cout << endl;

    // Calculate time taken for each algorithm
    auto bubbleDuration = duration_cast<microseconds>(bubbleEnd - bubbleStart).count() / 1000000.0;
    auto parallelBubbleDuration = duration_cast<microseconds>(parallelBubbleEnd - parallelBubbleStart).count() / 1000000.0;
    auto mergeDuration = duration_cast<microseconds>(mergeEnd - mergeStart).count() / 1000000.0;
    auto parallelMergeDuration = duration_cast<microseconds>(parallelMergeEnd - parallelMergeStart).count() / 1000000.0;

    cout << "Serial Bubble Sort Time (seconds): " << bubbleDuration << endl;
    cout << "Parallel Bubble Sort Time (seconds): " << parallelBubbleDuration << endl;
    cout << "Serial Merge Sort Time (seconds): " << mergeDuration << endl;
    cout << "Parallel Merge Sort Time (seconds): " << parallelMergeDuration << endl;

    // Print the time differences
    cout << "Time difference for Bubble Sort (Serial vs Parallel): " 
         << bubbleDuration - parallelBubbleDuration << " seconds" << endl;

    cout << "Time difference for Merge Sort (Serial vs Parallel): " 
         << mergeDuration - parallelMergeDuration << " seconds" << endl;

    return 0;
}
