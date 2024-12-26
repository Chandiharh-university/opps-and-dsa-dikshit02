#include <iostream>
#include <vector>
#include <algorithm> // For sort() in bucket sort
using namespace std;

// Merge Sort Implementation
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Bucket Sort Implementation
void bucketSort(vector<int>& arr, int bucketCount) {
    if (arr.empty()) return;

    int maxValue = *max_element(arr.begin(), arr.end());
    int minValue = *min_element(arr.begin(), arr.end());

    int range = (maxValue - minValue + 1) / bucketCount + 1;
    vector<vector<int>> buckets(bucketCount);

    for (int num : arr) {
        int bucketIndex = (num - minValue) / range;
        buckets[bucketIndex].push_back(num);
    }

    arr.clear();
    for (auto& bucket : buckets) {
        sort(bucket.begin(), bucket.end());
        arr.insert(arr.end(), bucket.begin(), bucket.end());
    }
}

// Main Function to Test Merge Sort and Bucket Sort
int main() {
    vector<int> arr1 = {42, 32, 24, 60, 15, 5, 90};
    vector<int> arr2 = arr1;

    cout << "Original Array: ";
    for (int num : arr1) cout << num << " ";
    cout << endl;

    // Using Merge Sort
    mergeSort(arr1, 0, arr1.size() - 1);
    cout << "Sorted Array using Merge Sort: ";
    for (int num : arr1) cout << num << " ";
    cout << endl;

    // Using Bucket Sort
    bucketSort(arr2, 5);
    cout << "Sorted Array using Bucket Sort: ";
    for (int num : arr2) cout << num << " ";
    cout << endl;

    return 0;
}
