#include <iostream>
using namespace std;

// Function to perform Insertion Sort
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        // Move elements of arr[0..i-1], that are greater than key,
        // to one position ahead of their current position
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Function to perform Bubble Sort
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j+1]
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to print an array
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int arr1[] = {12, 11, 13, 5, 6};
    int n1 = sizeof(arr1) / sizeof(arr1[0]);

    int arr2[] = {20, 7, 15, 3, 8};
    int n2 = sizeof(arr2) / sizeof(arr2[0]);

    cout << "Original array for Insertion Sort: ";
    printArray(arr1, n1);
    insertionSort(arr1, n1);
    cout << "Sorted array using Insertion Sort: ";
    printArray(arr1, n1);

    cout << "\nOriginal array for Bubble Sort: ";
    printArray(arr2, n2);
    bubbleSort(arr2, n2);
    cout << "Sorted array using Bubble Sort: ";
    printArray(arr2, n2);

    return 0;
}
