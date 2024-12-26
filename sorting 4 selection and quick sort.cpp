#include <iostream>
using namespace std;

// Function to perform Quick Sort
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Function to perform Selection Sort
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        swap(arr[i], arr[minIdx]);
    }
}

// Function to print an array
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int arr1[] = {10, 80, 30, 90, 40, 50, 70};
    int n1 = sizeof(arr1) / sizeof(arr1[0]);

    int arr2[] = {64, 25, 12, 22, 11};
    int n2 = sizeof(arr2) / sizeof(arr2[0]);

    cout << "Original array for Quick Sort: ";
    printArray(arr1, n1);
    quickSort(arr1, 0, n1 - 1);
    cout << "Sorted array using Quick Sort: ";
    printArray(arr1, n1);

    cout << "\nOriginal array for Selection Sort: ";
    printArray(arr2, n2);
    selectionSort(arr2, n2);
    cout << "Sorted array using Selection Sort: ";
    printArray(arr2, n2);

    return 0;
}
