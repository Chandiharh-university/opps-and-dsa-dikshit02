#include <iostream>
using namespace std;

int main() {
    int size;

    // Input the size of the array
    cout << "Enter the size of the array: ";
    cin >> size;

    // Declare the array
    int arr[size];

    // Input the array elements
    cout << "Enter " << size << " elements of the array: ";
    for (int i = 0; i < size; i++) {
        cin >> arr[i];
    }

    int index;
    // Input the index to access
    cout << "Enter the index to retrieve the value: ";
    cin >> index;

    // Check if the index is within bounds
    if (index >= 0 && index < size) {
        cout << "Value at index " << index << " is: " << arr[index] << endl;
    } else {
        cout << "Error: Index out of bounds!" << endl;
    }

    return 0;
}
