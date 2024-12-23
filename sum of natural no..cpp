#include <iostream>
using namespace std;

class NaturalNumberSum {
private:
    int n;

public:
    // Constructor to initialize the value of n
    NaturalNumberSum(int num) : n(num) {}

    // Method to calculate sum using iterative approach
    int sumIterative() {
        int sum = 0;
        for (int i = 1; i <= n; ++i) {
            sum += i;
        }
        return sum;
    }

    // Method to calculate sum using recursive approach
    int sumRecursive(int current) {
        if (current == 0) {
            return 0;
        }
        return current + sumRecursive(current - 1);
    }

    // Method to display the sum
    void displaySum() {
        cout << "Sum of first " << n << " natural numbers (Iterative): " 
             << sumIterative() << endl;

        cout << "Sum of first " << n << " natural numbers (Recursive): " 
             << sumRecursive(n) << endl;
    }
};

int main() {
    int num;

    // User input for the number of natural numbers
    cout << "Enter a positive integer: ";
    cin >> num;

    if (num <= 0) {
        cout << "Please enter a positive integer!" << endl;
        return 1;
    }

    // Create an object of NaturalNumberSum class
    NaturalNumberSum sumObj(num);

    // Display the sum using both methods
    sumObj.displaySum();

    return 0;
}
