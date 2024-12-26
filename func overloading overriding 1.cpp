#include <iostream>
using namespace std;

// Base class with a virtual function
class Base {
public:
    virtual int add(int a, int b, int c) {
        return a + b + c;
    }
};

// Derived class overriding the add function
class Derived : public Base {
public:
    int add(int a, int b, int c) override {
        return (a + b + c) * 2; // Modified behavior: Multiply the sum by 2
    }
};

// A class demonstrating function overloading
class OverloadDemo {
public:
    // Overloaded functions for `a + b + c`
    int add(int a, int b, int c) {
        return a + b + c;
    }

    double add(double a, double b, double c) {
        return a + b + c;
    }

    string add(string a, string b, string c) {
        return a + b + c;
    }
};

int main() {
    // Demonstrating function overloading
    OverloadDemo obj;
    cout << "Overloaded add (int): " << obj.add(1, 2, 3) << endl;
    cout << "Overloaded add (double): " << obj.add(1.1, 2.2, 3.3) << endl;
    cout << "Overloaded add (string): " << obj.add("Hello, ", "C++ ", "World!") << endl;

    // Demonstrating function overriding
    Base* basePtr;
    Derived derivedObj;

    basePtr = &derivedObj; // Base class pointer to Derived object
    cout << "Overridden add in Derived class: " << basePtr->add(1, 2, 3) << endl;

    return 0;
}
