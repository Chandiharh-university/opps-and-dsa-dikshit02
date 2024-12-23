#include <iostream>
using namespace std;
int main()
{
    float a;
    cout<<"Enter the number for table:";
    cin>>a;
    for(int i=1;i<=10;++i){
        cout<<a<<"*"<<i<<"="<<a*i<<endl;
    }
    return 0;
}