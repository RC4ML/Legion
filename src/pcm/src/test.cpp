#include <iostream>
using namespace std;
int a,b;
int *getvalue(){
    // cout<<&x<<endl;
    return &a;
} 
int main(){
    a = 10;
    b = 20;
    // int *x = getvalue(a);
    cout<<&a<<endl;
    int *c = getvalue();
    cout<<*c<<endl;
    return 0;
}