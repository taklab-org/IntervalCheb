#include <iostream>
#include <kv/defint-newtoncotes.hpp>
#include <chrono>

typedef kv::interval<double> itv;

struct Func {
    template <class T> T operator() (const T& x) {
        return 4  / (1 + x * x);
    }
};

int main() {
    using namespace std;
    chrono::system_clock::time_point start, end;
    std::cout.precision(17);

    std::cout << rad(kv::defint_newtoncotes(Func(), itv(0), itv(1), 2, 20)) << "\n";

    start = chrono::system_clock::now();
    (kv::defint_newtoncotes(Func(), itv(0), itv(1), 2, 20));
    end = chrono::system_clock::now();
    double extime = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()/ 1e6);
    printf("time: %lf[s]\n", extime);

    std::cout << rad(kv::defint_newtoncotes(Func(), itv(0), itv(1), 6, 0)) << "\n";

    start = chrono::system_clock::now();
    (kv::defint_newtoncotes(Func(), itv(0), itv(1), 6, 0));
    end = chrono::system_clock::now();
    extime = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()/ 1e6);
    printf("time: %lf[s]\n", extime);


}
