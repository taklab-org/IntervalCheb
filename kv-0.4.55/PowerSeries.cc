#include <iostream>
#include <kv/defint.hpp>
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

    std::cout << rad(kv::defint(Func(), itv(0), itv(1), 12, 10)) << "\n";

    start = chrono::system_clock::now();
    kv::defint(Func(), itv(0), itv(1), 12, 10);
    end = chrono::system_clock::now();
    double extime = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()/ 1e6);
    printf("time: %lf[s]\n", extime);



    std::cout << rad(kv::defint_autostep(Func(), itv(0), itv(1), 12)) << "\n";

    start = chrono::system_clock::now();
    kv::defint_autostep(Func(), itv(0), itv(1), 12);
    end = chrono::system_clock::now();
    extime = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()/ 1e6);
    printf("time: %lf[s]\n", extime);

}
