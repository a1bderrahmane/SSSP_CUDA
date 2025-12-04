#include <cuco/static_map.cuh>

int main() {
    cuco::static_map<int, int> map{
        10,
        cuco::empty_key<int>{-1},
        cuco::empty_value<int>{-1}
    };
    return 0;
}
