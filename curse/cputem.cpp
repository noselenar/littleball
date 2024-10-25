#include <random>
#include <chrono>
#include <iostream>
#include <unistd.h> // 使用 getpid() on Unix, Windows 上用 GetCurrentProcessId

int getHardwareBasedSeed() {
    // 组合当前进程 ID 和时间的纳秒部分生成种子
    auto time_point = std::chrono::high_resolution_clock::now();
    auto time_seed = time_point.time_since_epoch().count();
    return static_cast<int>((time_seed ^ getpid()) & 0xFFFFFFFF);
}

int main() {
    int seed = getHardwareBasedSeed();
    std::mt19937 rng(seed);
    
    // 生成 0.0 到 1.0 之间的随机 double
    std::uniform_real_distribution<double> dist(0.0, 1.0); 
    std::cout << dist(rng) << std::endl;

    // 生成特定范围内的随机 double，例如 5.0 到 10.0
    std::uniform_real_distribution<double> distRange(5.0, 10.0);
    std::cout << distRange(rng) << std::endl;

    return 0;
}
