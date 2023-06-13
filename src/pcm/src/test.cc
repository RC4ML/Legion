  /* filename: test.cc */
#include <iostream>
#include <cstdint>
#include "cpucounters.h"
#include "pcm-pcie.h"

using std::cout;
int main(void) {
    //新建一个PCM实例
    PCM* pcm_ = PCM::getInstance();
    auto status = pcm_->program();
    const int TEST_SCALE = 100000000;
    int * arr = new int[TEST_SCALE];
    uint32_t check_sum = 0;
    // PCIeCounterState pcie_count_before = getPCIeCounterState(0, 0);
    // cout<<"pcie_count:"<<pcie_count_before<<endl;
    // MySleepMs(uint(5000));
    //我们希望监控整个系统的硬件数值
    SystemCounterState before_sstate = getSystemCounterState();
        for(int i = 0; i < TEST_SCALE; i++) {
            check_sum = (check_sum + arr[i]) % (1 << 16 - 1);
        }
    SystemCounterState after_sstate = getSystemCounterState();
    // PCIeCounterState pcie_count_after = getPCIeCounterState(0, 0);
    // cout<<"pcie_count:"<<pcie_count_after<<endl;

    // 获取区间内的统计数值
    std::cout << "\tL3 misses: " << getL3CacheMisses(before_sstate, after_sstate) << "\n"
              << "\tDRAM Reads (bytes): " << getBytesReadFromMC(before_sstate, after_sstate) << "\n"
              << "\tDRAM Writes (bytes): " << getBytesWrittenToMC(before_sstate, after_sstate) << "\n";

	delete []arr;
	return 0;
}
