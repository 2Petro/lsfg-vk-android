#pragma once

#include <cstdint>
#include <vector>

#ifdef __ANDROID__

struct AHardwareBuffer;

namespace LSFG::HwMe {

    bool isEnabled();

    bool available();

    bool generate(AHardwareBuffer* prev, AHardwareBuffer* cur,
        const std::vector<AHardwareBuffer*>& outs,
        uint32_t width, uint32_t height);

    // single-output convenience
    bool generate(AHardwareBuffer* colA, AHardwareBuffer* colB,
        AHardwareBuffer* out, float alpha,
        uint32_t width, uint32_t height);

}

#endif
