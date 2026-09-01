#include "context.hpp"
#include "config/config.hpp"
#include "common/exception.hpp"
#include "extract/extract.hpp"
#include "extract/trans.hpp"
#include "utils/utils.hpp"
#include "hooks.hpp"
#include "layer.hpp"
#include <cmath>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <android/log.h>
#include "hwme.hpp"
#endif

#include <vulkan/vulkan_core.h>
#include <lsfg_3_1.hpp>
#include <lsfg_3_1p.hpp>

#include <filesystem>
#include <exception>
#include <functional>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <array>

LsContext::LsContext(const Hooks::DeviceInfo& info, VkSwapchainKHR swapchain,
        VkExtent2D extent, const std::vector<VkImage>& swapchainImages)
        : swapchain(swapchain), swapchainImages(swapchainImages),
          extent(extent) {
    // get updated configuration
    auto& conf = Config::activeConf;
    if (!conf.config_file.empty()
            && (
                    !std::filesystem::exists(conf.config_file)
                  || conf.timestamp != std::filesystem::last_write_time(conf.config_file)
            )) {
        std::cerr << "lsfg-vk: Rereading configuration, as it is no longer valid.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // reread configuration
        const std::string file = Utils::getConfigFile();
        const auto name = Utils::getProcessName();
        try {
            Config::updateConfig(file);
            conf = Config::getConfig(name);
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: Failed to update configuration, continuing using old:\n";
            std::cerr << "- " << e.what() << '\n';
        }

        LSFG_3_1P::finalize();
        LSFG_3_1::finalize();

        // print config
        std::cerr << "lsfg-vk: Reloaded configuration for " << name.second << ":\n";
        if (!conf.dll.empty()) std::cerr << "  Using DLL from: " << conf.dll << '\n';
        if (conf.targetFpsEnabled) std::cerr << "  Target FPS: " << conf.targetFps << " (enabled)\n";
        else std::cerr << "  Multiplier: " << conf.multiplier << '\n';
        std::cerr << "  Flow Scale: " << conf.flowScale << '\n';
        std::cerr << "  Performance Mode: " << (conf.performance ? "Enabled" : "Disabled") << '\n';
        std::cerr << "  HDR Mode: " << (conf.hdr ? "Enabled" : "Disabled") << '\n';
        if (conf.e_present != 2) std::cerr << "  ! Present Mode: " << conf.e_present << '\n';

        if (!conf.targetFpsEnabled && conf.multiplier <= 1) return;
    }
    // we could take the format from the swapchain,
    // but honestly this is safer.
    const VkFormat format = conf.hdr
        ? VK_FORMAT_R8G8B8A8_UNORM
        : VK_FORMAT_R16G16B16A16_SFLOAT;

#ifdef __ANDROID__
    // Android path: use AHardwareBuffer-backed images for sharing with framegen.
    // Turnip/Mesa on Android doesn't support OPAQUE_FD export, so we use the
    // AHB path (createContextFromAHB + presentContext with -1 + waitIdle).

    this->frame_0 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    this->frame_1 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    size_t genCount = 0;
    if (conf.targetFpsEnabled) {
        if (conf.targetBaseFps > 0) {
            int need = (conf.targetFps + conf.targetBaseFps -1)/ conf.targetBaseFps; // exact for fixed base, no worst-case overhead -> L5 not L13
            if (need<1) need=1; if (need>8) need=8;
            genCount = (need>0? (size_t)(need-1):0);
            if (genCount==0) genCount=1;
        } else {
            int maxNeed = (conf.targetFps + 9) / 10;
            if (maxNeed < 2) maxNeed = 2;
            if (maxNeed > 8) maxNeed = 8;
            genCount = static_cast<size_t>(maxNeed - 1);
            if (genCount==0) genCount=1;
        }
    } else genCount = static_cast<size_t>(conf.multiplier - 1);
    this->maxGenCount = genCount;
    for (size_t i = 0; i < genCount; ++i)
        this->out_n.emplace_back(info.device, info.physicalDevice,
            extent, format,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    // initialize lsfg
    const std::function<std::vector<uint8_t>(const std::string&)> shaderLoader =
        [](const std::string& name) {
            auto dxbc = Extract::getShader(name);
            auto spirv = Extract::translateShader(dxbc);
            return spirv;
        };

#ifdef __ANDROID__
    // Host-device mode: run framegen on the layer's hooked device instead of
    // an internal instance (which cannot discover the host's ICD, e.g. Turnip
    // injected by a wrapper). Enabled by default for testing; set
    // LSFG_HOST_DEVICE=0 to force the legacy internal-device path.
    const char* hostDeviceEnv = std::getenv("LSFG_HOST_DEVICE");
    this->hostSync = !(hostDeviceEnv && std::string(hostDeviceEnv) == "0");
    if (this->hostSync && !conf.performance) {
        std::cerr << "lsfg-vk: Host-device mode requires performance mode, using internal device.\n";
        this->hostSync = false;
    }
#endif

    setenv("DISABLE_LSFG", "1", 1); // NOLINT
    size_t initGen = this->maxGenCount;
#ifndef __ANDROID__
    if (!conf.targetFpsEnabled) initGen = static_cast<size_t>(conf.multiplier - 1);
#endif

#ifdef __ANDROID__
    if (this->hostSync) {
        LSFG_3_1P::initializeFromHost(
            info.instance, info.physicalDevice, info.device,
            conf.hdr, 1.0F / conf.flowScale, initGen,
            shaderLoader);
        std::cerr << "lsfg-vk: Host-device mode enabled (framegen shares the hooked device).\n";
    } else
#endif
    {
        auto* lsfgInitialize = LSFG_3_1::initialize;
        if (conf.performance)
            lsfgInitialize = LSFG_3_1P::initialize;
        lsfgInitialize(
            Utils::getDeviceUUID(info.physicalDevice),
            conf.hdr, 1.0F / conf.flowScale, initGen,
            shaderLoader
        );
    }

    // Create framegen context using AHB sharing
    std::vector<AHardwareBuffer*> outAhbs;
    outAhbs.reserve(this->maxGenCount);
    for (size_t i = 0; i < this->maxGenCount; ++i)
        outAhbs.push_back(this->out_n.at(i).getAhb());

    int32_t ctxId;
    if (conf.performance)
        ctxId = LSFG_3_1P::createContextFromAHB(
            this->frame_0.getAhb(), this->frame_1.getAhb(),
            outAhbs, extent, format);
    else
        ctxId = LSFG_3_1::createContextFromAHB(
            this->frame_0.getAhb(), this->frame_1.getAhb(),
            outAhbs, extent, format);

    this->lsfgCtxId = std::shared_ptr<int32_t>(
        new int32_t(ctxId),
        [perf = conf.performance](const int32_t* id) {
            if (perf) LSFG_3_1P::deleteContext(*id);
            else LSFG_3_1::deleteContext(*id);
        }
    );

    unsetenv("DISABLE_LSFG"); // NOLINT

    std::cerr << "lsfg-vk: Android AHB context created (id=" << ctxId << ")\n";

#else
    // Desktop Linux path: use OPAQUE_FD-based image sharing
    {
        size_t genCount = 0;
        if (conf.targetFpsEnabled) {
            if (conf.targetBaseFps>0) {
                int need=(conf.targetFps+conf.targetBaseFps-1)/conf.targetBaseFps; if(need<1)need=1; if(need>8)need=8; genCount=(need>0?(size_t)(need-1):0); if(genCount==0)genCount=1;
            } else {
                int maxNeeded = (conf.targetFps + 9) / 10;
                if (maxNeeded < 2) maxNeeded = 2;
                if (maxNeeded > 8) maxNeeded = 8;
                genCount = static_cast<size_t>(maxNeeded - 1);
                if (genCount==0) genCount=1;
            }
        } else genCount = static_cast<size_t>(conf.multiplier - 1);
        this->maxGenCount = genCount;
    }
    std::array<int, 2> fds{};
    this->frame_0 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(0));
    this->frame_1 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(1));

    std::vector<int> outFds(this->maxGenCount);
    for (size_t i = 0; i < this->maxGenCount; ++i)
        this->out_n.emplace_back(info.device, info.physicalDevice,
            extent, format,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
            &outFds.at(i));

    // initialize lsfg
    auto* lsfgInitialize = LSFG_3_1::initialize;
    auto* lsfgCreateContext = LSFG_3_1::createContext;
    auto* lsfgDeleteContext = LSFG_3_1::deleteContext;
    if (conf.performance) {
        lsfgInitialize = LSFG_3_1P::initialize;
        lsfgCreateContext = LSFG_3_1P::createContext;
        lsfgDeleteContext = LSFG_3_1P::deleteContext;
    }

    setenv("DISABLE_LSFG", "1", 1); // NOLINT

    lsfgInitialize(
        Utils::getDeviceUUID(info.physicalDevice),
        conf.hdr, 1.0F / conf.flowScale, this->maxGenCount,
        [](const std::string& name) {
            auto dxbc = Extract::getShader(name);
            auto spirv = Extract::translateShader(dxbc);
            return spirv;
        }
    );

    this->lsfgCtxId = std::shared_ptr<int32_t>(
        new int32_t(lsfgCreateContext(fds.at(0), fds.at(1), outFds, extent, format)),
        [lsfgDeleteContext = lsfgDeleteContext](const int32_t* id) {
            lsfgDeleteContext(*id);
        }
    );

    unsetenv("DISABLE_LSFG"); // NOLINT
#endif

    // prepare render passes
    this->cmdPool = Mini::CommandPool(info.device, info.queue.first);
    size_t poolGen = this->maxGenCount;
    if (poolGen==0) poolGen = conf.targetFpsEnabled ? 1 : static_cast<size_t>(conf.multiplier - 1);
    size_t allocGen = std::max<size_t>(poolGen, 7);
    for (size_t i = 0; i < 8; i++) {
        auto& pass = this->passInfos.at(i);
        pass.renderSemaphores.resize(allocGen);
        pass.acquireSemaphores.resize(allocGen);
        pass.postCopyBufs.resize(allocGen);
        pass.postCopySemaphores.resize(allocGen);
        pass.prevPostCopySemaphores.resize(allocGen);
    }
    this->smoothedRealFps = 60.0f;
    if (conf.targetFpsEnabled) this->smoothedRealFps = 30.0f;
}

VkResult LsContext::present(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx) {
    const auto& conf = Config::activeConf;
    auto& pass = this->passInfos.at(this->frameIdx % 8);

#ifdef __ANDROID__
    // Android path: synchronous frame generation using waitIdle()
    // instead of OPAQUE_FD semaphore export which Turnip doesn't support.
    auto tPresentStart = std::chrono::steady_clock::now();
    auto tGenStart = tPresentStart;
    auto tGenEnd = tPresentStart;
    bool timingEnabled = conf.timingDebug;
    LSFG::HwMe::setEnabled(conf.hwme);
    LSFG::HwMe::configure(conf.hwmeMaxMv, conf.hwmeDebug);

    // target fps pacing: compute neededTotal so R+G=T~=target even when R drops (shader compile)
    // e.g. target 60: R30->G30(2x), R25->G35(2.4x=>round 2=>~60), R20->G40(3x), R18->G42(3x=>54 avg, will dither 3/4 to hit 60)
    size_t neededTotal = conf.multiplier;
    size_t neededGen = 0;
    if (conf.targetFpsEnabled) {
        float base = 0;
        if (conf.targetBaseFps > 0) {
            base = (float)conf.targetBaseFps; // manual override for test: avoids FIFO throttle artifact (15 vs 30)
            this->smoothedRealFps = base;
        } else {
            if (this->hasLastPresent) {
                auto dtNs = std::chrono::duration_cast<std::chrono::nanoseconds>(tPresentStart - this->lastPresentTime).count();
                if (dtNs>0) {
                    float inst = 1e9f/(float)dtNs;
                    if (inst<5) inst=5; if (inst>480) inst=480;
                    if (this->frameIdx<3) this->smoothedRealFps=inst;
                    else this->smoothedRealFps = this->smoothedRealFps*0.6f + inst*0.4f;
                }
            } else if (this->frameIdx==0) this->smoothedRealFps = (float)conf.targetFps/2.0f;
            base = this->smoothedRealFps;
        }
        float ratio = (float)conf.targetFps / base;
        int need = (int)std::round(ratio);
        if (need<1) need=1; if (need<2 && ratio>1.35f) need=2;
        if (need>(int)(this->maxGenCount+1)) need=(int)(this->maxGenCount+1);
        if (need>8) need=8;
        if (conf.targetBaseFps==0) {
            static thread_local int pendNeed=0, pendCnt=0;
            if (need!=this->lastNeeded && this->frameIdx>3) {
                if (pendNeed!=need){pendNeed=need; pendCnt=1;}
                else pendCnt++;
                if (pendCnt<2) need=this->lastNeeded; else pendCnt=0;
            } else {pendNeed=need; pendCnt=0;}
        }
        neededTotal=(size_t)need;
        neededGen = neededTotal>0?neededTotal-1:0;
        this->lastNeeded=(int)neededTotal;
    } else {
        neededTotal=conf.multiplier;
        neededGen=neededTotal>0?neededTotal-1:0;
        if (neededGen>this->maxGenCount) neededGen=this->maxGenCount;
    }

    // 1. copy swapchain image to frame_0/frame_1
    //    Use a simple semaphore (no fd export) to synchronize the copy
    pass.preCopySemaphores.at(1) = Mini::Semaphore(info.device);
    pass.preCopyBuf = Mini::CommandBuffer(info.device, this->cmdPool);
    pass.preCopyBuf.begin();

    Utils::copyImage(pass.preCopyBuf.handle(),
        this->swapchainImages.at(presentIdx),
        this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
        this->extent.width, this->extent.height,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        true, false);

    pass.preCopyBuf.end();

    std::vector<VkSemaphore> gameRenderSemaphores2 = gameRenderSemaphores;
    if (this->frameIdx > 0)
        gameRenderSemaphores2.emplace_back(this->passInfos.at((this->frameIdx - 1) % 8)
            .preCopySemaphores.at(1).handle());

    // Submit the copy and wait for it to complete synchronously.
    // On Android we need the copy to finish before calling presentContext
    // because there's no FD-based cross-device semaphore to chain them.
    pass.preCopyBuf.submit(info.queue.second,
        gameRenderSemaphores2,
        { pass.preCopySemaphores.at(1).handle() });

    bool hwUsed = false;
    if (timingEnabled) tGenStart = std::chrono::steady_clock::now();
    if (neededGen>0 && this->frameIdx > 0 && LSFG::HwMe::isEnabled()) {
        if (LSFG::HwMe::available()) {
            Layer::ovkQueueSubmit(info.queue.second, 0, nullptr, VK_NULL_HANDLE);
            AHardwareBuffer* prev = (this->frameIdx % 2 == 0) ? this->frame_1.getAhb() : this->frame_0.getAhb();
            AHardwareBuffer* cur  = (this->frameIdx % 2 == 0) ? this->frame_0.getAhb() : this->frame_1.getAhb();
            std::vector<AHardwareBuffer*> outs;
            outs.reserve(neededGen);
            for (size_t oi=0; oi<neededGen && oi<this->out_n.size(); ++oi) outs.push_back(this->out_n.at(oi).getAhb());
            if (LSFG::HwMe::generate(prev, cur, outs, this->extent.width, this->extent.height)) {
                std::cerr << "lsfg-vk: HWME generated " << outs.size() << " frame(s)\n";
                hwUsed = true;
            } else {
                std::cerr << "lsfg-vk: HWME failed, falling back to SW\n";
            }
        } else {
            static bool logged = false;
            if (!logged) {
                std::cerr << "lsfg-vk: HWME requested but not available (no QCOM block), using SW\n";
                logged = true;
            }
        }
    }

    if (!hwUsed) {
        if (neededGen>0) {
            if (this->hostSync) {
                LSFG_3_1P::presentContextNative(*this->lsfgCtxId,
                    pass.preCopySemaphores.at(1).handle(), {});
                LSFG_3_1P::waitFrame(*this->lsfgCtxId);
            } else {
                Layer::ovkQueueSubmit(info.queue.second, 0, nullptr, VK_NULL_HANDLE);
                std::vector<int> noOutSems;
                if (conf.performance)
                    LSFG_3_1P::presentContext(*this->lsfgCtxId, -1, noOutSems);
                else
                    LSFG_3_1::presentContext(*this->lsfgCtxId, -1, noOutSems);
                if (conf.performance)
                    LSFG_3_1P::waitIdle();
                else
                    LSFG_3_1::waitIdle();
            }
        }
        if (timingEnabled) tGenEnd = std::chrono::steady_clock::now();
    } else {
        if (timingEnabled) tGenEnd = std::chrono::steady_clock::now();
    }

    // 4. Copy generated frames to swapchain images and present them (variable needGen for target mode keeps sync)
    for (size_t i = 0; i < neededGen; i++) {
        // acquire next swapchain image
        pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
        uint32_t imageIdx{};
        auto res = Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
            pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to acquire next swapchain image");

        // copy output image to swapchain image
        pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);
        pass.postCopyBufs.at(i).begin();

        Utils::copyImage(pass.postCopyBufs.at(i).handle(),
            this->out_n.at(i).handle(),
            this->swapchainImages.at(imageIdx),
            this->extent.width, this->extent.height,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            false, true);

        pass.postCopyBufs.at(i).end();
        pass.postCopyBufs.at(i).submit(info.queue.second,
            { pass.acquireSemaphores.at(i).handle() },
            { pass.postCopySemaphores.at(i).handle() });

        // present swapchain image
        VkSemaphore postCopySem = pass.postCopySemaphores.at(i).handle();
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = i == 0 ? pNext : nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &postCopySem,
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &imageIdx,
        };
        res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to present swapchain image");
    }

    // 5. present actual next frame (the real capture, not a generated one)
    VkResult res = VK_SUCCESS;
    if (neededGen>0) {
        pass.prevPostCopySemaphores.at(neededGen - 1) = Mini::Semaphore(info.device);
        VkSemaphore lastPostCopySem = pass.postCopySemaphores.at(neededGen - 1).handle();
        const VkPresentInfoKHR finalPresentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &lastPostCopySem,
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &presentIdx,
        };
        res = Layer::ovkQueuePresentKHR(queue, &finalPresentInfo);
    } else {
        VkSemaphore preCopySem = pass.preCopySemaphores.at(1).handle();
        const VkPresentInfoKHR finalPresentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = pNext,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &preCopySem,
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &presentIdx,
        };
        res = Layer::ovkQueuePresentKHR(queue, &finalPresentInfo);
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw LSFG::vulkan_error(res, "Failed to present swapchain image");

    if (timingEnabled) {
        auto tPresentEnd = std::chrono::steady_clock::now();
        int genLatencyMs = (int)std::chrono::duration<double, std::milli>(tGenEnd - tGenStart).count();
        int totalMs = (int)std::chrono::duration<double, std::milli>(tPresentEnd - tPresentStart).count();
        int frametimeMs = 0, fps = 0;
        int realFps = 0, genFps = 0, totalFps = 0;
        if (this->hasLastPresent) {
            auto dtNs = std::chrono::duration_cast<std::chrono::nanoseconds>(tPresentStart - this->lastPresentTime).count();
            if (dtNs > 0) {
                frametimeMs = (int)(dtNs / 1000000);
                realFps = (int)(1e9 / dtNs);
                genFps = realFps * (int)neededGen;
                totalFps = realFps * (int)neededTotal;
                fps = totalFps;
            }
        }
        this->lastPresentTime = tPresentStart;
        this->hasLastPresent = true;
        char type = (neededTotal > 1 ? 'G' : 'R');
        char mode = hwUsed ? 'H' : 'S';
        // R=real, G=gen, T=total -> target mode shows T~=target even when R drops (shader compile) => G substitutes
        if (conf.targetFpsEnabled)
            std::cerr << "lsfg-timing FPS (R " << realFps << " G " << genFps << " T " << totalFps << ") FT (" << frametimeMs << ") T (" << type << ") M (" << mode << ") L (" << genLatencyMs << ") total (" << totalMs << ") [" << this->extent.width << "x" << this->extent.height << " x" << neededTotal << " target " << conf.targetFps << "]\n";
        else
            std::cerr << "lsfg-timing FPS (R " << realFps << " G " << genFps << " T " << totalFps << ") FT (" << frametimeMs << ") T (" << type << ") M (" << mode << ") L (" << genLatencyMs << ") total (" << totalMs << ") [" << this->extent.width << "x" << this->extent.height << " x" << neededTotal << "]\n";
    }
    if (conf.targetFpsEnabled && !timingEnabled) {
        this->lastPresentTime = tPresentStart;
        this->hasLastPresent = true;
    }

    this->frameIdx++;
    return res;

#else
    // Desktop Linux path: OPAQUE_FD semaphore-based synchronization (also supports target mode)
    auto tPresentStart = std::chrono::steady_clock::now();
    size_t neededTotal = conf.multiplier;
    size_t neededGen = 0;
    if (conf.targetFpsEnabled) {
        float base = 0;
        if (conf.targetBaseFps > 0) { base=(float)conf.targetBaseFps; this->smoothedRealFps=base; }
        else {
            if (this->hasLastPresent) {
                auto dtNs = std::chrono::duration_cast<std::chrono::nanoseconds>(tPresentStart - this->lastPresentTime).count();
                if (dtNs>0){ float inst=1e9f/(float)dtNs; if(inst<5)inst=5; if(inst>480)inst=480; if(this->frameIdx<3) this->smoothedRealFps=inst; else this->smoothedRealFps=this->smoothedRealFps*0.6f+inst*0.4f; }
            } else if(this->frameIdx==0) this->smoothedRealFps=(float)conf.targetFps/2.0f;
            base=this->smoothedRealFps;
        }
        float ratio=(float)conf.targetFps/base; int need=(int)std::round(ratio); if(need<1)need=1; if(need<2&&ratio>1.35f)need=2; if(need>(int)(this->maxGenCount+1))need=(int)(this->maxGenCount+1); if(need>8)need=8; neededTotal=(size_t)need; neededGen=need>0?need-1:0; this->lastNeeded=need;
    } else { neededTotal=conf.multiplier; neededGen=neededTotal>0?neededTotal-1:0; if(neededGen>this->maxGenCount) neededGen=this->maxGenCount; }

    // 1. copy swapchain image to frame_0/frame_1
    int preCopySemaphoreFd{};
    pass.preCopySemaphores.at(0) = Mini::Semaphore(info.device, &preCopySemaphoreFd);
    pass.preCopySemaphores.at(1) = Mini::Semaphore(info.device);
    pass.preCopyBuf = Mini::CommandBuffer(info.device, this->cmdPool);
    pass.preCopyBuf.begin();

    Utils::copyImage(pass.preCopyBuf.handle(),
        this->swapchainImages.at(presentIdx),
        this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
        this->extent.width, this->extent.height,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        true, false);

    pass.preCopyBuf.end();

    std::vector<VkSemaphore> gameRenderSemaphores2 = gameRenderSemaphores;
    if (this->frameIdx > 0)
        gameRenderSemaphores2.emplace_back(this->passInfos.at((this->frameIdx - 1) % 8)
            .preCopySemaphores.at(1).handle());
    pass.preCopyBuf.submit(info.queue.second,
        gameRenderSemaphores2,
        { pass.preCopySemaphores.at(0).handle(),
          pass.preCopySemaphores.at(1).handle() });

    // 2. render intermediary frames
    bool skipGen = (neededGen==0);
    std::vector<int> renderSemaphoreFds(neededGen);
    for (size_t i = 0; i < neededGen; ++i)
        pass.renderSemaphores.at(i) = Mini::Semaphore(info.device, &renderSemaphoreFds.at(i));

    if (!skipGen) {
    if (conf.performance)
        LSFG_3_1P::presentContext(*this->lsfgCtxId,
            preCopySemaphoreFd,
            renderSemaphoreFds);
    else
        LSFG_3_1::presentContext(*this->lsfgCtxId,
            preCopySemaphoreFd,
            renderSemaphoreFds);
    }

    for (size_t i = 0; i < neededGen; i++) {
        // 3. acquire next swapchain image
        pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
        uint32_t imageIdx{};
        auto res = Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
            pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to acquire next swapchain image");

        // 4. copy output image to swapchain image
        pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.prevPostCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);
        pass.postCopyBufs.at(i).begin();

        Utils::copyImage(pass.postCopyBufs.at(i).handle(),
            this->out_n.at(i).handle(),
            this->swapchainImages.at(imageIdx),
            this->extent.width, this->extent.height,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            false, true);

        pass.postCopyBufs.at(i).end();
        pass.postCopyBufs.at(i).submit(info.queue.second,
            { pass.acquireSemaphores.at(i).handle(),
              pass.renderSemaphores.at(i).handle() },
            { pass.postCopySemaphores.at(i).handle(),
              pass.prevPostCopySemaphores.at(i).handle() });

        // 5. present swapchain image
        std::vector<VkSemaphore> waitSemaphores{ pass.postCopySemaphores.at(i).handle() };
        if (i != 0) waitSemaphores.emplace_back(pass.prevPostCopySemaphores.at(i - 1).handle());

        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = i == 0 ? pNext : nullptr, // only set on first present
            .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
            .pWaitSemaphores = waitSemaphores.data(),
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &imageIdx,
        };
        res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to present swapchain image");
    }

    // 6. present actual next frame
    VkResult resDesktop = VK_SUCCESS;
    if (neededGen>0) {
        VkSemaphore lastPrevPostCopySemaphore = pass.prevPostCopySemaphores.at(neededGen - 1).handle();
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &lastPrevPostCopySemaphore,
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &presentIdx,
        };
        resDesktop = Layer::ovkQueuePresentKHR(queue, &presentInfo);
    } else {
        VkSemaphore preCopySem = pass.preCopySemaphores.at(1).handle();
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &preCopySem,
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &presentIdx,
        };
        resDesktop = Layer::ovkQueuePresentKHR(queue, &presentInfo);
    }
    if (resDesktop != VK_SUCCESS && resDesktop != VK_SUBOPTIMAL_KHR)
        throw LSFG::vulkan_error(resDesktop, "Failed to present swapchain image");
    this->lastPresentTime = tPresentStart; this->hasLastPresent = true;
    this->frameIdx++;
    return resDesktop;
#endif
}
