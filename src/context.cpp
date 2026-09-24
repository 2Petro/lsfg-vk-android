#include "context.hpp"
#include "config/config.hpp"
#include "common/exception.hpp"
#include "extract/extract.hpp"
#include "extract/trans.hpp"
#include "utils/utils.hpp"
#include "hooks.hpp"
#include "layer.hpp"

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <android/log.h>
#endif

#include <vulkan/vulkan_core.h>
#include <lsfg_3_1.hpp>
#include <lsfg_3_1p.hpp>

#include <filesystem>
#include <exception>
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
        VkExtent2D extent, const std::vector<VkImage>& swapchainImages,
        VkFormat swapFormat)
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
        std::cerr << "  Multiplier: " << conf.multiplier << '\n';
        std::cerr << "  Flow Scale: " << conf.flowScale << '\n';
        std::cerr << "  Performance Mode: " << (conf.performance ? "Enabled" : "Disabled") << '\n';
        std::cerr << "  HDR Mode: " << (conf.hdr ? "Enabled" : "Disabled") << '\n';
        if (conf.e_present != 2) std::cerr << "  ! Present Mode: " << conf.e_present << '\n';

        if (conf.multiplier <= 1) return;
    }
    // Second framegen option: NPU (RIFE ONNX on Hexagon HTP). Fixed RGBA8
    // internal format — the pack-back shader is proven against 8-bit —
    // regardless of the HDR flag. The DLL/shader path keeps its logic below.
    this->useNpu = !conf.npu_model.empty();
    // we could take the format from the swapchain,
    // but honestly this is safer.
    VkFormat format = conf.hdr
        ? VK_FORMAT_R8G8B8A8_UNORM
        : VK_FORMAT_R16G16B16A16_SFLOAT;
    if (this->useNpu) {
        format = VK_FORMAT_R8G8B8A8_UNORM;
        this->npuOutFormat = format;
        std::cerr << "lsfg-vk: NPU framegen selected (" << conf.npu_model << ")\n";
    }

#ifdef __ANDROID__
    // Android path: use AHardwareBuffer-backed images for sharing with framegen.
    // Turnip/Mesa on Android doesn't support OPAQUE_FD export, so we use the
    // AHB path (createContextFromAHB + presentContext with -1 + waitIdle).

    this->frame_0 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    this->frame_1 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    for (size_t i = 0; i < static_cast<size_t>(conf.multiplier - 1); ++i)
        this->out_n.emplace_back(info.device, info.physicalDevice,
            extent, format,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    // initialize lsfg
    auto* lsfgInitialize = LSFG_3_1::initialize;
    auto* lsfgDeleteContext = LSFG_3_1::deleteContext;
        lsfgInitialize = LSFG_3_1P::initialize;
        lsfgDeleteContext = LSFG_3_1P::deleteContext;

    setenv("DISABLE_LSFG", "1", 1); // NOLINT

    lsfgInitialize(
        Utils::getDeviceUUID(info.physicalDevice),
        conf.hdr, 1.0F / conf.flowScale, conf.multiplier - 1,
        [](const std::string& name) {
            auto dxbc = Extract::getShader(name);
            auto spirv = Extract::translateShader(dxbc);
            return spirv;
        }
    );

    // Create framegen context using AHB sharing
    std::vector<AHardwareBuffer*> outAhbs;
    outAhbs.reserve(conf.multiplier - 1);
    for (size_t i = 0; i < static_cast<size_t>(conf.multiplier - 1); ++i)
        outAhbs.push_back(this->out_n.at(i).getAhb());

    int32_t ctxId;
        ctxId = LSFG_3_1P::createContextFromAHB(
            this->frame_0.getAhb(), this->frame_1.getAhb(),
            outAhbs, extent, format);

    this->lsfgCtxId = std::shared_ptr<int32_t>(
        new int32_t(ctxId),
        [lsfgDeleteContext = lsfgDeleteContext](const int32_t* id) {
            lsfgDeleteContext(*id);
        }
    );

    unsetenv("DISABLE_LSFG"); // NOLINT

    std::cerr << "lsfg-vk: Android AHB context created (id=" << ctxId << ")\n";

#else
    // Desktop Linux (glibc/Turnip) path.
    if (this->useNpu) {
        // NPU framegen: keep lsfg's frame-delivery skeleton (frame_0/frame_1
        // captures + out_n delivery images + swapchain handling below) but
        // replace ONLY the DLL/shader interpolation with the NPU backend.
        // out_n needs STORAGE usage for the pack-back compute write; the
        // existing postCopy (TRANSFER_SRC -> swapchain) is unchanged.
        std::array<int, 2> fds{};
        this->frame_0 = Mini::Image(info.device, info.physicalDevice,
            extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
            &fds.at(0));
        this->frame_1 = Mini::Image(info.device, info.physicalDevice,
            extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
            &fds.at(1));

        std::vector<int> outFds(conf.multiplier - 1);
        for (size_t i = 0; i < (conf.multiplier - 1); ++i)
            this->out_n.emplace_back(info.device, info.physicalDevice,
                extent, format,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                &outFds.at(i));

        // NPU worker dims come from the model graph (e.g. 400x300); swap
        // format is the app's real swapchain format for convert sampling.
        VkFormat srcFmt = swapFormat != VK_FORMAT_UNDEFINED
            ? swapFormat : VK_FORMAT_B8G8R8A8_SRGB;
        this->npuDeviceInfo = info;
        if (!this->npu_.init(info.device, info.physicalDevice,
                info.queue.second, info.queue.first,
                extent, srcFmt, conf.npu_model, conf.npu_bin)) {
            std::cerr << "lsfg-vk: NPU backend failed, falling back to passthrough\n";
            this->useNpu = false;
        } else {
            std::cerr << "lsfg-vk: NPU backend ready (swap fmt=" << (int)srcFmt << ")\n";
            // Model-size pack target (native res): TRANSFER_SRC for the
            // delivery upscale blit + STORAGE for the pack compute write.
            VkExtent2D packExt{ (uint32_t)this->npu_.modelW(),
                                (uint32_t)this->npu_.modelH() };
            int packFd = -1;
            this->npuPackTarget = Mini::Image(info.device, info.physicalDevice,
                packExt, format,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, &packFd);
            this->npu_.setPackTarget(this->npuPackTarget.handle());
            this->hasPackTarget = true;
            std::cerr << "lsfg-vk: NPU pack target " << packExt.width << "x"
                      << packExt.height << "\n";
        }
    }
    if (!this->useNpu) {
    // DLL/shader path: use OPAQUE_FD-based image sharing

    std::array<int, 2> fds{};
    this->frame_0 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(0));
    this->frame_1 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(1));

    std::vector<int> outFds(conf.multiplier - 1);
    for (size_t i = 0; i < (conf.multiplier - 1); ++i)
        this->out_n.emplace_back(info.device, info.physicalDevice,
            extent, format,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
            &outFds.at(i));

    // initialize lsfg
    auto* lsfgInitialize = LSFG_3_1::initialize;
    auto* lsfgCreateContext = LSFG_3_1::createContext;
    auto* lsfgDeleteContext = LSFG_3_1::deleteContext;
        lsfgInitialize = LSFG_3_1P::initialize;
        lsfgCreateContext = LSFG_3_1P::createContext;
        lsfgDeleteContext = LSFG_3_1P::deleteContext;

    setenv("DISABLE_LSFG", "1", 1); // NOLINT

    lsfgInitialize(
        Utils::getDeviceUUID(info.physicalDevice),
        conf.hdr, 1.0F / conf.flowScale, conf.multiplier - 1,
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
    } // end DLL/shader path (skipped when NPU active)
#endif

    // prepare render passes
    this->cmdPool = Mini::CommandPool(info.device, info.queue.first);
    for (size_t i = 0; i < 8; i++) {
        auto& pass = this->passInfos.at(i);
        pass.renderSemaphores.resize(conf.multiplier - 1);
        pass.acquireSemaphores.resize(conf.multiplier - 1);
        pass.postCopyBufs.resize(conf.multiplier - 1);
        pass.postCopySemaphores.resize(conf.multiplier - 1);
        pass.prevPostCopySemaphores.resize(conf.multiplier - 1);
    }
}

VkResult LsContext::present(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx) {
    const auto& conf = Config::activeConf;
    auto& pass = this->passInfos.at(this->frameIdx % 8);

#ifdef __ANDROID__
    // Android path: synchronous frame generation using waitIdle()
    // instead of OPAQUE_FD semaphore export which Turnip doesn't support.

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

    // Wait for the pre-copy to finish before telling framegen to start.
    // This is a device-wide idle wait — heavier than semaphore-based sync
    // but necessary because OPAQUE_FD is not available on Android.
    Layer::ovkQueueSubmit(info.queue.second, 0, nullptr, VK_NULL_HANDLE);

    // 2. Tell framegen to generate intermediary frames
    //    presentContext(id, -1, {}) — no semaphore FDs, synchronous
    std::vector<int> noOutSems;  // empty
        LSFG_3_1P::presentContext(*this->lsfgCtxId, -1, noOutSems);

    // 3. Wait for framegen's GPU work to finish before reading output images.
    //    framegen uses its own VkDevice internally, so we need waitIdle()
    //    to ensure cross-device synchronization.
        LSFG_3_1P::waitIdle();

    // 4. Copy generated frames to swapchain images and present them
    for (size_t i = 0; i < static_cast<size_t>(conf.multiplier - 1); i++) {
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
    //    Wait for the last post-copy to finish
    pass.prevPostCopySemaphores.at(conf.multiplier - 1 - 1) = Mini::Semaphore(info.device);
    VkSemaphore lastPostCopySem = pass.postCopySemaphores.at(conf.multiplier - 1 - 1).handle();
    const VkPresentInfoKHR finalPresentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &lastPostCopySem,
        .swapchainCount = 1,
        .pSwapchains = &this->swapchain,
        .pImageIndices = &presentIdx,
    };
    auto res = Layer::ovkQueuePresentKHR(queue, &finalPresentInfo);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw LSFG::vulkan_error(res, "Failed to present swapchain image");

    this->frameIdx++;
    return res;

#else
    // Desktop Linux path: OPAQUE_FD semaphore-based synchronization

    // --- STEP 1: PRE-COPY (CAPTURE) ---
    pass.preCopySemaphores.at(0) = Mini::Semaphore(info.device);
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

    // --- STEP 2: BUILD WAIT LIST (Original logic + NULL checks) ---
    std::vector<VkSemaphore> waitSems;
    for (auto s : gameRenderSemaphores) {
        if (s != VK_NULL_HANDLE) waitSems.push_back(s);
    }
    if (this->frameIdx > 0) {
        VkSemaphore prev = this->passInfos.at((this->frameIdx - 1) % 8).preCopySemaphores.at(1).handle();
        if (prev != VK_NULL_HANDLE) waitSems.push_back(prev);
    }

    // --- STEP 3: BUILD SIGNAL LIST ---
    std::vector<VkSemaphore> signalSems;
    if (pass.preCopySemaphores.at(0).handle() != VK_NULL_HANDLE)
        signalSems.push_back(pass.preCopySemaphores.at(0).handle());
    if (pass.preCopySemaphores.at(1).handle() != VK_NULL_HANDLE)
        signalSems.push_back(pass.preCopySemaphores.at(1).handle());

    // --- STEP 4: SUBMIT FIRST ---
    pass.preCopyBuf.submit(info.queue.second, waitSems, signalSems);

    // --- STEP 5: EXPORT SYNC FD (The Fix) ---
    // DLL path only: exporting moves the payload out of preCopySemaphores[0]
    // into the fd consumed by presentContext — the semaphore is left
    // unsignaled and must never be waited on afterwards. The NPU path skips
    // this (ordering comes from same-queue submission order + its own
    // fences), so both pre-copy semaphores stay signaled and waitable.
    int preCopySemaphoreFd = -1;
    if (!this->useNpu)
        pass.preCopySemaphores.at(0).exportSyncFd(info.device, &preCopySemaphoreFd);

    // --- STEP 6: FRAMEGEN & INTERMEDIARY FRAMES ---
    // NPU path: swap ONLY the generation call. Handoff runs purely on the
    // proven protocol inside Npu::Backend (dma_buf fds, GPU fence wait per
    // submit, socket ping-pong, wall-clock splits, FNV cross-check) — none
    // of lsfg's frame-timing sync is reused for it. Delivery below
    // (acquire/postCopy/present) is the untouched lsfg skeleton.
    std::vector<int> renderSemaphoreFds;
    bool haveGenerated = false;
    if (this->useNpu) {
        // Same-queue submission order guarantees convert runs after the
        // preCopy above (which waited on the game semaphores).
        haveGenerated = this->npu_.generate(
            this->swapchainImages.at(presentIdx),
            this->out_n.at(0).handle(), this->frameIdx);
        // NPU v1 is 2x: exactly one interpolated frame per present.
        if (haveGenerated && conf.multiplier > 1) {
            renderSemaphoreFds.resize(1, -1);
            pass.renderSemaphores.resize(1);
        }
    } else if (conf.multiplier > 1) {
        //printf("[LSFG_DEBUG] Entering multiplier > 1 block (Frame: %llu)\n", (unsigned long long)this->frameIdx); fflush(stdout);

        renderSemaphoreFds.resize(conf.multiplier - 1, -1);
        for (size_t i = 0; i < (conf.multiplier - 1); ++i) {
            pass.renderSemaphores.at(i) = Mini::Semaphore(info.device);
            pass.renderSemaphores.at(i).exportSyncFd(info.device, &renderSemaphoreFds.at(i));
        }

            LSFG_3_1P::presentContext(*this->lsfgCtxId, preCopySemaphoreFd, renderSemaphoreFds);

        // Immediate cleanup of FDs to try and stretch the life of the process
        for (int &fd : renderSemaphoreFds) { if (fd >= 0) { close(fd); fd = -1; } }
        if (preCopySemaphoreFd >= 0) { close(preCopySemaphoreFd); preCopySemaphoreFd = -1; }
        haveGenerated = true; // DLL path always yields multiplier-1 frames
    }

    // Shared delivery: copy generated frame(s) to fresh swapchain images and
    // present them (untouched lsfg skeleton + vsync pacing).
    double deliveryT0 = 0;
    bool timeDelivery = this->useNpu && getenv("LSFG_NPU_VERBOSE");
    if (timeDelivery) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        deliveryT0 = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
    }
    {
        const size_t nGen = this->useNpu ? 1 : (conf.multiplier - 1);
        const bool doGen = this->useNpu ? haveGenerated : (conf.multiplier > 1);
        for (size_t i = 0; doGen && i < nGen; i++) {
            //printf("[LSFG_DEBUG] Loop %zu: Acquire\n", i); fflush(stdout);
            pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
            uint32_t imageIdx{};
            Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
                pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);

            //printf("[LSFG_DEBUG] Loop %zu: Recording Copy\n", i); fflush(stdout);
            pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
            pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);

            pass.postCopyBufs.at(i).begin();
            if (this->useNpu && this->hasPackTarget) {
                // NPU delivery: HW-blit the model-size pack target up to the
                // swapchain image (LINEAR). Same layout dance as copyImage
                // but scaled — keeps pack fill-rate at model pixels.
                VkCommandBuffer cb = pass.postCopyBufs.at(i).handle();
                VkImage src = this->npuPackTarget.handle();
                VkImage dst = this->swapchainImages.at(imageIdx);
                uint32_t sw = (uint32_t)this->npu_.modelW();
                uint32_t sh = (uint32_t)this->npu_.modelH();
                uint32_t dw = this->extent.width, dh = this->extent.height;
                const VkImageMemoryBarrier srcB{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .image = src,
                    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                         .levelCount = 1, .layerCount = 1}};
                const VkImageMemoryBarrier dstB{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .image = dst,
                    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                         .levelCount = 1, .layerCount = 1}};
                const VkImageMemoryBarrier pre[2] = {srcB, dstB};
                Layer::ovkCmdPipelineBarrier(
                    cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                    2, pre);
                const VkImageBlit blit{
                    .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .layerCount = 1},
                    .srcOffsets = {{0, 0, 0},
                                   {(int32_t)sw, (int32_t)sh, 1}},
                    .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .layerCount = 1},
                    .dstOffsets = {{0, 0, 0},
                                   {(int32_t)dw, (int32_t)dh, 1}}};
                Layer::ovkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                    VK_FILTER_LINEAR);
                const VkImageMemoryBarrier srcBack{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    .image = src,
                    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                         .levelCount = 1, .layerCount = 1}};
                const VkImageMemoryBarrier dstGo{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    .image = dst,
                    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                         .levelCount = 1, .layerCount = 1}};
                const VkImageMemoryBarrier post[2] = {srcBack, dstGo};
                Layer::ovkCmdPipelineBarrier(
                    cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                    nullptr, 2, post);
            } else {
                Utils::copyImage(pass.postCopyBufs.at(i).handle(),
                    this->out_n.at(i).handle(),
                    this->swapchainImages.at(imageIdx),
                    this->extent.width, this->extent.height,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    false, true);
            }
            pass.postCopyBufs.at(i).end();

            // RAW SUBMISSION - The part that worked
            //printf("[LSFG_DEBUG] Loop %zu: Raw Submit\n", i); fflush(stdout);
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkCommandBuffer cb = pass.postCopyBufs.at(i).handle();
            VkSemaphore waitSem = pass.acquireSemaphores.at(i).handle();
            VkSemaphore sigSem = pass.postCopySemaphores.at(i).handle();

            VkSubmitInfo subInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = (waitSem != VK_NULL_HANDLE ? 1u : 0u),
                .pWaitSemaphores = &waitSem,
                .pWaitDstStageMask = &waitStage,
                .commandBufferCount = 1,
                .pCommandBuffers = &cb,
                .signalSemaphoreCount = (sigSem != VK_NULL_HANDLE ? 1u : 0u),
                .pSignalSemaphores = &sigSem
            };

            Layer::ovkQueueSubmit(info.queue.second, 1, &subInfo, VK_NULL_HANDLE);

            //printf("[LSFG_DEBUG] Loop %zu: Present\n", i); fflush(stdout);
            const VkPresentInfoKHR loopPresentInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = (sigSem != VK_NULL_HANDLE ? 1u : 0u),
                .pWaitSemaphores = &sigSem,
                .swapchainCount = 1,
                .pSwapchains = &this->swapchain,
                .pImageIndices = &imageIdx,
            };
            Layer::ovkQueuePresentKHR(queue, &loopPresentInfo);
        }
    }

    // FINAL FRAME
    //printf("[LSFG_DEBUG] Final Frame Present\n"); fflush(stdout);
    // NPU without a generated frame yet (first present): preCopySemaphores[0]
    // was never exported in NPU mode so both are intact — but only [1] is
    // guaranteed signaled-and-unconsumed for this pass; [0] is reserved for
    // the DLL export pattern. Never touch a default (empty) semaphore.
    const bool genDone = this->useNpu ? haveGenerated : (conf.multiplier > 1);
    VkSemaphore finalWait = VK_NULL_HANDLE;
    if (genDone) {
        finalWait = this->useNpu
            ? pass.postCopySemaphores.at(0).handle()
            : pass.postCopySemaphores.at(conf.multiplier - 2).handle();
    } else {
        finalWait = pass.preCopySemaphores.at(1).handle();
    }

    const VkPresentInfoKHR finalPresentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = (finalWait != VK_NULL_HANDLE ? 1u : 0u),
        .pWaitSemaphores = &finalWait,
        .swapchainCount = 1,
        .pSwapchains = &this->swapchain,
        .pImageIndices = &presentIdx,
    };

    auto res = Layer::ovkQueuePresentKHR(queue, &finalPresentInfo);
    if (timeDelivery) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double ms = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6 - deliveryT0;
        struct timespec ts2;
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        fprintf(stderr,
                "[LSFG-NPU t=%llu.%03llu] frame %llu delivery %.1fms%s\n",
                (unsigned long long)ts2.tv_sec,
                (unsigned long long)(ts2.tv_nsec / 1000000ULL),
                (unsigned long long)this->frameIdx, ms,
                haveGenerated ? "" : " (passthrough)");
    }
        // --- AGGRESSIVE BYPASS TEST ---
    // Instead of sleep, we waste cycles to ensure the driver has
    // time to process the command stream.
    this->frameIdx++;
    return res;
#endif
}
