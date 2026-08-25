#pragma once

#include <vulkan/vulkan_core.h>

#include <functional>
#include <cstdint>
#include <string>
#include <vector>

#ifdef __ANDROID__
struct AHardwareBuffer;
#endif

namespace LSFG_3_1P {

    ///
    /// Initialize the LSFG library.
    ///
    /// @param deviceUUID The UUID of the Vulkan device to use.
    /// @param isHdr Whether the images are in HDR format.
    /// @param flowScale Internal flow scale factor.
    /// @param generationCount Number of frames to generate.
    /// @param loader Function to load shader source code by name.
    ///
    /// @throws LSFG::vulkan_error if Vulkan objects fail to initialize.
    ///
    __attribute__((visibility("default")))
    void initialize(uint64_t deviceUUID,
        bool isHdr, float flowScale, uint64_t generationCount,
        const std::function<std::vector<uint8_t>(const std::string&)>& loader);

#ifdef __ANDROID__
    ///
    /// Initialize the LSFG library on an existing host-owned device instead
    /// of creating an internal instance/device. Used when the host (layer)
    /// and framegen must run on the same driver, e.g. Android Turnip setups
    /// where the internal instance cannot discover the host's ICD.
    ///
    /// @param instance Host instance, used to resolve device entry points.
    /// @param physicalDevice Host physical device.
    /// @param device Host logical device. Must support compute and have the
    ///        AHB external memory extensions enabled; not owned by LSFG.
    /// @param isHdr Whether the images are in HDR format.
    /// @param flowScale Internal flow scale factor.
    /// @param generationCount Number of frames to generate.
    /// @param loader Function to load shader source code by name.
    ///
    /// @throws LSFG::vulkan_error if initialization fails.
    ///
    __attribute__((visibility("default")))
    void initializeFromHost(
        VkInstance instance,
        VkPhysicalDevice physicalDevice, VkDevice device,
        bool isHdr, float flowScale, uint64_t generationCount,
        const std::function<std::vector<uint8_t>(const std::string&)>& loader);
#endif

    ///
    /// Create a new LSFG context on a swapchain.
    ///
    /// @param in0 File descriptor for the first input image.
    /// @param in1 File descriptor for the second input image.
    /// @param outN File descriptor for each output image. This defines the LSFG level.
    /// @param extent The size of the images
    /// @param format The format of the images.
    /// @return A unique identifier for the created context.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be created.
    ///
    __attribute__((visibility("default")))
    int32_t createContext(
        int in0, int in1, const std::vector<int>& outN,
        VkExtent2D extent, VkFormat format);

#ifdef __ANDROID__
    /// Android-specific variant: see LSFG_3_1::createContextFromAHB.
    __attribute__((visibility("default")))
    int32_t createContextFromAHB(
        AHardwareBuffer* in0, AHardwareBuffer* in1,
        const std::vector<AHardwareBuffer*>& outN,
        VkExtent2D extent, VkFormat format);
#endif

    ///
    /// Present a context.
    ///
    /// @param id Unique identifier of the context to present.
    /// @param inSem Semaphore to wait on before starting the generation.
    /// @param outSem Semaphores to signal once each output image is ready.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be presented.
    ///
    __attribute__((visibility("default")))
    void presentContext(int32_t id, int inSem, const std::vector<int>& outSem);

#ifdef __ANDROID__
    ///
    /// Present a context using native same-device semaphore handles
    /// (host-device mode, see initializeFromHost).
    ///
    /// @param id Unique identifier of the context to present.
    /// @param inSem Binary semaphore signaled when input is ready,
    ///        or VK_NULL_HANDLE for unsynchronized operation.
    /// @param outSems Binary semaphores signaled per generated frame,
    ///        empty for unsynchronized operation.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be presented.
    ///
    __attribute__((visibility("default")))
    void presentContextNative(int32_t id,
        VkSemaphore inSem, const std::vector<VkSemaphore>& outSems);

    ///
    /// Wait (scoped) for the completion fences of the most recent present
    /// of this context. Unlike waitIdle() this does not idle the whole
    /// device and is safe to use on a host-shared device.
    ///
    /// @param id Unique identifier of the context.
    ///
    /// @throws LSFG::vulkan_error on timeout.
    ///
    __attribute__((visibility("default")))
    void waitFrame(int32_t id);
#endif

    ///
    /// Delete an LSFG context.
    ///
    /// @param id Unique identifier of the context to delete.
    ///
    __attribute__((visibility("default")))
    void deleteContext(int32_t id);

    ///
    /// Deinitialize the LSFG library.
    ///
    __attribute__((visibility("default")))
    void finalize();

#ifdef __ANDROID__
    /// Block until framegen's internal Vulkan device is idle. See LSFG_3_1::waitIdle.
    __attribute__((visibility("default")))
    void waitIdle();
#endif

}
