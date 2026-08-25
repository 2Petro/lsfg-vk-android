#pragma once

#include "core/instance.hpp"

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

namespace LSFG::Core {

    class Image;

    ///
    /// C++ wrapper class for a Vulkan device.
    ///
    /// This class manages the lifetime of a Vulkan device.
    ///
    class Device {
    public:
        ///
        /// Create the device.
        ///
        /// @param instance Vulkan instance
        /// @param deviceUUID The UUID of the Vulkan device to use.
        ///
        /// @throws LSFG::vulkan_error if object creation fails.
        ///
        Device(const Instance& instance, uint64_t deviceUUID);

        ///
        /// Wrap an existing host-owned device instead of creating one.
        /// The wrapped handles are not owned and never destroyed.
        ///
        /// @param instance Host instance, used to resolve entry points.
        /// @param physicalDevice Host physical device.
        /// @param device Host logical device (must be usable for compute).
        ///
        /// @throws LSFG::vulkan_error if queue discovery fails.
        ///
        static Device fromHost(VkInstance instance,
            VkPhysicalDevice physicalDevice, VkDevice device);

        /// Get the Vulkan handle.
        [[nodiscard]] auto handle() const { return *this->device; }
        /// Get the physical device associated with this logical device.
        [[nodiscard]] VkPhysicalDevice getPhysicalDevice() const { return this->physicalDevice; }
        /// Get the compute queue family index.
        [[nodiscard]] uint32_t getComputeFamilyIdx() const { return this->computeFamilyIdx; }
        /// Get the compute queue.
        [[nodiscard]] VkQueue getComputeQueue() const { return this->computeQueue; }
        /// Whether the device supports null image descriptors.
        [[nodiscard]] bool supportsNullDescriptor() const { return this->nullDescriptorSupported; }
        /// Valid sampled image used for optional bindings when nullDescriptor is absent.
        [[nodiscard]] const Image& getFallbackDescriptorImage() const;

        // Trivially copyable, moveable and destructible
        Device(const Core::Device&) noexcept = default;
        Device& operator=(const Core::Device&) noexcept = default;
        Device(Device&&) noexcept = default;
        Device& operator=(Device&&) noexcept = default;
        ~Device() = default;
    private:
        Device() noexcept = default;

        std::shared_ptr<VkDevice> device;
        VkPhysicalDevice physicalDevice{};

        uint32_t computeFamilyIdx{0};

        VkQueue computeQueue{};
        bool nullDescriptorSupported{false};
        std::shared_ptr<Image> fallbackDescriptorImage;
    };

}
