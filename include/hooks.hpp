#pragma once

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <string>

namespace Hooks {

    /// Vulkan device information structure.
    struct DeviceInfo {
        VkDevice device;
        VkPhysicalDevice physicalDevice;
        VkInstance instance{VK_NULL_HANDLE};
        std::pair<uint32_t, VkQueue> queue; // graphics family
    };

    /// Map of hooked Vulkan functions.
    extern std::unordered_map<std::string, PFN_vkVoidFunction> hooks;

}
