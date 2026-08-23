#include "layer.hpp"
#include "common/exception.hpp"
#include "config/config.hpp"
#include "hooks.hpp"

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <sys/system_properties.h>
#endif

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan_core.h>

#include <unordered_map>
#include <exception>
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>

namespace {
    struct InstanceDispatchTable {
        PFN_vkGetInstanceProcAddr gpa{nullptr};
        PFN_vkDestroyInstance DestroyInstance{nullptr};
        PFN_vkCreateDevice CreateDevice{nullptr};
        PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties{nullptr};
        PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties{nullptr};
        PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties{nullptr};
        PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetPhysicalDeviceSurfaceCapabilitiesKHR{nullptr};
        PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices{nullptr};
    };

    struct DeviceDispatchTable {
        PFN_vkGetDeviceProcAddr gpa{nullptr};
        PFN_vkSetDeviceLoaderData setDeviceLoaderData{nullptr};
        PFN_vkDestroyDevice DestroyDevice{nullptr};
        PFN_vkCreateSwapchainKHR CreateSwapchainKHR{nullptr};
        PFN_vkQueuePresentKHR QueuePresentKHR{nullptr};
        PFN_vkDestroySwapchainKHR DestroySwapchainKHR{nullptr};
        PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR{nullptr};
        PFN_vkAllocateCommandBuffers AllocateCommandBuffers{nullptr};
        PFN_vkFreeCommandBuffers FreeCommandBuffers{nullptr};
        PFN_vkBeginCommandBuffer BeginCommandBuffer{nullptr};
        PFN_vkEndCommandBuffer EndCommandBuffer{nullptr};
        PFN_vkCreateCommandPool CreateCommandPool{nullptr};
        PFN_vkDestroyCommandPool DestroyCommandPool{nullptr};
        PFN_vkCreateImage CreateImage{nullptr};
        PFN_vkDestroyImage DestroyImage{nullptr};
        PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements{nullptr};
        PFN_vkBindImageMemory BindImageMemory{nullptr};
        PFN_vkAllocateMemory AllocateMemory{nullptr};
        PFN_vkFreeMemory FreeMemory{nullptr};
        PFN_vkCreateSemaphore CreateSemaphore{nullptr};
        PFN_vkDestroySemaphore DestroySemaphore{nullptr};
        PFN_vkGetMemoryFdKHR GetMemoryFdKHR{nullptr};
        PFN_vkGetSemaphoreFdKHR GetSemaphoreFdKHR{nullptr};
#ifdef __ANDROID__
        PFN_vkGetAndroidHardwareBufferPropertiesANDROID GetAndroidHardwareBufferPropertiesANDROID{nullptr};
#endif
        PFN_vkGetDeviceQueue GetDeviceQueue{nullptr};
        PFN_vkQueueSubmit QueueSubmit{nullptr};
        PFN_vkCmdPipelineBarrier CmdPipelineBarrier{nullptr};
        PFN_vkCmdBlitImage CmdBlitImage{nullptr};
        PFN_vkAcquireNextImageKHR AcquireNextImageKHR{nullptr};
    };

    std::mutex g_dispatch_mutex;
    std::unordered_map<VkInstance, InstanceDispatchTable> g_instance_dispatch;
    std::unordered_map<VkDevice, DeviceDispatchTable> g_device_dispatch;
    std::unordered_map<VkPhysicalDevice, VkInstance> g_phys_device_to_instance;
    std::unordered_map<VkQueue, VkDevice> g_queue_to_device;
    std::unordered_map<VkCommandBuffer, VkDevice> g_cmdbuf_to_device;

    thread_local PFN_vkGetInstanceProcAddr tls_next_gpa_instance = nullptr;
    thread_local PFN_vkGetDeviceProcAddr tls_next_gpa_device = nullptr;
    thread_local PFN_vkGetInstanceProcAddr tls_next_gipa_device_create = nullptr;
    thread_local PFN_vkSetDeviceLoaderData tls_next_set_loader_data = nullptr;

    InstanceDispatchTable* GetInstanceDispatchTable(VkInstance instance) {
        std::lock_guard<std::mutex> lock(g_dispatch_mutex);
        auto it = g_instance_dispatch.find(instance);
        if (it != g_instance_dispatch.end()) return &it->second;
        return nullptr;
    }

    DeviceDispatchTable* GetDeviceDispatchTable(VkDevice device) {
        std::lock_guard<std::mutex> lock(g_dispatch_mutex);
        auto it = g_device_dispatch.find(device);
        if (it != g_device_dispatch.end()) return &it->second;
        return nullptr;
    }

    VkInstance GetInstanceFromPhysicalDevice(VkPhysicalDevice pd) {
        std::lock_guard<std::mutex> lock(g_dispatch_mutex);
        auto it = g_phys_device_to_instance.find(pd);
        if (it != g_phys_device_to_instance.end()) return it->second;
        return nullptr;
    }
}

namespace {
    VkResult layer_vkCreateInstance(
            const VkInstanceCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkInstance* pInstance) {
        try {
            auto* layerDesc = const_cast<VkLayerInstanceCreateInfo*>(
                reinterpret_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext));
            while (layerDesc && (layerDesc->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
                    || layerDesc->function != VK_LAYER_LINK_INFO)) {
                layerDesc = const_cast<VkLayerInstanceCreateInfo*>(
                    reinterpret_cast<const VkLayerInstanceCreateInfo*>(layerDesc->pNext));
            }
            if (!layerDesc)
                throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
                    "No layer creation info found in pNext chain");

            PFN_vkGetInstanceProcAddr next_gpa = layerDesc->u.pLayerInfo->pfnNextGetInstanceProcAddr;
            layerDesc->u.pLayerInfo = layerDesc->u.pLayerInfo->pNext;

            tls_next_gpa_instance = next_gpa;

            if (!Config::activeConf.enable) {
                auto next_createInstance = reinterpret_cast<PFN_vkCreateInstance>(
                    next_gpa(nullptr, "vkCreateInstance"));
                return next_createInstance(pCreateInfo, pAllocator, pInstance);
            }

            auto* createInstanceHook = reinterpret_cast<PFN_vkCreateInstance>(
                Hooks::hooks["vkCreateInstance"]);
            auto res = createInstanceHook(pCreateInfo, pAllocator, pInstance);
            if (res != VK_SUCCESS)
                throw LSFG::vulkan_error(res, "Unknown error");

            InstanceDispatchTable idt{};
            idt.gpa = next_gpa;
            idt.DestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(next_gpa(*pInstance, "vkDestroyInstance"));
            idt.CreateDevice = reinterpret_cast<PFN_vkCreateDevice>(next_gpa(*pInstance, "vkCreateDevice"));
            idt.GetPhysicalDeviceQueueFamilyProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(next_gpa(*pInstance, "vkGetPhysicalDeviceQueueFamilyProperties"));
            idt.GetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(next_gpa(*pInstance, "vkGetPhysicalDeviceMemoryProperties"));
            idt.GetPhysicalDeviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(next_gpa(*pInstance, "vkGetPhysicalDeviceProperties"));
            idt.GetPhysicalDeviceSurfaceCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(next_gpa(*pInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
            idt.EnumeratePhysicalDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(next_gpa(*pInstance, "vkEnumeratePhysicalDevices"));

            {
                std::lock_guard<std::mutex> lock(g_dispatch_mutex);
                g_instance_dispatch[*pInstance] = idt;
            }

            if (idt.GetPhysicalDeviceProperties && idt.EnumeratePhysicalDevices) {
                uint32_t deviceCount = 0;
                if (idt.EnumeratePhysicalDevices(*pInstance, &deviceCount, nullptr) == VK_SUCCESS && deviceCount > 0) {
                    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
                    if (idt.EnumeratePhysicalDevices(*pInstance, &deviceCount, physicalDevices.data()) == VK_SUCCESS) {
                        {
                            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
                            for (auto pd : physicalDevices) {
                                g_phys_device_to_instance[pd] = *pInstance;
                            }
                        }
                        VkPhysicalDeviceProperties deviceProps{};
                        idt.GetPhysicalDeviceProperties(physicalDevices[0], &deviceProps);
                        fprintf(stderr, "lsfg-vk: Active Vulkan Driver/Device Name -> %s\n", deviceProps.deviceName);
                    }
                }
            }

            std::cerr << "lsfg-vk: Vulkan instance layer initialized successfully.\n";
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: An error occurred while initializing the Vulkan instance layer:\n";
            std::cerr << "- " << e.what() << '\n';
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return VK_SUCCESS;
    }

    VkResult layer_vkCreateDevice(
            VkPhysicalDevice physicalDevice,
            const VkDeviceCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkDevice* pDevice) {
        try {
            auto* layerDesc = const_cast<VkLayerDeviceCreateInfo*>(
                reinterpret_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext));
            while (layerDesc && (layerDesc->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                    || layerDesc->function != VK_LAYER_LINK_INFO)) {
                layerDesc = const_cast<VkLayerDeviceCreateInfo*>(
                    reinterpret_cast<const VkLayerDeviceCreateInfo*>(layerDesc->pNext));
            }
            if (!layerDesc)
                throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
                    "No layer creation info found in pNext chain");

            PFN_vkGetDeviceProcAddr next_gpa = layerDesc->u.pLayerInfo->pfnNextGetDeviceProcAddr;
            PFN_vkGetInstanceProcAddr next_gipa = layerDesc->u.pLayerInfo->pfnNextGetInstanceProcAddr;
            layerDesc->u.pLayerInfo = layerDesc->u.pLayerInfo->pNext;

            auto* layerDesc2 = const_cast<VkLayerDeviceCreateInfo*>(
                reinterpret_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext));
            while (layerDesc2 && (layerDesc2->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                    || layerDesc2->function != VK_LOADER_DATA_CALLBACK)) {
                layerDesc2 = const_cast<VkLayerDeviceCreateInfo*>(
                    reinterpret_cast<const VkLayerDeviceCreateInfo*>(layerDesc2->pNext));
            }

            tls_next_gpa_device = next_gpa;
            tls_next_gipa_device_create = next_gipa;
            tls_next_set_loader_data = layerDesc2 ? layerDesc2->u.pfnSetDeviceLoaderData : nullptr;

            VkInstance instance = GetInstanceFromPhysicalDevice(physicalDevice);

            if (!Config::activeConf.enable) {
                auto next_createDevice = reinterpret_cast<PFN_vkCreateDevice>(
                    next_gipa(instance, "vkCreateDevice"));
                return next_createDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
            }

            auto* createDeviceHook = reinterpret_cast<PFN_vkCreateDevice>(
                Hooks::hooks["vkCreateDevicePre"]);
            auto res = createDeviceHook(physicalDevice, pCreateInfo, pAllocator, pDevice);
            if (res != VK_SUCCESS)
                throw LSFG::vulkan_error(res, "Unknown error");

            DeviceDispatchTable ddt{};
            ddt.gpa = next_gpa;
            ddt.setDeviceLoaderData = tls_next_set_loader_data;
            ddt.DestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(next_gpa(*pDevice, "vkDestroyDevice"));
            ddt.CreateSwapchainKHR = reinterpret_cast<PFN_vkCreateSwapchainKHR>(next_gpa(*pDevice, "vkCreateSwapchainKHR"));
            ddt.QueuePresentKHR = reinterpret_cast<PFN_vkQueuePresentKHR>(next_gpa(*pDevice, "vkQueuePresentKHR"));
            ddt.DestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(next_gpa(*pDevice, "vkDestroySwapchainKHR"));
            ddt.GetSwapchainImagesKHR = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(next_gpa(*pDevice, "vkGetSwapchainImagesKHR"));
            ddt.AllocateCommandBuffers = reinterpret_cast<PFN_vkAllocateCommandBuffers>(next_gpa(*pDevice, "vkAllocateCommandBuffers"));
            ddt.FreeCommandBuffers = reinterpret_cast<PFN_vkFreeCommandBuffers>(next_gpa(*pDevice, "vkFreeCommandBuffers"));
            ddt.BeginCommandBuffer = reinterpret_cast<PFN_vkBeginCommandBuffer>(next_gpa(*pDevice, "vkBeginCommandBuffer"));
            ddt.EndCommandBuffer = reinterpret_cast<PFN_vkEndCommandBuffer>(next_gpa(*pDevice, "vkEndCommandBuffer"));
            ddt.CreateCommandPool = reinterpret_cast<PFN_vkCreateCommandPool>(next_gpa(*pDevice, "vkCreateCommandPool"));
            ddt.DestroyCommandPool = reinterpret_cast<PFN_vkDestroyCommandPool>(next_gpa(*pDevice, "vkDestroyCommandPool"));
            ddt.CreateImage = reinterpret_cast<PFN_vkCreateImage>(next_gpa(*pDevice, "vkCreateImage"));
            ddt.DestroyImage = reinterpret_cast<PFN_vkDestroyImage>(next_gpa(*pDevice, "vkDestroyImage"));
            ddt.GetImageMemoryRequirements = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(next_gpa(*pDevice, "vkGetImageMemoryRequirements"));
            ddt.BindImageMemory = reinterpret_cast<PFN_vkBindImageMemory>(next_gpa(*pDevice, "vkBindImageMemory"));
            ddt.GetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(next_gpa(*pDevice, "vkGetMemoryFdKHR"));
            ddt.AllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(next_gpa(*pDevice, "vkAllocateMemory"));
            ddt.FreeMemory = reinterpret_cast<PFN_vkFreeMemory>(next_gpa(*pDevice, "vkFreeMemory"));
            ddt.CreateSemaphore = reinterpret_cast<PFN_vkCreateSemaphore>(next_gpa(*pDevice, "vkCreateSemaphore"));
            ddt.DestroySemaphore = reinterpret_cast<PFN_vkDestroySemaphore>(next_gpa(*pDevice, "vkDestroySemaphore"));
            ddt.GetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(next_gpa(*pDevice, "vkGetSemaphoreFdKHR"));
#ifdef __ANDROID__
            ddt.GetAndroidHardwareBufferPropertiesANDROID = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(next_gpa(*pDevice, "vkGetAndroidHardwareBufferPropertiesANDROID"));
	    if (!ddt.GetAndroidHardwareBufferPropertiesANDROID) {
	        std::cerr << "[OkiLayer] (no function pointer for vkGetAndroidHardwareBufferPropertiesANDROID)\n";
	    } else {
            // Pointer is valid! Disable the layer property for any future instances.
	        int res = __system_property_set("debug.vulkan.layers", "");
        	if (res == 0) {
	            std::cout << "AHB pointer acquired! Successfully unset android debug.vulkan.layers\n";
        	} else {
	            std::cerr << "Failed to clear 'debug.vulkan.layers' (error code: " << res << ")\n";
        	}
	    }
#endif

            ddt.GetDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(next_gpa(*pDevice, "vkGetDeviceQueue"));
            ddt.QueueSubmit = reinterpret_cast<PFN_vkQueueSubmit>(next_gpa(*pDevice, "vkQueueSubmit"));
            ddt.CmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(next_gpa(*pDevice, "vkCmdPipelineBarrier"));
            ddt.CmdBlitImage = reinterpret_cast<PFN_vkCmdBlitImage>(next_gpa(*pDevice, "vkCmdBlitImage"));
            ddt.AcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(next_gpa(*pDevice, "vkAcquireNextImageKHR"));

            {
                std::lock_guard<std::mutex> lock(g_dispatch_mutex);
                g_device_dispatch[*pDevice] = ddt;
            }

            auto postCreateDeviceHook = reinterpret_cast<PFN_vkCreateDevice>(
                Hooks::hooks["vkCreateDevicePost"]);
            res = postCreateDeviceHook(physicalDevice, pCreateInfo, pAllocator, pDevice);
            if (res != VK_SUCCESS)
                throw LSFG::vulkan_error(res, "Unknown error");

            std::cerr << "lsfg-vk: Vulkan device layer initialized successfully.\n";
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: An error occurred while initializing the Vulkan device layer:\n";
            std::cerr << "- " << e.what() << '\n';
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return VK_SUCCESS;
    }
}

extern "C" __attribute__((visibility("default"))) PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName);
extern "C" __attribute__((visibility("default"))) PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName);

const std::unordered_map<std::string, PFN_vkVoidFunction> layerFunctions = {
    { "vkCreateInstance",
        reinterpret_cast<PFN_vkVoidFunction>(&layer_vkCreateInstance) },
    { "vkCreateDevice",
        reinterpret_cast<PFN_vkVoidFunction>(&layer_vkCreateDevice) },
    { "vkGetInstanceProcAddr",
        reinterpret_cast<PFN_vkVoidFunction>(&layer_vkGetInstanceProcAddr) },
    { "vkGetDeviceProcAddr",
        reinterpret_cast<PFN_vkVoidFunction>(&layer_vkGetDeviceProcAddr) },
};

extern "C" __attribute__((visibility("default"))) PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!pName) return nullptr;

    auto it = layerFunctions.find(pName);
    if (it != layerFunctions.end()) return it->second;

    it = Hooks::hooks.find(pName);
    if (it != Hooks::hooks.end() && Config::activeConf.enable) return it->second;

    auto idt = GetInstanceDispatchTable(instance);
    if (idt && idt->gpa) return idt->gpa(instance, pName);

    return nullptr;
}

extern "C" __attribute__((visibility("default"))) PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!pName) return nullptr;

    auto it = layerFunctions.find(pName);
    if (it != layerFunctions.end()) return it->second;

    it = Hooks::hooks.find(pName);
    if (it != Hooks::hooks.end() && Config::activeConf.enable) return it->second;

    auto ddt = GetDeviceDispatchTable(device);
    if (ddt && ddt->gpa) return ddt->gpa(device, pName);

    return nullptr;
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo,
                 const VkAllocationCallbacks* pAllocator,
                 VkInstance* pInstance) {
    return layer_vkCreateInstance(pCreateInfo, pAllocator, pInstance);
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDevice(VkPhysicalDevice physicalDevice,
               const VkDeviceCreateInfo* pCreateInfo,
               const VkAllocationCallbacks* pAllocator,
               VkDevice* pDevice) {
    return layer_vkCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceLayerProperties(uint32_t* pPropertyCount,
                                   VkLayerProperties* pProperties) {
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pProperties) {
        *pPropertyCount = 1;
        return VK_SUCCESS;
    }
    if (*pPropertyCount < 1) return VK_INCOMPLETE;

    VkLayerProperties& prop = pProperties[0];
    memset(&prop, 0, sizeof(prop));
    strncpy(prop.layerName, "VK_LAYER_LS_frame_generation", VK_MAX_EXTENSION_NAME_SIZE - 1);
    prop.specVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
    prop.implementationVersion = 1;
    strncpy(prop.description, "Lossless Scaling Vulkan layer", VK_MAX_DESCRIPTION_SIZE - 1);
    *pPropertyCount = 1;
    return VK_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceExtensionProperties(const char* pLayerName,
                                       uint32_t* pPropertyCount,
                                       VkExtensionProperties* pProperties) {
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (pLayerName && strcmp(pLayerName, "VK_LAYER_LS_frame_generation") != 0)
        return VK_ERROR_LAYER_NOT_PRESENT;

    if (!pProperties) {
        *pPropertyCount = 0;
        return VK_SUCCESS;
    }
    *pPropertyCount = 0;
    return VK_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                     const char* pLayerName,
                                     uint32_t* pPropertyCount,
                                     VkExtensionProperties* pProperties) {
    (void)physicalDevice;
    (void)pProperties;
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (pLayerName && strcmp(pLayerName, "VK_LAYER_LS_frame_generation") != 0)
        return VK_ERROR_LAYER_NOT_PRESENT;

    *pPropertyCount = 0;
    return VK_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice,
                                 uint32_t* pPropertyCount,
                                 VkLayerProperties* pProperties) {
    (void)physicalDevice;
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pProperties) {
        *pPropertyCount = 1;
        return VK_SUCCESS;
    }
    if (*pPropertyCount < 1) return VK_INCOMPLETE;

    VkLayerProperties& prop = pProperties[0];
    memset(&prop, 0, sizeof(prop));
    strncpy(prop.layerName, "VK_LAYER_LS_frame_generation", VK_MAX_EXTENSION_NAME_SIZE - 1);
    prop.specVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
    prop.implementationVersion = 1;
    strncpy(prop.description, "Lossless Scaling Vulkan layer", VK_MAX_DESCRIPTION_SIZE - 1);
    *pPropertyCount = 1;
    return VK_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkSetDeviceLoaderData(VkDevice device, void* object) {
    auto ddt = GetDeviceDispatchTable(device);
    if (ddt && ddt->setDeviceLoaderData)
        return ddt->setDeviceLoaderData(device, object);
    return VK_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* p) {
    if (!p || p->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (p->loaderLayerInterfaceVersion > 2)
        p->loaderLayerInterfaceVersion = 2;

    p->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
    p->pfnGetPhysicalDeviceProcAddr = nullptr;

    return VK_SUCCESS;
}

namespace Layer {
    VkResult ovkCreateInstance(
            const VkInstanceCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkInstance* pInstance) {
        if (!tls_next_gpa_instance) return VK_ERROR_INITIALIZATION_FAILED;
        auto createInstanceFunc = reinterpret_cast<PFN_vkCreateInstance>(
            tls_next_gpa_instance(nullptr, "vkCreateInstance"));
        if (!createInstanceFunc) return VK_ERROR_INITIALIZATION_FAILED;
        return createInstanceFunc(pCreateInfo, pAllocator, pInstance);
    }

    void ovkDestroyInstance(
            VkInstance instance,
            const VkAllocationCallbacks* pAllocator) {
        auto idt = GetInstanceDispatchTable(instance);
        if (idt && idt->DestroyInstance) {
            idt->DestroyInstance(instance, pAllocator);
        }
        std::lock_guard<std::mutex> lock(g_dispatch_mutex);
        g_instance_dispatch.erase(instance);
    }

    VkResult ovkCreateDevice(
            VkPhysicalDevice physicalDevice,
            const VkDeviceCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkDevice* pDevice) {
        VkInstance instance = GetInstanceFromPhysicalDevice(physicalDevice);

        if (tls_next_gipa_device_create) {
            auto createDeviceFunc = reinterpret_cast<PFN_vkCreateDevice>(
                tls_next_gipa_device_create(instance, "vkCreateDevice"));
            if (createDeviceFunc) {
                return createDeviceFunc(physicalDevice, pCreateInfo, pAllocator, pDevice);
            }
        }
        
        // Fallback to InstanceDispatchTable
        auto idt = GetInstanceDispatchTable(instance);
        if (idt && idt->CreateDevice) {
            return idt->CreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
        }
        
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkDestroyDevice(
            VkDevice device,
            const VkAllocationCallbacks* pAllocator) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->DestroyDevice) {
            ddt->DestroyDevice(device, pAllocator);
        }
        std::lock_guard<std::mutex> lock(g_dispatch_mutex);
        g_device_dispatch.erase(device);
    }

    VkResult ovkSetDeviceLoaderData(VkDevice device, void* object) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->setDeviceLoaderData) {
            return ddt->setDeviceLoaderData(device, object);
        }
        return VK_SUCCESS;
    }

    PFN_vkVoidFunction ovkGetInstanceProcAddr(
            VkInstance instance,
            const char* pName) {
        auto idt = GetInstanceDispatchTable(instance);
        if (idt && idt->gpa) {
            return idt->gpa(instance, pName);
        }
        return nullptr;
    }

    PFN_vkVoidFunction ovkGetDeviceProcAddr(
            VkDevice device,
            const char* pName) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->gpa) {
            return ddt->gpa(device, pName);
        }
        return nullptr;
    }

    void ovkGetPhysicalDeviceQueueFamilyProperties(
            VkPhysicalDevice physicalDevice,
            uint32_t* pQueueFamilyPropertyCount,
            VkQueueFamilyProperties* pQueueFamilyProperties) {
        VkInstance instance = GetInstanceFromPhysicalDevice(physicalDevice);
        auto idt = GetInstanceDispatchTable(instance);
        if (idt && idt->GetPhysicalDeviceQueueFamilyProperties) {
            idt->GetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
        }
    }

    void ovkGetPhysicalDeviceMemoryProperties(
            VkPhysicalDevice physicalDevice,
            VkPhysicalDeviceMemoryProperties* pMemoryProperties) {
        VkInstance instance = GetInstanceFromPhysicalDevice(physicalDevice);
        auto idt = GetInstanceDispatchTable(instance);
        if (idt && idt->GetPhysicalDeviceMemoryProperties) {
            idt->GetPhysicalDeviceMemoryProperties(physicalDevice, pMemoryProperties);
        }
    }

    void ovkGetPhysicalDeviceProperties(
            VkPhysicalDevice physicalDevice,
            VkPhysicalDeviceProperties* pProperties) {
        VkInstance instance = GetInstanceFromPhysicalDevice(physicalDevice);
        auto idt = GetInstanceDispatchTable(instance);
        if (idt && idt->GetPhysicalDeviceProperties) {
            idt->GetPhysicalDeviceProperties(physicalDevice, pProperties);
        }
    }

    VkResult ovkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            VkPhysicalDevice physicalDevice,
            VkSurfaceKHR surface,
            VkSurfaceCapabilitiesKHR* pSurfaceCapabilities) {
        VkInstance instance = GetInstanceFromPhysicalDevice(physicalDevice);
        auto idt = GetInstanceDispatchTable(instance);
        if (idt && idt->GetPhysicalDeviceSurfaceCapabilitiesKHR) {
            return idt->GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ovkCreateSwapchainKHR(
            VkDevice device,
            const VkSwapchainCreateInfoKHR* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkSwapchainKHR* pSwapchain) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->CreateSwapchainKHR) {
            return ddt->CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ovkQueuePresentKHR(
            VkQueue queue,
            const VkPresentInfoKHR* pPresentInfo) {
        VkDevice device = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
            auto it = g_queue_to_device.find(queue);
            if (it != g_queue_to_device.end()) device = it->second;
        }
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->QueuePresentKHR) {
            return ddt->QueuePresentKHR(queue, pPresentInfo);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkDestroySwapchainKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            const VkAllocationCallbacks* pAllocator) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->DestroySwapchainKHR) {
            ddt->DestroySwapchainKHR(device, swapchain, pAllocator);
        }
    }

    VkResult ovkGetSwapchainImagesKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            uint32_t* pSwapchainImageCount,
            VkImage* pSwapchainImages) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->GetSwapchainImagesKHR) {
            return ddt->GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ovkAllocateCommandBuffers(
            VkDevice device,
            const VkCommandBufferAllocateInfo* pAllocateInfo,
            VkCommandBuffer* pCommandBuffers) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->AllocateCommandBuffers) {
            VkResult res = ddt->AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
            if (res == VK_SUCCESS && pCommandBuffers && pAllocateInfo) {
                std::lock_guard<std::mutex> lock(g_dispatch_mutex);
                for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; ++i) {
                    g_cmdbuf_to_device[pCommandBuffers[i]] = device;
                }
            }
            return res;
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkFreeCommandBuffers(
            VkDevice device,
            VkCommandPool commandPool,
            uint32_t commandBufferCount,
            const VkCommandBuffer* pCommandBuffers) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->FreeCommandBuffers) {
            ddt->FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
            if (pCommandBuffers) {
                std::lock_guard<std::mutex> lock(g_dispatch_mutex);
                for (uint32_t i = 0; i < commandBufferCount; ++i) {
                    g_cmdbuf_to_device.erase(pCommandBuffers[i]);
                }
            }
        }
    }

    VkResult ovkBeginCommandBuffer(
            VkCommandBuffer commandBuffer,
            const VkCommandBufferBeginInfo* pBeginInfo) {
        VkDevice device = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
            auto it = g_cmdbuf_to_device.find(commandBuffer);
            if (it != g_cmdbuf_to_device.end()) device = it->second;
        }
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->BeginCommandBuffer) {
            return ddt->BeginCommandBuffer(commandBuffer, pBeginInfo);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ovkEndCommandBuffer(
            VkCommandBuffer commandBuffer) {
        VkDevice device = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
            auto it = g_cmdbuf_to_device.find(commandBuffer);
            if (it != g_cmdbuf_to_device.end()) device = it->second;
        }
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->EndCommandBuffer) {
            return ddt->EndCommandBuffer(commandBuffer);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ovkCreateCommandPool(
            VkDevice device,
            const VkCommandPoolCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkCommandPool* pCommandPool) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->CreateCommandPool) {
            return ddt->CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkDestroyCommandPool(
            VkDevice device,
            VkCommandPool commandPool,
            const VkAllocationCallbacks* pAllocator) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->DestroyCommandPool) {
            ddt->DestroyCommandPool(device, commandPool, pAllocator);
        }
    }

    VkResult ovkCreateImage(
            VkDevice device,
            const VkImageCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkImage* pImage) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->CreateImage) {
            return ddt->CreateImage(device, pCreateInfo, pAllocator, pImage);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkDestroyImage(
            VkDevice device,
            VkImage image,
            const VkAllocationCallbacks* pAllocator) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->DestroyImage) {
            ddt->DestroyImage(device, image, pAllocator);
        }
    }

    void ovkGetImageMemoryRequirements(
            VkDevice device,
            VkImage image,
            VkMemoryRequirements* pMemoryRequirements) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->GetImageMemoryRequirements) {
            ddt->GetImageMemoryRequirements(device, image, pMemoryRequirements);
        }
    }

    VkResult ovkBindImageMemory(
            VkDevice device,
            VkImage image,
            VkDeviceMemory memory,
            VkDeviceSize memoryOffset) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->BindImageMemory) {
            return ddt->BindImageMemory(device, image, memory, memoryOffset);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ovkAllocateMemory(
            VkDevice device,
            const VkMemoryAllocateInfo* pAllocateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkDeviceMemory* pMemory) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->AllocateMemory) {
            return ddt->AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkFreeMemory(
            VkDevice device,
            VkDeviceMemory memory,
            const VkAllocationCallbacks* pAllocator) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->FreeMemory) {
            ddt->FreeMemory(device, memory, pAllocator);
        }
    }

    VkResult ovkCreateSemaphore(
            VkDevice device,
            const VkSemaphoreCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkSemaphore* pSemaphore) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->CreateSemaphore) {
            return ddt->CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkDestroySemaphore(
            VkDevice device,
            VkSemaphore semaphore,
            const VkAllocationCallbacks* pAllocator) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->DestroySemaphore) {
            ddt->DestroySemaphore(device, semaphore, pAllocator);
        }
    }

    VkResult ovkGetMemoryFdKHR(
            VkDevice device,
            const VkMemoryGetFdInfoKHR* pGetFdInfo,
            int* pFd) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->GetMemoryFdKHR) {
            return ddt->GetMemoryFdKHR(device, pGetFdInfo, pFd);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult ovkGetSemaphoreFdKHR(
            VkDevice device,
            const VkSemaphoreGetFdInfoKHR* pGetFdInfo,
            int* pFd) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->GetSemaphoreFdKHR) {
            return ddt->GetSemaphoreFdKHR(device, pGetFdInfo, pFd);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

#ifdef __ANDROID__
    VkResult ovkGetAndroidHardwareBufferPropertiesANDROID(
            VkDevice device,
            const AHardwareBuffer* hardwareBuffer,
            VkAndroidHardwareBufferPropertiesANDROID* pProperties) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->GetAndroidHardwareBufferPropertiesANDROID) {
            return ddt->GetAndroidHardwareBufferPropertiesANDROID(device, hardwareBuffer, pProperties);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
#endif

    void ovkGetDeviceQueue(
            VkDevice device,
            uint32_t queueFamilyIndex,
            uint32_t queueIndex,
            VkQueue* pQueue) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->GetDeviceQueue) {
            ddt->GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
            if (pQueue && *pQueue) {
                std::lock_guard<std::mutex> lock(g_dispatch_mutex);
                g_queue_to_device[*pQueue] = device;
            }
        }
    }

    VkResult ovkQueueSubmit(
            VkQueue queue,
            uint32_t submitCount,
            const VkSubmitInfo* pSubmits,
            VkFence fence) {
        VkDevice device = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
            auto it = g_queue_to_device.find(queue);
            if (it != g_queue_to_device.end()) device = it->second;
        }
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->QueueSubmit) {
            return ddt->QueueSubmit(queue, submitCount, pSubmits, fence);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    void ovkCmdPipelineBarrier(
            VkCommandBuffer commandBuffer,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask,
            VkDependencyFlags dependencyFlags,
            uint32_t memoryBarrierCount,
            const VkMemoryBarrier* pMemoryBarriers,
            uint32_t bufferMemoryBarrierCount,
            const VkBufferMemoryBarrier* pBufferMemoryBarriers,
            uint32_t imageMemoryBarrierCount,
            const VkImageMemoryBarrier* pImageMemoryBarriers) {
        VkDevice device = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
            auto it = g_cmdbuf_to_device.find(commandBuffer);
            if (it != g_cmdbuf_to_device.end()) device = it->second;
        }
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->CmdPipelineBarrier) {
            ddt->CmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags,
                memoryBarrierCount, pMemoryBarriers,
                bufferMemoryBarrierCount, pBufferMemoryBarriers,
                imageMemoryBarrierCount, pImageMemoryBarriers);
        }
    }

    void ovkCmdBlitImage(
            VkCommandBuffer commandBuffer,
            VkImage srcImage,
            VkImageLayout srcImageLayout,
            VkImage dstImage,
            VkImageLayout dstImageLayout,
            uint32_t regionCount,
            const VkImageBlit* pRegions,
            VkFilter filter) {
        VkDevice device = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
            auto it = g_cmdbuf_to_device.find(commandBuffer);
            if (it != g_cmdbuf_to_device.end()) device = it->second;
        }
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->CmdBlitImage) {
            ddt->CmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
        }
    }

    VkResult ovkAcquireNextImageKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            uint64_t timeout,
            VkSemaphore semaphore,
            VkFence fence,
            uint32_t* pImageIndex) {
        auto ddt = GetDeviceDispatchTable(device);
        if (ddt && ddt->AcquireNextImageKHR) {
            return ddt->AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

extern "C" __attribute__((visibility("default"))) PFN_vkVoidFunction
layer_vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    return vkGetInstanceProcAddr(instance, pName);
}
extern "C" __attribute__((visibility("default"))) PFN_vkVoidFunction
layer_vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    return vkGetDeviceProcAddr(device, pName);
}
