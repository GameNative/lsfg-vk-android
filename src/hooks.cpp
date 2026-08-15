#include "hooks.hpp"
#include "common/exception.hpp"
#include "config/config.hpp"
#include "utils/utils.hpp"
#include "context.hpp"
#include "layer.hpp"

#include <vulkan/vulkan_core.h>

#include <unordered_map>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <exception>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace Hooks;

namespace {

    ///
    /// Add extensions to the instance create info.
    ///
    VkResult myvkCreateInstance(
            const VkInstanceCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkInstance* pInstance) {
        auto extensions = Utils::addExtensions(
            pCreateInfo->ppEnabledExtensionNames,
            pCreateInfo->enabledExtensionCount,
            {
                "VK_KHR_get_physical_device_properties2",
                "VK_KHR_external_memory_capabilities",
                "VK_KHR_external_semaphore_capabilities"
            }
        );
        VkInstanceCreateInfo createInfo = *pCreateInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        auto res = Layer::ovkCreateInstance(&createInfo, pAllocator, pInstance);
        if (res == VK_ERROR_EXTENSION_NOT_PRESENT)
            throw std::runtime_error(
                "Required Vulkan instance extensions are not present."
                "Your GPU driver is not supported.");
        return res;
    }

    /// Map of devices to related information.
    std::unordered_map<VkDevice, DeviceInfo> deviceToInfo;

    ///
    /// Add extensions to the device create info.
    /// (function pointers are not initialized yet)
    ///
    VkResult myvkCreateDevicePre(
            VkPhysicalDevice physicalDevice,
            const VkDeviceCreateInfo* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkDevice* pDevice) {
        // add extensions, skipping any the driver doesn't expose — appending
        // an unsupported name would fail the game's entire device creation
        std::vector<const char*> wanted{
            "VK_KHR_external_memory",
            "VK_KHR_external_memory_fd",
            "VK_KHR_external_semaphore",
            "VK_KHR_external_semaphore_fd"
        };
        uint32_t supportedCount{};
        std::vector<VkExtensionProperties> supportedExts;
        if (Layer::ovkEnumerateDeviceExtensionProperties(physicalDevice,
                &supportedCount, nullptr) && supportedCount > 0) {
            supportedExts.resize(supportedCount);
            if (!Layer::ovkEnumerateDeviceExtensionProperties(physicalDevice,
                    &supportedCount, supportedExts.data()))
                supportedExts.clear();
        }
        if (!supportedExts.empty()) {
            std::erase_if(wanted, [&supportedExts](const char* name) {
                const bool found = std::ranges::any_of(supportedExts,
                    [name](const VkExtensionProperties& p) {
                        return std::string(static_cast<const char*>(p.extensionName)) == name;
                    });
                if (!found)
                    std::cerr << "lsfg-vk: skipping unsupported device extension "
                        << name << "\n";
                return !found;
            });
        }
        auto extensions = Utils::addExtensions(
            pCreateInfo->ppEnabledExtensionNames,
            pCreateInfo->enabledExtensionCount,
            wanted);
        VkDeviceCreateInfo createInfo = *pCreateInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // Single-device framegen runs the LSFG compute chain on THIS device,
        // so its required features must be enabled here. Merge them into
        // whatever feature structure the game supplied, gated on physical
        // support. NOLINTBEGIN
        VkPhysicalDeviceFeatures supported{};
        VkPhysicalDeviceFeatures mergedFeatures{};
        VkPhysicalDeviceVulkan12Features vk12Supported{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        VkPhysicalDeviceFeatures2 supported2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vk12Supported };
        VkPhysicalDeviceVulkan12Features extraVk12{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        if (Layer::ovkGetPhysicalDeviceFeatures(physicalDevice, &supported)) {
            Layer::ovkGetPhysicalDeviceFeatures2(physicalDevice, &supported2);

            auto mergeCore = [&](VkPhysicalDeviceFeatures* target) {
                target->shaderStorageImageExtendedFormats |= supported.shaderStorageImageExtendedFormats;
                target->shaderStorageImageReadWithoutFormat |= supported.shaderStorageImageReadWithoutFormat;
                target->shaderStorageImageWriteWithoutFormat |= supported.shaderStorageImageWriteWithoutFormat;
                target->shaderInt16 |= supported.shaderInt16;
            };

            VkPhysicalDeviceFeatures2* features2InChain = nullptr;
            VkPhysicalDeviceVulkan12Features* vk12InChain = nullptr;
            for (auto* node = const_cast<VkBaseOutStructure*>(
                        reinterpret_cast<const VkBaseOutStructure*>(createInfo.pNext));
                    node != nullptr; node = node->pNext) {
                if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                    features2InChain = reinterpret_cast<VkPhysicalDeviceFeatures2*>(node);
                else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                    vk12InChain = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(node);
            }

            if (features2InChain != nullptr) {
                mergeCore(&features2InChain->features);
            } else if (createInfo.pEnabledFeatures != nullptr) {
                mergedFeatures = *createInfo.pEnabledFeatures;
                mergeCore(&mergedFeatures);
                createInfo.pEnabledFeatures = &mergedFeatures;
            } else {
                mergeCore(&mergedFeatures);
                createInfo.pEnabledFeatures = &mergedFeatures;
            }

            // fp16 shaders and the DXBC path's memory-model requirement live in
            // the 1.2 feature struct. Merge into the game's struct when it has
            // one; otherwise append ours (valid on 1.2+ devices, which every
            // supported ICD here is).
            if (vk12InChain != nullptr) {
                vk12InChain->shaderFloat16 |= vk12Supported.shaderFloat16;
                vk12InChain->vulkanMemoryModel |= vk12Supported.vulkanMemoryModel;
                vk12InChain->vulkanMemoryModelDeviceScope |= vk12Supported.vulkanMemoryModelDeviceScope;
            } else if (vk12Supported.shaderFloat16 || vk12Supported.vulkanMemoryModel) {
                extraVk12.shaderFloat16 = vk12Supported.shaderFloat16;
                extraVk12.vulkanMemoryModel = vk12Supported.vulkanMemoryModel;
                extraVk12.vulkanMemoryModelDeviceScope = vk12Supported.vulkanMemoryModelDeviceScope;
                extraVk12.pNext = const_cast<void*>(createInfo.pNext);
                createInfo.pNext = &extraVk12;
            }
        }
        // NOLINTEND

        auto res = Layer::ovkCreateDevice(physicalDevice, &createInfo, pAllocator, pDevice);
        if (res == VK_ERROR_EXTENSION_NOT_PRESENT)
            throw std::runtime_error(
                "Required Vulkan device extensions are not present."
                "Your GPU driver is not supported.");
        return res;
    }

    ///
    /// Add related device information after the device is created.
    ///
    VkResult myvkCreateDevicePost(
            VkPhysicalDevice physicalDevice,
            VkDeviceCreateInfo* pCreateInfo,
            const VkAllocationCallbacks*,
            VkDevice* pDevice) {
        deviceToInfo.emplace(*pDevice, DeviceInfo {
            .device = *pDevice,
            .physicalDevice = physicalDevice,
            .queue = Utils::findQueue(*pDevice, physicalDevice, pCreateInfo, VK_QUEUE_GRAPHICS_BIT)
        });
        return VK_SUCCESS;
    }

    /// Erase the device information when the device is destroyed.
    void myvkDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) noexcept {
        deviceToInfo.erase(device);
        Layer::ovkDestroyDevice(device, pAllocator);
    }

    std::unordered_map<VkSwapchainKHR, LsContext> swapchains;
    std::unordered_map<VkSwapchainKHR, VkDevice> swapchainToDeviceTable;
    std::unordered_map<VkSwapchainKHR, VkPresentModeKHR> swapchainToPresent;

    ///
    /// Adjust swapchain creation parameters and create a swapchain context.
    ///
    VkResult myvkCreateSwapchainKHR(
            VkDevice device,
            const VkSwapchainCreateInfoKHR* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkSwapchainKHR* pSwapchain) noexcept {
        // find device
        auto it = deviceToInfo.find(device);
        if (it == deviceToInfo.end()) {
            Utils::logLimitN("swapMap", 5, "Device not found in map");
            return Layer::ovkCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
        }
        Utils::resetLimitN("swapMap");
        auto& deviceInfo = it->second;

        // increase amount of images in swapchain
        VkSwapchainCreateInfoKHR createInfo = *pCreateInfo;
        const auto maxImages = Utils::getMaxImageCount(
            deviceInfo.physicalDevice, pCreateInfo->surface);
        createInfo.minImageCount = createInfo.minImageCount + 1
            + static_cast<uint32_t>(deviceInfo.queue.first);
        if (createInfo.minImageCount > maxImages) {
            createInfo.minImageCount = maxImages;
            Utils::logLimitN("swapCount", 10,
                "Requested image count (" +
                    std::to_string(pCreateInfo->minImageCount) + ") "
                "exceeds maximum allowed (" +
                    std::to_string(maxImages) + "). "
                "Continuing with maximum allowed image count. "
                "This might lead to performance degradation.");
        } else {
            Utils::resetLimitN("swapCount");
        }

        // allow copy operations on swapchain images
        createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        // enforce present mode
        createInfo.presentMode = Config::activeConf.e_present;

        // retire potential old swapchain
        if (pCreateInfo->oldSwapchain) {
            swapchains.erase(pCreateInfo->oldSwapchain);
            swapchainToDeviceTable.erase(pCreateInfo->oldSwapchain);
        }

        // create swapchain
        auto res = Layer::ovkCreateSwapchainKHR(device, &createInfo, pAllocator, pSwapchain);
        if (res != VK_SUCCESS)
            return res; // can't be caused by lsfg-vk (yet)

        try {
            swapchainToPresent.emplace(*pSwapchain, createInfo.presentMode);

            // get all swapchain images
            uint32_t imageCount{};
            res = Layer::ovkGetSwapchainImagesKHR(device, *pSwapchain, &imageCount, nullptr);
            if (res != VK_SUCCESS || imageCount == 0)
                throw LSFG::vulkan_error(res, "Failed to get swapchain image count");

            std::vector<VkImage> swapchainImages(imageCount);
            res = Layer::ovkGetSwapchainImagesKHR(device, *pSwapchain,
                &imageCount, swapchainImages.data());
            if (res != VK_SUCCESS)
                throw LSFG::vulkan_error(res, "Failed to get swapchain images");

            // create swapchain context
            swapchainToDeviceTable.emplace(*pSwapchain, device);
            swapchains.emplace(*pSwapchain, LsContext(
                deviceInfo, *pSwapchain, pCreateInfo->imageExtent,
                swapchainImages
            ));

            std::cerr << "lsfg-vk: Swapchain context " <<
                    (createInfo.oldSwapchain ? "recreated" : "created")
                << " (using " << imageCount << " images).\n";

            Utils::resetLimitN("swapCtxCreate");
        } catch (const std::exception& e) {
            Utils::logLimitN("swapCtxCreate", 5,
                "An error occurred while creating the swapchain wrapper:\n"
                "- " + std::string(e.what()));
            return VK_SUCCESS; // swapchain is still valid
        }
        return VK_SUCCESS;
    }

    ///
    /// Update presentation parameters and present the next frame(s).
    ///
    VkResult myvkQueuePresentKHR(
            VkQueue queue,
            const VkPresentInfoKHR* pPresentInfo) noexcept {
        // find swapchain device
        auto it = swapchainToDeviceTable.find(*pPresentInfo->pSwapchains);
        if (it == swapchainToDeviceTable.end()) {
            Utils::logLimitN("swapMap", 5,
                "Swapchain not found in map");
            return Layer::ovkQueuePresentKHR(queue, pPresentInfo);
        }

        // find device info
        auto it2 = deviceToInfo.find(it->second);
        if (it2 == deviceToInfo.end()) {
            Utils::logLimitN("swapMap", 5,
                "Device not found in map");
            return Layer::ovkQueuePresentKHR(queue, pPresentInfo);
        }
        auto& deviceInfo = it2->second;

        // find swapchain context
        auto it3 = swapchains.find(*pPresentInfo->pSwapchains);
        if (it3 == swapchains.end()) {
            Utils::logLimitN("swapMap", 5,
                "Swapchain context not found in map");
            return Layer::ovkQueuePresentKHR(queue, pPresentInfo);
        }
        auto& swapchain = it3->second;

        // find present mode
        auto it4 = swapchainToPresent.find(*pPresentInfo->pSwapchains);
        if (it4 == swapchainToPresent.end()) {
            Utils::logLimitN("swapMap", 5,
                "Swapchain present mode not found in map");
            return Layer::ovkQueuePresentKHR(queue, pPresentInfo);
        }
        auto& present = it4->second;

        // enforce present mode | NOLINTBEGIN
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
        const VkSwapchainPresentModeInfoEXT* presentModeInfo =
            reinterpret_cast<const VkSwapchainPresentModeInfoEXT*>(pPresentInfo->pNext);
        while (presentModeInfo) {
            if (presentModeInfo->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT) {
                for (size_t i = 0; i < presentModeInfo->swapchainCount; i++)
                    const_cast<VkPresentModeKHR*>(presentModeInfo->pPresentModes)[i] =
                        present;
            }
            presentModeInfo =
                reinterpret_cast<const VkSwapchainPresentModeInfoEXT*>(presentModeInfo->pNext);
        }
        #pragma clang diagnostic pop

        // NOLINTEND | present the next frame
        VkResult res{}; // might return VK_SUBOPTIMAL_KHR
        try {
            // ensure config is valid
            auto& conf = Config::activeConf;
            if (!conf.config_file.empty()
                    && (
                            !std::filesystem::exists(conf.config_file)
                          || conf.timestamp != std::filesystem::last_write_time(conf.config_file)
                    )) {
                // A change to fps_limit or the debug keys is applied in place,
                // without tearing down the swapchain. Anything else forces a
                // recreation so LsContext picks it up in its constructor.
                bool pacingOnly = false;
                if (std::filesystem::exists(conf.config_file)) {
                    try {
                        Config::updateConfig(conf.config_file);
                        auto fresh = Config::getConfig(Utils::getProcessName());
                        if (fresh.enable == conf.enable
                                && fresh.dll == conf.dll
                                && fresh.multiplier == conf.multiplier
                                && fresh.flowScale == conf.flowScale
                                && fresh.performance == conf.performance
                                && fresh.hdr == conf.hdr
                                && fresh.origMipmaps == conf.origMipmaps
                                && fresh.e_present == conf.e_present) {
                            conf.fpsLimit = fresh.fpsLimit;
                            conf.debugInputs = fresh.debugInputs;
                            conf.debugDump = fresh.debugDump;
                            conf.timestamp = fresh.timestamp;
                            pacingOnly = true;
                        }
                    } catch (const std::exception&) {
                        // fall through to the recreation path
                    }
                }
                if (!pacingOnly) {
                    Layer::ovkQueuePresentKHR(queue, pPresentInfo);
                    return VK_ERROR_OUT_OF_DATE_KHR;
                }
            }

            // ensure present mode is still valid
            if (present != conf.e_present) {
                Layer::ovkQueuePresentKHR(queue, pPresentInfo);
                return VK_ERROR_OUT_OF_DATE_KHR;
            }

            // no framegen at multiplier <= 1, but the fps cap still applies
            if (conf.multiplier <= 1)
                return swapchain.presentPassthrough(queue, pPresentInfo);

            // skip if framegen force-disabled itself after repeated sync failures
            if (swapchain.isDisabled())
                return swapchain.presentPassthrough(queue, pPresentInfo);

            // present the swapchain
            std::vector<VkSemaphore> semaphores(pPresentInfo->waitSemaphoreCount);
            std::copy_n(pPresentInfo->pWaitSemaphores, semaphores.size(), semaphores.data());

            res = swapchain.present(deviceInfo, pPresentInfo->pNext,
                queue, semaphores, *pPresentInfo->pImageIndices);

            Utils::resetLimitN("swapPresent");
        } catch (const std::exception& e) {
            Utils::logLimitN("swapPresent", 5,
                "An error occurred while presenting the swapchain:\n"
                "- " + std::string(e.what()));
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return res;
    }

    /// Erase the swapchain context and mapping when the swapchain is destroyed.
    void myvkDestroySwapchainKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            const VkAllocationCallbacks* pAllocator) noexcept {
        swapchains.erase(swapchain);
        swapchainToDeviceTable.erase(swapchain);
        swapchainToPresent.erase(swapchain);
        Layer::ovkDestroySwapchainKHR(device, swapchain, pAllocator);
    }
}

std::unordered_map<std::string, PFN_vkVoidFunction> Hooks::hooks = {
    // instance hooks
    {"vkCreateInstance", reinterpret_cast<PFN_vkVoidFunction>(myvkCreateInstance)},

    // device hooks
    {"vkCreateDevicePre", reinterpret_cast<PFN_vkVoidFunction>(myvkCreateDevicePre)},
    {"vkCreateDevicePost", reinterpret_cast<PFN_vkVoidFunction>(myvkCreateDevicePost)},
    {"vkDestroyDevice", reinterpret_cast<PFN_vkVoidFunction>(myvkDestroyDevice)},

    // swapchain hooks
    {"vkCreateSwapchainKHR", reinterpret_cast<PFN_vkVoidFunction>(myvkCreateSwapchainKHR)},
    {"vkQueuePresentKHR", reinterpret_cast<PFN_vkVoidFunction>(myvkQueuePresentKHR)},
    {"vkDestroySwapchainKHR", reinterpret_cast<PFN_vkVoidFunction>(myvkDestroySwapchainKHR)}
};
