#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "lsfg_3_1.hpp"
#include "v3_1/context.hpp"
#include "core/commandpool.hpp"
#include "core/descriptorpool.hpp"
#include "core/instance.hpp"
#include "pool/shaderpool.hpp"
#include "common/exception.hpp"
#include "common/utils.hpp"

#include <cstdint>
#include <optional>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace LSFG;
using namespace LSFG_3_1;

namespace {
    std::optional<Core::Instance> instance;
    std::optional<Vulkan> device;
    std::unordered_map<int32_t, Context> contexts;
    bool externalMode = false;

    // Bounded drain of the framegen queue: a fence-only submit completes when
    // all previously submitted work does. Used instead of vkQueueWaitIdle /
    // vkDeviceWaitIdle in external mode, where an unbounded wait on the game's
    // queue can freeze the present thread forever if the queue is wedged.
    void drainQueueBounded(VkDevice dev, VkQueue queue) {
        const VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence{};
        if (vkCreateFence(dev, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
            return;
        if (vkQueueSubmit(queue, 0, nullptr, fence) == VK_SUCCESS)
            vkWaitForFences(dev, 1, &fence, VK_TRUE, 2'000'000'000ULL);
        vkDestroyFence(dev, fence, nullptr);
    }
}

void LSFG_3_1::initialize(uint64_t deviceUUID,
        bool isHdr, float flowScale, uint64_t generationCount,
        const std::function<std::vector<uint8_t>(const std::string&)>& loader) {
    if (instance.has_value() || device.has_value())
        return;

    instance.emplace();
    device.emplace(Vulkan {
        .device{*instance, deviceUUID},
        .generationCount = generationCount,
        .flowScale = flowScale,
        .isHdr = isHdr
    });
    contexts = std::unordered_map<int32_t, Context>();

    device->commandPool = Core::CommandPool(device->device);
    device->descriptorPool = Core::DescriptorPool(device->device);

    device->resources = Pool::ResourcePool(device->isHdr, device->flowScale);
    device->shaders = Pool::ShaderPool(loader);

    std::srand(static_cast<uint32_t>(std::time(nullptr)));
}

void LSFG_3_1::initializeExternal(PFN_vkGetInstanceProcAddr gipa,
        VkInstance externalInstance, VkPhysicalDevice physicalDevice,
        VkDevice externalDevice, uint32_t queueFamilyIdx, VkQueue queue,
        bool isHdr, float flowScale, uint64_t generationCount,
        const std::function<std::vector<uint8_t>(const std::string&)>& loader) {
    if (instance.has_value() || device.has_value())
        return;

    externalMode = true;
    instance.emplace(gipa, externalInstance);
    device.emplace(Vulkan {
        .device{*instance, physicalDevice, externalDevice, queueFamilyIdx, queue},
        .generationCount = generationCount,
        .flowScale = flowScale,
        .isHdr = isHdr
    });
    contexts = std::unordered_map<int32_t, Context>();

    device->commandPool = Core::CommandPool(device->device);
    device->descriptorPool = Core::DescriptorPool(device->device);

    device->resources = Pool::ResourcePool(device->isHdr, device->flowScale);
    device->shaders = Pool::ShaderPool(loader);

    std::srand(static_cast<uint32_t>(std::time(nullptr)));
}

int32_t LSFG_3_1::createContext(
        int in0, int in1, const std::vector<int>& outN,
        VkExtent2D extent, VkFormat format) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    const int32_t id = std::rand();
    contexts.emplace(id, Context(*device, in0, in1, outN, extent, format));
    return id;
}

void LSFG_3_1::presentContext(int32_t id, int inSem, const std::vector<int>& outSem) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    auto it = contexts.find(id);
    if (it == contexts.end())
        throw LSFG::vulkan_error(VK_ERROR_UNKNOWN, "Context not found");

    it->second.present(*device, inSem, outSem);
}

void LSFG_3_1::deleteContext(int32_t id) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    auto it = contexts.find(id);
    if (it == contexts.end())
        throw LSFG::vulkan_error(VK_ERROR_DEVICE_LOST, "No such context");

    // external mode: the device belongs to the game — idling it from a layer
    // hook races the game's own queue access and can deadlock the driver.
    // All framegen work is submitted on this one queue, so drain just it,
    // with a bounded wait so a wedged queue can't freeze the present thread.
    if (externalMode)
        drainQueueBounded(device->device.handle(), device->device.getComputeQueue());
    else
        vkDeviceWaitIdle(device->device.handle());
    contexts.erase(it);
}

void LSFG_3_1::finalize() {
    if (!instance.has_value() || !device.has_value())
        return;

    // see deleteContext: never idle the game's whole device in external mode
    if (externalMode)
        drainQueueBounded(device->device.handle(), device->device.getComputeQueue());
    else
        vkDeviceWaitIdle(device->device.handle());
    contexts.clear();
    device.reset();
    instance.reset();
    externalMode = false;
}

#ifdef __ANDROID__

#include <android/hardware_buffer.h>

int32_t LSFG_3_1::createContextFromAHB(
        AHardwareBuffer* in0, AHardwareBuffer* in1,
        const std::vector<AHardwareBuffer*>& outN,
        VkExtent2D extent, VkFormat format) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    const int32_t id = std::rand();
    contexts.emplace(id, Context(*device, in0, in1, outN, extent, format));
    return id;
}

#endif // __ANDROID__

int32_t LSFG_3_1::createContextFromImages(
        VkImage in0, VkImage in1, const std::vector<VkImage>& outN,
        VkExtent2D extent, VkFormat format) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    const int32_t id = std::rand();
    contexts.emplace(id, Context(*device, in0, in1, outN, extent, format));
    return id;
}

#ifdef __ANDROID__
void LSFG_3_1::waitIdle() {
    if (!device.has_value()) return;
    // single-device mode: work is ordered on the caller's queue; idling the
    // caller's whole device here would stall the game for no benefit
    if (externalMode) return;
    vkDeviceWaitIdle(device->device.handle());
}
#endif
