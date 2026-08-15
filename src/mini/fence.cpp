#include "mini/fence.hpp"
#include "common/exception.hpp"
#include "layer.hpp"

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

using namespace Mini;

Fence::Fence(VkDevice device, bool signaled) : device(device) {
    // create fence
    const VkFenceCreateInfo desc{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : VkFenceCreateFlags{}
    };
    VkFence fenceHandle{};
    auto res = Layer::ovkCreateFence(device, &desc, nullptr, &fenceHandle);
    if (res != VK_SUCCESS || fenceHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Unable to create fence");

    // store fence in shared ptr
    this->fence = std::shared_ptr<VkFence>(
        new VkFence(fenceHandle),
        [dev = device](VkFence* fenceHandle) {
            Layer::ovkDestroyFence(dev, *fenceHandle, nullptr);
        }
    );
}

VkResult Fence::wait(uint64_t timeoutNs) const {
    return Layer::ovkWaitForFences(this->device, 1, &(*this->fence), VK_TRUE, timeoutNs);
}

VkResult Fence::reset() const {
    return Layer::ovkResetFences(this->device, 1, &(*this->fence));
}
