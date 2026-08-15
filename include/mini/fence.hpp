#pragma once

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

namespace Mini {

    ///
    /// C++ wrapper class for a Vulkan fence.
    ///
    /// This class manages the lifetime of a Vulkan fence.
    ///
    class Fence {
    public:
        Fence() noexcept = default;

        ///
        /// Create the fence.
        ///
        /// @param device Vulkan device
        /// @param signaled Whether the fence starts in the signaled state.
        ///
        /// @throws LSFG::vulkan_error if object creation fails.
        ///
        Fence(VkDevice device, bool signaled);

        ///
        /// Wait for the fence to become signaled.
        ///
        /// @param timeoutNs Maximum time to wait in nanoseconds.
        /// @return VK_SUCCESS, VK_TIMEOUT or an error code.
        ///
        [[nodiscard]] VkResult wait(uint64_t timeoutNs) const;

        ///
        /// Reset the fence to the unsignaled state.
        ///
        /// @return VK_SUCCESS or an error code.
        ///
        VkResult reset() const;

        /// Get the Vulkan handle.
        [[nodiscard]] auto handle() const { return *this->fence; }

        // Trivially copyable, moveable and destructible
        Fence(const Fence&) noexcept = default;
        Fence& operator=(const Fence&) noexcept = default;
        Fence(Fence&&) noexcept = default;
        Fence& operator=(Fence&&) noexcept = default;
        ~Fence() = default;
    private:
        VkDevice device{};
        std::shared_ptr<VkFence> fence;
    };

}
