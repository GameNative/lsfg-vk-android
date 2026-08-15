#pragma once

#include <vulkan/vulkan_core.h>

#include <memory>

namespace LSFG::Core {

    ///
    /// C++ wrapper class for a Vulkan instance.
    ///
    /// This class manages the lifetime of a Vulkan instance.
    ///
    class Instance {
    public:
        ///
        /// Create the instance.
        ///
        /// @throws LSFG::vulkan_error if object creation fails.
        ///
        Instance();

        ///
        /// Adopt an externally owned instance. Routes all Vulkan calls through
        /// the given loader entry point (e.g. a layer's next-in-chain
        /// vkGetInstanceProcAddr) and never destroys the instance.
        ///
        /// @param gipa Loader entry point to resolve Vulkan functions with.
        /// @param external Instance owned by the caller.
        ///
        Instance(PFN_vkGetInstanceProcAddr gipa, VkInstance external);

        /// Get the Vulkan handle.
        [[nodiscard]] auto handle() const { return this->instance ? *this->instance : VK_NULL_HANDLE; }

        /// Trivially copyable, moveable and destructible
        Instance(const Instance&) noexcept = default;
        Instance& operator=(const Instance&) noexcept = default;
        Instance(Instance&&) noexcept = default;
        Instance& operator=(Instance&&) noexcept = default;
        ~Instance() = default;
    private:
        std::shared_ptr<VkInstance> instance;
    };

}
