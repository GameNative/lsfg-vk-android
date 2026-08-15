#pragma once

#include <vulkan/vulkan_core.h>

#ifdef __ANDROID__
#include <vulkan/vulkan_android.h>
#include <android/hardware_buffer.h>
#endif

#include "hooks.hpp"
#include "mini/commandbuffer.hpp"
#include "mini/commandpool.hpp"
#include "mini/fence.hpp"
#include "mini/image.hpp"
#include "mini/semaphore.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

///
/// This class is the frame generation context. There should be one instance per swapchain.
///
class LsContext {
public:
    ///
    /// Create the swapchain context.
    ///
    /// @param info The device information to use.
    /// @param swapchain The Vulkan swapchain to use.
    /// @param extent The extent of the swapchain images.
    /// @param swapchainImages The swapchain images to use.
    ///
    /// @throws LSFG::vulkan_error if any Vulkan call fails.
    ///
    LsContext(const Hooks::DeviceInfo& info, VkSwapchainKHR swapchain,
        VkExtent2D extent, const std::vector<VkImage>& swapchainImages);

    ///
    /// Custom present logic.
    ///
    /// @param info The device information to use.
    /// @param pNext Unknown pointer set in the present info structure.
    /// @param queue The Vulkan queue to present the frame on.
    /// @param gameRenderSemaphores The semaphores to wait on before presenting.
    /// @param presentIdx The index of the swapchain image to present.
    /// @return The result of the Vulkan present operation, which can be VK_SUCCESS or VK_SUBOPTIMAL_KHR.
    ///
    /// @throws LSFG::vulkan_error if any Vulkan call fails.
    ///
    VkResult present(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx);

    /// Whether framegen was disabled for this swapchain after repeated sync
    /// failures. The present hook passes frames through untouched when set.
    [[nodiscard]] bool isDisabled() const { return this->forceDisabled; }

    ///
    /// Pace and forward a present without frame generation (multiplier <= 1).
    /// Applies the same vsync-locked fps limiter as the framegen path so the
    /// cap also works as a plain frame limiter.
    ///
    VkResult presentPassthrough(VkQueue queue, const VkPresentInfoKHR* pPresentInfo);

    // Non-copyable, trivially moveable and destructible
    LsContext(const LsContext&) = delete;
    LsContext& operator=(const LsContext&) = delete;
    LsContext(LsContext&&) = default;
    LsContext& operator=(LsContext&&) = default;
    ~LsContext() = default;
private:
    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    VkExtent2D extent;

    std::shared_ptr<int32_t> lsfgCtxId; // lsfg context id
    Mini::Image frame_0, frame_1; // frames shared with lsfg. write to frame_0 when fc % 2 == 0
    std::vector<Mini::Image> out_n; // output images shared with lsfg, indexed by framegen id

    Mini::CommandPool cmdPool;
    uint64_t frameIdx{0};

#ifdef __ANDROID__
    void paceBaseFrame(int fpsLimit);

    Mini::Fence preCopyFence;   // signaled when the swapchain -> frame_n copy completes
    uint32_t copyFenceTimeouts{0};
    bool forceDisabled{false};

    int64_t pacerAnchorNs{0};       // schedule anchor of the current real frame
    int64_t pacerNextDueNs{0};      // next vsync-grid slot for the real frame
    int64_t lastRealPresentNs{0};   // previous real-frame entry, for the EWMA
    uint64_t baseIntervalEwmaNs{0}; // measured real-frame interval

    // display vsync grid, published by the app (vsync.txt next to conf.toml);
    // Choreographer timestamps share CLOCK_MONOTONIC with the pacer
    int64_t vsyncPhaseNs{0};
    int64_t vsyncPeriodNs{0};
    int64_t vsyncLastReadNs{0};

    uint64_t statsPresents{0};      // presents (real + generated) in the window
    uint64_t statsRealPresents{0};  // real presents in the window
    int64_t statsWindowStartNs{0};

    // per-stage timing accumulators for the stats window
    uint64_t statsWorkNs{0};        // copy fence wait + framegen + waitIdle
    uint64_t statsGenSleepNs{0};    // slept before generated presents
    uint64_t statsGenLateNs{0};     // generated present lateness vs its slot
    uint64_t statsGenSkips{0};      // generated frames dropped to hold cadence
    uint64_t statsGenPresentNs{0};  // time blocked in generated queuePresent calls
    uint64_t statsRealPresentNs{0}; // time blocked in the real queuePresent call
    std::string statsSeq;           // recent image-index sequence (R=real, g=gen)

    void dumpFrameImages(const Hooks::DeviceInfo& info, uint32_t presentIdx);
    int dumpDoneFrames{0};          // presents already dumped (debug_dump)
    bool dumpFailed{false};
#else
    static constexpr bool forceDisabled = false;
#endif

    struct RenderPassInfo {
        Mini::CommandBuffer preCopyBuf; // copy from swapchain image to frame_0/frame_1
        std::array<Mini::Semaphore, 2> preCopySemaphores; // signal when preCopyBuf is done

        std::vector<Mini::Semaphore> renderSemaphores; // signal when lsfg is done with frame n

        std::vector<Mini::Semaphore> acquireSemaphores; // signal for swapchain image n

        std::vector<Mini::CommandBuffer> postCopyBufs; // copy from out_n to swapchain image
        std::vector<Mini::Semaphore> postCopySemaphores; // signal when postCopyBuf is done
        std::vector<Mini::Semaphore> prevPostCopySemaphores; // signal for previous postCopyBuf
    }; // data for a single render pass
    std::array<RenderPassInfo, 8> passInfos; // allocate 8 because why not
};
