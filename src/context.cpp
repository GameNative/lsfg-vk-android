#include "context.hpp"
#include "config/config.hpp"
#include "common/exception.hpp"
#include "extract/extract.hpp"
#include "extract/trans.hpp"
#include "extract/mipmaps_replacement_spv.h"
#include "utils/utils.hpp"
#include "hooks.hpp"
#include "layer.hpp"

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <android/log.h>
#endif

#include <vulkan/vulkan_core.h>
#include <lsfg_3_1.hpp>
#include <lsfg_3_1p.hpp>

#include <filesystem>
#include <exception>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <array>

#ifdef __ANDROID__
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cerrno>
#include <ctime>

namespace {
    // Every GPU wait is bounded so an incompatible sync path degrades to
    // passthrough presents instead of hanging the game's present thread.
    constexpr uint64_t kSyncFenceTimeoutNs = 500'000'000ULL;
    constexpr uint32_t kMaxFenceTimeouts = 6;

    int64_t nowNs() {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
    }

    void sleepUntilNs(int64_t targetNs) {
        const int64_t delta = targetNs - nowNs();
        if (delta <= 0) return;
        timespec req{
            .tv_sec = static_cast<time_t>(delta / 1'000'000'000LL),
            .tv_nsec = static_cast<long>(delta % 1'000'000'000LL)
        };
        while (nanosleep(&req, &req) == -1 && errno == EINTR) {}
    }

    void imageBarrier(VkCommandBuffer cmd, VkImage img,
            VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
            VkPipelineStageFlags dstStage, VkAccessFlags dstAccess,
            VkImageLayout oldLayout, VkImageLayout newLayout,
            uint32_t srcFamily = VK_QUEUE_FAMILY_IGNORED,
            uint32_t dstFamily = VK_QUEUE_FAMILY_IGNORED) {
        const VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = srcAccess,
            .dstAccessMask = dstAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = srcFamily,
            .dstQueueFamilyIndex = dstFamily,
            .image = img,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1
            }
        };
        Layer::ovkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
            0, nullptr, 0, nullptr, 1, &barrier);
    }

    void blitFull(VkCommandBuffer cmd, VkImage src, VkImage dst,
            uint32_t width, uint32_t height) {
        const VkImageBlit blit{
            .srcSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
            .srcOffsets = { {0, 0, 0},
                { static_cast<int32_t>(width), static_cast<int32_t>(height), 1 } },
            .dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
            .dstOffsets = { {0, 0, 0},
                { static_cast<int32_t>(width), static_cast<int32_t>(height), 1 } },
        };
        Layer::ovkCmdBlitImage(cmd,
            src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_NEAREST);
    }

    // Parse the vsync grid the app publishes next to conf.toml. Returns false
    // when the file is missing or malformed.
    bool readVsyncFile(const std::filesystem::path& configFile,
            int64_t* phaseNs, int64_t* periodNs) {
        if (configFile.empty()) return false;
        std::ifstream in(configFile.parent_path() / "vsync.txt");
        if (!in.is_open()) return false;
        int64_t phase = 0, period = 0;
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("vsync_ns=", 0) == 0)
                phase = std::strtoll(line.c_str() + 9, nullptr, 10);
            else if (line.rfind("period_ns=", 0) == 0)
                period = std::strtoll(line.c_str() + 10, nullptr, 10);
        }
        if (phase <= 0 || period < 1'000'000 || period > 100'000'000) return false;
        *phaseNs = phase;
        *periodNs = period;
        return true;
    }

    float halfToFloat(uint16_t h) {
        const uint32_t sign = (h & 0x8000U) << 16;
        uint32_t exp = (h >> 10) & 0x1FU;
        uint32_t mant = h & 0x3FFU;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((mant & 0x400U) == 0) { mant <<= 1; exp--; }
                mant &= 0x3FFU;
                bits = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000U | (mant << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    size_t dumpBytesPerPixel(VkFormat format) {
        switch (format) {
            case VK_FORMAT_R8_UNORM: return 1;
            case VK_FORMAT_R8G8B8A8_UNORM: return 4;
            case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
            default: return 0;
        }
    }

    // Write one image region of the mapped dump buffer as binary PPM.
    // signedMap remaps [-1,1] to [0,1] so negative flow values stay visible.
    bool writePpm(const std::filesystem::path& path, const uint8_t* pixels,
            uint32_t width, uint32_t height, VkFormat format, bool signedMap) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) return false;
        out << "P6\n" << width << ' ' << height << "\n255\n";
        std::vector<uint8_t> row(static_cast<size_t>(width) * 3);
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                if (format == VK_FORMAT_R8G8B8A8_UNORM) {
                    const uint8_t* px = pixels
                        + (static_cast<size_t>(y) * width + x) * 4;
                    row[x * 3 + 0] = px[0];
                    row[x * 3 + 1] = px[1];
                    row[x * 3 + 2] = px[2];
                } else if (format == VK_FORMAT_R8_UNORM) {
                    const uint8_t v = pixels[static_cast<size_t>(y) * width + x];
                    row[x * 3 + 0] = v;
                    row[x * 3 + 1] = v;
                    row[x * 3 + 2] = v;
                } else {
                    const uint16_t* px = reinterpret_cast<const uint16_t*>(pixels)
                        + (static_cast<size_t>(y) * width + x) * 4;
                    for (int c = 0; c < 3; c++) {
                        float v = halfToFloat(px[c]);
                        if (signedMap) v = v * 0.5F + 0.5F;
                        v = std::clamp(v, 0.0F, 1.0F);
                        row[x * 3 + c] = static_cast<uint8_t>(v * 255.0F + 0.5F);
                    }
                }
            }
            out.write(reinterpret_cast<const char*>(row.data()),
                static_cast<std::streamsize>(row.size()));
        }
        return out.good();
    }

    struct StatsDetail {
        double workMs, genSleepMs, genLateMs, genPresentMs, realPresentMs;
        uint64_t genSkips;
        std::string seq;
    };

    void writeStats(const std::filesystem::path& configFile, double fps, double baseFps,
            const StatsDetail& d) {
        if (configFile.empty()) return;
        std::ofstream out(configFile.parent_path() / "stats.txt", std::ios::trunc);
        if (!out.is_open()) return;
        out << "fps=" << fps << "\nbase=" << baseFps
            << "\nwork_ms=" << d.workMs
            << "\ngen_sleep_ms=" << d.genSleepMs
            << "\ngen_late_ms=" << d.genLateMs
            << "\ngen_present_ms=" << d.genPresentMs
            << "\nreal_present_ms=" << d.realPresentMs
            << "\ngen_skips=" << d.genSkips
            << "\nseq=" << d.seq << '\n';
    }
}
#endif

LsContext::LsContext(const Hooks::DeviceInfo& info, VkSwapchainKHR swapchain,
        VkExtent2D extent, const std::vector<VkImage>& swapchainImages)
        : swapchain(swapchain), swapchainImages(swapchainImages),
          extent(extent) {
    // get updated configuration
    auto& conf = Config::activeConf;
    if (!conf.config_file.empty()
            && (
                    !std::filesystem::exists(conf.config_file)
                  || conf.timestamp != std::filesystem::last_write_time(conf.config_file)
            )) {
        std::cerr << "lsfg-vk: Rereading configuration, as it is no longer valid.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // reread configuration
        const std::string file = Utils::getConfigFile();
        const auto name = Utils::getProcessName();
        try {
            Config::updateConfig(file);
            conf = Config::getConfig(name);
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: Failed to update configuration, continuing using old:\n";
            std::cerr << "- " << e.what() << '\n';
        }

        LSFG_3_1P::finalize();
        LSFG_3_1::finalize();

        // print config
        std::cerr << "lsfg-vk: Reloaded configuration for " << name.second << ":\n";
        if (!conf.dll.empty()) std::cerr << "  Using DLL from: " << conf.dll << '\n';
        std::cerr << "  Multiplier: " << conf.multiplier << '\n';
        std::cerr << "  Flow Scale: " << conf.flowScale << '\n';
        std::cerr << "  Performance Mode: " << (conf.performance ? "Enabled" : "Disabled") << '\n';
        std::cerr << "  HDR Mode: " << (conf.hdr ? "Enabled" : "Disabled") << '\n';
        if (conf.e_present != 2) std::cerr << "  ! Present Mode: " << conf.e_present << '\n';

        if (conf.multiplier <= 1) return;
    }
    // we could take the format from the swapchain,
    // but honestly this is safer.
    const VkFormat format = conf.hdr
        ? VK_FORMAT_R8G8B8A8_UNORM
        : VK_FORMAT_R16G16B16A16_SFLOAT;

#ifdef __ANDROID__
    // Android path: single-device framegen. All images live on the game's own
    // device and every dispatch runs on the game's queue — two different ICDs
    // (e.g. turnip for the game, the system driver for a private framegen
    // device) interpret shared AHB memory differently and scramble it, so no
    // cross-device sharing is used at all.

    this->preCopyFence = Mini::Fence(info.device, true);

    const VkImageUsageFlags sharedUsage = VK_IMAGE_USAGE_STORAGE_BIT
        | VK_IMAGE_USAGE_SAMPLED_BIT
        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    this->frame_0 = Mini::Image::createDeviceLocal(info.device, info.physicalDevice,
        extent, format, sharedUsage, VK_IMAGE_ASPECT_COLOR_BIT);
    this->frame_1 = Mini::Image::createDeviceLocal(info.device, info.physicalDevice,
        extent, format, sharedUsage, VK_IMAGE_ASPECT_COLOR_BIT);

    for (size_t i = 0; i < static_cast<size_t>(conf.multiplier - 1); ++i)
        this->out_n.emplace_back(Mini::Image::createDeviceLocal(info.device, info.physicalDevice,
            extent, format, sharedUsage, VK_IMAGE_ASPECT_COLOR_BIT));

    setenv("DISABLE_LSFG", "1", 1); // NOLINT

    const auto shaderLoader = [](const std::string& name) {
        // debug override: a raw SPIR-V file next to conf.toml replaces the
        // translated DLL shader (override_<name>.spv, brackets -> underscores)
        const auto& confFile = Config::activeConf.config_file;
        if (!confFile.empty()) {
            std::string fname = "override_" + name + ".spv";
            for (auto& c : fname) if (c == '[' || c == ']') c = '_';
            const auto path = std::filesystem::path(confFile).parent_path() / fname;
            std::ifstream in(path, std::ios::binary);
            if (in.is_open()) {
                std::vector<uint8_t> spirv(
                    (std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
                if (spirv.size() >= 20) {
                    std::cerr << "lsfg-vk: using shader override " << path
                        << " (" << spirv.size() << " bytes)\n";
                    return spirv;
                }
            }
        }
        if (!Config::activeConf.origMipmaps
                && (name == "mipmaps" || name == "p_mipmaps")) {
            return std::vector<uint8_t>(mipmaps_replacement_spv,
                mipmaps_replacement_spv + mipmaps_replacement_spv_len);
        }
        auto dxbc = Extract::getShader(name);
        auto spirv = Extract::translateShader(dxbc);
        return spirv;
    };
    if (conf.performance)
        LSFG_3_1P::initializeExternal(Layer::ovkGetInstanceProcAddr, Layer::ovkInstance(),
            info.physicalDevice, info.device,
            static_cast<uint32_t>(info.queue.first), info.queue.second,
            conf.hdr, 1.0F / conf.flowScale, conf.multiplier - 1, shaderLoader);
    else
        LSFG_3_1::initializeExternal(Layer::ovkGetInstanceProcAddr, Layer::ovkInstance(),
            info.physicalDevice, info.device,
            static_cast<uint32_t>(info.queue.first), info.queue.second,
            conf.hdr, 1.0F / conf.flowScale, conf.multiplier - 1, shaderLoader);

    std::vector<VkImage> outImages;
    outImages.reserve(conf.multiplier - 1);
    for (size_t i = 0; i < static_cast<size_t>(conf.multiplier - 1); ++i)
        outImages.push_back(this->out_n.at(i).handle());

    int32_t ctxId;
    if (conf.performance)
        ctxId = LSFG_3_1P::createContextFromImages(
            this->frame_0.handle(), this->frame_1.handle(),
            outImages, extent, format);
    else
        ctxId = LSFG_3_1::createContextFromImages(
            this->frame_0.handle(), this->frame_1.handle(),
            outImages, extent, format);

    auto* lsfgDeleteContext = conf.performance
        ? LSFG_3_1P::deleteContext : LSFG_3_1::deleteContext;
    this->lsfgCtxId = std::shared_ptr<int32_t>(
        new int32_t(ctxId),
        [lsfgDeleteContext = lsfgDeleteContext](const int32_t* id) {
            lsfgDeleteContext(*id);
        }
    );

    unsetenv("DISABLE_LSFG"); // NOLINT

    // one-time transition of the shared images to GENERAL, the layout they are
    // handed back and forth in
    {
        this->cmdPool = Mini::CommandPool(info.device, info.queue.first);
        Mini::CommandBuffer initCmd(info.device, this->cmdPool);
        Mini::Fence initFence(info.device, false);
        initCmd.begin();
        auto toGeneral = [&](VkImage img) {
            imageBarrier(initCmd.handle(), img,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        };
        toGeneral(this->frame_0.handle());
        toGeneral(this->frame_1.handle());
        for (auto& out : this->out_n) toGeneral(out.handle());
        initCmd.end();
        initCmd.submit(info.queue.second, {}, {}, initFence.handle());
        if (initFence.wait(1'000'000'000ull) != VK_SUCCESS)
            throw LSFG::vulkan_error(VK_TIMEOUT, "Timed out initializing shared image layouts");
    }

    std::cerr << "lsfg-vk: single-device framegen context created (id=" << ctxId << ")\n";

#else
    // Desktop Linux path: use OPAQUE_FD-based image sharing

    std::array<int, 2> fds{};
    this->frame_0 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(0));
    this->frame_1 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(1));

    std::vector<int> outFds(conf.multiplier - 1);
    for (size_t i = 0; i < (conf.multiplier - 1); ++i)
        this->out_n.emplace_back(info.device, info.physicalDevice,
            extent, format,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
            &outFds.at(i));

    // initialize lsfg
    auto* lsfgInitialize = LSFG_3_1::initialize;
    auto* lsfgCreateContext = LSFG_3_1::createContext;
    auto* lsfgDeleteContext = LSFG_3_1::deleteContext;
    if (conf.performance) {
        lsfgInitialize = LSFG_3_1P::initialize;
        lsfgCreateContext = LSFG_3_1P::createContext;
        lsfgDeleteContext = LSFG_3_1P::deleteContext;
    }

    setenv("DISABLE_LSFG", "1", 1); // NOLINT

    lsfgInitialize(
        Utils::getDeviceUUID(info.physicalDevice),
        conf.hdr, 1.0F / conf.flowScale, conf.multiplier - 1,
        [](const std::string& name) {
            auto dxbc = Extract::getShader(name);
            auto spirv = Extract::translateShader(dxbc);
            return spirv;
        }
    );

    this->lsfgCtxId = std::shared_ptr<int32_t>(
        new int32_t(lsfgCreateContext(fds.at(0), fds.at(1), outFds, extent, format)),
        [lsfgDeleteContext = lsfgDeleteContext](const int32_t* id) {
            lsfgDeleteContext(*id);
        }
    );

    unsetenv("DISABLE_LSFG"); // NOLINT
#endif

    // prepare render passes
    this->cmdPool = Mini::CommandPool(info.device, info.queue.first);
    for (size_t i = 0; i < 8; i++) {
        auto& pass = this->passInfos.at(i);
        pass.renderSemaphores.resize(conf.multiplier - 1);
        pass.acquireSemaphores.resize(conf.multiplier - 1);
        pass.postCopyBufs.resize(conf.multiplier - 1);
        pass.postCopySemaphores.resize(conf.multiplier - 1);
        pass.prevPostCopySemaphores.resize(conf.multiplier - 1);
    }
}

#ifdef __ANDROID__
// Pace the REAL game frame. Phase-locked to the display's vsync grid when the
// app publishes it (vsync.txt) — a free-running sleep clock beats against the
// display and drops a vblank at every phase crossing. The cap quantizes to
// whole vsync counts (30 = every 2nd vsync at 60Hz, every 4th at 120Hz).
void LsContext::paceBaseFrame(int fpsLimit) {
    const auto& conf = Config::activeConf;
    const int64_t targetNs = fpsLimit > 0
        ? 1'000'000'000LL / static_cast<int64_t>(fpsLimit) : 0;
    const int64_t entryNs = nowNs();
    if (targetNs > 0 && entryNs - this->vsyncLastReadNs > 1'000'000'000LL) {
        this->vsyncLastReadNs = entryNs;
        if (!readVsyncFile(conf.config_file, &this->vsyncPhaseNs, &this->vsyncPeriodNs)) {
            this->vsyncPhaseNs = 0;
            this->vsyncPeriodNs = 0;
        }
    }
    if (targetNs > 0 && this->vsyncPeriodNs > 0) {
        const int64_t stepNs = std::max<int64_t>(1,
            (targetNs + this->vsyncPeriodNs / 2) / this->vsyncPeriodNs) * this->vsyncPeriodNs;
        int64_t due = this->pacerNextDueNs;
        if (due <= 0 || due < entryNs - stepNs || due > entryNs + 2 * stepNs) {
            due = this->vsyncPhaseNs
                + ((entryNs - this->vsyncPhaseNs) / stepNs + 1) * stepNs;
        }
        sleepUntilNs(due);
        this->pacerAnchorNs = due;
        this->pacerNextDueNs = due + stepNs;
    } else if (targetNs > 0 && this->pacerAnchorNs != 0
            && entryNs < this->pacerAnchorNs + targetNs) {
        sleepUntilNs(this->pacerAnchorNs + targetNs);
        this->pacerAnchorNs += targetNs;
        this->pacerNextDueNs = 0;
    } else {
        this->pacerAnchorNs = nowNs();
        this->pacerNextDueNs = 0;
    }
}
// debug_dump: read frame_0, frame_1 and every out_n back to the CPU and write
// them as PPMs under <config dir>/dump/. Runs synchronously (fence wait) —
// the frame stalls, which is fine for a pixel-inspection mode.
void LsContext::dumpFrameImages(const Hooks::DeviceInfo& info, uint32_t presentIdx) {
    const auto& conf = Config::activeConf;
    if (conf.config_file.empty()) { this->dumpFailed = true; return; }

    const VkFormat layerFormat = conf.hdr
        ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R16G16B16A16_SFLOAT;

    struct DumpEntry {
        VkImage image;
        std::string name;
        VkExtent2D extent;
        VkFormat format;
        bool signedMap;
        size_t offset;
        size_t bytes;
    };
    std::vector<DumpEntry> entries;
    entries.push_back({ this->frame_0.handle(), "frame0",
        this->extent, layerFormat, false, 0, 0 });
    entries.push_back({ this->frame_1.handle(), "frame1",
        this->extent, layerFormat, false, 0, 0 });
    for (size_t i = 0; i < this->out_n.size(); i++)
        entries.push_back({ this->out_n.at(i).handle(), "out" + std::to_string(i),
            this->extent, layerFormat, false, 0, 0 });
    if (conf.performance)
        for (const auto& dbg : LSFG_3_1P::debugInternalImages(*this->lsfgCtxId))
            entries.push_back({ dbg.image, dbg.name, dbg.extent, dbg.format,
                dbg.format == VK_FORMAT_R16G16B16A16_SFLOAT, 0, 0 });

    size_t bufferBytes = 0;
    for (auto& e : entries) {
        const size_t bpp = dumpBytesPerPixel(e.format);
        if (bpp == 0) {
            std::cerr << "lsfg-vk: debug_dump skipping " << e.name
                << " (unsupported format " << e.format << ")\n";
            e.image = VK_NULL_HANDLE;
            continue;
        }
        e.bytes = static_cast<size_t>(e.extent.width) * e.extent.height * bpp;
        e.offset = (bufferBytes + 7) & ~size_t{7};
        bufferBytes = e.offset + e.bytes;
    }
    entries.erase(std::remove_if(entries.begin(), entries.end(),
        [](const DumpEntry& e) { return e.image == VK_NULL_HANDLE; }), entries.end());
    if (entries.empty() || bufferBytes == 0) {
        this->dumpFailed = true;
        return;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    auto cleanup = [&]() {
        if (buffer != VK_NULL_HANDLE)
            Layer::ovkDestroyBuffer(info.device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE)
            Layer::ovkFreeMemory(info.device, memory, nullptr);
    };
    auto fail = [&](const char* what, VkResult res) {
        std::cerr << "lsfg-vk: debug_dump disabled: " << what
            << " (" << res << ")\n";
        this->dumpFailed = true;
        cleanup();
    };

    const VkBufferCreateInfo bufInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bufferBytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult res = Layer::ovkCreateBuffer(info.device, &bufInfo, nullptr, &buffer);
    if (res != VK_SUCCESS) return fail("vkCreateBuffer", res);

    VkMemoryRequirements reqs{};
    Layer::ovkGetBufferMemoryRequirements(info.device, buffer, &reqs);
    VkPhysicalDeviceMemoryProperties memProps{};
    Layer::ovkGetPhysicalDeviceMemoryProperties(info.physicalDevice, &memProps);
    uint32_t typeIdx = UINT32_MAX;
    VkMemoryPropertyFlags typeFlags = 0;
    const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (!(reqs.memoryTypeBits & (1U << i))) continue;
        const auto flags = memProps.memoryTypes[i].propertyFlags;
        if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if (typeIdx == UINT32_MAX || (flags & wanted) == wanted) {
            typeIdx = i;
            typeFlags = flags;
            if ((flags & wanted) == wanted) break;
        }
    }
    if (typeIdx == UINT32_MAX)
        return fail("no host-visible memory type", VK_ERROR_FEATURE_NOT_PRESENT);

    const VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size,
        .memoryTypeIndex = typeIdx,
    };
    res = Layer::ovkAllocateMemory(info.device, &allocInfo, nullptr, &memory);
    if (res != VK_SUCCESS) return fail("vkAllocateMemory", res);
    res = Layer::ovkBindBufferMemory(info.device, buffer, memory, 0);
    if (res != VK_SUCCESS) return fail("vkBindBufferMemory", res);

    Mini::CommandBuffer cmd(info.device, this->cmdPool);
    Mini::Fence fence(info.device, false);
    cmd.begin();
    for (const auto& e : entries) {
        imageBarrier(cmd.handle(), e.image,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        const VkBufferImageCopy region{
            .bufferOffset = e.offset,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
            .imageExtent = { e.extent.width, e.extent.height, 1 },
        };
        Layer::ovkCmdCopyImageToBuffer(cmd.handle(), e.image,
            VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
        imageBarrier(cmd.handle(), e.image,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    }
    cmd.end();
    cmd.submit(info.queue.second, {}, {}, fence.handle());
    res = fence.wait(2'000'000'000ull);
    if (res != VK_SUCCESS) return fail("dump fence wait", res);

    void* mapped = nullptr;
    res = Layer::ovkMapMemory(info.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (res != VK_SUCCESS) return fail("vkMapMemory", res);
    if (!(typeFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        const VkMappedMemoryRange range{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = memory,
            .size = VK_WHOLE_SIZE,
        };
        Layer::ovkInvalidateMappedMemoryRanges(info.device, 1, &range);
    }

    const auto dumpDir = std::filesystem::path(conf.config_file).parent_path() / "dump";
    std::error_code ec;
    std::filesystem::create_directories(dumpDir, ec);
    bool ok = !ec;
    for (const auto& e : entries) {
        if (!ok) break;
        const auto path = dumpDir /
            ("f" + std::to_string(this->frameIdx) + "_" + e.name + ".ppm");
        ok = writePpm(path, static_cast<const uint8_t*>(mapped) + e.offset,
            e.extent.width, e.extent.height, e.format, e.signedMap);
    }
    Layer::ovkUnmapMemory(info.device, memory);
    cleanup();
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    if (!ok) return fail("writing dump files", VK_ERROR_UNKNOWN);

    std::ofstream index(dumpDir / "index.txt", std::ios::app);
    index << "frame=" << this->frameIdx
        << " input=" << (this->frameIdx % 2 == 0 ? "frame0" : "frame1")
        << " presentIdx=" << presentIdx
        << " mult=" << conf.multiplier << '\n';
    this->dumpDoneFrames++;
    std::cerr << "lsfg-vk: dumped frame " << this->frameIdx
        << " (" << this->dumpDoneFrames << "/" << conf.debugDump << ")\n";
}
#endif

VkResult LsContext::presentPassthrough(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
#ifdef __ANDROID__
    this->paceBaseFrame(Config::activeConf.fpsLimit);
#endif
    return Layer::ovkQueuePresentKHR(queue, pPresentInfo);
}

VkResult LsContext::present(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx) {
    const auto& conf = Config::activeConf;
    auto& pass = this->passInfos.at(this->frameIdx % 8);

#ifdef __ANDROID__
    // Android path: synchronous frame generation using waitIdle()
    // instead of OPAQUE_FD semaphore export which Turnip doesn't support.

    // Base frame-rate limiter (see paceBaseFrame): the on-screen rate becomes
    // fps_limit * multiplier.
    this->paceBaseFrame(conf.fpsLimit);

    // Track the delivered real-frame interval as an EWMA. A cap is a ceiling,
    // not a promise on base performance, so generated-frame spacing follows
    // the measured cadence.
    if (this->lastRealPresentNs != 0 && this->pacerAnchorNs > this->lastRealPresentNs) {
        const uint64_t delta = static_cast<uint64_t>(
            this->pacerAnchorNs - this->lastRealPresentNs);
        // A delta this large is a pause/resume, not a slow frame; feeding it
        // into the EWMA would mis-pace generated frames for seconds while it
        // decays. Re-learn from scratch instead.
        if (delta > 250'000'000ull)
            this->baseIntervalEwmaNs = 0;
        else
            this->baseIntervalEwmaNs = this->baseIntervalEwmaNs == 0
                ? delta : (this->baseIntervalEwmaNs * 7 + delta) / 8;
    }
    this->lastRealPresentNs = this->pacerAnchorNs;

    // If the previous frame's copy still hasn't signaled, the sync path is
    // stuck; pass the frame through untouched. Enough consecutive timeouts
    // disable framegen for this swapchain entirely.
    const auto fenceRes = this->preCopyFence.wait(kSyncFenceTimeoutNs);
    if (fenceRes == VK_SUCCESS)
        this->copyFenceTimeouts = 0;
    if (fenceRes != VK_SUCCESS) {
        if (++this->copyFenceTimeouts >= kMaxFenceTimeouts && !this->forceDisabled) {
            this->forceDisabled = true;
            std::cerr << "lsfg-vk: disabling frame generation for this swapchain after "
                << this->copyFenceTimeouts << " consecutive copy fence timeouts\n";
        }
        const VkPresentInfoKHR passInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = pNext,
            .waitSemaphoreCount = static_cast<uint32_t>(gameRenderSemaphores.size()),
            .pWaitSemaphores = gameRenderSemaphores.data(),
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &presentIdx,
        };
        return Layer::ovkQueuePresentKHR(queue, &passInfo);
    }
    this->preCopyFence.reset();

    // 1. copy swapchain image to frame_0/frame_1
    //    Use a simple semaphore (no fd export) to synchronize the copy.
    //    The input AHB is written by this device and read by framegen's
    //    separate device, so the write must be RELEASED to the external queue
    //    family — without the ownership transfer the other device may read
    //    stale data (Adreno caches are not coherent across devices).
    pass.preCopySemaphores.at(1) = Mini::Semaphore(info.device);
    pass.preCopyBuf = Mini::CommandBuffer(info.device, this->cmdPool);
    pass.preCopyBuf.begin();
    {
        VkCommandBuffer cmd = pass.preCopyBuf.handle();
        VkImage inputImg = this->frameIdx % 2 == 0
            ? this->frame_0.handle() : this->frame_1.handle();
        imageBarrier(cmd, this->swapchainImages.at(presentIdx),
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        imageBarrier(cmd, inputImg,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        blitFull(cmd, this->swapchainImages.at(presentIdx), inputImg,
            this->extent.width, this->extent.height);
        imageBarrier(cmd, this->swapchainImages.at(presentIdx),
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        imageBarrier(cmd, inputImg,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    }
    pass.preCopyBuf.end();

    std::vector<VkSemaphore> gameRenderSemaphores2 = gameRenderSemaphores;
    if (this->frameIdx > 0)
        gameRenderSemaphores2.emplace_back(this->passInfos.at((this->frameIdx - 1) % 8)
            .preCopySemaphores.at(1).handle());

    // Submit the copy. Same-queue submission order sequences framegen after
    // it; the fence is only the next frame's watchdog. The second signal
    // semaphore lets the real present wait for the copy (and thus the game's
    // rendering) even when every generated frame is skipped.
    pass.preCopySemaphores.at(0) = Mini::Semaphore(info.device);
    pass.preCopyBuf.submit(info.queue.second,
        gameRenderSemaphores2,
        { pass.preCopySemaphores.at(0).handle(),
          pass.preCopySemaphores.at(1).handle() },
        this->preCopyFence.handle());
    const int64_t workStartNs = nowNs();

    // 2. Tell framegen to generate intermediary frames
    //    presentContext(id, -1, {}) — no semaphore FDs, synchronous
    std::vector<int> noOutSems;  // empty
    if (conf.performance)
        LSFG_3_1P::presentContext(*this->lsfgCtxId, -1, noOutSems);
    else
        LSFG_3_1::presentContext(*this->lsfgCtxId, -1, noOutSems);

    // 3. No CPU-side wait: framegen submitted on the same queue, so the
    //    generated-frame copies below are ordered after it by submission
    //    order plus the image barriers.
    this->statsWorkNs += static_cast<uint64_t>(nowNs() - workStartNs);

    if (conf.debugDump > 0 && this->dumpDoneFrames < conf.debugDump
            && !this->dumpFailed)
        this->dumpFrameImages(info, presentIdx);

    // Late-frame guard interval: 1/multiplier of the measured base interval,
    // anchored on the paced real frame. Presents are not delayed to it — a
    // generated frame far past its slot is dropped to hold cadence.
    uint64_t baseIntervalNs = this->baseIntervalEwmaNs;
    if (baseIntervalNs == 0 && conf.fpsLimit > 0)
        baseIntervalNs = 1'000'000'000ull / static_cast<uint64_t>(conf.fpsLimit);
    const uint64_t outIntervalNs = baseIntervalNs / conf.multiplier;

    // 4. Copy generated frames to swapchain images and present them
    int lastPresentedGen = -1;
    for (size_t i = 0; i < static_cast<size_t>(conf.multiplier - 1); i++) {
        // A generated frame far past its slot is dropped to hold cadence
        // (presenting it late would push everything after it off-pace).
        // Present as early as possible and let the FIFO queue space the frames
        // across vblanks (desktop-path behavior): early enqueue maximizes the
        // margin to each frame's vblank. Sleeping toward the nominal slot was
        // measured to HALVE that margin and cause periodic drops. Only a frame
        // already past its slot is dropped, to hold cadence.
        // Space the presents across the base period: generated frame i at
        // anchor + i*out, the real frame at anchor + (mult-1)*out below. When
        // the total present rate saturates the display this sleep is a no-op
        // (queue back-pressure spaces them); when it doesn't (e.g. cap 30 on a
        // 120Hz panel) it prevents the pair landing on adjacent vblanks.
        if (outIntervalNs > 0) {
            const int64_t dueNs = this->pacerAnchorNs
                + static_cast<int64_t>(i * outIntervalNs);
            const int64_t nowGenNs = nowNs();
            if (nowGenNs > dueNs + static_cast<int64_t>(outIntervalNs) * 3 / 2) {
                this->statsGenSkips++;
                continue;
            }
            if (nowGenNs > dueNs)
                this->statsGenLateNs += static_cast<uint64_t>(nowGenNs - dueNs);
            else
                this->statsGenSleepNs += static_cast<uint64_t>(dueNs - nowGenNs);
            sleepUntilNs(dueNs);
        }

        // acquire next swapchain image
        pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
        uint32_t imageIdx{};
        auto res = Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
            pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to acquire next swapchain image");
        if (this->statsSeq.size() < 200)
            this->statsSeq += 'a' + static_cast<char>(imageIdx % 26);

        // copy output image to swapchain image (same device: plain
        // compute->transfer barriers order the read after framegen's write)
        pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.prevPostCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);
        pass.postCopyBufs.at(i).begin();
        {
            VkCommandBuffer cmd = pass.postCopyBufs.at(i).handle();
            // debug_inputs: 1 = previous-frame input image, 2 = the real
            // swapchain image itself (pure same-format round trip)
            VkImage genSrc = this->out_n.at(i).handle();
            if (conf.debugInputs == 1)
                genSrc = this->frameIdx % 2 == 0
                    ? this->frame_1.handle() : this->frame_0.handle();
            else if (conf.debugInputs == 2)
                genSrc = this->swapchainImages.at(presentIdx);
            imageBarrier(cmd, genSrc,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                conf.debugInputs == 2 ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            imageBarrier(cmd, this->swapchainImages.at(imageIdx),
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            blitFull(cmd, genSrc,
                this->swapchainImages.at(imageIdx),
                this->extent.width, this->extent.height);
            imageBarrier(cmd, genSrc,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                conf.debugInputs == 2 ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL);
            imageBarrier(cmd, this->swapchainImages.at(imageIdx),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
        pass.postCopyBufs.at(i).end();
        pass.postCopyBufs.at(i).submit(info.queue.second,
            { pass.acquireSemaphores.at(i).handle() },
            { pass.postCopySemaphores.at(i).handle(),
              pass.prevPostCopySemaphores.at(i).handle() });

        // present swapchain image
        VkSemaphore postCopySem = pass.postCopySemaphores.at(i).handle();
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = i == 0 ? pNext : nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &postCopySem,
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &imageIdx,
        };
        const int64_t genPresentStartNs = nowNs();
        res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
        this->statsGenPresentNs += static_cast<uint64_t>(nowNs() - genPresentStartNs);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to present swapchain image");
        lastPresentedGen = static_cast<int>(i);
        this->statsPresents++;
    }

    // 5. present actual next frame (the real capture, not a generated one).
    //    Waits on the last generated frame's second post-copy semaphore
    //    (the first one was consumed by that frame's own present). If every
    //    generated frame was skipped no semaphore is needed: the copy fence
    //    wait above already proved the game's rendering finished.
    if (outIntervalNs > 0 && conf.fpsLimit > 0)
        sleepUntilNs(this->pacerAnchorNs
            + static_cast<int64_t>((conf.multiplier - 1) * outIntervalNs));
    if (this->statsSeq.size() < 200) {
        this->statsSeq += 'A' + static_cast<char>(presentIdx % 26);
        this->statsSeq += '0' + static_cast<char>(gameRenderSemaphores.size() % 10);
    }
    std::vector<VkSemaphore> finalWaits = { pass.preCopySemaphores.at(0).handle() };
    if (lastPresentedGen >= 0)
        finalWaits.push_back(pass.prevPostCopySemaphores
            .at(static_cast<size_t>(lastPresentedGen)).handle());
    const VkPresentInfoKHR finalPresentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = static_cast<uint32_t>(finalWaits.size()),
        .pWaitSemaphores = finalWaits.data(),
        .swapchainCount = 1,
        .pSwapchains = &this->swapchain,
        .pImageIndices = &presentIdx,
    };
    const int64_t realPresentStartNs = nowNs();
    auto res = Layer::ovkQueuePresentKHR(queue, &finalPresentInfo);
    this->statsRealPresentNs += static_cast<uint64_t>(nowNs() - realPresentStartNs);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw LSFG::vulkan_error(res, "Failed to present swapchain image");
    this->statsPresents++;
    this->statsRealPresents++;

    // Publish measured fps + per-stage timing for the HUD and diagnosis.
    const int64_t statsNow = nowNs();
    if (this->statsWindowStartNs == 0) {
        this->statsWindowStartNs = statsNow;
    } else if (statsNow - this->statsWindowStartNs >= 500'000'000LL) {
        const double windowSec =
            static_cast<double>(statsNow - this->statsWindowStartNs) / 1e9;
        const double frames = static_cast<double>(this->statsRealPresents);
        writeStats(conf.config_file,
            static_cast<double>(this->statsPresents) / windowSec,
            frames / windowSec,
            StatsDetail{
                .workMs        = static_cast<double>(this->statsWorkNs) / 1e6 / frames,
                .genSleepMs    = static_cast<double>(this->statsGenSleepNs) / 1e6 / frames,
                .genLateMs     = static_cast<double>(this->statsGenLateNs) / 1e6 / frames,
                .genPresentMs  = static_cast<double>(this->statsGenPresentNs) / 1e6 / frames,
                .realPresentMs = static_cast<double>(this->statsRealPresentNs) / 1e6 / frames,
                .genSkips      = this->statsGenSkips,
                .seq           = this->statsSeq,
            });
        this->statsSeq.clear();
        this->statsPresents = 0;
        this->statsRealPresents = 0;
        this->statsWorkNs = 0;
        this->statsGenSleepNs = 0;
        this->statsGenLateNs = 0;
        this->statsGenPresentNs = 0;
        this->statsRealPresentNs = 0;
        this->statsGenSkips = 0;
        this->statsWindowStartNs = statsNow;
    }

    this->frameIdx++;
    return res;

#else
    // Desktop Linux path: OPAQUE_FD semaphore-based synchronization

    // 1. copy swapchain image to frame_0/frame_1
    int preCopySemaphoreFd{};
    pass.preCopySemaphores.at(0) = Mini::Semaphore(info.device, &preCopySemaphoreFd);
    pass.preCopySemaphores.at(1) = Mini::Semaphore(info.device);
    pass.preCopyBuf = Mini::CommandBuffer(info.device, this->cmdPool);
    pass.preCopyBuf.begin();

    Utils::copyImage(pass.preCopyBuf.handle(),
        this->swapchainImages.at(presentIdx),
        this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
        this->extent.width, this->extent.height,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        true, false);

    pass.preCopyBuf.end();

    std::vector<VkSemaphore> gameRenderSemaphores2 = gameRenderSemaphores;
    if (this->frameIdx > 0)
        gameRenderSemaphores2.emplace_back(this->passInfos.at((this->frameIdx - 1) % 8)
            .preCopySemaphores.at(1).handle());
    pass.preCopyBuf.submit(info.queue.second,
        gameRenderSemaphores2,
        { pass.preCopySemaphores.at(0).handle(),
          pass.preCopySemaphores.at(1).handle() });

    // 2. render intermediary frames
    std::vector<int> renderSemaphoreFds(conf.multiplier - 1);
    for (size_t i = 0; i < (conf.multiplier - 1); ++i)
        pass.renderSemaphores.at(i) = Mini::Semaphore(info.device, &renderSemaphoreFds.at(i));

    if (conf.performance)
        LSFG_3_1P::presentContext(*this->lsfgCtxId,
            preCopySemaphoreFd,
            renderSemaphoreFds);
    else
        LSFG_3_1::presentContext(*this->lsfgCtxId,
            preCopySemaphoreFd,
            renderSemaphoreFds);

    for (size_t i = 0; i < (conf.multiplier - 1); i++) {
        // 3. acquire next swapchain image
        pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
        uint32_t imageIdx{};
        auto res = Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
            pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to acquire next swapchain image");

        // 4. copy output image to swapchain image
        pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.prevPostCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);
        pass.postCopyBufs.at(i).begin();

        Utils::copyImage(pass.postCopyBufs.at(i).handle(),
            this->out_n.at(i).handle(),
            this->swapchainImages.at(imageIdx),
            this->extent.width, this->extent.height,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            false, true);

        pass.postCopyBufs.at(i).end();
        pass.postCopyBufs.at(i).submit(info.queue.second,
            { pass.acquireSemaphores.at(i).handle(),
              pass.renderSemaphores.at(i).handle() },
            { pass.postCopySemaphores.at(i).handle(),
              pass.prevPostCopySemaphores.at(i).handle() });

        // 5. present swapchain image
        std::vector<VkSemaphore> waitSemaphores{ pass.postCopySemaphores.at(i).handle() };
        if (i != 0) waitSemaphores.emplace_back(pass.prevPostCopySemaphores.at(i - 1).handle());

        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = i == 0 ? pNext : nullptr, // only set on first present
            .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
            .pWaitSemaphores = waitSemaphores.data(),
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &imageIdx,
        };
        res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to present swapchain image");
    }

    // 6. present actual next frame
    VkSemaphore lastPrevPostCopySemaphore =
        pass.prevPostCopySemaphores.at(conf.multiplier - 1 - 1).handle();
    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &lastPrevPostCopySemaphore,
        .swapchainCount = 1,
        .pSwapchains = &this->swapchain,
        .pImageIndices = &presentIdx,
    };
    auto res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw LSFG::vulkan_error(res, "Failed to present swapchain image");

    this->frameIdx++;
    return res;
#endif
}
