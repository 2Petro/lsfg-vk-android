#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <chrono>
#include <cstddef>
#include <string>

namespace Config {

    /// lsfg-vk configuration
    struct Configuration {
        /// Whether lsfg-vk should be loaded in the first place.
        bool enable{false};
        /// Path to Lossless.dll.
        std::string dll;

        /// The frame generation muliplier
        size_t multiplier{2};
        /// The internal flow scale factor
        float flowScale{1.0F};
        /// Whether performance mode is enabled
        bool performance{false};
        /// Whether HDR is enabled
        bool hdr{false};

        /// Experimental flag for overriding the synchronization method.
        VkPresentModeKHR e_present;

        /// HWME (Adreno GL_QCOM_motion_estimation) enable, default true
        bool hwme{true};
        /// HWME max motion clamp, default 128
        float hwmeMaxMv{128.0F};
        /// HWME debug level 0-5, default 0
        int hwmeDebug{0};
        /// Timing debug single-line log, default false
        bool timingDebug{false};

        /// Fixed target FPS mode: when enabled, ignore multiplier and gen to hit target
        bool targetFpsEnabled{false};
        int targetFps{60};
        /// When >0, use this as real/base fps instead of auto-estimating R (fixes FIFO throttle artifacts)
        int targetBaseFps{0};

        /// NPU framegen (RIFE ONNX on Hexagon HTP) second option. Empty
        /// npu_model = DLL/shader path (default). All TOML-driven: no
        /// environment is available in the Android app process.
        /// Path to the NPU ONNX model (e.g. rife46_400x300_ft_slim_fp16).
        std::string npu_model;
        /// Worker binary override (bionic npu_interp).
        std::string npu_bin;
        /// Android launch script: sh <worker_sh> <model> <bin> <sockfd>.
        /// Sets up env (LD_LIBRARY_PATH, ADSP_LIBRARY_PATH) and execs the
        /// worker. Empty = exec workerBin directly (bionic) — glibc uses
        /// the linker64 dispatch instead.
        std::string worker_sh;
        /// FNV cross-check cadence in runs (0 = off after proof).
        long npu_verify_every{240};
        /// HTP power/perf (QNN htp_performance_mode): burst|balanced|
        /// power-saver|sustained_high_performance|...; forwarded to the
        /// worker via the launch script.
        std::string npu_perf_mode{"burst"};
        /// Cap NPU runs/sec (0 = uncapped). Paces run starts; e.g. 32
        /// holds a steady rhythm and saves power past display rate.
        double npu_target_inf{0};
        /// Per-frame stage logging (goes to lsfg.log on Android).
        bool npu_verbose{false};
        /// Passthrough: run skeleton/delivery without NPU work.
        bool npu_dryrun{false};
        /// Skip DMA_BUF_SYNC ioctls around handoffs.
        bool npu_nosync{false};

        /// Path to the configuration file.
        std::filesystem::path config_file;
        /// File timestamp of the configuration file
        std::chrono::time_point<std::chrono::file_clock> timestamp;
    };

    /// Active configuration. Must be set in main.cpp.
    extern Configuration activeConf;

    ///
    /// Read the configuration file while preserving the previous configuration
    /// in case of an error.
    ///
    /// @param file The path to the configuration file.
    ///
    /// @throws std::runtime_error if an error occurs while loading the configuration file.
    ///
    void updateConfig(const std::string& file);

    ///
    /// Get the configuration for a game.
    ///
    /// @param name The name of the executable to fetch.
    /// @return The configuration for the game or global configuration.
    ///
    /// @throws std::runtime_error if the configuration is invalid.
    ///
    Configuration getConfig(const std::pair<std::string, std::string>& name);

}
