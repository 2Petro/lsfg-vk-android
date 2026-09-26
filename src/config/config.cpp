#include "config/config.hpp"
#include "common/exception.hpp"

#include "config/default_conf.hpp"

#include <vulkan/vulkan_core.h>
#include <toml11/find.hpp>
#include <toml11/parser.hpp>
#include <toml.hpp>

#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <optional>
#include <fstream>
#include <cstdlib>
#include <utility>
#include <string>

using namespace Config;

namespace {
    Configuration globalConf{};
    std::optional<std::unordered_map<std::string, Configuration>> gameConfs;
}

Configuration Config::activeConf{};

namespace {
    /// Turn a string into a VkPresentModeKHR enum value.
    VkPresentModeKHR into_present(const std::string& mode) {
        if (mode == "fifo" || mode == "vsync")
            return VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR;
        if (mode == "mailbox")
            return VkPresentModeKHR::VK_PRESENT_MODE_MAILBOX_KHR;
        if (mode == "immediate")
            return VkPresentModeKHR::VK_PRESENT_MODE_IMMEDIATE_KHR;
        return VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR;
    }

    /// Numeric TOML knob accepting both integer and floating values.
    /// (toml11 find_or<T> does not coerce int<->double; a plain
    /// find_or<double> on `npu_target_inf = 32` silently keeps 0.0.)
    double into_number(const toml::value& table, const std::string& key,
                       double dflt) {
        const auto& t = table.as_table();
        auto it = t.find(key);
        if (it == t.end())
            return dflt;
        const toml::value& v = it->second;
        if (v.is_floating())
            return v.as_floating();
        if (v.is_integer())
            return static_cast<double>(v.as_integer());
        return dflt;
    }
}

void Config::updateConfig(const std::string& file) {
    if (!std::filesystem::exists(file)) {
        std::cerr << "lsfg-vk: Placing default configuration file at " << file << '\n';
        const auto parent = std::filesystem::path(file).parent_path();
        if (!std::filesystem::exists(parent))
            if (!std::filesystem::create_directories(parent))
                throw std::runtime_error("Unable to create configuration directory at " + parent.string());

        std::ofstream out(file);
        if (!out.is_open())
            throw std::runtime_error("Unable to create configuration file at " + file);
        out << DEFAULT_CONFIG;
        out.close();
    }

    // parse config file
    std::optional<toml::value> parsed;
    try {
        parsed.emplace(toml::parse(file));
        if (!parsed->contains("version"))
            throw std::runtime_error("Configuration file is missing 'version' field");
        if (parsed->at("version").as_integer() != 1)
            throw std::runtime_error("Configuration file version is not supported, expected 1");
    } catch (const std::exception& e) {
        throw LSFG::rethrowable_error("Unable to parse configuration file", e);
    }
    auto& toml = *parsed;

    // parse global configuration
    const toml::value globalTable = toml::find_or_default<toml::table>(toml, "global");
    Configuration global{
        .dll =   toml::find_or(globalTable, "dll", std::string()),
        .hwme = toml::find_or(globalTable, "hwme", true),
        .hwmeMaxMv = static_cast<float>(toml::find_or(globalTable, "hwme_maxmv", 128.0)),
        .hwmeDebug = toml::find_or(globalTable, "hwme_debug", 0),
        .timingDebug = toml::find_or(globalTable, "timing_debug", false),
        .targetFpsEnabled = toml::find_or(globalTable, "target_fps_enabled", false),
        .targetFps = toml::find_or(globalTable, "target_fps", 60),
        .targetBaseFps = toml::find_or(globalTable, "target_base_fps", 0),
        .npu_model = toml::find_or(globalTable, "npu_model", std::string()),
        .npu_bin = toml::find_or(globalTable, "npu_bin", std::string()),
        .worker_sh = toml::find_or(globalTable, "worker_sh", std::string()),
        .npu_verify_every = toml::find_or(globalTable, "npu_verify_every", 240L),
        .npu_perf_mode = toml::find_or(globalTable, "npu_perf_mode", std::string("burst")),
        .npu_target_inf = into_number(globalTable, "npu_target_inf", 0.0),
        .npu_verbose = toml::find_or(globalTable, "npu_verbose", false),
        .npu_dryrun = toml::find_or(globalTable, "npu_dryrun", false),
        .npu_nosync = toml::find_or(globalTable, "npu_nosync", false),
        .config_file = file,
        .timestamp = std::filesystem::last_write_time(file)
    };

    // validate global configuration
    if (global.multiplier < 2)
        throw std::runtime_error("Global Multiplier cannot be less than 2");
    if (global.flowScale < 0.25F || global.flowScale > 1.0F)
        throw std::runtime_error("Flow scale must be between 0.25 and 1.0");
    if (global.targetFps < 10 || global.targetFps > 480)
        throw std::runtime_error("target_fps must be between 10 and 480");
    if (global.targetBaseFps < 0 || global.targetBaseFps > 480)
        throw std::runtime_error("target_base_fps must be between 0 and 480");

    // parse game-specific configuration
    std::unordered_map<std::string, Configuration> games;
    const toml::value gamesList = toml::find_or_default<toml::array>(toml, "game");
    for (const auto& gameTable : gamesList.as_array()) {
        if (!gameTable.is_table())
            throw std::runtime_error("Invalid game configuration entry");
        if (!gameTable.contains("exe"))
            throw std::runtime_error("Game override missing 'exe' field");

        const std::string exe = toml::find<std::string>(gameTable, "exe");
        Configuration game{
            .enable = true,
            .dll = global.dll,
            .multiplier = toml::find_or(gameTable, "multiplier", 2U),
            .flowScale = toml::find_or(gameTable, "flow_scale", 1.0F),
            .performance = toml::find_or(gameTable, "performance_mode", false),
            .hdr = toml::find_or(gameTable, "hdr_mode", false),
            .e_present =   into_present(toml::find_or(gameTable, "experimental_present_mode", "")),
            .hwme = toml::find_or(gameTable, "hwme", global.hwme),
            .hwmeMaxMv = static_cast<float>(toml::find_or(gameTable, "hwme_maxmv", static_cast<double>(global.hwmeMaxMv))),
            .hwmeDebug = toml::find_or(gameTable, "hwme_debug", global.hwmeDebug),
            .timingDebug = toml::find_or(gameTable, "timing_debug", global.timingDebug),
            .targetFpsEnabled = toml::find_or(gameTable, "target_fps_enabled", global.targetFpsEnabled),
            .targetFps = toml::find_or(gameTable, "target_fps", global.targetFps),
            .targetBaseFps = toml::find_or(gameTable, "target_base_fps", global.targetBaseFps),
            .npu_model = toml::find_or(gameTable, "npu_model", global.npu_model),
            .npu_bin = toml::find_or(gameTable, "npu_bin", global.npu_bin),
            .worker_sh = toml::find_or(gameTable, "worker_sh", global.worker_sh),
            .npu_verify_every = toml::find_or(gameTable, "npu_verify_every", global.npu_verify_every),
            .npu_perf_mode = toml::find_or(gameTable, "npu_perf_mode", global.npu_perf_mode),
            .npu_target_inf = into_number(gameTable, "npu_target_inf", global.npu_target_inf),
            .npu_verbose = toml::find_or(gameTable, "npu_verbose", global.npu_verbose),
            .npu_dryrun = toml::find_or(gameTable, "npu_dryrun", global.npu_dryrun),
            .npu_nosync = toml::find_or(gameTable, "npu_nosync", global.npu_nosync),
            .config_file = file,
            .timestamp = global.timestamp
        };

        // validate the configuration
        if (game.multiplier < 1)
            throw std::runtime_error("Multiplier cannot be less than 1");
        if (game.flowScale < 0.25F || game.flowScale > 1.0F)
            throw std::runtime_error("Flow scale must be between 0.25 and 1.0");
        if (game.targetFps < 10 || game.targetFps > 480)
            throw std::runtime_error("target_fps must be between 10 and 480");
        if (game.targetBaseFps < 0 || game.targetBaseFps > 480)
            throw std::runtime_error("target_base_fps must be between 0 and 480");
        games[exe] = std::move(game);
    }

    // store configurations
    globalConf = global;
    gameConfs = std::move(games);
}

Configuration Config::getConfig(const std::pair<std::string, std::string>& name) {
    // process legacy environment variables
    if (std::getenv("LSFG_LEGACY")) {
        Configuration conf{
            .enable = true,
            .multiplier = 2,
            .flowScale = 1.0F,
            .e_present = VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR
        };

        const char* dll = std::getenv("LSFG_DLL_PATH");
        if (dll) conf.dll = std::string(dll);
        const char* multiplier = std::getenv("LSFG_MULTIPLIER");
        if (multiplier) conf.multiplier = std::stoul(multiplier);
        const char* flow_scale = std::getenv("LSFG_FLOW_SCALE");
        if (flow_scale) conf.flowScale = std::stof(flow_scale);
        const char* performance = std::getenv("LSFG_PERFORMANCE_MODE");
        if (performance) conf.performance = std::string(performance) == "1";
        const char* hdr = std::getenv("LSFG_HDR_MODE");
        if (hdr) conf.hdr = std::string(hdr) == "1";
        const char* e_present = std::getenv("LSFG_EXPERIMENTAL_PRESENT_MODE");
        if (e_present) conf.e_present = into_present(std::string(e_present));
        const char* ten = std::getenv("LSFG_TARGET_FPS_ENABLED");
        if (ten) conf.targetFpsEnabled = std::string(ten)=="1" || std::string(ten)=="true";
        const char* tf = std::getenv("LSFG_TARGET_FPS");
        if (tf) conf.targetFps = std::stoi(tf);
        const char* tbf = std::getenv("LSFG_TARGET_BASE_FPS");
        if (tbf) conf.targetBaseFps = std::stoi(tbf);

        return conf;
    }

    // process new configuration system
    if (!gameConfs.has_value())
        return globalConf;

    const auto& games = *gameConfs;
    auto it = std::ranges::find_if(games, [&name](const auto& pair) {
        return name.first.ends_with(pair.first) || (name.second == pair.first);
    });
    if (it != games.end())
        return it->second;

    return globalConf;
}
