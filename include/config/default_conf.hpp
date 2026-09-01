#pragma once

#include <string>

const std::string DEFAULT_CONFIG = R"(version = 1
[global]
# override the location of Lossless Scaling
# dll = "/games/Lossless Scaling/Lossless.dll"
# hwme = true
# hwme_maxmv = 128.0
# hwme_debug = 0
# timing_debug = false
# target_fps_enabled = false
# target_fps = 60
# target_base_fps = 0  # 0=auto estimate R, else fixed base (e.g. 30) -> gen = target/base

# [[game]] # example entry
# exe = "Game.exe"
#
# multiplier = 3
# flow_scale = 0.7
# performance_mode = true
# hdr_mode = false
#
# experimental_present_mode = "fifo"

[[game]] # default vkcube entry
exe = "vkcube"

multiplier = 4
performance_mode = true

[[game]] # default benchmark entry
exe = "benchmark"

multiplier = 4
performance_mode = false

[[game]] # override Genshin Impact
exe = "Genshin"

multiplier = 3
)";
