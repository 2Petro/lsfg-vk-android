// npu_interp.cpp - bionic NPU interpolation service for the Vulkan layer.
//
// Loads a 2-input RIFE model (shapes read from graph, e.g. 400x300), allocs
// HTP pool in0/in1/out, warms up, then serves a line protocol over a unix
// socket passed as argv[2] (SCM_RIGHTS carries the dma fds to the layer):
//   worker -> "NPU_FDS in0 in1 out insz outsz inW inH outW outH\n" + 3 fds
//   worker -> "NPU_READY\n"
//   layer  -> 'R' : run pair -> worker replies "D <ms>\n"
//   layer  -> 'V' : run pair + CPU FNV -> worker replies "D <ms> 0x<fnv>\n"
//   EOF -> clean exit(0). No disk I/O anywhere in the loop.
//
// Build (NDK r29, cwd = onnx dir):
//   NDK=/opt/r29
//   CLANG=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++
//   $CLANG -std=c++17 -O3 -ffast-math -fPIE -pie -Iheaders npu_interp.cpp \
//       -Llib/jni/arm64-v8a -lonnxruntime -static-libstdc++ -o npu_interp
// Smoke (linker env like run.sh; serves one run then EOF quits):
//   (printf 'R'; sleep 3) | ./npu_interp models/rife_model/rife46_400x300_ft_slim_fp16.onnx 99
// NOTE: socket fd 99 trick above does NOT carry fds (shell pipes can't
// SCM_RIGHTS); full test happens through the layer. Smoke only checks init.

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <filesystem>
#include <onnxruntime_cxx_api.h>

inline bool file_exists(const std::string& name) {
    return std::filesystem::exists(name) && std::filesystem::file_size(name) > 0;
}

size_t GetTypeSize(ONNXTensorElementDataType elem_type) {
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return 4;
        default: return 4;
    }
}

struct ManagedBuffer {
    void* data = nullptr;
    size_t size = 0;
    bool from_shared_allocator = false;
    Ort::Allocator* shared_allocator = nullptr;
    std::vector<char> cpu_storage;
    void Allocate(size_t bytes, Ort::Allocator* shared) {
        size = bytes;
        if (shared) {
            try {
                data = shared->Alloc(bytes);
                from_shared_allocator = true;
                shared_allocator = shared;
                std::memset(data, 0, bytes);
                return;
            } catch (const Ort::Exception&) {}
        }
        cpu_storage.assign(bytes, 0);
        data = cpu_storage.data();
        from_shared_allocator = false;
    }
    ~ManagedBuffer() {
        if (from_shared_allocator && shared_allocator && data) shared_allocator->Free(data);
    }
};

std::unordered_map<std::string, std::string> BuildQnnOptions(bool shared) {
    std::unordered_map<std::string, std::string> opts = {
        {"backend_path", "libQnnHtp.so"},
        {"htp_graph_finalization_optimization_mode", "3"},
        {"htp_arch", "73"},
        {"qnn_context_priority", "high"},
        {"rpc_control_latency", "100"},
        {"vtcm_mb", "8"},
        {"htp_performance_mode", "burst"},
    };
    if (shared) opts["enable_htp_shared_memory_allocator"] = "1";
    return opts;
}

// Send one line + N fds over a unix socket (SCM_RIGHTS).
static bool send_fds(int sock, const std::string& line, int* fds, int nfds) {
    struct msghdr m = {};
    struct iovec io = {(void*)line.data(), line.size()};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    char ctl[CMSG_SPACE(sizeof(int) * 4)];
    m.msg_control = ctl;
    m.msg_controllen = sizeof(ctl);
    struct cmsghdr* c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
    memcpy(CMSG_DATA(c), fds, sizeof(int) * nfds);
    m.msg_controllen = c->cmsg_len;
    return sendmsg(sock, &m, 0) == (ssize_t)line.size();
}

static uint64_t fnv_of(const void* data, size_t n) {
    uint64_t fnv = 1469598103934665603ULL;
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; i++) { fnv ^= p[i]; fnv *= 1099511628211ULL; }
    return fnv;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.onnx> <sockfd>\n";
        return 1;
    }
    std::string model_path = argv[1];
    int sock = std::atoi(argv[2]);
    std::string ctx_path = model_path + ".ctx.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "NPU_INTERP");
    if (!file_exists(ctx_path) && model_path.find(".ctx.onnx") == std::string::npos) {
        std::cerr << "[NPUW] compiling graph -> " << ctx_path << "...\n";
        Ort::SessionOptions go;
        go.SetLogSeverityLevel(4);
        go.AddConfigEntry("ep.context_enable", "1");
        go.AddConfigEntry("ep.context_file_path", ctx_path.c_str());
        go.AddConfigEntry("ep.context_embed_mode", "1");
        go.AddConfigEntry("session.disable_cpu_ep_fallback", "0");
        auto q = BuildQnnOptions(true);
        q.erase("htp_performance_mode");
        go.AppendExecutionProvider("QNN", q);
        Ort::Session gs(env, model_path.c_str(), go);
        std::cerr << "[NPUW] ctx generated.\n";
    }
    std::string active = file_exists(ctx_path) ? ctx_path : model_path;
    std::unique_ptr<Ort::Session> session;
    bool want_shared = true;
    for (auto at : {std::make_pair(true, false), std::make_pair(false, false),
                    std::make_pair(true, true), std::make_pair(false, true)}) {
        try {
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(1);
            so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            so.DisableProfiling();
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            so.SetLogSeverityLevel(4);
            so.AddConfigEntry("session.disable_cpu_ep_fallback", at.second ? "0" : "1");
            so.AppendExecutionProvider("QNN", BuildQnnOptions(at.first));
            session = std::make_unique<Ort::Session>(env, active.c_str(), so);
            want_shared = at.first;
            std::cerr << "[NPUW] session loaded (shared " << (at.first ? "on" : "off")
                      << ", fallback " << (at.second ? "ALLOW" : "off") << ")\n";
            break;
        } catch (const Ort::Exception& e) {
            std::cerr << "[NPUW] load failed: " << e.what() << "\n";
        }
    }
    if (!session) return 1;

    std::unique_ptr<Ort::Allocator> htp_alloc;
    Ort::MemoryInfo htp_info(nullptr);
    if (want_shared) {
        try {
            htp_info = Ort::MemoryInfo("QnnHtpShared", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
            htp_alloc = std::make_unique<Ort::Allocator>(*session, htp_info);
        } catch (const Ort::Exception& e) {
            std::cerr << "[NPUW] shared allocator unavailable: " << e.what() << "\n";
        }
    }
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo cpu_info("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    size_t n_in = session->GetInputCount(), n_out = session->GetOutputCount();
    if (n_in != 2 || n_out < 1) {
        std::cerr << "[NPUW] need a 2-input model\n";
        return 1;
    }
    struct IO { std::string name; ManagedBuffer buf; std::vector<int64_t> shape; ONNXTensorElementDataType et; };
    std::vector<IO> ins(n_in);
    std::vector<Ort::Value> in_tensors;
    Ort::IoBinding bind(*session);
    for (size_t i = 0; i < n_in; i++) {
        auto np = session->GetInputNameAllocated(i, alloc);
        ins[i].name = np.get();
        auto ti = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        ins[i].et = ti.GetElementType();
        ins[i].shape = ti.GetShape();
        size_t els = 1;
        for (auto d : ins[i].shape) els *= (size_t)d;
        ins[i].buf.Allocate(els * GetTypeSize(ins[i].et), htp_alloc.get());
        const OrtMemoryInfo* mi = ins[i].buf.from_shared_allocator ? (const OrtMemoryInfo*)htp_info : (const OrtMemoryInfo*)cpu_info;
        in_tensors.push_back(Ort::Value::CreateTensor(mi, ins[i].buf.data, ins[i].buf.size,
            ins[i].shape.data(), ins[i].shape.size(), ins[i].et));
        bind.BindInput(ins[i].name.c_str(), in_tensors.back());
    }
    auto onp = session->GetOutputNameAllocated(0, alloc);
    std::string out_name = onp.get();
    auto oti = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    std::vector<int64_t> oshp = oti.GetShape();
    size_t oels = 1;
    for (auto d : oshp) oels *= (size_t)d;
    ManagedBuffer out_buf;
    out_buf.Allocate(oels * GetTypeSize(oti.GetElementType()), htp_alloc.get());
    Ort::Value out_tensor = Ort::Value::CreateTensor(
        out_buf.from_shared_allocator ? (const OrtMemoryInfo*)htp_info : (const OrtMemoryInfo*)cpu_info,
        out_buf.data, out_buf.size, oshp.data(), oshp.size(), oti.GetElementType());
    bind.BindOutput(out_name.c_str(), out_tensor);
    if (!ins[0].buf.from_shared_allocator || !ins[1].buf.from_shared_allocator ||
        !out_buf.from_shared_allocator) {
        std::cerr << "[NPUW] REQUIRE HTP pool buffers; aborting\n";
        return 1;
    }

    Ort::RunOptions ro;
    ro.SetRunLogSeverityLevel(4);
    ro.SetRunLogVerbosityLevel(0);
    ro.AddConfigEntry("qnn.rpc_control_latency", "100");
    for (int i = 0; i < 15; i++) session->Run(ro, bind); // warmup on zeros
    std::cerr << "[NPUW] warmup done\n";

    void* h = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_NOLOAD);
    auto to_fd = h ? (int (*)(void*))dlsym(h, "rpcmem_to_fd") : nullptr;
    if (!to_fd) {
        std::cerr << "[NPUW] rpcmem_to_fd missing\n";
        return 1;
    }
    int fds[3] = {to_fd(ins[0].buf.data), to_fd(ins[1].buf.data), to_fd(out_buf.data)};
    if (fds[0] < 0 || fds[1] < 0 || fds[2] < 0) {
        std::cerr << "[NPUW] fd export failed\n";
        return 1;
    }
    // input shape [1,3,H,W], output shape [1,3,oH,oW]
    long long iW = (long long)ins[0].shape[3], iH = (long long)ins[0].shape[2];
    long long oW = (long long)oshp[3], oH = (long long)oshp[2];
    char line[256];
    snprintf(line, sizeof line, "NPU_FDS %d %d %d %zu %zu %lld %lld %lld %lld\n",
             fds[0], fds[1], fds[2], ins[0].buf.size, out_buf.size, iW, iH, oW, oH);
    if (!send_fds(sock, line, fds, 3)) {
        std::cerr << "[NPUW] fd send failed\n";
        return 1;
    }
    const char* ready = "NPU_READY\n";
    if (send(sock, ready, strlen(ready), 0) != (ssize_t)strlen(ready)) return 1;
    std::cerr << "[NPUW] serving on socket " << sock << "\n";

    // serve: 'R' = run, 'V' = run + FNV, EOF = quit
    char c;
    while (recv(sock, &c, 1, 0) == 1) {
        if (c != 'R' && c != 'V') continue;
        auto t0 = std::chrono::high_resolution_clock::now();
        session->Run(ro, bind);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        char out[128];
        int n;
        if (c == 'V')
            n = snprintf(out, sizeof out, "D %.4f 0x%llx\n", ms,
                         (unsigned long long)fnv_of(out_buf.data, out_buf.size));
        else
            n = snprintf(out, sizeof out, "D %.4f\n", ms);
        size_t off = 0;
        while (off < (size_t)n) {
            ssize_t w = send(sock, out + off, n - off, 0);
            if (w <= 0) return 0;
            off += (size_t)w;
        }
    }
    return 0;
}
