// NPU backend: glibc/Turnip side of the Vulkan->NPU->Vulkan zero-copy loop.
//
// Protocol mirrors vk_npu_loop.cpp feeder + npu_interp worker:
//   worker --"NPU_FDS ..." + 3 dma fds (SCM_RIGHTS)--> layer
//   worker --"NPU_READY"--> layer
//   layer --'R'/'V'--> worker --"D <ms> [0x<fnv>]"--> layer
// Sync per handoff: GPU fence wait per submit + DMA_BUF_SYNC ioctl +
// wall-clock splits + FNV cross-check. No lsfg semaphore reuse for timing.

#include "npu/npu_backend.hpp"
#include "npu/convert_spv.hpp"
#include "npu/pack_spv.hpp"
#include "layer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dlfcn.h>
#include <signal.h>

#ifndef DMA_BUF_BASE
#define DMA_BUF_BASE 'b'
struct dma_buf_sync { uint64_t flags; };
#define DMA_BUF_SYNC_READ (1 << 0)
#define DMA_BUF_SYNC_WRITE (2 << 0)
#define DMA_BUF_SYNC_RW (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START (0 << 2)
#define DMA_BUF_SYNC_END (1 << 2)
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#endif

namespace Npu {

static void nlog(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    fprintf(stderr, "[LSFG-NPU t=%llu.%03llu] %s\n",
            (unsigned long long)ts.tv_sec,
            (unsigned long long)(ts.tv_nsec / 1000000ULL), buf);
}

// Per-frame stage logs are verbose (4 lines/frame); gated by the TOML
// npu_verbose knob (setVerbose). Verify + stats + errors always print.
#define NVLOG(...)                      \
    do {                                \
        if (verbose_)                   \
            nlog(__VA_ARGS__);          \
    } while (0)

double Backend::nowMs() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

double Backend::dmaSync(int fd, int start) {
    if (noSync_)
        return 0.0;
    struct dma_buf_sync s;
    s.flags = (uint64_t)(DMA_BUF_SYNC_RW | (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    static bool warned = false;
    if (r && !warned) {
        nlog("dma sync ioctl unsupported, continuing without");
        warned = true;
    }
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

static bool sockReadline(int sock, std::string& out) {
    out.clear();
    char c;
    while (out.size() < 4096) {
        ssize_t r = recv(sock, &c, 1, 0);
        if (r <= 0)
            return false;
        if (c == '\n')
            return true;
        out += c;
    }
    return false;
}

static void npuAnchor() {}
static std::string libDir() {
    Dl_info di;
    // any symbol in this lib works as the dladdr anchor
    if (dladdr((void*)npuAnchor, &di) && di.dli_fname) {
        std::string p = di.dli_fname;
        auto s = p.find_last_of('/');
        if (s != std::string::npos)
            return p.substr(0, s);
    }
    return ".";
}

Backend::~Backend() {
    release();
}

Backend& Backend::operator=(Backend&& o) noexcept {
    if (this == &o)
        return *this;
    release();
    // transfer ownership member-wise; null the source so its dtor no-ops
#define MV(x) x = o.x
    MV(p_);
    MV(ready_);
    MV(dead_);
    MV(dev_);
    MV(phys_);
    MV(queue_);
    MV(qf_);
    MV(srcExt_);
    MV(srcFmt_);
    MV(pid_);
    MV(sock_);
    inFds_[0] = o.inFds_[0];
    inFds_[1] = o.inFds_[1];
    MV(outFd_);
    MV(inSz_);
    MV(outSz_);
    MV(inW_);
    MV(inH_);
    MV(outW_);
    MV(outH_);
    MV(sampler_);
    MV(dsLayout_);
    MV(pipeLayout_);
    MV(pipe_);
    MV(pool_);
    ds_[0] = o.ds_[0];
    ds_[1] = o.ds_[1];
    MV(packDsLayout_);
    MV(packPipeLayout_);
    MV(packPipe_);
    MV(packPool_);
    MV(packDs_);
    MV(packSwapRB_);
    in_[0] = o.in_[0];
    in_[1] = o.in_[1];
    MV(out_);
    MV(stg_);
    MV(stgMem_);
    MV(stgPtr_);
    MV(stgCoherent_);
    MV(cmdPool_);
    MV(cb_);
    MV(cb2_);
    MV(cb3_);
    MV(fence_);
    MV(views_);
    MV(packViews_);
    MV(outLayouts_);
    MV(outFmt_);
    MV(packTarget_);
    o.packTarget_ = VK_NULL_HANDLE;
    MV(workerSh_);
    MV(perfMode_);
    MV(verbose_);
    MV(dryRun_);
    MV(noSync_);
    MV(verifyEvery_);
    MV(frames_);
    MV(runs_);
    MV(lastEnter_);
    MV(tFill_);
    MV(tNpu_);
    MV(tPresent_);
    MV(syncMs_);
    MV(wall0_);
    MV(wConvMx_);
    MV(wNpuMx_);
    MV(wPresMx_);
    lastFnv_.store(o.lastFnv_.load());
    o.lastFnv_.store(0);
    verifyThread_ = std::move(o.verifyThread_);
#undef MV
    o.sock_ = -1;
    o.pid_ = -1;
    o.inFds_[0] = o.inFds_[1] = -1;
    o.outFd_ = -1;
    o.in_[0] = o.in_[1] = Buf{};
    o.out_ = Buf{};
    o.stg_ = VK_NULL_HANDLE;
    o.stgMem_ = VK_NULL_HANDLE;
    o.stgPtr_ = nullptr;
    o.sampler_ = VK_NULL_HANDLE;
    o.dsLayout_ = VK_NULL_HANDLE;
    o.pipeLayout_ = VK_NULL_HANDLE;
    o.pipe_ = VK_NULL_HANDLE;
    o.pool_ = VK_NULL_HANDLE;
    o.ds_[0] = o.ds_[1] = VK_NULL_HANDLE;
    o.packDsLayout_ = VK_NULL_HANDLE;
    o.packPipeLayout_ = VK_NULL_HANDLE;
    o.packPipe_ = VK_NULL_HANDLE;
    o.packPool_ = VK_NULL_HANDLE;
    o.packDs_ = VK_NULL_HANDLE;
    o.cmdPool_ = VK_NULL_HANDLE;
    o.cb_ = o.cb2_ = o.cb3_ = VK_NULL_HANDLE;
    o.fence_ = VK_NULL_HANDLE;
    o.views_.clear();
    o.packViews_.clear();
    o.outLayouts_.clear();
    o.dev_ = VK_NULL_HANDLE;
    o.ready_ = false;
    o.dead_ = true;
    o.frames_ = o.runs_ = 0;
    return *this;
}

void Backend::release() {
    report(true);
    // Reap the async verify thread before freeing the staging buffer it
    // may be reading (verifies are 240+ frames apart: never blocks in
    // practice; bounded by one FNV pass on teardown).
    if (verifyThread_.joinable())
        verifyThread_.join();
    auto freeBuf = [&](Buf& b) {
        if (b.b && p_.DestroyBuffer)
            p_.DestroyBuffer(dev_, b.b, nullptr);
        if (b.m && p_.FreeMem)
            p_.FreeMem(dev_, b.m, nullptr);
        b.b = VK_NULL_HANDLE;
        b.m = VK_NULL_HANDLE;
    };
    freeBuf(in_[0]);
    freeBuf(in_[1]);
    freeBuf(out_);
    if (stg_ && p_.DestroyBuffer)
        p_.DestroyBuffer(dev_, stg_, nullptr);
    if (stgMem_ && p_.FreeMem)
        p_.FreeMem(dev_, stgMem_, nullptr);
    for (auto& kv : views_)
        if (p_.DestroyImgView)
            p_.DestroyImgView(dev_, kv.second, nullptr);
    views_.clear();
    for (auto& kv : packViews_)
        if (p_.DestroyImgView)
            p_.DestroyImgView(dev_, kv.second, nullptr);
    packViews_.clear();
    if (pipe_ && p_.DestroyPipe)
        p_.DestroyPipe(dev_, pipe_, nullptr);
    if (pipeLayout_ && p_.DestroyPipeLayout)
        p_.DestroyPipeLayout(dev_, pipeLayout_, nullptr);
    if (dsLayout_ && p_.DestroyDsLayout)
        p_.DestroyDsLayout(dev_, dsLayout_, nullptr);
    if (pool_ && p_.DestroyDescPool)
        p_.DestroyDescPool(dev_, pool_, nullptr);
    if (packPipe_ && p_.DestroyPipe)
        p_.DestroyPipe(dev_, packPipe_, nullptr);
    if (packPipeLayout_ && p_.DestroyPipeLayout)
        p_.DestroyPipeLayout(dev_, packPipeLayout_, nullptr);
    if (packDsLayout_ && p_.DestroyDsLayout)
        p_.DestroyDsLayout(dev_, packDsLayout_, nullptr);
    if (packPool_ && p_.DestroyDescPool)
        p_.DestroyDescPool(dev_, packPool_, nullptr);
    if (sampler_ && p_.DestroySampler)
        p_.DestroySampler(dev_, sampler_, nullptr);
    if (cmdPool_ && p_.DestroyPool)
        p_.DestroyPool(dev_, cmdPool_, nullptr);
    if (fence_ && p_.DestroyFence)
        p_.DestroyFence(dev_, fence_, nullptr);
    if (sock_ >= 0)
        close(sock_);
}

bool Backend::spawnWorker(const std::string& modelPath, const std::string& workerBin) {
    std::string bin = workerBin;
    std::string model = modelPath;
    // No environment fallback: all paths arrive via TOML (Android has no
    // usable process env). Filesystem fallbacks only.
    if (bin.empty()) {
        std::string sib = libDir() + "/npu_interp";
        struct stat st;
        if (!stat(sib.c_str(), &st))
            bin = sib;
        else if (!stat("./npu_interp", &st))
            bin = "./npu_interp";
        else
            bin = sib; // report missing below
    }
    if (model.empty())
        model = libDir() + "/models/rife_model/rife46_400x300_ft_slim_fp16.onnx";
    struct stat st;
    if (stat(bin.c_str(), &st)) {
        nlog("worker binary missing: %s (set npu_bin in conf.toml)", bin.c_str());
        return false;
    }
    if (stat(model.c_str(), &st)) {
        nlog("model missing: %s (set npu_model in conf.toml)", model.c_str());
        return false;
    }
    if (!workerSh_.empty() && stat(workerSh_.c_str(), &st)) {
        nlog("worker script missing: %s (set worker_sh in conf.toml)", workerSh_.c_str());
        return false;
    }
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) {
        nlog("socketpair failed");
        return false;
    }
    signal(SIGPIPE, SIG_IGN);
    pid_t pid = fork();
    if (pid < 0) {
        nlog("fork failed");
        return false;
    }
    if (pid == 0) {
        close(sv[0]);
        char sf[16];
        snprintf(sf, sizeof sf, "%d", sv[1]);
#ifdef __ANDROID__
        // Bionic: already native, no linker trick. With a launch script
        // (TOML worker_sh) the script sets up env (LD_LIBRARY_PATH,
        // ADSP_LIBRARY_PATH) and execs the worker; otherwise exec the
        // worker binary directly. Contract: sh <script> <model> <bin> <sockfd>
        if (!workerSh_.empty()) {
            execl("/system/bin/sh", "sh", workerSh_.c_str(), model.c_str(),
                  bin.c_str(), sf, perfMode_.c_str(), (char*)0);
        } else {
            execl(bin.c_str(), bin.c_str(), model.c_str(), sf,
                  perfMode_.c_str(), (char*)0);
        }
        _exit(127);
#else
        std::string wd = (slash == std::string::npos) ? "." : bin.substr(0, slash);
        // npu_interp lives next to the onnx workspace root in our setup;
        // its sibling lib/ dirs hold the bionic + QNN stack (mirrors run.sh).
        std::string ws = wd;
        // if bin is ./npu_interp, wd is "." already the workspace
        std::string ld = ws + "/lib/aarch64-android:" + ws + "/lib/jni/arm64-v8a:"
                         "/proc/1/root/apex/com.android.runtime/lib64/bionic:"
                         "/proc/1/root/apex/com.android.runtime/lib64:"
                         "/proc/1/root/system/lib64";
        setenv("LD_LIBRARY_PATH", ld.c_str(), 1);
        setenv("ADSP_LIBRARY_PATH", (ws + "/lib/hexagon-v73/unsigned").c_str(), 1);
        char sf[16];
        snprintf(sf, sizeof sf, "%d", sv[1]);
        execl("/proc/1/root/apex/com.android.runtime/bin/linker64", "linker64",
              bin.c_str(), model.c_str(), sf, perfMode_.c_str(), (char*)0);
        _exit(127);
#endif
    }
    close(sv[1]);
    pid_ = pid;
    sock_ = sv[0];
    std::string acc;
    std::vector<int> fds;
    char buf[1024];
    while (acc.find('\n') == std::string::npos) {
        struct msghdr m = {};
        struct iovec io = {buf, sizeof buf};
        m.msg_iov = &io;
        m.msg_iovlen = 1;
        char ctl[128];
        m.msg_control = ctl;
        m.msg_controllen = sizeof ctl;
        ssize_t r = recvmsg(sock_, &m, 0);
        if (r <= 0) {
            nlog("FDS recv failed");
            return false;
        }
        acc.append(buf, (size_t)r);
        for (struct cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
                size_t n = ((size_t)c->cmsg_len >= CMSG_LEN(0))
                               ? (c->cmsg_len - CMSG_LEN(0)) / sizeof(int)
                               : 0;
                int* pp = (int*)CMSG_DATA(c);
                for (size_t i = 0; i < n; i++)
                    fds.push_back(pp[i]);
            }
    }
    int a, b, c, inW, inH, outW, outH;
    size_t insz, outsz;
    if (sscanf(acc.c_str(), "NPU_FDS %d %d %d %zu %zu %d %d %d %d", &a, &b, &c,
               &insz, &outsz, &inW, &inH, &outW, &outH) != 9 ||
        fds.size() < 3) {
        nlog("bad FDS line: %s", acc.c_str());
        return false;
    }
    std::string line;
    if (!sockReadline(sock_, line) || line.find("NPU_READY") == std::string::npos) {
        nlog("no READY from worker");
        return false;
    }
    inFds_[0] = fds[0];
    inFds_[1] = fds[1];
    outFd_ = fds[2];
    inSz_ = insz;
    outSz_ = outsz;
    inW_ = inW;
    inH_ = inH;
    outW_ = outW;
    outH_ = outH;
    nlog("worker pid=%d fds=%d,%d,%d in=%dx%d out=%dx%d model=%s", pid, fds[0],
         fds[1], fds[2], inW, inH, outW, outH, model.c_str());
    return true;
}

bool Backend::importOne(int fd, size_t sz, VkBuffer* ob, VkDeviceMemory* om) {
    VkExternalMemoryBufferCreateInfo ebci = {};
    ebci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ebci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ebci;
    bci.size = sz;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (p_.CreateBuffer(dev_, &bci, nullptr, ob) != VK_SUCCESS)
        return false;
    VkMemoryRequirements mr;
    p_.GetBufMemReq(dev_, *ob, &mr);
    VkMemoryDedicatedAllocateInfo dai = {};
    dai.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dai.buffer = *ob;
    for (uint32_t t = 0; t < 32; t++) {
        if (!(mr.memoryTypeBits & (1u << t)))
            continue;
        int tryfd = dup(fd);
        if (tryfd < 0)
            continue;
        VkImportMemoryFdInfoKHR imi = {};
        imi.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        imi.pNext = &dai;
        imi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        imi.fd = tryfd;
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &imi;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = t;
        if (p_.AllocMem(dev_, &mai, nullptr, om) != VK_SUCCESS)
            continue;
        if (p_.BindBufMem(dev_, *ob, *om, 0) != VK_SUCCESS) {
            p_.FreeMem(dev_, *om, nullptr);
            *om = VK_NULL_HANDLE;
            return false; // fd consumed; bail like vk_npu_loop feeder
        }
        return true;
    }
    p_.DestroyBuffer(dev_, *ob, nullptr);
    *ob = VK_NULL_HANDLE;
    return false;
}

bool Backend::setupGpu(VkExtent2D swapExtent, VkFormat swapFormat) {
    srcExt_ = swapExtent;
    srcFmt_ = swapFormat;
    auto fail = [&](const char* m) {
        nlog("setup: %s", m);
        return false;
    };
    auto gp = [&](const char* name) {
        return Layer::ovkGetDeviceProcAddr(dev_, name);
    };
#define RESOLVE(field, name)                                  \
    p_.field = (decltype(p_.field))gp(name);                  \
    if (!p_.field)                                            \
        return fail("missing " name)
    RESOLVE(CreateBuffer, "vkCreateBuffer");
    RESOLVE(DestroyBuffer, "vkDestroyBuffer");
    RESOLVE(GetBufMemReq, "vkGetBufferMemoryRequirements");
    RESOLVE(AllocMem, "vkAllocateMemory");
    RESOLVE(FreeMem, "vkFreeMemory");
    RESOLVE(BindBufMem, "vkBindBufferMemory");
    RESOLVE(MapMem, "vkMapMemory");
    RESOLVE(UnmapMem, "vkUnmapMemory");
    RESOLVE(FlushMapped, "vkFlushMappedMemoryRanges");
    RESOLVE(InvalidateMapped, "vkInvalidateMappedMemoryRanges");
    RESOLVE(CreatePool, "vkCreateCommandPool");
    RESOLVE(DestroyPool, "vkDestroyCommandPool");
    RESOLVE(AllocCb, "vkAllocateCommandBuffers");
    RESOLVE(BeginCb, "vkBeginCommandBuffer");
    RESOLVE(EndCb, "vkEndCommandBuffer");
    RESOLVE(Barrier, "vkCmdPipelineBarrier");
    RESOLVE(CopyBuf, "vkCmdCopyBuffer");
    RESOLVE(Submit, "vkQueueSubmit");
    RESOLVE(CreateFence, "vkCreateFence");
    RESOLVE(DestroyFence, "vkDestroyFence");
    RESOLVE(WaitFence, "vkWaitForFences");
    RESOLVE(ResetFence, "vkResetFences");
    RESOLVE(CreateShader, "vkCreateShaderModule");
    RESOLVE(DestroyShader, "vkDestroyShaderModule");
    RESOLVE(CreateSampler, "vkCreateSampler");
    RESOLVE(DestroySampler, "vkDestroySampler");
    RESOLVE(CreateDsLayout, "vkCreateDescriptorSetLayout");
    RESOLVE(DestroyDsLayout, "vkDestroyDescriptorSetLayout");
    RESOLVE(CreatePipeLayout, "vkCreatePipelineLayout");
    RESOLVE(DestroyPipeLayout, "vkDestroyPipelineLayout");
    RESOLVE(CreateCompute, "vkCreateComputePipelines");
    RESOLVE(DestroyPipe, "vkDestroyPipeline");
    RESOLVE(CreateDescPool, "vkCreateDescriptorPool");
    RESOLVE(DestroyDescPool, "vkDestroyDescriptorPool");
    RESOLVE(AllocDescSets, "vkAllocateDescriptorSets");
    RESOLVE(UpdateDesc, "vkUpdateDescriptorSets");
    RESOLVE(CreateImgView, "vkCreateImageView");
    RESOLVE(DestroyImgView, "vkDestroyImageView");
    RESOLVE(BindPipe, "vkCmdBindPipeline");
    RESOLVE(BindDescSets, "vkCmdBindDescriptorSets");
    RESOLVE(Dispatch, "vkCmdDispatch");
    RESOLVE(PushConst, "vkCmdPushConstants");
#undef RESOLVE
    p_.ResetCmd =
        (PFN_vkResetCommandBuffer)gp("vkResetCommandBuffer"); // optional
    // sampler
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0;
    if (p_.CreateSampler(dev_, &sci, nullptr, &sampler_) != VK_SUCCESS)
        return fail("sampler");
    // convert layout: binding0 sampled img, binding1 storage buf; pc 16B
    VkDescriptorSetLayoutBinding bl[2] = {};
    bl[0].binding = 0;
    bl[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bl[0].descriptorCount = 1;
    bl[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bl[1].binding = 1;
    bl[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bl[1].descriptorCount = 1;
    bl[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci = {};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2;
    lci.pBindings = bl;
    if (p_.CreateDsLayout(dev_, &lci, nullptr, &dsLayout_) != VK_SUCCESS)
        return fail("dslayout");
    VkPushConstantRange pc = {};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 16;
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pc;
    if (p_.CreatePipeLayout(dev_, &pli, nullptr, &pipeLayout_) != VK_SUCCESS)
        return fail("pipelayout");
    VkShaderModuleCreateInfo smi = {};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sizeof(kNpuConvertSpv);
    smi.pCode = kNpuConvertSpv;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (p_.CreateShader(dev_, &smi, nullptr, &mod) != VK_SUCCESS)
        return fail("shader");
    VkComputePipelineCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = pipeLayout_;
    bool ok = p_.CreateCompute(dev_, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe_) ==
              VK_SUCCESS;
    p_.DestroyShader(dev_, mod, nullptr);
    if (!ok)
        return fail("pipeline");
    VkDescriptorPoolSize psz[2] = {};
    psz[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    psz[0].descriptorCount = 2;
    psz[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz[1].descriptorCount = 2;
    VkDescriptorPoolCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 2;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = psz;
    if (p_.CreateDescPool(dev_, &pci, nullptr, &pool_) != VK_SUCCESS)
        return fail("pool");
    VkDescriptorSetAllocateInfo aai = {};
    aai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    aai.descriptorPool = pool_;
    aai.descriptorSetCount = 1;
    aai.pSetLayouts = &dsLayout_;
    for (int s = 0; s < 2; s++)
        if (p_.AllocDescSets(dev_, &aai, &ds_[s]) != VK_SUCCESS)
            return fail("sets");
    // import NPU dma_bufs (zero-copy: same physical pages as HTP pool)
    if (!importOne(inFds_[0], inSz_, &in_[0].b, &in_[0].m))
        return fail("import in0");
    if (!importOne(inFds_[1], inSz_, &in_[1].b, &in_[1].m))
        return fail("import in1");
    if (!importOne(outFd_, outSz_, &out_.b, &out_.m))
        return fail("import out");
    // staging for FNV verify readback
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = outSz_;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (p_.CreateBuffer(dev_, &bci, nullptr, &stg_) != VK_SUCCESS)
        return fail("staging");
    VkMemoryRequirements mr;
    p_.GetBufMemReq(dev_, stg_, &mr);
    VkPhysicalDeviceMemoryProperties memProps = {};
    Layer::ovkGetPhysicalDeviceMemoryProperties(phys_, &memProps);
    bool okm = false;
    uint32_t pickedType = 0xFFFFFFFFu;
    // Prefer a CPU-cached heap for staging: the verify FNV reads back the
    // whole frame on the present thread, and uncached reads stall it.
    for (int pass = 0; pass < 2 && !okm; pass++) {
        for (uint32_t t = 0; t < 32 && !okm; t++) {
            if (!(mr.memoryTypeBits & (1u << t)))
                continue;
            VkMemoryPropertyFlags fl = memProps.memoryTypes[t].propertyFlags;
            bool cached = (fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
            if ((pass == 0) != cached)
                continue;
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = t;
        VkDeviceMemory tmp = VK_NULL_HANDLE;
        if (p_.AllocMem(dev_, &mai, nullptr, &tmp) != VK_SUCCESS)
            continue;
        void* pp = nullptr;
        if (p_.MapMem(dev_, tmp, 0, outSz_, 0, &pp) != VK_SUCCESS) {
            p_.FreeMem(dev_, tmp, nullptr);
            continue;
        }
        p_.UnmapMem(dev_, tmp);
        p_.FreeMem(dev_, tmp, nullptr);
        if (p_.AllocMem(dev_, &mai, nullptr, &stgMem_) != VK_SUCCESS)
            continue;
        if (p_.MapMem(dev_, stgMem_, 0, outSz_, 0, &stgPtr_) != VK_SUCCESS) {
            p_.FreeMem(dev_, stgMem_, nullptr);
            stgMem_ = VK_NULL_HANDLE;
            continue;
        }
        pickedType = t;
        okm = true;
        } // end per-type loop
    } // end cached-first / uncached-fallback passes
    if (!okm)
        return fail("staging map");
    stgCoherent_ =
        (memProps.memoryTypes[pickedType].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    nlog("staging memtype %u flags=0x%x%s", pickedType,
         memProps.memoryTypes[pickedType].propertyFlags,
         stgCoherent_ ? "" : " (non-coherent: invalidate before CPU read)");
    if (p_.BindBufMem(dev_, stg_, stgMem_, 0) != VK_SUCCESS)
        return fail("staging bind");
    VkCommandPoolCreateInfo cpi2 = {};
    cpi2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi2.queueFamilyIndex = qf_;
    if (p_.CreatePool(dev_, &cpi2, nullptr, &cmdPool_) != VK_SUCCESS)
        return fail("cmdpool");
    VkCommandBufferAllocateInfo cai = {};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = cmdPool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (p_.AllocCb(dev_, &cai, &cb_) != VK_SUCCESS)
        return fail("cb");
    if (p_.AllocCb(dev_, &cai, &cb2_) != VK_SUCCESS)
        return fail("cb2");
    if (p_.AllocCb(dev_, &cai, &cb3_) != VK_SUCCESS)
        return fail("cb3");
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (p_.CreateFence(dev_, &fci, nullptr, &fence_) != VK_SUCCESS)
        return fail("fence");
    // pack pipeline: binding0 storage buf (NPU out), binding1 storage img
    packSwapRB_ = (swapFormat == VK_FORMAT_B8G8R8A8_UNORM ||
                   swapFormat == VK_FORMAT_B8G8R8A8_SRGB)
                      ? 1u
                      : 0u;
    VkDescriptorSetLayoutBinding pbl[2] = {};
    pbl[0].binding = 0;
    pbl[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pbl[0].descriptorCount = 1;
    pbl[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pbl[1].binding = 1;
    pbl[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pbl[1].descriptorCount = 1;
    pbl[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    plci.bindingCount = 2;
    plci.pBindings = pbl;
    if (p_.CreateDsLayout(dev_, &plci, nullptr, &packDsLayout_) != VK_SUCCESS)
        return fail("packdslayout");
    VkPushConstantRange ppc = {};
    ppc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ppc.offset = 0;
    ppc.size = 20;
    VkPipelineLayoutCreateInfo ppli = {};
    ppli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ppli.setLayoutCount = 1;
    ppli.pSetLayouts = &packDsLayout_;
    ppli.pushConstantRangeCount = 1;
    ppli.pPushConstantRanges = &ppc;
    if (p_.CreatePipeLayout(dev_, &ppli, nullptr, &packPipeLayout_) != VK_SUCCESS)
        return fail("packpipelayout");
    VkShaderModuleCreateInfo psmi = {};
    psmi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    psmi.codeSize = sizeof(kNpuPackSpv);
    psmi.pCode = kNpuPackSpv;
    VkShaderModule pmod = VK_NULL_HANDLE;
    if (p_.CreateShader(dev_, &psmi, nullptr, &pmod) != VK_SUCCESS)
        return fail("packshader");
    VkComputePipelineCreateInfo pcpi = {};
    pcpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pcpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pcpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pcpi.stage.module = pmod;
    pcpi.stage.pName = "main";
    pcpi.layout = packPipeLayout_;
    bool pok = p_.CreateCompute(dev_, VK_NULL_HANDLE, 1, &pcpi, nullptr,
                                &packPipe_) == VK_SUCCESS;
    p_.DestroyShader(dev_, pmod, nullptr);
    if (!pok)
        return fail("packpipeline");
    VkDescriptorPoolSize ppsz[2] = {};
    ppsz[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ppsz[0].descriptorCount = 1;
    ppsz[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ppsz[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo ppci = {};
    ppci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ppci.maxSets = 1;
    ppci.poolSizeCount = 2;
    ppci.pPoolSizes = ppsz;
    if (p_.CreateDescPool(dev_, &ppci, nullptr, &packPool_) != VK_SUCCESS)
        return fail("packpool");
    VkDescriptorSetAllocateInfo paai = {};
    paai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    paai.descriptorPool = packPool_;
    paai.descriptorSetCount = 1;
    paai.pSetLayouts = &packDsLayout_;
    if (p_.AllocDescSets(dev_, &paai, &packDs_) != VK_SUCCESS)
        return fail("packds");
    nlog("convert ready %ux%u -> %dx%d (pack ready, swapRB=%u)", swapExtent.width,
         swapExtent.height, inW_, inH_, packSwapRB_);
    return true;
}

bool Backend::init(VkDevice device, VkPhysicalDevice physDevice, VkQueue queue,
                   uint32_t queueFamily, VkExtent2D swapExtent,
                   VkFormat swapFormat, const std::string& modelPath,
                   const std::string& workerBin,
                   const std::string& workerSh) {
    dev_ = device;
    phys_ = physDevice;
    queue_ = queue;
    qf_ = queueFamily;
    workerSh_ = workerSh;
    if (!spawnWorker(modelPath, workerBin))
        return false;
    if (!setupGpu(swapExtent, swapFormat)) {
        dead_ = true;
        return false;
    }
    ready_ = true;
    wall0_ = nowMs();
    return true;
}

VkImageView Backend::viewFor(VkImage img) {
    auto it = views_.find(img);
    if (it != views_.end())
        return it->second;
    bool bgr = (srcFmt_ == VK_FORMAT_B8G8R8A8_UNORM ||
                srcFmt_ == VK_FORMAT_B8G8R8A8_SRGB);
    VkComponentMapping comp = {VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY};
    if (bgr) {
        comp.r = VK_COMPONENT_SWIZZLE_B;
        comp.g = VK_COMPONENT_SWIZZLE_G;
        comp.b = VK_COMPONENT_SWIZZLE_R;
    }
    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = srcFmt_;
    vci.components = comp;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (p_.CreateImgView(dev_, &vci, nullptr, &v) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    views_[img] = v;
    return v;
}

VkImageView Backend::packViewFor(VkImage img, VkFormat fmt) {
    auto it = packViews_.find(img);
    if (it != packViews_.end())
        return it->second;
    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (p_.CreateImgView(dev_, &vci, nullptr, &v) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    packViews_[img] = v;
    outLayouts_[img] = VK_IMAGE_LAYOUT_UNDEFINED;
    return v;
}

bool Backend::runWorker(char req, double& ms, uint64_t& fnv, bool& hasFnv) {
    if (send(sock_, &req, 1, 0) != 1)
        return false;
    std::string line;
    if (!sockReadline(sock_, line))
        return false;
    unsigned long long f = 0;
    hasFnv = false;
    if (sscanf(line.c_str(), "D %lf 0x%llx", &ms, &f) == 2) {
        fnv = f;
        hasFnv = true;
        return true;
    }
    return sscanf(line.c_str(), "D %lf", &ms) == 1;
}

bool Backend::generate(VkImage curSwapImage, VkImage outImage,
                       uint64_t frameIdx) {
    if (!ready_ || dead_)
        return false;
    if (dryRun_) // passthrough: prove skeleton/delivery (TOML npu_dryrun)
        return false;
    {
        double t = nowMs();
        double gap = lastEnter_ > 0 ? t - lastEnter_ : 0;
        NVLOG("frame %llu enter gap=%.1fms", (unsigned long long)frames_, gap);
        lastEnter_ = t;
    }
    if (frames_ == 0)
        wall0_ = nowMs();
    uint64_t k = frames_++;
    int slot = (int)(k % 2);
    VkImageView view = viewFor(curSwapImage);
    if (view == VK_NULL_HANDLE)
        return false;
    // descriptors: sampled swapchain img -> NPU in[slot]
    VkDescriptorImageInfo ii = {};
    ii.sampler = sampler_;
    ii.imageView = view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo bi = {};
    bi.buffer = in_[slot].b;
    bi.offset = 0;
    bi.range = inSz_;
    VkWriteDescriptorSet wds[2] = {};
    wds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds[0].dstSet = ds_[slot];
    wds[0].dstBinding = 0;
    wds[0].descriptorCount = 1;
    wds[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wds[0].pImageInfo = &ii;
    wds[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds[1].dstSet = ds_[slot];
    wds[1].dstBinding = 1;
    wds[1].descriptorCount = 1;
    wds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds[1].pBufferInfo = &bi;
    p_.UpdateDesc(dev_, 2, wds, 0, nullptr);

    // Verify cadence: the FNV cross-check plus a staging copy. Kept rare
    // (async FNV thread, TOML npu_verify_every, 0 = off after proof) so it
    // can't knock FIFO acquire phase off a vblank beat.
    bool verifyThis =
        verifyEvery_ > 0 && ((runs_ + 1) % (uint64_t)verifyEvery_ == 0);
    // ---- convert submit (GPU fence wait: proven protocol) ----
    double t0 = nowMs();
    VkCommandBufferBeginInfo bbi = {};
    bbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (p_.ResetCmd)
        p_.ResetCmd(cb_, 0);
    if (p_.BeginCb(cb_, &bbi) != VK_SUCCESS)
        return false;
    VkImageMemoryBarrier b1 = {}, b2 = {};
    b1.sType = b2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b1.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b1.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b1.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b1.srcQueueFamilyIndex = b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.image = curSwapImage;
    b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    p_.Barrier(cb_, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
               1, &b1);
    p_.BindPipe(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
    p_.BindDescSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1,
                    &ds_[slot], 0, nullptr);
    uint32_t pc[4] = {(uint32_t)inW_, (uint32_t)inH_, srcExt_.width,
                      srcExt_.height};
    p_.PushConst(cb_, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    p_.Dispatch(cb_, (uint32_t)((inW_ + 15) / 16),
                (uint32_t)((inH_ + 15) / 16), 1);
    b2 = b1;
    b2.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b2.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b2.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b2.dstAccessMask = 0;
    VkBufferMemoryBarrier bbIn = {};
    bbIn.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bbIn.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bbIn.dstAccessMask = 0;
    bbIn.srcQueueFamilyIndex = bbIn.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    bbIn.buffer = in_[slot].b;
    bbIn.offset = 0;
    bbIn.size = inSz_;
    p_.Barrier(cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &bbIn, 1,
               &b2);
    if (p_.EndCb(cb_) != VK_SUCCESS)
        return false;
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb_;
    if (p_.Submit(queue_, 1, &si, fence_) != VK_SUCCESS)
        return false;
    {
        double wt0 = nowMs();
        VkResult wr =
            p_.WaitFence(dev_, 1, &fence_, VK_TRUE, 5000000000ull);
        NVLOG("frame %llu convert fence=%d %.1fms", (unsigned long long)k, (int)wr,
             nowMs() - wt0);
        if (wr != VK_SUCCESS)
            return false;
    }
    p_.ResetFence(dev_, 1, &fence_);
    double dC = nowMs() - t0;
    tFill_ += dC;
    if (dC > wConvMx_)
        wConvMx_ = dC;
    // release GPU-written input so the DSP reads fresh bytes
    tFill_ += dmaSync(inFds_[slot], 0);

    if (k < 1)
        return false; // need a pair before first run
    // ---- NPU run (socket ping-pong, wall-clock) ----
    double npuMs = 0;
    uint64_t wfnv = 0;
    bool hasFnv = false;
    char req = verifyThis ? 'V' : 'R';
    if (!runWorker(req, npuMs, wfnv, hasFnv)) {
        dead_ = true;
        nlog("worker comm failed");
        return false;
    }
    NVLOG("frame %llu npu run %.2fms", (unsigned long long)k, npuMs);
    tNpu_ += npuMs;
    if (npuMs > wNpuMx_)
        wNpuMx_ = npuMs;
    // ---- present-back barrier (+ staging copy for verify) ----
    // ---- present-back + pack in ONE submit (was two fence roundtrips).
    // Barrier the NPU-written buffer (+ staging copy on verify frames),
    // then pack it straight into outImage. Single submit/wait keeps the
    // GPU fed and the present thread off the CPU.
    t0 = nowMs();
    tPresent_ += dmaSync(outFd_, 1);
    // Pack destination: model-size target when set (caller HW-blits it up),
    // else the swap-size out image. Dst extent follows the target.
    VkImage packDst =
        (packTarget_ != VK_NULL_HANDLE) ? packTarget_ : outImage;
    VkExtent2D packExt = (packTarget_ != VK_NULL_HANDLE)
                             ? VkExtent2D{(uint32_t)inW_, (uint32_t)inH_}
                             : srcExt_;
    VkImageView pv = packViewFor(packDst, outFmt_);
    if (pv == VK_NULL_HANDLE)
        return false;
    VkDescriptorBufferInfo pbi = {};
    pbi.buffer = out_.b;
    pbi.offset = 0;
    pbi.range = outSz_;
    VkDescriptorImageInfo pii = {};
    pii.imageView = pv;
    pii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet pwds[2] = {};
    pwds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    pwds[0].dstSet = packDs_;
    pwds[0].dstBinding = 0;
    pwds[0].descriptorCount = 1;
    pwds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pwds[0].pBufferInfo = &pbi;
    pwds[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    pwds[1].dstSet = packDs_;
    pwds[1].dstBinding = 1;
    pwds[1].descriptorCount = 1;
    pwds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pwds[1].pImageInfo = &pii;
    p_.UpdateDesc(dev_, 2, pwds, 0, nullptr);
    if (p_.ResetCmd)
        p_.ResetCmd(cb2_, 0);
    if (p_.BeginCb(cb2_, &bbi) != VK_SUCCESS)
        return false;
    VkBufferMemoryBarrier bmb = {};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    bmb.srcQueueFamilyIndex = bmb.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = out_.b;
    bmb.offset = 0;
    bmb.size = outSz_;
    p_.Barrier(cb2_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &bmb, 0,
               nullptr);
    if (verifyThis) {
        VkBufferCopy cp = {};
        cp.size = outSz_;
        p_.CopyBuf(cb2_, out_.b, stg_, 1, &cp);
    }
    VkImageLayout oldL = outLayouts_[packDst];
    VkImageMemoryBarrier pb = {};
    pb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pb.oldLayout = oldL;
    pb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pb.srcAccessMask =
        (oldL == VK_IMAGE_LAYOUT_UNDEFINED) ? (VkAccessFlags)0
                                            : (VkAccessFlags)VK_ACCESS_MEMORY_READ_BIT;
    pb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pb.srcQueueFamilyIndex = pb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pb.image = packDst;
    pb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    p_.Barrier(cb2_,
               oldL == VK_IMAGE_LAYOUT_UNDEFINED
                   ? (VkPipelineStageFlags)VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                   : (VkPipelineStageFlags)VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
               1, &pb);
    p_.BindPipe(cb2_, VK_PIPELINE_BIND_POINT_COMPUTE, packPipe_);
    p_.BindDescSets(cb2_, VK_PIPELINE_BIND_POINT_COMPUTE, packPipeLayout_, 0, 1,
                    &packDs_, 0, nullptr);
    // pack shader maps NPU out (inW/inH) to the pack target extent.
    uint32_t ppc[5] = {(uint32_t)inW_, (uint32_t)inH_, packExt.width,
                       packExt.height, packSwapRB_};
    p_.PushConst(cb2_, packPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20,
                 ppc);
    p_.Dispatch(cb2_, (packExt.width + 15) / 16, (packExt.height + 15) / 16, 1);
    pb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    pb.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    pb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pb.dstAccessMask = 0;
    p_.Barrier(cb2_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
               &pb);
    if (p_.EndCb(cb2_) != VK_SUCCESS)
        return false;
    si.pCommandBuffers = &cb2_;
    if (p_.Submit(queue_, 1, &si, fence_) != VK_SUCCESS)
        return false;
    {
        double wt0 = nowMs();
        VkResult wr =
            p_.WaitFence(dev_, 1, &fence_, VK_TRUE, 5000000000ull);
        NVLOG("frame %llu back+pack fence=%d %.1fms", (unsigned long long)k,
             (int)wr, nowMs() - wt0);
        if (wr != VK_SUCCESS)
            return false;
    }
    p_.ResetFence(dev_, 1, &fence_);
    double dP = nowMs() - t0;
    tPresent_ += dP;
    if (dP > wPresMx_)
        wPresMx_ = dP;
    tPresent_ += dmaSync(outFd_, 0);
    outLayouts_[packDst] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    runs_++;
    if (verifyThis) {
        // Reap the previous verify (240+ frames ago: never blocks), then
        // hash off-thread: the staging buffer is ours until the next
        // verify, so the present thread pays ~0ms here.
        if (verifyThread_.joinable())
            verifyThread_.join();
        if (!stgCoherent_) {
            // Non-coherent heap: pull the GPU-written copy into CPU view
            // before the FNV thread reads it (else stale cache -> MISMATCH).
            VkMappedMemoryRange range = {};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = stgMem_;
            range.offset = 0;
            range.size = outSz_;
            if (p_.InvalidateMapped(dev_, 1, &range) != VK_SUCCESS)
                nlog("staging invalidate failed");
        }
        uint64_t runNo = runs_;
        void* base = stgPtr_;
        size_t n = outSz_;
        verifyThread_ = std::thread(
            [this, runNo, base, n, wfnv, hasFnv] {
                uint64_t g = 1469598103934665603ULL;
                const uint8_t* bytes = static_cast<const uint8_t*>(base);
                for (size_t i = 0; i < n; i++) {
                    g ^= bytes[i];
                    g *= 1099511628211ULL;
                }
                lastFnv_.store(g);
                nlog("verify run %llu GPU=0x%llx NPU=0x%llx %s",
                     (unsigned long long)runNo, (unsigned long long)g,
                     (unsigned long long)wfnv,
                     (hasFnv && g == wfnv) ? "MATCH" : "MISMATCH");
            });
    }
    // (merged above: barrier + staging copy + pack share cb2_ in one submit;
    // cb3_ retained allocated but unused.)
    (void)frameIdx;
    if (runs_ % 60 == 0)
        report(false);
    return true;
}

void Backend::report(bool final) {
    if (!frames_ && !runs_)
        return;
    double wall = nowMs() - wall0_;
    double pure = (tFill_ + tNpu_ + tPresent_) / (runs_ ? runs_ : 1);
    nlog("%s runs=%llu frames=%llu wall=%.0fms convert=%.2f npu=%.2f present=%.2f "
         "sync=%.2f ms/frame PURE %.3fms = %.1f inf/sec winmax conv/npu/pres="
         "%.1f/%.1f/%.1f FNV=0x%llx",
         final ? "FINAL" : "stats", (unsigned long long)runs_,
         (unsigned long long)frames_, wall, runs_ ? tFill_ / runs_ : 0,
         runs_ ? tNpu_ / runs_ : 0, runs_ ? tPresent_ / runs_ : 0,
         runs_ ? syncMs_ / runs_ : 0, pure,
         pure > 0 ? 1000.0 / pure : 0, wConvMx_, wNpuMx_, wPresMx_,
         lastFnv_.load());
}

} // namespace Npu
