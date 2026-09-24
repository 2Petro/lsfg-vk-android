#pragma once
// NPU frame-generation backend for lsfg-vk (Android/bionic + glibc/Turnip).
//
// Second framegen option alongside the Lossless.dll shader path: GPU converts
// swapchain frames into NPU-shared dma_bufs, a bionic npu_interp worker runs
// RIFE on Hexagon HTP, GPU packs the interpolated frame back. Dataflow +
// sync mirror the proven feeder loop:
//
//   dma_buf fd inherit (SCM_RIGHTS) -> GPU fence wait per submit ->
//   socket ping-pong per iteration -> wall-clock per-iter timing,
//   FNV cross-check GPU vs NPU.
//
// All NPU-shared memory is DMA-BUF external memory (imported from the HTP
// pool via DMA_BUF_BIT_EXT); AHB stays app-facing only. Deliberately
// synchronous + lockstep (no private swapchain, no pacing thread):
// generate() runs on lsfg's present thread between its preCopy and postCopy,
// so lsfg's instance/device/swapchain handling, vsync pacing and
// acquire/present logic stay untouched. Only the framegen call is swapped.
//
// Configuration is TOML-driven (LsContext fills Options from Config);
// environment variables are never consulted on Android.

#include <vulkan/vulkan_core.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/types.h>

namespace Npu {

class Backend {
public:
    Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&& o) noexcept { *this = std::move(o); }
    Backend& operator=(Backend&& o) noexcept;
    ~Backend();

    // Spawn the bionic worker, import its dma fds, build convert/pack
    // pipelines. Returns false on any failure (caller falls back to
    // passthrough present; layer keeps running).
    // workerSh: Android launch script (sh <script> <model> <bin> <sockfd>)
    // that sets up env and execs the worker; empty = direct exec of
    // workerBin (bionic) or linker64 dispatch (glibc chroot).
    bool init(VkDevice device, VkPhysicalDevice physDevice, VkQueue queue,
              uint32_t queueFamily, VkExtent2D swapExtent, VkFormat swapFormat,
              const std::string& modelPath, const std::string& workerBin,
              const std::string& workerSh = "");

    // TOML-driven knobs (set from Config; no env on Android).
    void setVerbose(bool v) { verbose_ = v; }
    void setDryRun(bool v) { dryRun_ = v; } // passthrough, keeps skeleton
    void setNoSync(bool v) { noSync_ = v; } // skip DMA_BUF_SYNC ioctls
    void setVerifyEvery(long n) { verifyEvery_ = n; } // 0 = verify off

    // Convert curSwapImage into NPU input slot (frameIdx%2); when frameIdx>=1
    // run the worker on the pair and pack the interpolated frame into
    // outImage (one of lsfg's out_n images, left in PRESENT_SRC for the
    // existing postCopy). Returns true when outImage holds a fresh generated
    // frame, false on first frame / error (present real frame instead).
    bool generate(VkImage curSwapImage, VkImage outImage, uint64_t frameIdx);

    // Pack-at-native: direct pack output into a model-size target (caller
    // upscales with a fixed-function blit). Set after init (dims known).
    void setPackTarget(VkImage img) { packTarget_ = img; }
    int modelW() const { return inW_; }
    int modelH() const { return inH_; }

    bool alive() const { return ready_; }
    void report(bool final);

private:
    bool spawnWorker(const std::string& modelPath, const std::string& workerBin);
    bool setupGpu(VkExtent2D swapExtent, VkFormat swapFormat);
    bool importOne(int fd, size_t sz, VkBuffer* ob, VkDeviceMemory* om);
    VkImageView viewFor(VkImage img);
    VkImageView packViewFor(VkImage img, VkFormat fmt);
    bool runWorker(char req, double& ms, uint64_t& fnv, bool& hasFnv);
    void release();
    static double nowMs();
    double dmaSync(int fd, int start);

    // Resolved at init via Layer::ovkGetDeviceProcAddr (forwards to driver;
    // lsfg hooks none of these, so no layer.cpp changes needed).
    struct Procs {
        PFN_vkCreateBuffer CreateBuffer = nullptr;
        PFN_vkDestroyBuffer DestroyBuffer = nullptr;
        PFN_vkGetBufferMemoryRequirements GetBufMemReq = nullptr;
        PFN_vkAllocateMemory AllocMem = nullptr;
        PFN_vkFreeMemory FreeMem = nullptr;
        PFN_vkBindBufferMemory BindBufMem = nullptr;
        PFN_vkMapMemory MapMem = nullptr;
        PFN_vkUnmapMemory UnmapMem = nullptr;
        PFN_vkFlushMappedMemoryRanges FlushMapped = nullptr;
        PFN_vkInvalidateMappedMemoryRanges InvalidateMapped = nullptr;
        PFN_vkCreateCommandPool CreatePool = nullptr;
        PFN_vkDestroyCommandPool DestroyPool = nullptr;
        PFN_vkAllocateCommandBuffers AllocCb = nullptr;
        PFN_vkResetCommandBuffer ResetCmd = nullptr;
        PFN_vkBeginCommandBuffer BeginCb = nullptr;
        PFN_vkEndCommandBuffer EndCb = nullptr;
        PFN_vkCmdPipelineBarrier Barrier = nullptr;
        PFN_vkCmdCopyBuffer CopyBuf = nullptr;
        PFN_vkQueueSubmit Submit = nullptr;
        PFN_vkCreateFence CreateFence = nullptr;
        PFN_vkDestroyFence DestroyFence = nullptr;
        PFN_vkWaitForFences WaitFence = nullptr;
        PFN_vkResetFences ResetFence = nullptr;
        PFN_vkCreateShaderModule CreateShader = nullptr;
        PFN_vkDestroyShaderModule DestroyShader = nullptr;
        PFN_vkCreateSampler CreateSampler = nullptr;
        PFN_vkDestroySampler DestroySampler = nullptr;
        PFN_vkCreateDescriptorSetLayout CreateDsLayout = nullptr;
        PFN_vkDestroyDescriptorSetLayout DestroyDsLayout = nullptr;
        PFN_vkCreatePipelineLayout CreatePipeLayout = nullptr;
        PFN_vkDestroyPipelineLayout DestroyPipeLayout = nullptr;
        PFN_vkCreateComputePipelines CreateCompute = nullptr;
        PFN_vkDestroyPipeline DestroyPipe = nullptr;
        PFN_vkCreateDescriptorPool CreateDescPool = nullptr;
        PFN_vkDestroyDescriptorPool DestroyDescPool = nullptr;
        PFN_vkAllocateDescriptorSets AllocDescSets = nullptr;
        PFN_vkUpdateDescriptorSets UpdateDesc = nullptr;
        PFN_vkCreateImageView CreateImgView = nullptr;
        PFN_vkDestroyImageView DestroyImgView = nullptr;
        PFN_vkCmdBindPipeline BindPipe = nullptr;
        PFN_vkCmdBindDescriptorSets BindDescSets = nullptr;
        PFN_vkCmdDispatch Dispatch = nullptr;
        PFN_vkCmdPushConstants PushConst = nullptr;
    } p_;

    struct Buf {
        VkBuffer b = VK_NULL_HANDLE;
        VkDeviceMemory m = VK_NULL_HANDLE;
    };

    bool ready_ = false;
    bool dead_ = false;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qf_ = 0;
    VkExtent2D srcExt_{0, 0};
    VkFormat srcFmt_ = VK_FORMAT_UNDEFINED;

    pid_t pid_ = -1;
    int sock_ = -1;
    int inFds_[2] = {-1, -1};
    int outFd_ = -1;
    size_t inSz_ = 0, outSz_ = 0;
    int inW_ = 0, inH_ = 0, outW_ = 0, outH_ = 0;

    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSetLayout packDsLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout packPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline packPipe_ = VK_NULL_HANDLE;
    VkDescriptorPool packPool_ = VK_NULL_HANDLE;
    VkDescriptorSet packDs_ = VK_NULL_HANDLE;
    uint32_t packSwapRB_ = 0;

    Buf in_[2];
    Buf out_;
    VkBuffer stg_ = VK_NULL_HANDLE;
    VkDeviceMemory stgMem_ = VK_NULL_HANDLE;
    void* stgPtr_ = nullptr;
    // Staging heap may be non-coherent (device showed flags=0xb: cached but
    // not coherent) — CPU FNV must invalidate after the GPU copy.
    bool stgCoherent_ = true;

    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cb_ = VK_NULL_HANDLE;   // convert
    VkCommandBuffer cb2_ = VK_NULL_HANDLE;  // present-back barrier
    VkCommandBuffer cb3_ = VK_NULL_HANDLE;  // pack
    VkFence fence_ = VK_NULL_HANDLE;

    std::unordered_map<VkImage, VkImageView> views_;
    std::unordered_map<VkImage, VkImageView> packViews_;
    std::unordered_map<VkImage, VkImageLayout> outLayouts_;
    VkFormat outFmt_ = VK_FORMAT_R8G8B8A8_UNORM;
    VkImage packTarget_ = VK_NULL_HANDLE; // model-size pack dst (0 = outImage)

    // TOML-driven behavior (see setters; defaults = production).
    std::string workerSh_;
    bool verbose_ = false;
    bool dryRun_ = false;
    bool noSync_ = false;
    long verifyEvery_ = 240;

    // timing splits (proven-protocol measurement: NPU ms + present ms +
    // sync ms, FNV cross-check GPU vs CPU)
    uint64_t frames_ = 0, runs_ = 0;
    double lastEnter_ = 0; // last generate() entry (monotonic ms)
    double tFill_ = 0, tNpu_ = 0, tPresent_ = 0;
    double syncMs_ = 0;
    double wall0_ = 0;
    double wConvMx_ = 0, wNpuMx_ = 0, wPresMx_ = 0;
    std::atomic<uint64_t> lastFnv_{0};
    // Async verify: the FNV readback burns ~5ms of CPU on 1.4MB. It runs on
    // a side thread so the present thread never stalls (no FIFO phase hit).
    // The staging buffer is only reused at the next verify (240+ frames
    // later), so no race; joined on reuse and teardown.
    std::thread verifyThread_;
};

} // namespace Npu
