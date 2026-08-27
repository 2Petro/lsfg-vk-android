#ifdef __ANDROID__

#define EGL_EGLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#include "hwme.hpp"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>

#include <cstdint>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <vector>

namespace {

    typedef void (*PFN_TexEstimateMotionQCOM)(GLuint ref, GLuint target, GLuint output);

    constexpr GLint QCOM_BLOCK_X = 0x8C90;
    constexpr GLint QCOM_BLOCK_Y = 0x8C91;

    constexpr GLenum IMPORT_TEXTURE = 0x1000;
    constexpr GLenum IMPORT_RBO     = 0x1001;

    PFN_TexEstimateMotionQCOM texEstimateMotion = nullptr;
    EGLDisplay g_dpy = EGL_NO_DISPLAY;

    struct GlResources {
        bool valid{false};
        float debugLevel{0.0f};
        uint32_t width{0};
        uint32_t height{0};
        uint32_t blockX{16};
        uint32_t blockY{16};

        GLuint lumaA{0};
        GLuint lumaB{0};
        GLuint fboLuma{0};

        GLuint mvTex[2]{0, 0};
        int mvSlot{-1};
        AHardwareBuffer* pairA{nullptr};
        AHardwareBuffer* pairB{nullptr};
        bool hasPairMv{false};
        int pairsSeen{0};
        uint32_t mvW{0};
        uint32_t mvH{0};

        GLuint progLuma{0};
        GLuint progBlend{0};
        GLuint progPattern{0};
        GLuint hashRbo{0};
        GLuint fboHash{0};
        GLuint fboOut{0};
    } res;

    std::unordered_map<AHardwareBuffer*, GLuint> importObjects;

    const char* VS_SRC = R"(#version 300 es
out vec2 vUv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

    const char* LUMA_FS_SRC = R"(#version 300 es
precision mediump float;
uniform sampler2D uColor;
uniform float uScale;
in vec2 vUv;
out vec4 o;
void main() {
    vec3 c = texture(uColor, vUv * uScale).rgb;
    float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
    o = vec4(y, y, y, 1.0);
}
)";

    const char* PATTERN_FS_SRC = R"(#version 300 es
precision mediump float;
uniform float uShift;
in vec2 vUv;
out vec4 o;
void main() {
    float x = vUv.x * 1080.0 + uShift;
    float stripe = step(0.5, fract(x / 64.0));
    o = vec4(vec3(stripe), 1.0);
}
)";

    const char* BLEND_FS_SRC = R"(#version 300 es
precision mediump float;
uniform sampler2D uColA;
uniform sampler2D uColB;
uniform sampler2D uMv;
uniform sampler2D uMvPrev;
uniform sampler2D uLumaDbg;
uniform sampler2D uLumaDbgB;
uniform float uHasPrevMv;
uniform float uAlpha;
uniform float uDebug;
uniform float uMaxMotion;
uniform vec2 uMvSign;
uniform vec2 uWorkToFull;
uniform vec2 uFullSize;
uniform vec2 uBlockSize;
uniform vec2 uMvSize;
uniform vec2 uWorkSize;
in vec2 vUv;
out vec4 o;
void main() {
    vec2 uvw = vUv / uWorkToFull;
    vec2 mc = uvw * uWorkSize / uBlockSize - 0.5;
    vec2 f = fract(mc);
    ivec2 base = ivec2(floor(mc));
    ivec2 mvMax = ivec2(uMvSize) - 1;
    vec2 m00 = texelFetch(uMv, clamp(base + ivec2(0, 0), ivec2(0), mvMax), 0).rg;
    vec2 m10 = texelFetch(uMv, clamp(base + ivec2(1, 0), ivec2(0), mvMax), 0).rg;
    vec2 m01 = texelFetch(uMv, clamp(base + ivec2(0, 1), ivec2(0), mvMax), 0).rg;
    vec2 m11 = texelFetch(uMv, clamp(base + ivec2(1, 1), ivec2(0), mvMax), 0).rg;
    vec2 m = mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);

    vec2 mpix = vec2(0.0);
    if (uHasPrevMv > 0.5) {
        vec2 p00 = texelFetch(uMvPrev, clamp(base + ivec2(0, 0), ivec2(0), mvMax), 0).rg;
        vec2 p10 = texelFetch(uMvPrev, clamp(base + ivec2(1, 0), ivec2(0), mvMax), 0).rg;
        vec2 p01 = texelFetch(uMvPrev, clamp(base + ivec2(0, 1), ivec2(0), mvMax), 0).rg;
        vec2 p11 = texelFetch(uMvPrev, clamp(base + ivec2(1, 1), ivec2(0), mvMax), 0).rg;
        mpix = mix(mix(p00, p10, f.x), mix(p01, p11, f.x), f.y);

        float a2 = uAlpha * uAlpha;
        vec2 pa2 = vUv - ((uAlpha * 0.5) * (m + mpix)
                 + (a2 * 0.5) * (m - mpix)) / uFullSize;
        vec2 pb2 = pa2 + m / uFullSize;
        if (all(greaterThanEqual(pa2, vec2(0.0))) && all(lessThanEqual(pa2, vec2(1.0)))
         && all(greaterThanEqual(pb2, vec2(0.0))) && all(lessThanEqual(pb2, vec2(1.0)))) {
            vec3 cat = texture(uColA, pa2).rgb;
            vec3 cbt = texture(uColB, pb2).rgb;
            vec3 diff = abs(cbt - cat);
            float lumaDiff = diff.r * 0.5 + diff.g + diff.b * 0.5;
            if (lumaDiff < 0.01) {
                o = vec4(uAlpha < 0.5 ? cat : cbt, 1.0);
            } else {
                o = vec4(mix(cat, cbt, uAlpha), 1.0);
            }
            return;
        }
    }

    m *= uMvSign;
    mpix *= uMvSign;

    float mag = length(m);
    if (mag > uMaxMotion)
        m *= uMaxMotion / mag;

    if (uDebug > 2.5) {
        o = vec4(vec3(texture(uColA, vUv).r), 1.0);
        return;
    }
    if (uDebug > 1.5) {
        o = vec4(vec3(texture(uLumaDbg, uvw).r), 1.0);
        return;
    }
    if (uDebug > 4.5) {
        float d = abs(texture(uLumaDbg, uvw).r - texture(uLumaDbgB, uvw).r);
        o = vec4(vec3(min(d * 8.0, 1.0)), 1.0);
        return;
    }
    if (uDebug > 3.5) {
        o = vec4(vec3(min(mag / 4.0, 1.0)), 1.0);
        return;
    }
    if (uDebug > 0.5) {
        o = vec4(vec3(min(mag / 32.0, 1.0)), 1.0);
        return;
    }

    vec2 pa = vUv - (uAlpha * m + (uHasPrevMv > 0.5 ? uAlpha * 0.5 * mpix : vec2(0.0))) / uFullSize;
    vec2 pb = vUv + ((1.0 - uAlpha) * m - (uHasPrevMv > 0.5 ? (1.0 - uAlpha) * 0.5 * mpix : vec2(0.0))) / uFullSize;
    vec3 ca = texture(uColA, pa).rgb;
    vec3 cb = texture(uColB, pb).rgb;
    o = vec4(mix(ca, cb, uAlpha), 1.0);
}
)";

    GLuint compileShader(GLenum type, const char* src) {
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok{};
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512]{};
            glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
            std::cerr << "lsfg-hwme: shader compile failed: " << log << "\n";
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    }

    GLuint linkProgram(const char* vsSrc, const char* fsSrc) {
        GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
        if (!vs || !fs) return 0;
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint ok{};
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[512]{};
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            std::cerr << "lsfg-hwme: program link failed: " << log << "\n";
            glDeleteProgram(prog);
            return 0;
        }
        return prog;
    }

    struct Job {
        AHardwareBuffer* colA{nullptr};
        AHardwareBuffer* colB{nullptr};
        std::vector<AHardwareBuffer*> outs;
        std::vector<float> alphas;
        uint32_t width{0};
        uint32_t height{0};
        int inFenceFd{-1};
        bool ok{false};
        bool done{false};
        std::mutex* doneMtx{nullptr};
        std::condition_variable* doneCv{nullptr};
    };

    std::thread workerThread;
    std::atomic<bool> probeDone{false};
    std::atomic<bool> probeOk{false};

    std::mutex jobMtx;
    std::deque<Job*> jobQueue;
    std::condition_variable jobCv;

    bool getImported(AHardwareBuffer* ahb, GLenum mode, GLuint* outObj) {
        auto it = importObjects.find(ahb);
        if (it == importObjects.end()) {
            if (mode == IMPORT_RBO) {
                GLuint rbo{};
                glGenRenderbuffers(1, &rbo);
                it = importObjects.emplace(ahb, rbo).first;
            } else {
                GLuint tex{};
                glGenTextures(1, &tex);
                glBindTexture(GL_TEXTURE_2D, tex);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                it = importObjects.emplace(ahb, tex).first;
            }
        }
        GLuint obj = it->second;

        EGLClientBuffer clientBuf = eglGetNativeClientBufferANDROID(ahb);
        if (!clientBuf) {
            static bool l1 = false;
            if (!l1) { std::cerr << "lsfg-hwme: eglGetNativeClientBufferANDROID failed\n"; l1 = true; }
            return false;
        }
        const EGLint imgAttribs[] = { EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE };
        EGLImage img = eglCreateImageKHR(g_dpy, EGL_NO_CONTEXT,
            EGL_NATIVE_BUFFER_ANDROID, clientBuf, imgAttribs);
        if (img == EGL_NO_IMAGE_KHR) {
            static bool l2 = false;
            if (!l2) {
                std::cerr << "lsfg-hwme: eglCreateImageKHR failed ("
                          << std::hex << eglGetError() << std::dec << ")\n";
                l2 = true;
            }
            return false;
        }

        GLenum err;
        if (mode == IMPORT_RBO) {
            glBindRenderbuffer(GL_RENDERBUFFER, obj);
            glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, img);
            err = glGetError();
            glBindRenderbuffer(GL_RENDERBUFFER, 0);
        } else {
            glBindTexture(GL_TEXTURE_2D, obj);
            glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img);
            err = glGetError();
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        eglDestroyImageKHR(g_dpy, img);

        if (err != GL_NO_ERROR) {
            static bool l3 = false;
            if (!l3) {
                std::cerr << "lsfg-hwme: EGLImage retarget failed, error=0x"
                          << std::hex << err << std::dec << "\n";
                l3 = true;
            }
            return false;
        }
        *outObj = obj;
        return true;
    }

    void ensureSized(uint32_t width, uint32_t height) {
        if (res.valid && res.width == width && res.height == height)
            return;

        if (res.lumaA) glDeleteTextures(1, &res.lumaA);
        if (res.lumaB) glDeleteTextures(1, &res.lumaB);
        if (res.mvTex[0]) glDeleteTextures(2, res.mvTex);
        res.mvSlot = -1;
        res.pairA = nullptr;
        res.pairB = nullptr;
        res.hasPairMv = false;

        res.width = width;
        res.height = height;
        uint32_t workW = ((width + res.blockX - 1) / res.blockX) * res.blockX;
        uint32_t workH = ((height + res.blockY - 1) / res.blockY) * res.blockY;
        res.mvW = workW / res.blockX;
        res.mvH = workH / res.blockY;

        glGenTextures(1, &res.lumaA);
        glBindTexture(GL_TEXTURE_2D, res.lumaA);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, workW, workH);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenTextures(1, &res.lumaB);
        glBindTexture(GL_TEXTURE_2D, res.lumaB);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, workW, workH);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        for (int i = 0; i < 2; i++) {
            glGenTextures(1, &res.mvTex[i]);
            glBindTexture(GL_TEXTURE_2D, res.mvTex[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, res.mvW, res.mvH);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        res.valid = true;
        std::cerr << "lsfg-hwme: resources sized "
                  << width << "x" << height
                  << " (work " << workW << "x" << workH
                  << ", mv " << res.mvW << "x" << res.mvH << ")\n";
    }

    void extractLuma(GLuint colorTex, GLuint lumaTex, uint32_t workW, uint32_t workH,
            float scaleToFull) {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, res.fboLuma);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, lumaTex, 0);
        glViewport(0, 0, workW, workH);
        glUseProgram(res.progLuma);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glUniform1i(glGetUniformLocation(res.progLuma, "uColor"), 0);
        glUniform1f(glGetUniformLocation(res.progLuma, "uScale"),
            scaleToFull > 1.0f ? 1.0f : scaleToFull);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    bool runJob(Job* job) {
        if (job->inFenceFd >= 0) {
            const EGLint attribs[] = {
                EGL_SYNC_NATIVE_FENCE_FD_ANDROID, job->inFenceFd, EGL_NONE
            };
            EGLSyncKHR sync = eglCreateSyncKHR(g_dpy,
                EGL_SYNC_NATIVE_FENCE_ANDROID, attribs);
            if (sync != EGL_NO_SYNC_KHR) {
                eglWaitSyncKHR(g_dpy, sync, 0);
                eglDestroySyncKHR(g_dpy, sync);
            } else {
                close(job->inFenceFd);
            }
            job->inFenceFd = -1;
        } else if (job->inFenceFd == -2) {
            // special marker: already waited via CPU
        }

        ensureSized(job->width, job->height);
        uint32_t workW = ((job->width + res.blockX - 1) / res.blockX) * res.blockX;
        uint32_t workH = ((job->height + res.blockY - 1) / res.blockY) * res.blockY;

        GLuint colATex{}, colBTex{};
        if (!getImported(job->colA, IMPORT_TEXTURE, &colATex)) return false;
        if (!getImported(job->colB, IMPORT_TEXTURE, &colBTex)) return false;

        extractLuma(colATex, res.lumaA, workW, workH,
            static_cast<float>(job->width) / static_cast<float>(workW));
        extractLuma(colBTex, res.lumaB, workW, workH,
            static_cast<float>(job->width) / static_cast<float>(workW));

        const char* testPatEnv = std::getenv("LSFG_HWME_TESTPATTERN");
        bool testPattern = testPatEnv && testPatEnv[0] == '1';
        if (testPattern) {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, res.fboLuma);
            glViewport(0, 0, workW, workH);
            glUseProgram(res.progPattern);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D, res.lumaA, 0);
            glUniform1f(glGetUniformLocation(res.progPattern, "uShift"), 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D, res.lumaB, 0);
            glUniform1f(glGetUniformLocation(res.progPattern, "uShift"), 16.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, res.fboLuma);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D, res.lumaA, 0);
        }
        while (glGetError() != GL_NO_ERROR) {}

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

        bool newPair = job->colA != res.pairA || job->colB != res.pairB;
        if (newPair) {
            res.mvSlot = (res.mvSlot + 1) & 1;
            texEstimateMotion(res.lumaA, res.lumaB, res.mvTex[res.mvSlot]);
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                std::cerr << "lsfg-hwme: TexEstimateMotionQCOM error=0x"
                          << std::hex << err << std::dec << "\n";
                return false;
            }
            res.pairA = job->colA;
            res.pairB = job->colB;
            res.hasPairMv = true;
            res.pairsSeen++;
        }
        GLuint mvCur = res.mvTex[res.mvSlot];
        bool hasPrev = res.pairsSeen >= 2
            && res.pairA == job->colA && res.pairB == job->colB;
        GLuint mvPrev = res.mvTex[res.mvSlot ^ 1];

        for (size_t oi = 0; oi < job->outs.size(); ++oi) {
            float alpha = oi < job->alphas.size() ? job->alphas[oi] : 0.5f;
            GLuint outRbo{};
            if (!getImported(job->outs[oi], IMPORT_RBO, &outRbo)) return false;

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, res.fboOut);
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER, outRbo);
            if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                std::cerr << "lsfg-hwme: output FBO incomplete\n";
                return false;
            }

            glViewport(0, 0, job->width, job->height);
            glUseProgram(res.progBlend);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, colATex);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, colBTex);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, mvCur);
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, res.lumaA);
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, mvPrev);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, res.lumaB);

            glUniform1i(glGetUniformLocation(res.progBlend, "uColA"), 0);
            glUniform1i(glGetUniformLocation(res.progBlend, "uColB"), 1);
            glUniform1i(glGetUniformLocation(res.progBlend, "uMv"), 2);
            glUniform1i(glGetUniformLocation(res.progBlend, "uLumaDbg"), 3);
            glUniform1i(glGetUniformLocation(res.progBlend, "uMvPrev"), 4);
            glUniform1i(glGetUniformLocation(res.progBlend, "uLumaDbgB"), 5);
            glUniform1f(glGetUniformLocation(res.progBlend, "uHasPrevMv"),
                hasPrev ? 1.0f : 0.0f);
            glUniform1f(glGetUniformLocation(res.progBlend, "uAlpha"), alpha);
            glUniform1f(glGetUniformLocation(res.progBlend, "uDebug"), res.debugLevel);
            {
                const char* maxMv = std::getenv("LSFG_HWME_MAXMV");
                float v = maxMv ? static_cast<float>(atof(maxMv)) : 128.0f;
                glUniform1f(glGetUniformLocation(res.progBlend, "uMaxMotion"), v);

                float sx = 1.0f, sy = 1.0f;
                const char* flipXY = std::getenv("LSFG_HWME_FLIPXY");
                if (flipXY && *flipXY == '1') { sx = -1.0f; sy = -1.0f; }
                const char* flipY = std::getenv("LSFG_HWME_FLIPY");
                if (flipY && *flipY == '1') sy = -sy;
                glUniform2f(glGetUniformLocation(res.progBlend, "uMvSign"), sx, sy);
            }
            glUniform2f(glGetUniformLocation(res.progBlend, "uWorkToFull"),
                static_cast<float>(workW) / static_cast<float>(job->width),
                static_cast<float>(workH) / static_cast<float>(job->height));
            glUniform2f(glGetUniformLocation(res.progBlend, "uWorkSize"),
                static_cast<float>(workW), static_cast<float>(workH));
            glUniform2f(glGetUniformLocation(res.progBlend, "uFullSize"),
                static_cast<float>(job->width), static_cast<float>(job->height));
            glUniform2f(glGetUniformLocation(res.progBlend, "uBlockSize"),
                static_cast<float>(res.blockX), static_cast<float>(res.blockY));
            glUniform2f(glGetUniformLocation(res.progBlend, "uMvSize"),
                static_cast<float>(res.mvW), static_cast<float>(res.mvH));

            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glFinish();
        return true;
    }

    void workerMain() {
        EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        g_dpy = dpy;
        if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, nullptr, nullptr)) {
            std::cerr << "lsfg-hwme: EGL init failed\n";
            probeDone = true;
            return;
        }

        const EGLint cfgAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_NONE
        };
        EGLConfig cfg{};
        EGLint count{};
        if (!eglChooseConfig(dpy, cfgAttribs, &cfg, 1, &count) || count < 1) {
            std::cerr << "lsfg-hwme: no EGL config\n";
            probeDone = true;
            return;
        }

        eglBindAPI(EGL_OPENGL_ES_API);

        const EGLint pbAttribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
        EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pbAttribs);

        const EGLint ctxAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttribs);
        if (ctx == EGL_NO_CONTEXT || surf == EGL_NO_SURFACE ||
                !eglMakeCurrent(dpy, surf, surf, ctx)) {
            std::cerr << "lsfg-hwme: GLES3 context setup failed ("
                      << std::hex << eglGetError() << std::dec << ")\n";
            probeDone = true;
            return;
        }

        std::cerr << "lsfg-hwme: GL "
                  << glGetString(GL_VERSION)
                  << " | GLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION)
                  << " | " << glGetString(GL_VENDOR) << "\n";

        auto fn = eglGetProcAddress("glTexEstimateMotionQCOM");
        if (!fn) {
            std::cerr << "lsfg-hwme: GL_QCOM_motion_estimation not available\n";
            probeDone = true;
            return;
        }
        texEstimateMotion = reinterpret_cast<PFN_TexEstimateMotionQCOM>(fn);

        GLint bx{}, by{};
        glGetIntegerv(QCOM_BLOCK_X, &bx);
        glGetIntegerv(QCOM_BLOCK_Y, &by);
        if (bx > 0) res.blockX = static_cast<uint32_t>(bx);
        if (by > 0) res.blockY = static_cast<uint32_t>(by);

        const char* dbg = std::getenv("LSFG_HWME_DEBUG");
        res.debugLevel = (dbg && *dbg) ? static_cast<float>(atof(dbg)) : 0.0f;

        res.progLuma = linkProgram(VS_SRC, LUMA_FS_SRC);
        res.progBlend = linkProgram(VS_SRC, BLEND_FS_SRC);
        res.progPattern = linkProgram(VS_SRC, PATTERN_FS_SRC);
        if (!res.progLuma || !res.progBlend) {
            std::cerr << "lsfg-hwme: shader setup failed\n";
            texEstimateMotion = nullptr;
            probeDone = true;
            return;
        }
        glGenFramebuffers(1, &res.fboLuma);
        glGenFramebuffers(1, &res.fboOut);

        glGenRenderbuffers(1, &res.hashRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, res.hashRbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 16, 16);
        glGenFramebuffers(1, &res.fboHash);

        std::cerr << "lsfg-hwme: Adreno Motion Engine ready "
                  << "(block " << res.blockX << "x" << res.blockY << ")\n";
        probeOk = true;
        probeDone = true;

        for (;;) {
            Job* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(jobMtx);
                jobCv.wait(lock, [] { return !jobQueue.empty(); });
                job = jobQueue.front();
                jobQueue.pop_front();
            }
            bool ok = runJob(job);
            job->ok = ok;
            {
                std::lock_guard<std::mutex> lk(*job->doneMtx);
                job->done = true;
            }
            job->doneCv->notify_all();
        }
    }

    void ensureWorker() {
        if (probeDone.load()) return;
        static std::once_flag once;
        std::call_once(once, [] {
            std::thread(workerMain).detach();
        });
        while (!probeDone.load())
            std::this_thread::yield();
    }

}

namespace LSFG::HwMe {

    bool isEnabled() {
        const char* e = std::getenv("LSFG_HWME");
        if (e && e[0] == '0' && e[1] == '\0') return false;
        if (e && (e[0] == 'n' || e[0] == 'N')) return false; // no/false
        return true;
    }

    bool available() {
        if (!isEnabled()) return false;
        ensureWorker();
        return probeOk.load();
    }

    bool generate(AHardwareBuffer* prev, AHardwareBuffer* cur,
        const std::vector<AHardwareBuffer*>& outs,
        uint32_t width, uint32_t height) {
        if (!isEnabled()) return false;
        ensureWorker();
        if (!probeOk.load()) return false;
        if (!prev || !cur || outs.empty() || width==0 || height==0) return false;

        std::vector<float> alphas;
        alphas.reserve(outs.size());
        size_t n = outs.size();
        for (size_t i=0;i<n;++i) alphas.push_back(float(i+1)/float(n+1));

        std::mutex doneMtx;
        std::condition_variable doneCv;
        Job job;
        job.colA = prev;
        job.colB = cur;
        job.outs = outs;
        job.alphas = alphas;
        job.width = width;
        job.height = height;
        job.inFenceFd = -2; // already waited via CPU in caller
        job.doneMtx = &doneMtx;
        job.doneCv = &doneCv;

        {
            std::lock_guard<std::mutex> lock(jobMtx);
            jobQueue.push_back(&job);
        }
        jobCv.notify_one();
        std::unique_lock<std::mutex> lk(doneMtx);
        doneCv.wait(lk, [&]{ return job.done; });
        return job.ok;
    }

    bool generate(AHardwareBuffer* colA, AHardwareBuffer* colB,
        AHardwareBuffer* out, float alpha,
        uint32_t width, uint32_t height) {
        if (!isEnabled()) return false;
        ensureWorker();
        if (!probeOk.load()) return false;
        if (!colA || !colB || !out || width==0 || height==0) return false;

        std::mutex doneMtx;
        std::condition_variable doneCv;
        Job job;
        job.colA = colA;
        job.colB = colB;
        job.outs = std::vector<AHardwareBuffer*>{out};
        job.alphas = std::vector<float>{alpha};
        job.width = width;
        job.height = height;
        job.inFenceFd = -2;
        job.doneMtx = &doneMtx;
        job.doneCv = &doneCv;

        {
            std::lock_guard<std::mutex> lock(jobMtx);
            jobQueue.push_back(&job);
        }
        jobCv.notify_one();
        std::unique_lock<std::mutex> lk(doneMtx);
        doneCv.wait(lk, [&]{ return job.done; });
        return job.ok;
    }

}

#endif
