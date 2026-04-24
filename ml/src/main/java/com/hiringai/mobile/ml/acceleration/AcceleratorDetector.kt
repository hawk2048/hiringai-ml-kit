package com.hiringai.mobile.ml.acceleration

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.opengl.GLES20
import android.os.Build
import android.os.Process
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.FileReader

/**
 * 加速器检测服务
 *
 * 检测设备可用的硬件加速器：GPU、NPU、DSP
 * 提供 Vulkan 支持、OpenGL ES 版本检测
 * 识别 Qualcomm Snapdragon NPU (通过 NNAPI)
 * 获取 GPU 内存、计算单元信息
 * 生成设备指纹用于加速推荐
 */
class AcceleratorDetector(private val context: Context) {

    companion object {
        private const val TAG = "AcceleratorDetector"

        // GPU 厂商标识
        private const val GPU_ADRENO = "Adreno"
        private const val GPU_MALI = "Mali"
        private const val GPU_POWERVR = "PowerVR"
        private const val GPU_NVIDIA = "NVIDIA"

        // SoC 厂商标识 (从 /sys/devices/soc0/ 检测)
        private val SNAPDRAGON_PATTERNS = listOf(
            "snapdragon", "sdm", "sm", "qcom", "qualcomm", "msm"
        )
        private val MEDIATEK_PATTERNS = listOf("mt", "mediatek")
        private val EXYNOS_PATTERNS = listOf("exynos", "samsung")
        private val KIRIN_PATTERNS = listOf("kirin", "hi3660", "hi3670")
        private val TENSOR_PATTERNS = listOf("tensor", "gs101", "gs201", "gs301")
    }

    /**
     * GPU 信息
     */
    data class GpuInfo(
        val name: String,
        val vendor: GpuVendor,
        val openGLESVersion: String,
        val vulkanVersion: String?,
        val computeUnits: Int = 0,
        val memoryMB: Long = 0,
        val supportsCompute: Boolean = false,
        val supportsFP16: Boolean = false
    )

    /**
     * GPU 厂商枚举
     */
    enum class GpuVendor {
        ADRENO,      // Qualcomm
        MALI,        // ARM (Samsung, MediaTek)
        POWERVR,     // Imagination
        NVIDIA,      // Tegra
        APPLE,       // Apple GPU (iOS only, not used here)
        INTEL,       // Intel HD Graphics
        UNKNOWN
    }

    /**
     * NPU 信息
     */
    data class NpuInfo(
        val available: Boolean,
        val name: String,
        val vendor: NpuVendor,
        val nnapiSupported: Boolean,
        val computeCapability: Int = 0  // 0-100 相对算力评分
    )

    /**
     * NPU 厂商枚举
     */
    enum class NpuVendor {
        SNAPDRAGON_HTP,    // Qualcomm Hexagon Tensor Processor
        MEDIATEK_APU,      // MediaTek AI Processing Unit
        SAMSUNG_NPU,       // Samsung Neural Processing Unit
        GOOGLE_TPU,        // Google Tensor Processing Unit
        HUAWEI_NPU,        // Huawei Da Vinci NPU
        NONE,
        UNKNOWN
    }

    /**
     * 设备加速指纹
     */
    data class DeviceAccelerationFingerprint(
        val socModel: String,
        val gpu: GpuInfo,
        val npu: NpuInfo,
        val totalRAMGB: Long,
        val recommendedBackend: AccelerationConfig.Backend,
        val recommendedGpuLayers: Int,
        val accelerationScore: Int  // 0-100 综合加速能力评分
    ) {
        fun getSummary(): String = buildString {
            appendLine("🔧 设备加速指纹:")
            appendLine("  SoC: $socModel")
            appendLine("  GPU: ${gpu.name} (${gpu.vendor})")
            appendLine("  OpenGL ES: ${gpu.openGLESVersion}")
            appendLine("  Vulkan: ${gpu.vulkanVersion ?: "不支持"}")
            appendLine("  NPU: ${if (npu.available) "${npu.name} (${npu.vendor})" else "不可用"}")
            appendLine("  NNAPI: ${if (npu.nnapiSupported) "✓" else "✗"}")
            appendLine("  内存: ${totalRAMGB}GB")
            appendLine("  推荐后端: ${recommendedBackend.displayName}")
            appendLine("  推荐 GPU Layers: $recommendedGpuLayers")
            appendLine("  加速能力评分: $accelerationScore/100")
        }
    }

    /**
     * 检测所有加速器
     */
    suspend fun detectAllAccelerators(): DeviceAccelerationFingerprint = withContext(Dispatchers.IO) {
        val gpu = detectGpu()
        val npu = detectNpu()
        val socModel = detectSocModel()
        val totalRAM = getTotalRAMGB()

        val recommendation = recommendBackend(gpu, npu, totalRAM)
        val gpuLayers = recommendGpuLayers(gpu, totalRAM)
        val score = calculateAccelerationScore(gpu, npu, totalRAM)

        DeviceAccelerationFingerprint(
            socModel = socModel,
            gpu = gpu,
            npu = npu,
            totalRAMGB = totalRAM,
            recommendedBackend = recommendation,
            recommendedGpuLayers = gpuLayers,
            accelerationScore = score
        )
    }

    /**
     * 检测 GPU 信息
     */
    fun detectGpu(): GpuInfo {
        val (glVersion, glRenderer, glVendor) = getOpenGLEnfo()
        val vendor = identifyGpuVendor(glRenderer, glVendor)
        val vulkanVersion = checkVulkanSupport()
        val (computeUnits, gpuMemory) = getGpuHardwareInfo(vendor)
        val supportsCompute = checkComputeShaderSupport()
        val supportsFP16 = checkFP16Support(vendor)

        return GpuInfo(
            name = glRenderer,
            vendor = vendor,
            openGLESVersion = glVersion,
            vulkanVersion = vulkanVersion,
            computeUnits = computeUnits,
            memoryMB = gpuMemory,
            supportsCompute = supportsCompute,
            supportsFP16 = supportsFP16
        )
    }

    /**
     * 检测 NPU 信息
     */
    fun detectNpu(): NpuInfo {
        val nnapiSupported = checkNnapiSupport()
        val (available, vendor, name) = detectNpuHardware()

        val computeCapability = if (available) {
            estimateNpuCapability(vendor)
        } else {
            0
        }

        return NpuInfo(
            available = available,
            name = name,
            vendor = vendor,
            nnapiSupported = nnapiSupported,
            computeCapability = computeCapability
        )
    }

    /**
     * 获取 OpenGL ES 信息
     */
    private fun getOpenGLEnfo(): Triple<String, String, String> {
        var glVersion = "Unknown"
        var glRenderer = "Unknown"
        var glVendor = "Unknown"

        try {
            // 创建临时 EGL 上下文获取 GPU 信息
            val egl = javax.microedition.khronos.egl.EGLContext.getEGL() as javax.microedition.khronos.egl.EGL10
            val display = egl.eglGetDisplay(javax.microedition.khronos.egl.EGL10.EGL_DEFAULT_DISPLAY)

            val version = IntArray(2)
            egl.eglInitialize(display, version)

            val configSpec = intArrayOf(
                javax.microedition.khronos.egl.EGL10.EGL_RENDERABLE_TYPE,
                4, // EGL_OPENGL_ES2_BIT
                javax.microedition.khronos.egl.EGL10.EGL_NONE
            )

            val configs = arrayOfNulls<javax.microedition.khronos.egl.EGLConfig>(1)
            val numConfigs = IntArray(1)
            egl.eglChooseConfig(display, configSpec, configs, 1, numConfigs)

            if (numConfigs[0] > 0) {
                val config = configs[0]
                val contextAttribs = intArrayOf(
                    0x3098, // EGL_CONTEXT_CLIENT_VERSION
                    2,
                    javax.microedition.khronos.egl.EGL10.EGL_NONE
                )

                val context = egl.eglCreateContext(display, config, javax.microedition.khronos.egl.EGL10.EGL_NO_CONTEXT, contextAttribs)

                if (context != javax.microedition.khronos.egl.EGL10.EGL_NO_CONTEXT) {
                    val surfaceAttribs = intArrayOf(javax.microedition.khronos.egl.EGL10.EGL_NONE)
                    val surface = egl.eglCreatePbufferSurface(display, config, surfaceAttribs)

                    egl.eglMakeCurrent(display, surface, surface, context)

                    glVersion = GLES20.glGetString(GLES20.GL_VERSION) ?: "Unknown"
                    glRenderer = GLES20.glGetString(GLES20.GL_RENDERER) ?: "Unknown"
                    glVendor = GLES20.glGetString(GLES20.GL_VENDOR) ?: "Unknown"

                    egl.eglMakeCurrent(display,
                        javax.microedition.khronos.egl.EGL10.EGL_NO_SURFACE,
                        javax.microedition.khronos.egl.EGL10.EGL_NO_SURFACE,
                        javax.microedition.khronos.egl.EGL10.EGL_NO_CONTEXT)
                    egl.eglDestroySurface(display, surface)
                    egl.eglDestroyContext(display, context)
                }
            }
            egl.eglTerminate(display)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get OpenGL ES info", e)
        }

        return Triple(glVersion, glRenderer, glVendor)
    }

    /**
     * 识别 GPU 厂商
     */
    private fun identifyGpuVendor(renderer: String, vendor: String): GpuVendor {
        val combined = "$renderer $vendor".lowercase()

        return when {
            combined.contains("adreno") -> GpuVendor.ADRENO
            combined.contains("mali") -> GpuVendor.MALI
            combined.contains("powervr") || combined.contains("rogue") -> GpuVendor.POWERVR
            combined.contains("nvidia") || combined.contains("tegra") -> GpuVendor.NVIDIA
            combined.contains("intel") -> GpuVendor.INTEL
            else -> GpuVendor.UNKNOWN
        }
    }

    /**
     * 检查 Vulkan 支持
     */
    private fun checkVulkanSupport(): String? {
        return try {
            // Android 7.0+ 支持 Vulkan
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                val packageManager = context.packageManager
                if (packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL, 1)) {
                    val version = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        if (packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 1)) "1.1" else "1.0"
                    } else {
                        "1.0"
                    }
                    version
                } else null
            } else null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check Vulkan support", e)
            null
        }
    }

    /**
     * 获取 GPU 硬件信息 (计算单元、显存)
     */
    private fun getGpuHardwareInfo(vendor: GpuVendor): Pair<Int, Long> {
        var computeUnits = 0
        var memoryMB: Long = 0

        try {
            when (vendor) {
                GpuVendor.ADRENO -> {
                    // Adreno GPU 信息
                    // 从 /sys/class/kgsl/kgsl-3d0/ 获取
                    computeUnits = readSysfsInt("/sys/class/kgsl/kgsl-3d0/gpu_model")
                        .let { estimateAdrenoCUs(it) }
                    memoryMB = readSysfsLong("/sys/class/kgsl/kgsl-3d0/devfreq/gpuclk")
                        .let { estimateAdrenoMemory(it) }
                }
                GpuVendor.MALI -> {
                    // Mali GPU 信息
                    computeUnits = readSysfsInt("/sys/class/misc/mali0/device/num_shader_cores")
                    memoryMB = getTotalRAMGB() * 1024 / 4  // 估算共享内存的 1/4
                }
                else -> {
                    // 未知 GPU，使用默认估算
                    computeUnits = 4
                    memoryMB = 512
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get GPU hardware info", e)
        }

        return Pair(computeUnits.coerceAtLeast(1), memoryMB.coerceAtLeast(256))
    }

    /**
     * 检查计算着色器支持
     */
    private fun checkComputeShaderSupport(): Boolean {
        return try {
            // OpenGL ES 3.1+ 支持计算着色器
            val egl = javax.microedition.khronos.egl.EGLContext.getEGL() as javax.microedition.khronos.egl.EGL10
            val display = egl.eglGetDisplay(javax.microedition.khronos.egl.EGL10.EGL_DEFAULT_DISPLAY)

            val version = IntArray(2)
            egl.eglInitialize(display, version)

            val configSpec = intArrayOf(
                javax.microedition.khronos.egl.EGL10.EGL_RENDERABLE_TYPE,
                0x40, // EGL_OPENGL_ES3_BIT
                javax.microedition.khronos.egl.EGL10.EGL_NONE
            )

            val configs = arrayOfNulls<javax.microedition.khronos.egl.EGLConfig>(1)
            val numConfigs = IntArray(1)
            egl.eglChooseConfig(display, configSpec, configs, 1, numConfigs)
            egl.eglTerminate(display)

            numConfigs[0] > 0
        } catch (e: Exception) {
            false
        }
    }

    /**
     * 检查 FP16 支持
     */
    private fun checkFP16Support(vendor: GpuVendor): Boolean {
        // Adreno 5xx+, Mali G72+, PowerVR Series 8XE+ 支持 FP16
        return vendor != GpuVendor.UNKNOWN
    }

    /**
     * 检查 NNAPI 支持
     */
    private fun checkNnapiSupport(): Boolean {
        return try {
            // NNAPI requires Android 8.1+ (API 27+)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                // FEATURE_NEURAL_NETWORKS was added in API 27
                @Suppress("DEPRECATION")
                context.packageManager.hasSystemFeature("android.hardware.neural_networks")
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    /**
     * 检测 NPU 硬件
     */
    private fun detectNpuHardware(): Triple<Boolean, NpuVendor, String> {
        val socModel = detectSocModel().lowercase()

        return when {
            matchesPatterns(socModel, SNAPDRAGON_PATTERNS) -> {
                // Snapdragon SoC with Hexagon DSP/HTP
                val hasNpu = isSnapdragonNpuAvailable()
                Triple(hasNpu, NpuVendor.SNAPDRAGON_HTP, "Qualcomm Hexagon NPU")
            }
            matchesPatterns(socModel, MEDIATEK_PATTERNS) -> {
                Triple(true, NpuVendor.MEDIATEK_APU, "MediaTek APU")
            }
            matchesPatterns(socModel, EXYNOS_PATTERNS) -> {
                // Exynos 990+ has NPU
                val hasNpu = socModel.contains("990") || socModel.contains("2100") ||
                    socModel.contains("2200") || socModel.contains("2400")
                Triple(hasNpu, NpuVendor.SAMSUNG_NPU, "Samsung NPU")
            }
            matchesPatterns(socModel, TENSOR_PATTERNS) -> {
                // Google Pixel with Tensor chip
                Triple(true, NpuVendor.GOOGLE_TPU, "Google TPU")
            }
            matchesPatterns(socModel, KIRIN_PATTERNS) -> {
                // Kirin 970+ has NPU
                Triple(true, NpuVendor.HUAWEI_NPU, "Huawei Da Vinci NPU")
            }
            else -> {
                // 检查是否有通用 NNAPI 加速器
                val hasNnapi = checkNnapiSupport()
                Triple(hasNnapi, NpuVendor.UNKNOWN, if (hasNnapi) "NNAPI Accelerator" else "None")
            }
        }
    }

    /**
     * 检查 Snapdragon NPU 可用性
     */
    private fun isSnapdragonNpuAvailable(): Boolean {
        return try {
            // 检查 Hexagon DSP 可用性
            val hexagonPath = "/dev/ion"
            java.io.File(hexagonPath).exists() && checkNnapiSupport()
        } catch (e: Exception) {
            false
        }
    }

    /**
     * 估算 NPU 算力
     */
    private fun estimateNpuCapability(vendor: NpuVendor): Int {
        return when (vendor) {
            NpuVendor.GOOGLE_TPU -> 90
            NpuVendor.SNAPDRAGON_HTP -> 80
            NpuVendor.HUAWEI_NPU -> 75
            NpuVendor.SAMSUNG_NPU -> 70
            NpuVendor.MEDIATEK_APU -> 60
            NpuVendor.UNKNOWN -> 40
            NpuVendor.NONE -> 0
        }
    }

    /**
     * 检测 SoC 型号
     */
    private fun detectSocModel(): String {
        return try {
            // 尝试从 /sys/devices/soc0/ 读取
            var soc = readSysfsString("/sys/devices/soc0/machine")
            if (soc.isNotEmpty()) return soc

            soc = readSysfsString("/sys/devices/virtual/misc/hw_info/hardware_info")
            if (soc.isNotEmpty()) return soc

            // 从 build.prop 获取
            soc = Build.HARDWARE
            if (soc.isNotEmpty()) return soc

            // 从品牌和型号推断
            "${Build.MANUFACTURER} ${Build.MODEL}"
        } catch (e: Exception) {
            "${Build.MANUFACTURER} ${Build.MODEL}"
        }
    }

    /**
     * 获取总 RAM (GB)
     */
    private fun getTotalRAMGB(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return memInfo.totalMem / (1024 * 1024 * 1024)
    }

    /**
     * 推荐加速后端
     */
    private fun recommendBackend(gpu: GpuInfo, npu: NpuInfo, totalRAMGB: Long): AccelerationConfig.Backend {
        // NPU 可用时优先使用 NNAPI
        if (npu.available && npu.nnapiSupported && npu.computeCapability >= 50) {
            return AccelerationConfig.Backend.NNAPI
        }

        // GPU 支持 Vulkan/Compute Shader 时使用 GPU
        if (gpu.vulkanVersion != null && gpu.supportsCompute && totalRAMGB >= 4) {
            return AccelerationConfig.Backend.GPU
        }

        // XNNPACK 优化 CPU
        if (gpu.supportsFP16) {
            return AccelerationConfig.Backend.XNNPACK
        }

        return AccelerationConfig.Backend.CPU
    }

    /**
     * 推荐 GPU Layers
     */
    private fun recommendGpuLayers(gpu: GpuInfo, totalRAMGB: Long): Int {
        if (gpu.vulkanVersion == null && !gpu.supportsCompute) {
            return 0
        }

        // 根据内存和 GPU 能力推荐
        return when {
            totalRAMGB >= 8 && gpu.computeUnits >= 16 -> 35
            totalRAMGB >= 6 && gpu.computeUnits >= 8 -> 20
            totalRAMGB >= 4 -> 10
            else -> 0
        }
    }

    /**
     * 计算加速能力评分
     */
    private fun calculateAccelerationScore(gpu: GpuInfo, npu: NpuInfo, totalRAMGB: Long): Int {
        var score = 0

        // NPU 分数 (最高 40 分)
        if (npu.available) {
            score += minOf(40, npu.computeCapability * 4 / 10)
        }

        // GPU 分数 (最高 35 分)
        score += when {
            gpu.vulkanVersion != null && gpu.supportsCompute -> 35
            gpu.vulkanVersion != null -> 25
            gpu.supportsCompute -> 20
            else -> 10
        }

        // GPU 厂商加分 (最高 15 分)
        score += when (gpu.vendor) {
            GpuVendor.ADRENO -> if (gpu.computeUnits >= 8) 15 else 10
            GpuVendor.MALI -> if (gpu.computeUnits >= 8) 12 else 8
            GpuVendor.POWERVR -> 10
            else -> 5
        }

        // 内存加分 (最高 10 分)
        score += when {
            totalRAMGB >= 8 -> 10
            totalRAMGB >= 6 -> 7
            totalRAMGB >= 4 -> 4
            else -> 2
        }

        return score.coerceIn(0, 100)
    }

    // ========== 辅助方法 ==========

    private fun matchesPatterns(text: String, patterns: List<String>): Boolean {
        return patterns.any { text.contains(it, ignoreCase = true) }
    }

    private fun readSysfsString(path: String): String {
        return try {
            BufferedReader(FileReader(path)).use { it.readLine()?.trim() ?: "" }
        } catch (e: Exception) {
            ""
        }
    }

    private fun readSysfsInt(path: String): Int {
        return readSysfsString(path).toIntOrNull() ?: 0
    }

    private fun readSysfsLong(path: String): Long {
        return readSysfsString(path).toLongOrNull() ?: 0L
    }

    private fun estimateAdrenoCUs(modelCode: Int): Int {
        // Adreno GPU 计算单元估算
        return when {
            modelCode >= 700 -> 8   // Adreno 7xx
            modelCode >= 600 -> 6   // Adreno 6xx
            modelCode >= 500 -> 4   // Adreno 5xx
            else -> 2
        }
    }

    private fun estimateAdrenoMemory(clockRate: Long): Long {
        // 根据 GPU 时钟估算显存 (MB)
        return (clockRate / 1000000).coerceIn(256, 2048)
    }
}
