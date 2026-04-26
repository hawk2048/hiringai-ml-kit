package com.hiringai.mobile.ml

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.io.RandomAccessFile

/**
 * 设备能力检测服务 v2
 *
 * 新增：
 *  - SoC 型号检测（/sys/devices/soc0/）
 *  - CPU 大/小核频率分离
 *  - NPU 粗检测（通过 NNAPI Feature 和 /proc/cpuinfo）
 *  - 当前温度读取（thermal zone 扫描）
 *  - 改进 benchmark score：用 NEON 友好的矩阵计算替代标量 sqrt
 */
class DeviceCapabilityDetector(private val context: Context) {

    data class DeviceCapabilities(
        val cpuCores: Int,
        val cpuArchitecture: String,
        val cpuBigCoreMaxFreqMHz: Int,      // 大核最大频率 MHz
        val cpuLittleCoreMaxFreqMHz: Int,   // 小核最大频率 MHz
        val totalRAM: Long,                 // MB（原为 GB，改为 MB 更精确）
        val availableRAM: Long,             // MB
        val hasVulkan: Boolean,
        val hasOpenGLES3: Boolean,
        val gpuName: String,
        val is64Bit: Boolean,
        val socModel: String,               // SoC 型号字符串
        val npuAvailable: Boolean,          // 是否检测到 NPU/NNAPI
        val benchmarkScore: Int,            // 0-100 相对性能评分
        val cpuTempCelsius: Float           // CPU 温度（-1 = 无法读取）
    ) {
        // 向后兼容：以 GB 为单位的属性
        val totalRAMGB: Long get() = totalRAM / 1024
        val availableRAMGB: Long get() = availableRAM / 1024
    }

    data class ModelRecommendation(
        val modelName: String,
        val isRecommended: Boolean,
        val requiredRAM: Int,
        val reason: String
    )

    companion object {
        private const val TAG = "DeviceCapability"

        private const val MIN_RAM_MB_FOR_LARGE  = 6 * 1024
        private const val MIN_RAM_MB_FOR_MEDIUM = 4 * 1024
        private const val MIN_RAM_MB_FOR_SMALL  = 2 * 1024
    }

    suspend fun detectCapabilities(): DeviceCapabilities = withContext(Dispatchers.IO) {
        val cpuCores = Runtime.getRuntime().availableProcessors()
        val cpuArch = getCPUArchitecture()
        val totalRAM = getTotalRAM() / (1024 * 1024)        // → MB
        val availableRAM = getAvailableRAM() / (1024 * 1024) // → MB
        val (hasVulkan, hasOpenGLES3, gpuName) = detectGPU()
        val is64Bit = Build.SUPPORTED_ABIS.any { it.contains("64") }
        val (bigFreq, littleFreq) = readCpuFrequencies()
        val socModel = readSocModel()
        val npuAvailable = detectNpu()
        val cpuTemp = readCpuTemp()
        val benchmarkScore = runBenchmark(cpuCores)

        DeviceCapabilities(
            cpuCores = cpuCores,
            cpuArchitecture = cpuArch,
            cpuBigCoreMaxFreqMHz = bigFreq,
            cpuLittleCoreMaxFreqMHz = littleFreq,
            totalRAM = totalRAM,
            availableRAM = availableRAM,
            hasVulkan = hasVulkan,
            hasOpenGLES3 = hasOpenGLES3,
            gpuName = gpuName,
            is64Bit = is64Bit,
            socModel = socModel,
            npuAvailable = npuAvailable,
            benchmarkScore = benchmarkScore,
            cpuTempCelsius = cpuTemp
        )
    }

    private fun getCPUArchitecture(): String {
        return Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
    }

    private fun getTotalRAM(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return memInfo.totalMem
    }

    private fun getAvailableRAM(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return memInfo.availMem
    }

    /**
     * 读取所有 CPU 核心的最大频率，分为大核（最高频）和小核（最低频）
     */
    private fun readCpuFrequencies(): Pair<Int, Int> {
        val maxFreqs = mutableListOf<Int>()
        for (core in 0..7) {
            val path = "/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_max_freq"
            try {
                val khz = File(path).readText().trim().toIntOrNull() ?: continue
                maxFreqs.add(khz / 1000)  // kHz → MHz
            } catch (_: Exception) {}
        }
        if (maxFreqs.isEmpty()) return Pair(0, 0)
        return Pair(maxFreqs.max(), maxFreqs.min())
    }

    /**
     * 读取 SoC 型号：优先 /sys/devices/soc0/，回退到 ro.board.platform
     */
    private fun readSocModel(): String {
        val socFiles = listOf(
            "/sys/devices/soc0/soc_id",
            "/sys/devices/soc0/family",
            "/sys/devices/soc0/machine",
            "/sys/devices/system/soc/soc0/family"
        )
        for (path in socFiles) {
            try {
                val content = File(path).readText().trim()
                if (content.isNotEmpty() && content != "unknown") return content
            } catch (_: Exception) {}
        }
        // 通过 /proc/cpuinfo 中的 Hardware 字段
        try {
            BufferedReader(FileReader("/proc/cpuinfo")).useLines { lines ->
                lines.forEach { line ->
                    if (line.startsWith("Hardware")) {
                        return line.substringAfter(":").trim()
                    }
                }
            }
        } catch (_: Exception) {}
        return "${Build.MANUFACTURER}_${Build.HARDWARE}"
    }

    /**
     * NPU 粗检测：NNAPI Feature + 已知 NPU 相关库文件
     */
    private fun detectNpu(): Boolean {
        // 1. NNAPI 支持 API 27+
        if (Build.VERSION.SDK_INT >= 27) {
            val pm = context.packageManager
            if (pm.hasSystemFeature("android.hardware.neural_networks")) return true
        }
        // 2. 检测已知 NPU 驱动文件
        val npuDriverPaths = listOf(
            "/vendor/lib64/libqnnhtpv73.so",    // Qualcomm HTP
            "/vendor/lib64/libhta_qcnpu.so",    // Qualcomm HTA
            "/vendor/lib64/libnnrt.so",          // MediaTek APU
            "/vendor/lib64/libtflite_npu.so",   // Generic NPU
            "/system/lib64/libneural_networks_v1.so"
        )
        return npuDriverPaths.any { File(it).exists() }
    }

    /**
     * 读取 CPU 温度（thermal zone 扫描）
     */
    private fun readCpuTemp(): Float {
        val cpuTypes = setOf("cpu", "cpu0", "tsens_tz_sensor", "soc_max", "cpuss",
            "cpu-1-0-usr", "cpu-1-2-usr", "cpu0-silver-usr", "cpu4-gold-usr")
        try {
            val thermalDir = File("/sys/class/thermal")
            if (thermalDir.exists()) {
                thermalDir.listFiles { f -> f.isDirectory && f.name.startsWith("thermal_zone") }
                    ?.sortedBy { it.name }
                    ?.forEach { zone ->
                        val type = File(zone, "type").runCatching { readText().trim().lowercase() }.getOrNull() ?: return@forEach
                        if (cpuTypes.any { type.contains(it) }) {
                            val raw = File(zone, "temp").runCatching { readText().trim().toLongOrNull() }.getOrNull() ?: return@forEach
                            return if (raw > 200L) raw / 1000f else raw.toFloat()
                        }
                    }
            }
        } catch (_: Exception) {}
        return -1f
    }

    private fun detectGPU(): Triple<Boolean, Boolean, String> {
        var hasVulkan = false
        var hasOpenGLES3 = false
        var gpuName = "Unknown"

        try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val configInfo = activityManager.deviceConfigurationInfo

            if (configInfo.reqGlEsVersion >= 0x30000) {
                hasOpenGLES3 = true
            }

            val gpuFiles = listOf(
                "/sys/class/gpu/gpu0/model",
                "/sys/class/kgsl/kgsl-3d0/gpu_model",
                "/sys/devices/soc0/build_id"
            )
            for (file in gpuFiles) {
                try {
                    val content = FileReader(file).use { it.readText().trim() }
                    if (content.isNotEmpty() && content.length < 100) {
                        gpuName = content
                        break
                    }
                } catch (_: Exception) {}
            }

            if (gpuName == "Unknown") {
                gpuName = "${Build.MANUFACTURER} ${Build.MODEL}"
            }

            hasVulkan = Build.VERSION.SDK_INT >= Build.VERSION_CODES.N
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting GPU", e)
            gpuName = "${Build.MANUFACTURER} ${Build.MODEL}"
        }

        return Triple(hasVulkan, hasOpenGLES3, gpuName)
    }

    /**
     * 多线程 benchmark score（模拟 MatMul 负载，比 sqrt 更贴近 AI 推理）
     *
     * 使用 4 线程并发计算，结果稳定且能体现多核性能
     */
    private fun runBenchmark(cores: Int): Int {
        return try {
            val n = 128
            val threads = minOf(cores, 4)
            val startTime = System.nanoTime()

            val tasks = (0 until threads).map { t ->
                Thread {
                    val a = FloatArray(n * n) { it * 0.001f }
                    val b = FloatArray(n * n) { it * 0.001f }
                    val c = FloatArray(n * n)
                    // 简化 MatMul n×n
                    for (i in 0 until n) {
                        for (k in 0 until n) {
                            val aik = a[i * n + k]
                            for (j in 0 until n) {
                                c[i * n + j] += aik * b[k * n + j]
                            }
                        }
                    }
                }
            }
            tasks.forEach { it.start() }
            tasks.forEach { it.join() }

            val elapsed = (System.nanoTime() - startTime) / 1_000_000_000.0
            // 基准：旗舰机 ~0.05s → score 90；低端机 ~0.5s → score ~30
            (100 - (elapsed * 120).toInt()).coerceIn(10, 100)
        } catch (_: Exception) {
            50
        }
    }

    fun recommendModels(capabilities: DeviceCapabilities): List<ModelRecommendation> {
        val recommendations = mutableListOf<ModelRecommendation>()

        for (model in LocalLLMService.AVAILABLE_MODELS) {
            val requiredMB = model.requiredRAM * 1024L
            val (isRecommended, reason) = when {
                capabilities.availableRAM < requiredMB -> {
                    Pair(false, "可用内存不足 (需要${model.requiredRAM}GB，当前可用 ${capabilities.availableRAM / 1024}GB)")
                }
                capabilities.benchmarkScore < 30 -> {
                    Pair(false, "设备 CPU 性能较低 (score=${capabilities.benchmarkScore})")
                }
                capabilities.totalRAM >= MIN_RAM_MB_FOR_LARGE && capabilities.benchmarkScore >= 60 -> {
                    Pair(true, "设备性能充足")
                }
                capabilities.totalRAM >= MIN_RAM_MB_FOR_MEDIUM -> {
                    Pair(model.requiredRAM <= 2, "可根据需求选择")
                }
                else -> {
                    Pair(model.requiredRAM <= 1, "建议使用轻量模型")
                }
            }

            recommendations.add(
                ModelRecommendation(
                    modelName = model.name,
                    isRecommended = isRecommended,
                    requiredRAM = model.requiredRAM,
                    reason = reason
                )
            )
        }

        return recommendations
    }

    fun getDeviceSummary(capabilities: DeviceCapabilities): String {
        return buildString {
            append("📱 设备信息:\n")
            append("  SoC  : ${capabilities.socModel}\n")
            append("  CPU  : ${capabilities.cpuCores}核 ${capabilities.cpuArchitecture}\n")
            append("       大核: ${capabilities.cpuBigCoreMaxFreqMHz} MHz  小核: ${capabilities.cpuLittleCoreMaxFreqMHz} MHz\n")
            append("  内存 : ${capabilities.totalRAM}MB (可用: ${capabilities.availableRAM}MB)\n")
            append("  GPU  : ${capabilities.gpuName}\n")
            append("  NPU  : ${if (capabilities.npuAvailable) "✓ 可用" else "✗ 未检测到"}\n")
            append("  Vulkan: ${if (capabilities.hasVulkan) "✓" else "✗"}\n")
            append("  64位 : ${if (capabilities.is64Bit) "✓" else "✗"}\n")
            if (capabilities.cpuTempCelsius > 0) {
                append("  温度 : ${"%.1f".format(capabilities.cpuTempCelsius)}°C\n")
            }
            append("  性能评分: ${capabilities.benchmarkScore}/100")
        }
    }
}
