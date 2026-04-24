package com.hiringai.mobile.ml

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.FileReader
import java.io.RandomAccessFile

/**
 * 设备能力检测服务
 * 用于检测设备硬件能力，推荐适合的模型
 */
class DeviceCapabilityDetector(private val context: Context) {

    data class DeviceCapabilities(
        val cpuCores: Int,
        val cpuArchitecture: String,
        val totalRAM: Long,          // GB
        val availableRAM: Long,      // GB
        val hasVulkan: Boolean,
        val hasOpenGLES3: Boolean,
        val gpuName: String,
        val is64Bit: Boolean,
        val benchmarkScore: Int      // 0-100 相对性能评分
    )

    data class ModelRecommendation(
        val modelName: String,
        val isRecommended: Boolean,
        val requiredRAM: Int,
        val reason: String
    )

    companion object {
        private const val TAG = "DeviceCapability"

        // 推荐模型的内存需求阈值
        private const val MIN_RAM_FOR_LARGE = 6  // GB
        private const val MIN_RAM_FOR_MEDIUM = 4 // GB
        private const val MIN_RAM_FOR_SMALL = 2  // GB
    }

    suspend fun detectCapabilities(): DeviceCapabilities = withContext(Dispatchers.IO) {
        val cpuCores = Runtime.getRuntime().availableProcessors()
        val cpuArch = getCPUArchitecture()
        val totalRAM = getTotalRAM() / (1024 * 1024 * 1024) // Convert to GB
        val availableRAM = getAvailableRAM() / (1024 * 1024 * 1024)
        val (hasVulkan, hasOpenGLES3, gpuName) = detectGPU()
        val is64Bit = Build.SUPPORTED_ABIS.any { it.contains("64") }
        val benchmarkScore = runSimpleBenchmark()

        DeviceCapabilities(
            cpuCores = cpuCores,
            cpuArchitecture = cpuArch,
            totalRAM = totalRAM,
            availableRAM = availableRAM,
            hasVulkan = hasVulkan,
            hasOpenGLES3 = hasOpenGLES3,
            gpuName = gpuName,
            is64Bit = is64Bit,
            benchmarkScore = benchmarkScore
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

    private fun detectGPU(): Triple<Boolean, Boolean, String> {
        var hasVulkan = false
        var hasOpenGLES3 = false
        var gpuName = "Unknown"

        try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val configInfo = activityManager.deviceConfigurationInfo

            // Check OpenGL ES version
            if (configInfo.reqGlEsVersion >= 0x30000) {
                hasOpenGLES3 = true
            }

            // Try to read GPU info from /sys
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
                } catch (e: Exception) {
                    // Continue to next file
                }
            }

            if (gpuName == "Unknown") {
                gpuName = "${Build.MANUFACTURER} ${Build.MODEL}"
            }

            // Vulkan support - all Android 8.0+ devices support Vulkan
            hasVulkan = true

        } catch (e: Exception) {
            Log.e(TAG, "Error detecting GPU", e)
            gpuName = "${Build.MANUFACTURER} ${Build.MODEL}"
        }

        return Triple(hasVulkan, hasOpenGLES3, gpuName)
    }

    private fun runSimpleBenchmark(): Int {
        // Simple benchmark: compute-intensive task
        // Returns a score 0-100 based on device performance
        return try {
            val startTime = System.nanoTime()
            var result = 0.0

            // Perform some computation
            for (i in 1..100000) {
                result += kotlin.math.sqrt(i.toDouble())
                result *= 1.0001
                result -= 0.0001
            }

            val elapsed = (System.nanoTime() - startTime) / 1_000_000_000.0

            // Normalize to 0-100 score (faster = higher score)
            // Assume 2 seconds is baseline (score 50)
            val score = (100 / (elapsed / 2.0 + 0.5)).toInt().coerceIn(10, 100)
            score
        } catch (e: Exception) {
            50 // Default score if benchmark fails
        }
    }

    fun recommendModels(capabilities: DeviceCapabilities): List<ModelRecommendation> {
        val recommendations = mutableListOf<ModelRecommendation>()

        for (model in LocalLLMService.AVAILABLE_MODELS) {
            val (isRecommended, reason) = when {
                capabilities.availableRAM < model.requiredRAM -> {
                    Pair(false, "可用内存不足 (需要${model.requiredRAM}GB)")
                }
                capabilities.benchmarkScore < 30 -> {
                    Pair(false, "设备性能较低")
                }
                capabilities.totalRAM >= MIN_RAM_FOR_LARGE && capabilities.benchmarkScore >= 60 -> {
                    Pair(true, "设备性能充足")
                }
                capabilities.totalRAM >= MIN_RAM_FOR_MEDIUM -> {
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
            append("  CPU: ${capabilities.cpuCores}核 ${capabilities.cpuArchitecture}\n")
            append("  内存: ${capabilities.totalRAM}GB (可用: ${capabilities.availableRAM}GB)\n")
            append("  GPU: ${capabilities.gpuName}\n")
            append("  Vulkan: ${if (capabilities.hasVulkan) "✓" else "✗"}\n")
            append("  OpenGL ES 3.0: ${if (capabilities.hasOpenGLES3) "✓" else "✗"}\n")
            append("  64位: ${if (capabilities.is64Bit) "✓" else "✗"}\n")
            append("  性能评分: ${capabilities.benchmarkScore}/100")
        }
    }
}