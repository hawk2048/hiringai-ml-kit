package com.hiringai.mobile.ml.acceleration

import android.content.Context
import android.os.Build
import android.util.Log
import ai.onnxruntime.OrtSession
import java.util.Locale

/**
 * NNAPI 硬件加速管理器
 *
 * 负责 NNAPI 设备枚举、安全选择和会话配置。
 *
 * 安全策略：
 * 1. 黑名单机制过滤已知崩溃驱动（高通 DSP 在小米设备上 SIGILL）
 * 2. 运行时设备能力检测
 * 3. 自动降级到 CPU 以保证稳定性
 */
object NNAPIManager {

    private const val TAG = "NNAPIManager"

    // ========== 黑名单配置 ==========

    /**
     * 已知不稳定的高通 NNAPI 驱动名称
     *
     * 背景：小米/Redmi/POCO 设备上的 Qualcomm DSP 驱动 (qti-dsp, qti-hta)
     * 会导致原生层 SIGILL/SIGSEGV 崩溃，无法被 Java try-catch 捕获。
     * 参考：https://github.com/microsoft/onnxruntime/issues/15880
     */
    private val BLACKLISTED_DRIVERS = setOf(
        "qti-default",    // Qualcomm 默认驱动
        "qti-hta",        // Qualcomm HTA (Hexagon Tensor Accelerator)
        "qti-dsp",        // Qualcomm DSP - 已知在小米设备上崩溃
        "qti-gpu",        // Qualcomm GPU 驱动（部分设备不稳定）
        "libnpu",         // NPU 驱动（部分设备兼容性问题）
    )

    /**
     * 已知存在 NNAPI 稳定性问题的设备制造商
     *
     * 这些厂商的设备使用高通芯片时，NNAPI 驱动可能不稳定。
     * 列表会根据用户反馈持续更新。
     */
    private val BLACKLISTED_MANUFACTURERS = setOf(
        "xiaomi",
        "redmi",
        "poco",
        "meizu",
        "realme"
    )

    // ========== 设备信息缓存 ==========

    private var cachedDeviceInfo: CachedDeviceInfo? = null
    private var isNNAPISafeCache: Boolean? = null

    // ========== 数据类 ==========

    /**
     * NNAPI 设备信息
     */
    data class NNAPIDeviceInfo(
        val name: String,
        val driverName: String,
        val deviceType: DeviceType,
        val isAvailable: Boolean,
        val isBlacklisted: Boolean,
        val blacklistedReason: String? = null
    )

    /**
     * 设备类型枚举
     */
    enum class DeviceType {
        CPU,            // CPU 后端
        GPU,            // GPU 后端 (Adreno, Mali 等)
        DSP,            // 数字信号处理器 (Hexagon 等)
        NPU,            // 神经网络处理器
        ACCELERATOR,    // 专用加速器
        UNKNOWN         // 未知类型
    }

    /**
     * 加速后端选项
     */
    enum class AccelerationBackend {
        CPU,            // 纯 CPU 执行
        NNAPI_SAFE,     // 安全的 NNAPI 设备（经过黑名单过滤）
        NNAPI_ALL       // 所有 NNAPI 设备（高风险）
    }

    /**
     * 性能模式
     */
    enum class PerformanceMode {
        LOW_LATENCY,    // 低延迟优先
        THROUGHPUT,     // 吞吐量优先
        BALANCED        // 平衡模式
    }

    /**
     * 精度模式
     */
    enum class PrecisionMode {
        ACCURACY,       // 精度优先（FP32）
        PERFORMANCE,    // 性能优先（FP16/INT8）
        AUTO            // 自动选择
    }

    // ========== 公共 API ==========

    /**
     * 检查当前设备是否支持 NNAPI
     */
    fun isNNAPIAvailable(): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
    }

    /**
     * 检查当前设备是否可以安全使用 NNAPI
     *
     * 安全条件：
     * 1. Android 9.0+ (API 28+)
     * 2. 设备制造商不在黑名单中
     * 3. 至少有一个非黑名单驱动可用
     */
    fun isNNAPISafe(context: Context): Boolean {
        isNNAPISafeCache?.let { return it }

        val result = checkNNAPISafety(context)
        isNNAPISafeCache = result
        return result
    }

    /**
     * 获取所有可用的 NNAPI 设备列表
     */
    fun getAvailableDevices(context: Context): List<NNAPIDeviceInfo> {
        if (!isNNAPIAvailable()) {
            return emptyList()
        }

        if (cachedDeviceInfo != null) {
            return cachedDeviceInfo!!.devices
        }

        val devices = enumerateDevices(context)
        cachedDeviceInfo = CachedDeviceInfo(
            devices = devices,
            hasSafeDevice = devices.any { it.isAvailable && !it.isBlacklisted },
            recommendedBackend = selectRecommendedBackend(context, devices)
        )

        return devices
    }

    /**
     * 创建安全的 NNAPI 会话选项
     *
     * @param context Android Context
     * @param performanceMode 性能模式
     * @param precisionMode 精度模式
     * @return 配置好的 OrtSession.SessionOptions，如果 NNAPI 不安全则返回 null
     */
    fun createSafeSessionOptions(
        context: Context,
        performanceMode: PerformanceMode = PerformanceMode.BALANCED,
        precisionMode: PrecisionMode = PrecisionMode.AUTO
    ): OrtSession.SessionOptions? {
        if (!isNNAPISafe(context)) {
            Log.i(TAG, "NNAPI not safe for this device, falling back to CPU")
            return createCPUOnlySessionOptions(performanceMode)
        }

        return try {
            val options = OrtSession.SessionOptions()

            // 设置优化级别
            options.setOptimizationLevel(
                when (performanceMode) {
                    PerformanceMode.LOW_LATENCY -> OrtSession.SessionOptions.OptLevel.ALL_OPT
                    PerformanceMode.THROUGHPUT -> OrtSession.SessionOptions.OptLevel.ALL_OPT
                    PerformanceMode.BALANCED -> OrtSession.SessionOptions.OptLevel.ALL_OPT
                }
            )

            // NNAPI 执行提供程序配置
            // 注意：实际 NNAPI 集成需要 ONNX Runtime 的 NNAPI EP
            // 这里返回基础配置，由调用者根据需要添加 NNAPI EP
            configureNNAPIOptions(options, performanceMode, precisionMode)

            Log.i(TAG, "Created safe NNAPI session options with $performanceMode mode")
            options
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create NNAPI session options", e)
            null
        }
    }

    /**
     * 创建纯 CPU 会话选项
     */
    fun createCPUOnlySessionOptions(
        performanceMode: PerformanceMode = PerformanceMode.BALANCED
    ): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()

        options.setOptimizationLevel(
            when (performanceMode) {
                PerformanceMode.LOW_LATENCY -> OrtSession.SessionOptions.OptLevel.ALL_OPT
                PerformanceMode.THROUGHPUT -> OrtSession.SessionOptions.OptLevel.ALL_OPT
                PerformanceMode.BALANCED -> OrtSession.SessionOptions.OptLevel.ALL_OPT
            }
        )

        // 启用所有 CPU 优化
        options.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
        options.setInterOpNumThreads(Runtime.getRuntime().availableProcessors())

        Log.i(TAG, "Created CPU-only session options")
        return options
    }

    /**
     * 获取推荐的加速后端
     */
    fun getRecommendedBackend(context: Context): AccelerationBackend {
        val deviceInfo = cachedDeviceInfo ?: getAvailableDevices(context).let {
            CachedDeviceInfo(
                devices = it,
                hasSafeDevice = it.any { d -> d.isAvailable && !d.isBlacklisted },
                recommendedBackend = selectRecommendedBackend(context, it)
            )
        }

        return deviceInfo.recommendedBackend
    }

    /**
     * 清除缓存（用于测试或配置更新）
     */
    fun clearCache() {
        cachedDeviceInfo = null
        isNNAPISafeCache = null
    }

    // ========== 内部实现 ==========

    /**
     * 缓存的 NNAPI 系统信息
     */
    private data class CachedDeviceInfo(
        val devices: List<NNAPIDeviceInfo>,
        val hasSafeDevice: Boolean,
        val recommendedBackend: AccelerationBackend
    )

    private fun checkNNAPISafety(context: Context): Boolean {
        // 检查 Android 版本
        if (!isNNAPIAvailable()) {
            Log.i(TAG, "NNAPI not available: Android version < 9.0")
            return false
        }

        // 检查设备制造商
        val manufacturer = Build.MANUFACTURER.lowercase(Locale.ROOT)
        if (BLACKLISTED_MANUFACTURERS.contains(manufacturer)) {
            Log.w(TAG, "Device manufacturer '$manufacturer' is blacklisted for NNAPI stability")
            return false
        }

        // 检查是否有安全的 NNAPI 设备
        val devices = enumerateDevices(context)
        val safeDevices = devices.filter { it.isAvailable && !it.isBlacklisted }

        if (safeDevices.isEmpty()) {
            Log.w(TAG, "No safe NNAPI devices available")
            return false
        }

        Log.i(TAG, "NNAPI is safe: ${safeDevices.size} safe device(s) available")
        return true
    }

    private fun enumerateDevices(context: Context): List<NNAPIDeviceInfo> {
        val devices = mutableListOf<NNAPIDeviceInfo>()

        if (!isNNAPIAvailable()) {
            return devices
        }

        try {
            // Android NNAPI 设备枚举
            // 实际实现需要使用 NnApiDelegate 或反射访问 NNAPI API
            // 这里提供基于已知设备的模拟实现

            // 常见的 NNAPI 设备类型
            val knownDeviceTypes = mapOf(
                "cpu" to DeviceType.CPU,
                "gpu" to DeviceType.GPU,
                "dsp" to DeviceType.DSP,
                "npu" to DeviceType.NPU,
                "nnapi-reference" to DeviceType.ACCELERATOR
            )

            // 模拟设备枚举（实际实现需要 NnApiDelegate.getDevices()）
            // ONNX Runtime 通过 NnApiDelegate 访问 NNAPI
            val simulatedDevices = getSimulatedDevices()

            for ((name, type) in simulatedDevices) {
                val driverName = extractDriverName(name)
                val isBlacklisted = isDriverBlacklisted(driverName)
                val blacklistedReason = if (isBlacklisted) {
                    getBlacklistReason(driverName)
                } else null

                devices.add(
                    NNAPIDeviceInfo(
                        name = name,
                        driverName = driverName,
                        deviceType = type,
                        isAvailable = true,
                        isBlacklisted = isBlacklisted,
                        blacklistedReason = blacklistedReason
                    )
                )
            }

            // 添加默认 CPU 设备
            devices.add(
                NNAPIDeviceInfo(
                    name = "CPU (fallback)",
                    driverName = "cpu",
                    deviceType = DeviceType.CPU,
                    isAvailable = true,
                    isBlacklisted = false
                )
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to enumerate NNAPI devices", e)
        }

        return devices
    }

    /**
     * 获取模拟的设备列表
     *
     * 实际实现应使用 NnApiDelegate 或 Android NDK NNAPI API
     */
    private fun getSimulatedDevices(): Map<String, DeviceType> {
        val devices = mutableMapOf<String, DeviceType>()

        // 基于 SoC 推测可用设备
        val socInfo = getSocInfo()

        when {
            // 高通 Snapdragon
            socInfo.contains("snapdragon", ignoreCase = true) ||
            socInfo.contains("qcom", ignoreCase = true) -> {
                devices["qti-default"] = DeviceType.ACCELERATOR
                devices["qti-dsp"] = DeviceType.DSP
                devices["qti-gpu"] = DeviceType.GPU
                devices["qti-hta"] = DeviceType.NPU
            }

            // 联发科 MediaTek
            socInfo.contains("mt", ignoreCase = true) ||
            socInfo.contains("mediatek", ignoreCase = true) -> {
                devices["mtk-mdla"] = DeviceType.NPU
                devices["mtk-vpu"] = DeviceType.DSP
            }

            // 三星 Exynos
            socInfo.contains("exynos", ignoreCase = true) -> {
                devices["exynos-npu"] = DeviceType.NPU
                devices["exynos-gpu"] = DeviceType.GPU
            }

            // 华为 Kirin
            socInfo.contains("kirin", ignoreCase = true) ||
            socInfo.contains("hisilicon", ignoreCase = true) -> {
                devices["kirin-npu"] = DeviceType.NPU
            }

            // 其他 ARM 设备
            else -> {
                devices["nnapi-reference"] = DeviceType.CPU
            }
        }

        return devices
    }

    /**
     * 获取 SoC 信息
     */
    private fun getSocInfo(): String {
        return try {
            val hw = Build.HARDWARE.lowercase(Locale.ROOT)
            val board = Build.BOARD.lowercase(Locale.ROOT)
            val device = Build.DEVICE.lowercase(Locale.ROOT)
            "$hw $board $device"
        } catch (e: Exception) {
            ""
        }
    }

    private fun extractDriverName(deviceName: String): String {
        return deviceName.lowercase(Locale.ROOT)
            .replace(Regex("[^a-z0-9-]"), "-")
            .split("-").firstOrNull() ?: deviceName
    }

    private fun isDriverBlacklisted(driverName: String): Boolean {
        val normalized = driverName.lowercase(Locale.ROOT)
        return BLACKLISTED_DRIVERS.any { blacklisted ->
            normalized.contains(blacklisted.lowercase(Locale.ROOT))
        }
    }

    private fun getBlacklistReason(driverName: String): String {
        return when {
            driverName.contains("qti-dsp") || driverName.contains("qti-hta") ->
                "Known SIGILL/SIGSEGV crash on Xiaomi devices"
            driverName.contains("qti") ->
                "Qualcomm NNAPI driver stability issues reported"
            else ->
                "Driver in stability blacklist"
        }
    }

    private fun selectRecommendedBackend(
        context: Context,
        devices: List<NNAPIDeviceInfo>
    ): AccelerationBackend {
        val safeDevices = devices.filter { it.isAvailable && !it.isBlacklisted }

        return when {
            safeDevices.isEmpty() -> {
                Log.i(TAG, "No safe NNAPI devices, recommending CPU backend")
                AccelerationBackend.CPU
            }
            safeDevices.any { it.deviceType == DeviceType.NPU } -> {
                Log.i(TAG, "NPU available and safe, recommending NNAPI")
                AccelerationBackend.NNAPI_SAFE
            }
            safeDevices.any { it.deviceType == DeviceType.GPU } -> {
                Log.i(TAG, "GPU available and safe, recommending NNAPI")
                AccelerationBackend.NNAPI_SAFE
            }
            else -> {
                Log.i(TAG, "Safe NNAPI devices available, recommending NNAPI_SAFE")
                AccelerationBackend.NNAPI_SAFE
            }
        }
    }

    private fun configureNNAPIOptions(
        options: OrtSession.SessionOptions,
        performanceMode: PerformanceMode,
        precisionMode: PrecisionMode
    ) {
        // NNAPI 特定配置
        // 实际实现需要通过 ONNX Runtime 的 NNAPI Execution Provider

        when (performanceMode) {
            PerformanceMode.LOW_LATENCY -> {
                // 低延迟配置
                options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            }
            PerformanceMode.THROUGHPUT -> {
                // 吞吐量配置
                options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
            }
            PerformanceMode.BALANCED -> {
                // 平衡配置
                options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            }
        }

        // 线程数配置
        val numThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        options.setIntraOpNumThreads(numThreads)
    }
}
