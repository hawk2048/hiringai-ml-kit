package com.hiringai.mobile.ml.acceleration

import android.content.Context
import android.content.SharedPreferences
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * 硬件加速配置
 *
 * 定义推理加速的后端类型、回退链和每模型加速偏好
 * 支持序列化持久化到 SharedPreferences
 */
@Serializable
data class AccelerationConfig(
    /** 首选加速后端 */
    val preferredBackend: Backend = Backend.CPU,
    /** 回退链：按优先级尝试的后端列表 */
    val fallbackChain: List<Backend> = listOf(Backend.CPU),
    /** 每模型加速偏好覆盖 (modelName -> config) */
    val modelOverrides: Map<String, ModelAccelerationConfig> = emptyMap(),
    /** 全局设置 */
    val globalSettings: GlobalAccelerationSettings = GlobalAccelerationSettings()
) {
    /**
     * 加速后端枚举
     */
    @Serializable
    enum class Backend(val displayName: String, val priority: Int) {
        /** CPU 推理 (最兼容) */
        CPU("CPU", 0),
        /** XNNPACK 优化的 CPU 推理 */
        XNNPACK("XNNPACK (优化CPU)", 1),
        /** GPU 加速 (OpenCL/Vulkan) */
        GPU("GPU", 2),
        /** Android Neural Networks API (NPU/DSP 加速) */
        NNAPI("NNAPI (NPU/DSP)", 3);

        companion object {
            fun fromName(name: String): Backend =
                entries.find { it.name.equals(name, ignoreCase = true) } ?: CPU
        }
    }

    /**
     * 单模型加速配置
     */
    @Serializable
    data class ModelAccelerationConfig(
        val backend: Backend? = null,
        val gpuLayers: Int = 0,
        val threads: Int? = null,
        val batchSize: Int? = null,
        val enabled: Boolean = true
    )

    /**
     * 全局加速设置
     */
    @Serializable
    data class GlobalAccelerationSettings(
        /** 自动检测最佳后端 */
        val autoDetect: Boolean = true,
        /** 内存不足时自动切换到 CPU */
        val autoFallbackOnLowMemory: Boolean = true,
        /** 低内存阈值 (MB) */
        val lowMemoryThresholdMB: Int = 512,
        /** 是否允许 GPU 分配内存 */
        val allowGpuMemoryAllocation: Boolean = true,
        /** GPU 层数 (用于 llama.cpp) */
        val defaultGpuLayers: Int = 0,
        /** 默认线程数 */
        val defaultThreads: Int = 4,
        /** NNAPI 加速开关 */
        val nnapiEnabled: Boolean = true,
        /** XNNPACK 开关 */
        val xnnpackEnabled: Boolean = true
    )

    /**
     * 获取指定模型的加速配置
     */
    fun getModelConfig(modelName: String): ModelAccelerationConfig {
        return modelOverrides[modelName] ?: ModelAccelerationConfig(
            backend = preferredBackend,
            gpuLayers = globalSettings.defaultGpuLayers,
            threads = globalSettings.defaultThreads
        )
    }

    /**
     * 获取有效后端链 (考虑禁用的后端)
     */
    fun getEffectiveFallbackChain(): List<Backend> {
        val disabledBackends = mutableSetOf<Backend>()
        if (!globalSettings.nnapiEnabled) disabledBackends.add(Backend.NNAPI)
        if (!globalSettings.xnnpackEnabled) disabledBackends.add(Backend.XNNPACK)

        return fallbackChain.filter { it !in disabledBackends }
    }

    companion object {
        private const val PREFS_NAME = "acceleration_config"
        private const val KEY_CONFIG = "config_json"

        private val json = Json {
            ignoreUnknownKeys = true
            encodeDefaults = true
            isLenient = true
        }

        /**
         * 默认配置
         */
        val DEFAULT = AccelerationConfig(
            preferredBackend = Backend.CPU,
            fallbackChain = listOf(Backend.CPU),
            globalSettings = GlobalAccelerationSettings()
        )

        /**
         * 高性能配置 (有 NPU/GPU)
         */
        val HIGH_PERFORMANCE = AccelerationConfig(
            preferredBackend = Backend.NNAPI,
            fallbackChain = listOf(Backend.NNAPI, Backend.GPU, Backend.XNNPACK, Backend.CPU),
            globalSettings = GlobalAccelerationSettings(
                autoDetect = true,
                defaultGpuLayers = 32,
                defaultThreads = 8
            )
        )

        /**
         * 省电配置
         */
        val POWER_SAVING = AccelerationConfig(
            preferredBackend = Backend.CPU,
            fallbackChain = listOf(Backend.CPU),
            globalSettings = GlobalAccelerationSettings(
                autoDetect = false,
                defaultGpuLayers = 0,
                defaultThreads = 2
            )
        )

        /**
         * 从 SharedPreferences 加载配置
         */
        fun load(context: Context): AccelerationConfig {
            return try {
                val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                val jsonStr = prefs.getString(KEY_CONFIG, null)
                if (jsonStr != null) {
                    json.decodeFromString<AccelerationConfig>(jsonStr)
                } else {
                    DEFAULT
                }
            } catch (e: Exception) {
                android.util.Log.e("AccelerationConfig", "Failed to load config", e)
                DEFAULT
            }
        }

        /**
         * 保存配置到 SharedPreferences
         */
        fun save(context: Context, config: AccelerationConfig) {
            try {
                val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                val jsonStr = json.encodeToString(AccelerationConfig.serializer(), config)
                prefs.edit().putString(KEY_CONFIG, jsonStr).apply()
            } catch (e: Exception) {
                android.util.Log.e("AccelerationConfig", "Failed to save config", e)
            }
        }

        /**
         * 重置为默认配置
         */
        fun reset(context: Context) {
            val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().clear().apply()
        }
    }
}
