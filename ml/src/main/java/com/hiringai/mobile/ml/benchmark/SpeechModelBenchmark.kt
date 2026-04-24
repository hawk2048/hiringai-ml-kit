package com.hiringai.mobile.ml.benchmark

import android.content.Context
import android.util.Log
import com.hiringai.mobile.ml.speech.LocalSpeechService
import com.hiringai.mobile.ml.speech.SpeechModelConfig
import com.hiringai.mobile.ml.speech.SpeechModelType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/**
 * 语音模型基准测试结果
 */
data class SpeechModelBenchmarkResult(
    val modelName: String,
    val modelType: SpeechModelType,
    val loadTimeMs: Long,
    val inferenceTimeMs: Long,
    val memoryMB: Long,
    val throughput: Float, // RTF (Real-Time Factor) or inf/s
    val success: Boolean,
    val errorMessage: String? = null,
    val additionalMetrics: Map<String, Any> = emptyMap()
) {
    fun toSummary(): String = buildString {
        val typeStr = when (modelType) {
            SpeechModelType.WHISPER -> "Whisper"
            SpeechModelType.PARAFORMER -> "Paraformer"
            SpeechModelType.CAM_PLUS -> "Cam++ VAD"
            SpeechModelType.TTS -> "TTS"
            SpeechModelType.VAD -> "VAD"
        }
        appendLine("📊 $modelName ($typeStr)")
        appendLine("   加载时间: ${loadTimeMs}ms")
        appendLine("   推理延迟: ${inferenceTimeMs}ms")
        appendLine("   内存占用: ${memoryMB}MB")
        appendLine("   吞吐量: %.2fx RTF".format(throughput))
        appendLine("   状态: ${if (success) "✅ 成功" else "❌ 失败: $errorMessage"}")
    }
}

/**
 * 批量语音模型基准测试报告
 */
data class SpeechBatchReport(
    val results: List<SpeechModelBenchmarkResult>,
    val deviceInfo: String,
    val totalDurationMs: Long,
    val timestamp: Long = System.currentTimeMillis()
) {
    fun toExportText(): String = buildString {
        appendLine("=".repeat(60))
        appendLine("语音模型基准测试报告")
        appendLine("=".repeat(60))
        appendLine("设备: $deviceInfo")
        appendLine("测试时间: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(java.util.Date(timestamp))}")
        appendLine("总耗时: ${totalDurationMs}ms")
        appendLine("测试模型数: ${results.size}")
        appendLine("成功: ${results.count { it.success }}")
        appendLine("失败: ${results.count { !it.success }}")
        appendLine()

        // 按类型分组
        val grouped = results.groupBy { it.modelType }
        grouped.forEach { (type, typeResults) ->
            val typeStr = when (type) {
                SpeechModelType.WHISPER -> "Whisper (语音识别)"
                SpeechModelType.PARAFORMER -> "Paraformer (中文语音识别)"
                SpeechModelType.CAM_PLUS -> "Cam++ VAD (语音活动检测)"
                SpeechModelType.TTS -> "TTS (语音合成)"
                SpeechModelType.VAD -> "VAD (语音活动检测)"
            }
            appendLine("【$typeStr】")
            appendLine("-".repeat(40))
            typeResults.forEach { appendLine(it.toSummary()) }
            appendLine()
        }

        appendLine("=".repeat(60))
    }
}

/**
 * 语音模型基准测试运行器
 * 支持 Whisper, Paraformer, Cam++ 等模型
 */
class SpeechModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "SpeechModelBenchmark"

        // 测试音频数据生成
        fun generateTestAudio(durationSeconds: Float, sampleRate: Int = 16000): FloatArray {
            val numSamples = (durationSeconds * sampleRate).toInt()
            return FloatArray(numSamples) { i ->
                // 生成简单的正弦波测试音频 (440Hz A音符)
                val t = i.toFloat() / sampleRate
                (Math.sin(2 * Math.PI * 440 * t) * 0.5).toFloat()
            }
        }
    }

    private val speechService = LocalSpeechService.getInstance(context)

    /**
     * 获取设备信息
     */
    fun getDeviceInfo(): String {
        val deviceName = android.os.Build.MODEL
        val manufacturer = android.os.Build.MANUFACTURER
        val sdkVersion = android.os.Build.VERSION.SDK_INT

        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val totalRamGB = memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)

        val cpuCores = Runtime.getRuntime().availableProcessors()

        return "$manufacturer $deviceName (SDK $sdkVersion, ${cpuCores}核, ${"%.1f".format(totalRamGB)}GB RAM)"
    }

    /**
     * 获取当前内存使用 (MB)
     */
    private fun getCurrentMemoryMB(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
    }

    /**
     * 运行单个语音模型基准测试
     */
    suspend fun benchmarkModel(
        config: SpeechModelConfig,
        testDurationSeconds: Float = 5.0f
    ): SpeechModelBenchmarkResult = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        val initialMemory = getCurrentMemoryMB()

        try {
            // 检查模型是否已下载
            if (!speechService.isModelDownloaded(config.name)) {
                return@withContext SpeechModelBenchmarkResult(
                    modelName = config.name,
                    modelType = config.type,
                    loadTimeMs = 0,
                    inferenceTimeMs = 0,
                    memoryMB = 0,
                    throughput = 0f,
                    success = false,
                    errorMessage = "模型未下载"
                )
            }

            // 加载模型并计时
            val loadStartTime = System.currentTimeMillis()
            val loadSuccess = speechService.loadModel(config)
            val loadTimeMs = System.currentTimeMillis() - loadStartTime

            if (!loadSuccess) {
                return@withContext SpeechModelBenchmarkResult(
                    modelName = config.name,
                    modelType = config.type,
                    loadTimeMs = loadTimeMs,
                    inferenceTimeMs = 0,
                    memoryMB = getCurrentMemoryMB() - initialMemory,
                    throughput = 0f,
                    success = false,
                    errorMessage = "模型加载失败"
                )
            }

            // 生成测试音频
            val testAudio = generateTestAudio(testDurationSeconds, config.sampleRate)

            // 运行推理并计时
            val inferenceStartTime = System.currentTimeMillis()

            when (config.type) {
                SpeechModelType.WHISPER, SpeechModelType.PARAFORMER -> {
                    speechService.transcribe(testAudio, config.sampleRate)
                }
                SpeechModelType.CAM_PLUS, SpeechModelType.VAD -> {
                    speechService.detectVoiceActivity(testAudio, config.sampleRate)
                }
                SpeechModelType.TTS -> {
                    speechService.synthesize("测试语音合成")
                }
            }

            val inferenceTimeMs = System.currentTimeMillis() - inferenceStartTime

            // 计算 RTF (Real-Time Factor)
            val rtf = if (testDurationSeconds > 0) {
                inferenceTimeMs / 1000f / testDurationSeconds
            } else 0f

            // 卸载模型
            speechService.unloadModel()

            val memoryMB = getCurrentMemoryMB() - initialMemory

            Log.d(TAG, "Benchmark completed for ${config.name}: RTF=$rtf")

            SpeechModelBenchmarkResult(
                modelName = config.name,
                modelType = config.type,
                loadTimeMs = loadTimeMs,
                inferenceTimeMs = inferenceTimeMs,
                memoryMB = memoryMB,
                throughput = if (rtf > 0) 1f / rtf else 0f, // 1/RTF = 实时倍数
                success = true,
                additionalMetrics = mapOf(
                    "audioDuration" to testDurationSeconds,
                    "rtf" to rtf
                )
            )
        } catch (e: Exception) {
            Log.e(TAG, "Benchmark failed for ${config.name}", e)
            speechService.unloadModel()

            SpeechModelBenchmarkResult(
                modelName = config.name,
                modelType = config.type,
                loadTimeMs = System.currentTimeMillis() - startTime,
                inferenceTimeMs = 0,
                memoryMB = getCurrentMemoryMB() - initialMemory,
                throughput = 0f,
                success = false,
                errorMessage = e.message
            )
        }
    }

    /**
     * 批量测试多个语音模型
     */
    suspend fun runBatchBenchmark(
        models: List<SpeechModelConfig> = LocalSpeechService.AVAILABLE_MODELS,
        testDurationSeconds: Float = 5.0f,
        onProgress: (Int, SpeechModelBenchmarkResult?) -> Unit = { _, _ -> }
    ): SpeechBatchReport = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        val results = mutableListOf<SpeechModelBenchmarkResult>()

        models.forEachIndexed { index, config ->
            onProgress((index * 100 / models.size), null)

            val result = benchmarkModel(config, testDurationSeconds)
            results.add(result)

            onProgress(((index + 1) * 100 / models.size), result)
        }

        val totalDuration = System.currentTimeMillis() - startTime

        SpeechBatchReport(
            results = results,
            deviceInfo = getDeviceInfo(),
            totalDurationMs = totalDuration
        )
    }

    /**
     * 仅测试 Whisper 模型
     */
    suspend fun benchmarkWhisperModels(
        testDurationSeconds: Float = 5.0f,
        onProgress: (Int, SpeechModelBenchmarkResult?) -> Unit = { _, _ -> }
    ): SpeechBatchReport {
        val whisperModels = LocalSpeechService.AVAILABLE_MODELS
            .filter { it.type == SpeechModelType.WHISPER }
        return runBatchBenchmark(whisperModels, testDurationSeconds, onProgress)
    }

    /**
     * 仅测试 Paraformer 模型
     */
    suspend fun benchmarkParaformerModels(
        testDurationSeconds: Float = 5.0f,
        onProgress: (Int, SpeechModelBenchmarkResult?) -> Unit = { _, _ -> }
    ): SpeechBatchReport {
        val paraformerModels = LocalSpeechService.AVAILABLE_MODELS
            .filter { it.type == SpeechModelType.PARAFORMER }
        return runBatchBenchmark(paraformerModels, testDurationSeconds, onProgress)
    }

    /**
     * 仅测试 Cam++ VAD 模型
     */
    suspend fun benchmarkVADModels(
        testDurationSeconds: Float = 5.0f,
        onProgress: (Int, SpeechModelBenchmarkResult?) -> Unit = { _, _ -> }
    ): SpeechBatchReport {
        val vadModels = LocalSpeechService.AVAILABLE_MODELS
            .filter { it.type == SpeechModelType.CAM_PLUS || it.type == SpeechModelType.VAD }
        return runBatchBenchmark(vadModels, testDurationSeconds, onProgress)
    }
}
