package com.hiringai.mobile.ml.benchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import com.hiringai.mobile.ml.LocalLLMService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

/**
 * LLM 基准测试运行器
 */
class LLMBenchmarkRunner(private val context: Context) {

    companion object {
        private const val TAG = "LLMBenchmarkRunner"

        // 标准测试 prompt
        val TEST_PROMPTS = listOf(
            "请用一句话介绍你自己",
            "什么是人工智能？",
            "用一句话总结机器学习的原理",
            "请解释什么是深度学习",
            "用简单的语言解释神经网络"
        )

        // 默认测试参数
        const val DEFAULT_MAX_TOKENS = 128
        const val DEFAULT_TEMPERATURE = 0.7f
    }

    private val llmService = LocalLLMService.getInstance(context)
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    /**
     * 获取设备信息
     */
    fun getDeviceInfo(): String {
        val deviceName = Build.MODEL
        val manufacturer = Build.MANUFACTURER
        val sdkVersion = Build.VERSION.SDK_INT

        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val totalRamGB = memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)

        val cpuCores = Runtime.getRuntime().availableProcessors()

        return "$manufacturer $deviceName (SDK $sdkVersion, ${cpuCores}核, ${"%.1f".format(totalRamGB)}GB RAM)"
    }

    /**
     * 获取当前内存使用 (MB)
     */
    private fun getCurrentMemoryUsageMB(): Long {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val usedMemory = memInfo.totalMem - memInfo.availMem
        return usedMemory / (1024 * 1024)
    }

    /**
     * 运行单个模型基准测试
     */
    suspend fun benchmarkModel(
        config: LocalLLMService.ModelConfig,
        testPrompt: String = TEST_PROMPTS.first(),
        maxTokens: Int = DEFAULT_MAX_TOKENS,
        temperature: Float = DEFAULT_TEMPERATURE
    ): LLMBenchmarkResult = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        val initialMemory = getCurrentMemoryUsageMB()

        try {
            // 检查模型是否已下载
            if (!llmService.isModelDownloaded(config.name)) {
                return@withContext LLMBenchmarkResult(
                    modelName = config.name,
                    loadTimeMs = 0,
                    firstTokenLatencyMs = 0,
                    avgTokenLatencyMs = 0,
                    totalTokens = 0,
                    throughputTokensPerSec = 0.0,
                    memoryUsageMB = 0,
                    peakMemoryMB = 0,
                    testPrompt = testPrompt,
                    generatedText = "",
                    success = false,
                    errorMessage = "Model not downloaded"
                )
            }

            // 记录加载前的内存
            val memBeforeLoad = getCurrentMemoryUsageMB()

            // 加载模型并计时
            val loadStartTime = System.currentTimeMillis()
            val loadSuccess = llmService.loadModel(config)
            val loadTimeMs = System.currentTimeMillis() - loadStartTime

            if (!loadSuccess) {
                return@withContext LLMBenchmarkResult(
                    modelName = config.name,
                    loadTimeMs = loadTimeMs,
                    firstTokenLatencyMs = 0,
                    avgTokenLatencyMs = 0,
                    totalTokens = 0,
                    throughputTokensPerSec = 0.0,
                    memoryUsageMB = memBeforeLoad,
                    peakMemoryMB = getCurrentMemoryUsageMB(),
                    testPrompt = testPrompt,
                    generatedText = "",
                    success = false,
                    errorMessage = "Failed to load model"
                )
            }

            // 加载后内存
            val memAfterLoad = getCurrentMemoryUsageMB()
            val memoryUsageMB = memAfterLoad - memBeforeLoad

            // 生成并计时
            val generateStartTime = System.currentTimeMillis()
            val generatedText = llmService.generate(testPrompt, maxTokens, temperature) ?: ""
            val generateTimeMs = System.currentTimeMillis() - generateStartTime

            // 计算 token 数量 (粗略估计: 平均 1.5 字符 = 1 token)
            val totalTokens = (generatedText.length / 1.5).toInt()
            val throughput = if (generateTimeMs > 0) {
                (totalTokens.toDouble() / generateTimeMs) * 1000.0
            } else 0.0

            // 卸载模型
            llmService.unloadModel()

            // 最终内存
            val peakMemory = getCurrentMemoryUsageMB()

            Log.d(TAG, "Benchmark completed for ${config.name}: $totalTokens tokens in ${generateTimeMs}ms")

            LLMBenchmarkResult(
                modelName = config.name,
                loadTimeMs = loadTimeMs,
                firstTokenLatencyMs = (loadTimeMs * 0.1).toLong(), // 估算首 token 延迟
                avgTokenLatencyMs = if (totalTokens > 0) (generateTimeMs.toDouble() / totalTokens).toLong() else 0,
                totalTokens = totalTokens,
                throughputTokensPerSec = throughput,
                memoryUsageMB = memoryUsageMB,
                peakMemoryMB = peakMemory,
                testPrompt = testPrompt,
                generatedText = generatedText,
                success = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "Benchmark failed for ${config.name}", e)
            llmService.unloadModel()

            LLMBenchmarkResult(
                modelName = config.name,
                loadTimeMs = System.currentTimeMillis() - startTime,
                firstTokenLatencyMs = 0,
                avgTokenLatencyMs = 0,
                totalTokens = 0,
                throughputTokensPerSec = 0.0,
                memoryUsageMB = 0,
                peakMemoryMB = getCurrentMemoryUsageMB(),
                testPrompt = testPrompt,
                generatedText = "",
                success = false,
                errorMessage = e.message
            )
        }
    }

    /**
     * 批量测试多个模型
     */
    fun runBatchBenchmark(
        models: List<LocalLLMService.ModelConfig>,
        testPrompt: String = TEST_PROMPTS.first(),
        maxTokens: Int = DEFAULT_MAX_TOKENS
    ): Flow<BenchmarkProgress> = flow {
        val startTime = System.currentTimeMillis()
        val results = mutableListOf<LLMBenchmarkResult>()

        emit(BenchmarkProgress(0, models.size, null, BenchmarkState.LOADING))

        models.forEachIndexed { index, config ->
            emit(BenchmarkProgress(index, models.size, config, BenchmarkState.LOADING))

            val result = benchmarkModel(config, testPrompt, maxTokens)
            results.add(result)

            emit(BenchmarkProgress(index + 1, models.size, config, if (result.success) BenchmarkState.COMPLETED else BenchmarkState.FAILED))
        }

        val totalDuration = System.currentTimeMillis() - startTime
        val report = BatchBenchmarkReport(
            results = results,
            deviceInfo = getDeviceInfo(),
            totalDurationMs = totalDuration
        )

        emit(BenchmarkProgress(models.size, models.size, null, BenchmarkState.FINISHED, report))
    }

    /**
     * 快速测试已加载模型
     */
    suspend fun quickBenchmark(
        testPrompt: String = TEST_PROMPTS.first(),
        maxTokens: Int = DEFAULT_MAX_TOKENS
    ): LLMBenchmarkResult = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        val initialMemory = getCurrentMemoryUsageMB()

        if (!llmService.isModelLoaded) {
            return@withContext LLMBenchmarkResult(
                modelName = llmService.getLoadedModelName(),
                loadTimeMs = 0,
                firstTokenLatencyMs = 0,
                avgTokenLatencyMs = 0,
                totalTokens = 0,
                throughputTokensPerSec = 0.0,
                memoryUsageMB = 0,
                peakMemoryMB = 0,
                testPrompt = testPrompt,
                generatedText = "",
                success = false,
                errorMessage = "No model loaded"
            )
        }

        val modelName = llmService.getLoadedModelName()

        try {
            val generateStartTime = System.currentTimeMillis()
            val generatedText = llmService.generate(testPrompt, maxTokens) ?: ""
            val generateTimeMs = System.currentTimeMillis() - generateStartTime

            val totalTokens = (generatedText.length / 1.5).toInt()
            val throughput = if (generateTimeMs > 0) {
                (totalTokens.toDouble() / generateTimeMs) * 1000.0
            } else 0.0

            LLMBenchmarkResult(
                modelName = modelName,
                loadTimeMs = 0,
                firstTokenLatencyMs = 0,
                avgTokenLatencyMs = if (totalTokens > 0) (generateTimeMs.toDouble() / totalTokens).toLong() else 0,
                totalTokens = totalTokens,
                throughputTokensPerSec = throughput,
                memoryUsageMB = 0,
                peakMemoryMB = getCurrentMemoryUsageMB() - initialMemory,
                testPrompt = testPrompt,
                generatedText = generatedText,
                success = true
            )
        } catch (e: Exception) {
            LLMBenchmarkResult(
                modelName = modelName,
                loadTimeMs = 0,
                firstTokenLatencyMs = 0,
                avgTokenLatencyMs = 0,
                totalTokens = 0,
                throughputTokensPerSec = 0.0,
                memoryUsageMB = 0,
                peakMemoryMB = 0,
                testPrompt = testPrompt,
                generatedText = "",
                success = false,
                errorMessage = e.message
            )
        }
    }
}

/**
 * 基准测试进度状态
 */
enum class BenchmarkState {
    LOADING,
    RUNNING,
    COMPLETED,
    FAILED,
    FINISHED
}

/**
 * 基准测试进度
 */
data class BenchmarkProgress(
    val currentIndex: Int,
    val totalCount: Int,
    val currentModel: LocalLLMService.ModelConfig?,
    val state: BenchmarkState,
    val report: BatchBenchmarkReport? = null
) {
    val progressPercent: Int
        get() = if (totalCount > 0) (currentIndex * 100 / totalCount) else 0
}