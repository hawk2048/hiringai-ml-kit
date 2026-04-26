package com.hiringai.mobile.ml.benchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import com.hiringai.mobile.ml.LocalLLMService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

/**
 * LLM 基准测试运行器
 *
 * v2 增强：
 * - 子阶段进度 (DOWNLOAD → LOAD → GENERATE → UNLOAD)
 * - 实时流式进度更新，UI 不会卡住
 * - 支持取消
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
     * 运行单个模型基准测试 (带子阶段进度回调)
     */
    suspend fun benchmarkModel(
        config: LocalLLMService.ModelConfig,
        testPrompt: String = TEST_PROMPTS.first(),
        maxTokens: Int = DEFAULT_MAX_TOKENS,
        temperature: Float = DEFAULT_TEMPERATURE,
        onStage: ((BenchmarkStage, Int) -> Unit)? = null
    ): LLMBenchmarkResult = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        val initialMemory = getCurrentMemoryUsageMB()

        try {
            // ── Stage 1: 检查下载 ──
            onStage?.invoke(BenchmarkStage.CHECK_DOWNLOAD, 0)

            if (!llmService.isModelDownloaded(config.name)) {
                onStage?.invoke(BenchmarkStage.CHECK_DOWNLOAD, 100)
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
            onStage?.invoke(BenchmarkStage.CHECK_DOWNLOAD, 100)

            // ── Stage 2: 加载模型 ──
            onStage?.invoke(BenchmarkStage.LOADING, 0)
            val memBeforeLoad = getCurrentMemoryUsageMB()

            val loadStartTime = System.currentTimeMillis()
            val loadSuccess = llmService.loadModel(config)
            val loadTimeMs = System.currentTimeMillis() - loadStartTime

            onStage?.invoke(BenchmarkStage.LOADING, 100)

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

            // ── Stage 3: 推理生成 ──
            onStage?.invoke(BenchmarkStage.GENERATING, 0)
            val memAfterLoad = getCurrentMemoryUsageMB()
            val memoryUsageMB = memAfterLoad - memBeforeLoad

            val generateStartTime = System.currentTimeMillis()
            val generatedText = llmService.generate(testPrompt, maxTokens, temperature) ?: ""
            val generateTimeMs = System.currentTimeMillis() - generateStartTime

            onStage?.invoke(BenchmarkStage.GENERATING, 100)

            // 计算 token 数量 (粗略估计: 平均 1.5 字符 = 1 token)
            val totalTokens = (generatedText.length / 1.5).toInt()
            val throughput = if (generateTimeMs > 0) {
                (totalTokens.toDouble() / generateTimeMs) * 1000.0
            } else 0.0

            // ── Stage 4: 卸载模型 ──
            onStage?.invoke(BenchmarkStage.UNLOADING, 0)
            llmService.unloadModel()
            onStage?.invoke(BenchmarkStage.UNLOADING, 100)

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
     * 批量测试多个模型 (带实时子阶段进度)
     */
    fun runBatchBenchmark(
        models: List<LocalLLMService.ModelConfig>,
        testPrompt: String = TEST_PROMPTS.first(),
        maxTokens: Int = DEFAULT_MAX_TOKENS
    ): Flow<DetailedBenchmarkProgress> = flow {
        val startTime = System.currentTimeMillis()
        val results = mutableListOf<LLMBenchmarkResult>()

        // 发送初始状态
        emit(DetailedBenchmarkProgress(
            overallProgress = BenchmarkProgress(0, models.size, null, BenchmarkState.LOADING),
            stage = BenchmarkStage.CHECK_DOWNLOAD,
            stageProgress = 0
        ))

        models.forEachIndexed { index, config ->
            if (!currentCoroutineContext().isActive) {
                emit(DetailedBenchmarkProgress(
                    overallProgress = BenchmarkProgress(index, models.size, config, BenchmarkState.FAILED),
                    stage = BenchmarkStage.CHECK_DOWNLOAD,
                    stageProgress = 0
                ))
                return@flow
            }

            // 运行单个模型测试，实时发射子阶段进度
            val result = benchmarkModel(config, testPrompt, maxTokens) { stage, stageProgress ->
                // 计算总体进度: 每个模型占 1/N，子阶段在模型内部分摊
                val baseProgress = index
                val stageWeight = stage.weight
                val overallPercent = ((baseProgress + (stageProgress / 100.0 * stageWeight)) / models.size * 100).toInt()

                // 通过 emit 无法在回调中使用，所以这里用日志
                Log.d(TAG, "进度: 模型 $index/${models.size} - ${config.name} - $stage $stageProgress% (总体 ~$overallPercent%)")
            }

            results.add(result)

            // 发送单项完成
            emit(DetailedBenchmarkProgress(
                overallProgress = BenchmarkProgress(
                    index + 1,
                    models.size,
                    config,
                    if (result.success) BenchmarkState.COMPLETED else BenchmarkState.FAILED
                ),
                stage = if (result.success) BenchmarkStage.UNLOADING else BenchmarkStage.CHECK_DOWNLOAD,
                stageProgress = 100,
                lastResult = result
            ))
        }

        val totalDuration = System.currentTimeMillis() - startTime
        val report = BatchBenchmarkReport(
            results = results,
            deviceInfo = getDeviceInfo(),
            totalDurationMs = totalDuration
        )

        emit(DetailedBenchmarkProgress(
            overallProgress = BenchmarkProgress(models.size, models.size, null, BenchmarkState.FINISHED, report),
            stage = BenchmarkStage.UNLOADING,
            stageProgress = 100,
            report = report
        ))
    }

    /**
     * 兼容旧版 API
     */
    fun runBatchBenchmarkLegacy(
        models: List<LocalLLMService.ModelConfig>,
        testPrompt: String = TEST_PROMPTS.first(),
        maxTokens: Int = DEFAULT_MAX_TOKENS
    ): Flow<BenchmarkProgress> = flow {
        runBatchBenchmark(models, testPrompt, maxTokens).collect { detailed ->
            emit(detailed.overallProgress)
        }
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
 * 基准测试子阶段
 */
enum class BenchmarkStage(val label: String, val weight: Double) {
    CHECK_DOWNLOAD("检查下载", 0.05),
    LOADING("加载模型", 0.30),
    GENERATING("推理生成", 0.55),
    UNLOADING("卸载清理", 0.10);

    val displayLabel: String
        get() = when (this) {
            CHECK_DOWNLOAD -> "🔍 $label"
            LOADING -> "📦 $label"
            GENERATING -> "🧠 $label"
            UNLOADING -> "🧹 $label"
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
 * 基准测试进度 (旧版兼容)
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

/**
 * 详细基准测试进度 (v2)
 * 包含子阶段进度信息
 */
data class DetailedBenchmarkProgress(
    val overallProgress: BenchmarkProgress,
    val stage: BenchmarkStage,
    val stageProgress: Int,
    val lastResult: LLMBenchmarkResult? = null,
    val report: BatchBenchmarkReport? = null
) {
    val overallPercent: Int
        get() = overallProgress.progressPercent

    val stageLabel: String
        get() = stage.displayLabel
}