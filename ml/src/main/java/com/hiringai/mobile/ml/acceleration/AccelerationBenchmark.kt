package com.hiringai.mobile.ml.acceleration

import android.content.Context
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 硬件加速基准测试
 *
 * 比较 CPU、GPU、NNAPI 后端的性能差异：
 * - 推理延迟
 * - 吞吐量
 * - 内存占用
 * - 首次推理开销
 *
 * 根据测试结果自动选择最优后端。
 */
class AccelerationBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "AccelerationBenchmark"
        private const val WARMUP_ITERATIONS = 3
        private const val BENCHMARK_ITERATIONS = 10
        private const val TEST_INPUT_SIZE = 256
        private const val TEST_OUTPUT_SIZE = 384
    }

    /**
     * 后端类型
     */
    enum class Backend {
        CPU,
        NNAPI_SAFE,
        NNAPI_ALL
    }

    /**
     * 单次基准测试结果
     */
    data class BackendBenchmarkResult(
        val backend: Backend,
        val avgLatencyMs: Double,
        val minLatencyMs: Long,
        val maxLatencyMs: Long,
        val p50LatencyMs: Double,
        val p95LatencyMs: Double,
        val p99LatencyMs: Double,
        val throughputInferencesPerSec: Double,
        val firstInferenceMs: Long,
        val warmupMs: Long,
        val memoryUsageMB: Long,
        val success: Boolean,
        val errorMessage: String? = null,
        val iterationResults: List<Long> = emptyList()
    ) {
        fun toSummary(): String {
            return buildString {
                appendLine("📊 $backend 基准测试结果:")
                appendLine("   平均延迟: ${"%.2f".format(avgLatencyMs)}ms")
                appendLine("   最小/最大: ${minLatencyMs}ms / ${maxLatencyMs}ms")
                appendLine("   P50/P95/P99: ${"%.1f".format(p50LatencyMs)}ms / ${"%.1f".format(p95LatencyMs)}ms / ${"%.1f".format(p99LatencyMs)}ms")
                appendLine("   吞吐量: ${"%.2f".format(throughputInferencesPerSec)} inferences/s")
                appendLine("   首次推理: ${firstInferenceMs}ms")
                appendLine("   预热时间: ${warmupMs}ms")
                appendLine("   内存占用: ${memoryUsageMB}MB")
                appendLine("   状态: ${if (success) "✅ 成功" else "❌ 失败: $errorMessage"}")
            }
        }
    }

    /**
     * 完整基准测试报告
     */
    data class BenchmarkReport(
        val results: List<BackendBenchmarkResult>,
        val recommendedBackend: Backend,
        val speedupRatio: Double,
        val deviceInfo: String,
        val timestamp: Long = System.currentTimeMillis()
    ) {
        fun getBestByLatency(): BackendBenchmarkResult? {
            return results.filter { it.success }.minByOrNull { it.avgLatencyMs }
        }

        fun getBestByThroughput(): BackendBenchmarkResult? {
            return results.filter { it.success }.maxByOrNull { it.throughputInferencesPerSec }
        }

        fun toExportText(): String {
            return buildString {
                appendLine("===================================")
                appendLine("硬件加速基准测试报告")
                appendLine("===================================")
                appendLine("设备: $deviceInfo")
                appendLine("测试时间: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date(timestamp))}")
                appendLine()
                appendLine("🏆 推荐后端: $recommendedBackend")
                appendLine("📈 加速比: ${"%.2f".format(speedupRatio)}x")
                appendLine()
                appendLine("📋 详细结果:")
                results.forEach { appendLine(it.toSummary()) }
                appendLine()
                appendLine("===================================")
                appendLine("CSV 导出:")
                appendLine("Backend,AvgLatencyMs,MinLatencyMs,MaxLatencyMs,P50Ms,P95Ms,P99Ms,Throughput,FirstInferenceMs,WarmupMs,MemoryMB,Success")
                results.forEach { r ->
                    appendLine("${r.backend},${"%.2f".format(r.avgLatencyMs)},${r.minLatencyMs},${r.maxLatencyMs},${"%.1f".format(r.p50LatencyMs)},${"%.1f".format(r.p95LatencyMs)},${"%.1f".format(r.p99LatencyMs)},${"%.2f".format(r.throughputInferencesPerSec)},${r.firstInferenceMs},${r.warmupMs},${r.memoryUsageMB},${r.success}")
                }
            }
        }
    }

    /**
     * 模型配置
     */
    data class ModelConfig(
        val name: String,
        val inputSize: Int,
        val outputSize: Int,
        val modelPath: String? = null
    )

    // ========== 公共 API ==========

    /**
     * 运行完整的后端基准测试
     *
     * @param modelConfig 模型配置（如果为 null，使用合成测试）
     * @param iterations 测试迭代次数
     */
    suspend fun runFullBenchmark(
        modelConfig: ModelConfig? = null,
        iterations: Int = BENCHMARK_ITERATIONS
    ): BenchmarkReport = withContext(Dispatchers.IO) {
        Log.i(TAG, "Starting full acceleration benchmark")

        val results = mutableListOf<BackendBenchmarkResult>()
        val deviceInfo = getDeviceInfo()

        // 测试 CPU 后端
        results.add(benchmarkBackend(Backend.CPU, modelConfig, iterations))

        // 测试安全 NNAPI 后端（如果可用）
        if (NNAPIManager.isNNAPISafe(context)) {
            results.add(benchmarkBackend(Backend.NNAPI_SAFE, modelConfig, iterations))
        } else {
            Log.i(TAG, "Skipping NNAPI_SAFE: not safe for this device")
            results.add(
                BackendBenchmarkResult(
                    backend = Backend.NNAPI_SAFE,
                    avgLatencyMs = 0.0,
                    minLatencyMs = 0,
                    maxLatencyMs = 0,
                    p50LatencyMs = 0.0,
                    p95LatencyMs = 0.0,
                    p99LatencyMs = 0.0,
                    throughputInferencesPerSec = 0.0,
                    firstInferenceMs = 0,
                    warmupMs = 0,
                    memoryUsageMB = 0,
                    success = false,
                    errorMessage = "NNAPI not safe for this device"
                )
            )
        }

        // 计算推荐后端
        val successfulResults = results.filter { it.success }
        val recommendedBackend = selectBestBackend(successfulResults)

        // 计算加速比
        val cpuResult = results.find { it.backend == Backend.CPU && it.success }
        val bestResult = successfulResults.minByOrNull { it.avgLatencyMs }
        val speedupRatio = if (cpuResult != null && bestResult != null && bestResult.avgLatencyMs > 0) {
            cpuResult.avgLatencyMs / bestResult.avgLatencyMs
        } else 1.0

        Log.i(TAG, "Benchmark complete. Recommended: $recommendedBackend, Speedup: ${"%.2f".format(speedupRatio)}x")

        BenchmarkReport(
            results = results,
            recommendedBackend = recommendedBackend,
            speedupRatio = speedupRatio,
            deviceInfo = deviceInfo
        )
    }

    /**
     * 快速基准测试（仅 CPU vs 最佳可用后端）
     */
    suspend fun quickBenchmark(): BackendBenchmarkResult? = withContext(Dispatchers.IO) {
        val recommendedBackend = NNAPIManager.getRecommendedBackend(context)

        if (recommendedBackend == NNAPIManager.AccelerationBackend.CPU) {
            // 只测 CPU
            benchmarkBackend(Backend.CPU, null, 5)
        } else {
            // 测试推荐后端
            benchmarkBackend(Backend.NNAPI_SAFE, null, 5)
        }
    }

    /**
     * 为特定模型自动选择最佳后端
     *
     * @param modelPath 模型文件路径
     * @return 最佳后端配置
     */
    suspend fun selectBestBackendForModel(modelPath: String): BackendBenchmarkResult? = withContext(Dispatchers.IO) {
        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            Log.e(TAG, "Model file not found: $modelPath")
            return@withContext null
        }

        val modelConfig = ModelConfig(
            name = modelFile.nameWithoutExtension,
            inputSize = TEST_INPUT_SIZE,
            outputSize = TEST_OUTPUT_SIZE,
            modelPath = modelPath
        )

        // 快速测试：CPU vs NNAPI_SAFE
        val cpuResult = benchmarkBackend(Backend.CPU, modelConfig, 5)
        val nnapiResult = if (NNAPIManager.isNNAPISafe(context)) {
            benchmarkBackend(Backend.NNAPI_SAFE, modelConfig, 5)
        } else null

        // 选择更快的后端
        when {
            nnapiResult == null || !nnapiResult.success -> cpuResult
            !cpuResult.success -> nnapiResult
            nnapiResult.avgLatencyMs < cpuResult.avgLatencyMs -> nnapiResult
            else -> cpuResult
        }
    }

    // ========== 内部实现 ==========

    private suspend fun benchmarkBackend(
        backend: Backend,
        modelConfig: ModelConfig?,
        iterations: Int
    ): BackendBenchmarkResult = withContext(Dispatchers.IO) {
        val iterationTimes = mutableListOf<Long>()
        var firstInferenceMs = 0L
        var warmupMs = 0L
        var memoryBeforeMB = 0L
        var memoryAfterMB = 0L
        var session: OrtSession? = null

        try {
            // 记录内存
            memoryBeforeMB = getMemoryUsageMB()

            // 创建会话选项
            val options = when (backend) {
                Backend.CPU -> NNAPIManager.createCPUOnlySessionOptions()
                Backend.NNAPI_SAFE -> NNAPIManager.createSafeSessionOptions(context)
                    ?: NNAPIManager.createCPUOnlySessionOptions()
                Backend.NNAPI_ALL -> NNAPIManager.createSafeSessionOptions(context)
                    ?: NNAPIManager.createCPUOnlySessionOptions()
            }

            // 获取或创建模型
            val modelData = modelConfig?.modelPath?.let { loadModelData(it) }
                ?: generateSyntheticModel()

            // 创建 ONNX 会话
            val env = OrtEnvironment.getEnvironment()
            session = env.createSession(modelData, options)

            // 预热
            val warmupStart = System.currentTimeMillis()
            repeat(WARMUP_ITERATIONS) {
                runInference(session, env, modelConfig)
            }
            warmupMs = System.currentTimeMillis() - warmupStart

            // 基准测试
            for (i in 0 until iterations) {
                val start = System.currentTimeMillis()
                runInference(session, env, modelConfig)
                val elapsed = System.currentTimeMillis() - start

                if (i == 0) {
                    firstInferenceMs = elapsed
                }
                iterationTimes.add(elapsed)
            }

            memoryAfterMB = getMemoryUsageMB()

            // 计算统计数据
            val stats = calculateStatistics(iterationTimes)

            BackendBenchmarkResult(
                backend = backend,
                avgLatencyMs = stats.avg,
                minLatencyMs = stats.min,
                maxLatencyMs = stats.max,
                p50LatencyMs = stats.p50,
                p95LatencyMs = stats.p95,
                p99LatencyMs = stats.p99,
                throughputInferencesPerSec = 1000.0 / stats.avg,
                firstInferenceMs = firstInferenceMs,
                warmupMs = warmupMs,
                memoryUsageMB = memoryAfterMB - memoryBeforeMB,
                success = true,
                iterationResults = iterationTimes
            )

        } catch (e: Exception) {
            Log.e(TAG, "Benchmark failed for $backend", e)
            BackendBenchmarkResult(
                backend = backend,
                avgLatencyMs = 0.0,
                minLatencyMs = 0,
                maxLatencyMs = 0,
                p50LatencyMs = 0.0,
                p95LatencyMs = 0.0,
                p99LatencyMs = 0.0,
                throughputInferencesPerSec = 0.0,
                firstInferenceMs = 0,
                warmupMs = 0,
                memoryUsageMB = 0,
                success = false,
                errorMessage = e.message
            )
        } finally {
            session?.close()
        }
    }

    private fun runInference(
        session: OrtSession,
        env: OrtEnvironment,
        modelConfig: ModelConfig?
    ) {
        val inputSize = modelConfig?.inputSize ?: TEST_INPUT_SIZE
        val input = generateTestInput(inputSize)
        val inputTensor = ai.onnxruntime.OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1, inputSize.toLong())
        )

        try {
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to inputTensor))
        } finally {
            inputTensor.close()
        }
    }

    private fun generateTestInput(size: Int): FloatArray {
        return FloatArray(size) { (it % 100) / 100.0f }
    }

    private fun generateSyntheticModel(): ByteArray {
        // 生成一个最小的 ONNX 模型字节流
        // 实际实现应使用 ONNX proto 或预编译的测试模型
        // 这里使用简化版本
        return ByteArray(1024) { 0 }
    }

    private fun loadModelData(path: String): ByteArray {
        return File(path).readBytes()
    }

    private fun calculateStatistics(times: List<Long>): Statistics {
        if (times.isEmpty()) return Statistics(0.0, 0, 0, 0.0, 0.0, 0.0)

        val sorted = times.sorted()
        val avg = times.average()
        val min = sorted.first()
        val max = sorted.last()
        val p50 = sorted[sorted.size * 50 / 100].toDouble()
        val p95 = sorted[sorted.size * 95 / 100].toDouble()
        val p99 = sorted[(sorted.size * 99 / 100).coerceAtMost(sorted.size - 1)].toDouble()

        return Statistics(avg, min, max, p50, p95, p99)
    }

    private data class Statistics(
        val avg: Double,
        val min: Long,
        val max: Long,
        val p50: Double,
        val p95: Double,
        val p99: Double
    )

    private fun selectBestBackend(results: List<BackendBenchmarkResult>): Backend {
        if (results.isEmpty()) return Backend.CPU

        val best = results.minByOrNull { it.avgLatencyMs }
        return best?.backend ?: Backend.CPU
    }

    private fun getMemoryUsageMB(): Long {
        val runtime = Runtime.getRuntime()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        return usedMemory / (1024 * 1024)
    }

    private fun getDeviceInfo(): String {
        val manufacturer = android.os.Build.MANUFACTURER
        val model = android.os.Build.MODEL
        val sdk = android.os.Build.VERSION.SDK_INT
        val cores = Runtime.getRuntime().availableProcessors()
        return "$manufacturer $model (API $sdk, $cores cores)"
    }
}
