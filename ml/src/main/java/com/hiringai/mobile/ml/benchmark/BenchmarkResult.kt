package com.hiringai.mobile.ml.benchmark

import com.hiringai.mobile.ml.LocalLLMService

/**
 * LLM 基准测试结果数据类
 */
data class LLMBenchmarkResult(
    val modelName: String,
    val loadTimeMs: Long,
    val firstTokenLatencyMs: Long,
    val avgTokenLatencyMs: Long,
    val totalTokens: Int,
    val throughputTokensPerSec: Double,
    val memoryUsageMB: Long,
    val peakMemoryMB: Long,
    val testPrompt: String,
    val generatedText: String,
    val success: Boolean,
    val errorMessage: String? = null,
    val timestamp: Long = System.currentTimeMillis()
) {
    fun toCsvRow(): String {
        return "$modelName,$loadTimeMs,$firstTokenLatencyMs,$avgTokenLatencyMs,$totalTokens,$throughputTokensPerSec,$memoryUsageMB,$peakMemoryMB,$success"
    }

    fun toSummary(): String {
        return buildString {
            appendLine("📊 $modelName 基准测试结果:")
            appendLine("   模型加载: ${loadTimeMs}ms")
            appendLine("   首Token延迟: ${firstTokenLatencyMs}ms")
            appendLine("   平均Token延迟: ${avgTokenLatencyMs}ms")
            appendLine("   生成Token数: $totalTokens")
            appendLine("   吞吐量: ${"%.2f".format(throughputTokensPerSec)} tokens/s")
            appendLine("   内存占用: ${memoryUsageMB}MB (峰值: ${peakMemoryMB}MB)")
            appendLine("   状态: ${if (success) "✅ 成功" else "❌ 失败: $errorMessage"}")
        }
    }

    companion object {
        fun getCsvHeader(): String {
            return "Model,LoadTimeMs,FirstTokenLatencyMs,AvgTokenLatencyMs,TotalTokens,ThroughputTokensPerSec,MemoryUsageMB,PeakMemoryMB,Success"
        }
    }
}

/**
 * 批量基准测试报告
 */
data class BatchBenchmarkReport(
    val results: List<LLMBenchmarkResult>,
    val deviceInfo: String,
    val totalDurationMs: Long,
    val timestamp: Long = System.currentTimeMillis()
) {
    fun getBestByThroughput(): LLMBenchmarkResult? {
        return results.filter { it.success }.maxByOrNull { it.throughputTokensPerSec }
    }

    fun getBestByLatency(): LLMBenchmarkResult? {
        return results.filter { it.success }.minByOrNull { it.avgTokenLatencyMs }
    }

    fun getBestByMemory(): LLMBenchmarkResult? {
        return results.filter { it.success }.minByOrNull { it.peakMemoryMB }
    }

    fun toExportText(): String {
        return buildString {
            appendLine("===================================")
            appendLine("LLM 批量基准测试报告")
            appendLine("===================================")
            appendLine("设备: $deviceInfo")
            appendLine("测试时间: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date(timestamp))}")
            appendLine("总耗时: ${totalDurationMs}ms")
            appendLine("测试模型数: ${results.size}")
            appendLine("成功: ${results.count { it.success }}")
            appendLine("失败: ${results.count { !it.success }}")
            appendLine()
            appendLine("📈 性能排名:")
            getBestByThroughput()?.let { appendLine("   最高吞吐量: ${it.modelName} (${"%.2f".format(it.throughputTokensPerSec)} tokens/s)") }
            getBestByLatency()?.let { appendLine("   最低延迟: ${it.modelName} (${it.avgTokenLatencyMs}ms)") }
            getBestByMemory()?.let { appendLine("   最低内存: ${it.modelName} (${it.peakMemoryMB}MB)") }
            appendLine()
            appendLine("📋 详细结果:")
            results.forEach { appendLine(it.toSummary()) }
            appendLine()
            appendLine("===================================")
            appendLine("CSV 导出:")
            appendLine(LLMBenchmarkResult.getCsvHeader())
            results.forEach { appendLine(it.toCsvRow()) }
        }
    }
}