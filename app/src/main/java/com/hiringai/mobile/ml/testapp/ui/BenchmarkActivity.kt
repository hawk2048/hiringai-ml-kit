package com.hiringai.mobile.ml.testapp.ui

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.hiringai.mobile.ml.LocalEmbeddingService
import com.hiringai.mobile.ml.LocalLLMService
import com.hiringai.mobile.ml.benchmark.*
import com.hiringai.mobile.ml.logging.MlLogger
import com.hiringai.mobile.ml.testapp.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 基准测试界面
 *
 * 特性：
 * - 实时子阶段进度 (下载 → 加载 → 推理 → 卸载)
 * - 批量模型选择
 * - 结果排行展示
 */
class BenchmarkActivity : AppCompatActivity() {

    private lateinit var logger: MlLogger
    private lateinit var llmRunner: LLMBenchmarkRunner

    private lateinit var progressCard: View
    private lateinit var progressBar: ProgressBar
    private lateinit var progressPercent: TextView
    private lateinit var progressStage: TextView
    private lateinit var progressModel: TextView
    private lateinit var resultContainer: LinearLayout
    private lateinit var btnStartBenchmark: Button
    private lateinit var btnBatchBenchmark: Button
    private lateinit var modelCheckBoxContainer: LinearLayout

    private val modelCheckBoxes = mutableMapOf<String, CheckBox>()
    private var isRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_benchmark)

        logger = MlLogger.getInstance(this)
        llmRunner = LLMBenchmarkRunner(this)

        initViews()
        populateModelList()
    }

    private fun initViews() {
        progressCard = findViewById(R.id.progressCard)
        progressBar = findViewById(R.id.benchmarkProgressBar)
        progressPercent = findViewById(R.id.progressPercent)
        progressStage = findViewById(R.id.progressStage)
        progressModel = findViewById(R.id.progressModel)
        resultContainer = findViewById(R.id.resultContainer)
        modelCheckBoxContainer = findViewById(R.id.modelCheckBoxContainer)

        btnStartBenchmark = findViewById(R.id.btnStartBenchmark)
        btnBatchBenchmark = findViewById(R.id.btnBatchBenchmark)

        btnStartBenchmark.setOnClickListener { startSingleBenchmark() }
        btnBatchBenchmark.setOnClickListener { startBatchBenchmark() }

        progressCard.visibility = View.GONE
    }

    private fun populateModelList() {
        modelCheckBoxContainer.removeAllViews()
        modelCheckBoxes.clear()

        LocalLLMService.AVAILABLE_MODELS.forEach { config ->
            val cb = CheckBox(this).apply {
                text = "${config.name} (${formatSize(config.size)})"
                isChecked = false
                tag = config.name
            }
            modelCheckBoxes[config.name] = cb
            modelCheckBoxContainer.addView(cb)
        }

        // Embedding models
        val embedHeader = TextView(this).apply {
            text = "── 嵌入模型 ──"
            setPadding(0, 16, 0, 8)
            setTextAppearance(android.R.style.TextAppearance_Medium)
        }
        modelCheckBoxContainer.addView(embedHeader)

        LocalEmbeddingService.AVAILABLE_MODELS.forEach { config ->
            val cb = CheckBox(this).apply {
                text = "${config.name} (${formatSize(config.modelSize)})"
                isChecked = false
                tag = config.name
            }
            modelCheckBoxes[config.name] = cb
            modelCheckBoxContainer.addView(cb)
        }

        // 全选/取消按钮
        val selectAllBtn = Button(this).apply {
            text = "全选"
            setOnClickListener {
                val allChecked = modelCheckBoxes.values.all { it.isChecked }
                modelCheckBoxes.values.forEach { it.isChecked = !allChecked }
                text = if (allChecked) "全选" else "取消全选"
            }
        }
        modelCheckBoxContainer.addView(selectAllBtn, 0)
    }

    private fun startSingleBenchmark() {
        val selectedModel = modelCheckBoxes.entries.firstOrNull { it.value.isChecked }?.key
        if (selectedModel == null) {
            Toast.makeText(this, "请选择至少一个模型", Toast.LENGTH_SHORT).show()
            return
        }

        val config = LocalLLMService.AVAILABLE_MODELS.find { it.name == selectedModel }
        if (config == null) {
            Toast.makeText(this, "找不到模型配置", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            runBenchmarks(listOf(config))
        }
    }

    private fun startBatchBenchmark() {
        val selectedModels = modelCheckBoxes.entries
            .filter { it.value.isChecked }
            .mapNotNull { entry ->
                LocalLLMService.AVAILABLE_MODELS.find { it.name == entry.key }
            }

        if (selectedModels.isEmpty()) {
            Toast.makeText(this, "请选择至少一个模型", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            runBenchmarks(selectedModels)
        }
    }

    private suspend fun runBenchmarks(models: List<LocalLLMService.ModelConfig>) {
        if (isRunning) return
        isRunning = true
        setButtonsEnabled(false)

        logger.info("Benchmark", "开始批量基准测试，共 ${models.size} 个模型")

        resultContainer.removeAllViews()
        progressCard.visibility = View.VISIBLE

        llmRunner.runBatchBenchmark(models)
            .catch { e ->
                logger.error("Benchmark", "基准测试异常", e)
                withContext(Dispatchers.Main) {
                    progressStage.text = "❌ 出错: ${e.message}"
                }
            }
            .collect { progress ->
                withContext(Dispatchers.Main) {
                    updateDetailedProgress(progress)
                    val state = progress.overallProgress.state
                    if (state == BenchmarkState.COMPLETED || state == BenchmarkState.FAILED) {
                        addResultCard(progress)
                    }
                    if (state == BenchmarkState.FINISHED && progress.report != null) {
                        showSummary(progress.report!!)
                    }
                }
            }

        isRunning = false
        setButtonsEnabled(true)
        progressStage.text = "✅ 测试完成"
    }

    private fun updateDetailedProgress(progress: DetailedBenchmarkProgress) {
        val overall = progress.overallProgress
        progressBar.max = 100
        progressBar.progress = progress.overallPercent

        progressPercent.text = "${progress.overallPercent}%"
        progressModel.text = overall.currentModel?.name ?: "准备中..."

        progressStage.text = when (overall.state) {
            BenchmarkState.LOADING -> "⏳ 加载模型... ${progress.stageLabel} ${progress.stageProgress}%"
            BenchmarkState.RUNNING -> "🔄 运行推理... ${progress.stageLabel} ${progress.stageProgress}%"
            BenchmarkState.COMPLETED -> "✅ 单项完成"
            BenchmarkState.FAILED -> "❌ 单项失败"
            BenchmarkState.FINISHED -> "🏁 全部完成"
        }

        logger.debug("Benchmark", "进度: ${overall.currentIndex}/${overall.totalCount} - ${overall.currentModel?.name} - ${overall.state} - ${progress.stageLabel}")
    }

    private fun addResultCard(progress: DetailedBenchmarkProgress) {
        // 注意：这个方法在 COMPLETED/FAILED 状态调用，但 progress 中没有单个结果
        // 结果汇总在 FINISHED 的 report 中
    }

    private fun showSummary(report: BatchBenchmarkReport) {
        val summaryText = TextView(this).apply {
            text = report.toExportText().take(2000)
            setPadding(16, 16, 16, 16)
            setTextAppearance(android.R.style.TextAppearance_Medium)
        }
        resultContainer.addView(summaryText)

        logger.info("Benchmark", "基准测试完成: ${report.results.size} 模型, 耗时 ${report.totalDurationMs}ms")
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        btnStartBenchmark.isEnabled = enabled
        btnBatchBenchmark.isEnabled = enabled
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1_000_000.0)
            else -> "%.0f KB".format(bytes / 1_000.0)
        }
    }
}
