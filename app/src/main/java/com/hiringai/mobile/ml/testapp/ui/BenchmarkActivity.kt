package com.hiringai.mobile.ml.testapp.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
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
 * Benchmark Activity - Modern UI for ML Model Benchmarking
 *
 * Features:
 * - Real-time sub-stage progress (Check -> Load -> Generate -> Unload)
 * - Batch model selection
 * - Beautiful result cards with metrics
 * - Summary statistics
 *
 * Intent extras:
 * - EXTRA_PRESELECT_MODEL: String - 模型名称，跳转时自动选中该模型
 */
class BenchmarkActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_PRESELECT_MODEL = "preselect_model"
    }

    private lateinit var logger: MlLogger
    private lateinit var llmRunner: LLMBenchmarkRunner

    private lateinit var progressCard: View
    private lateinit var progressBar: ProgressBar
    private lateinit var progressPercent: TextView
    private lateinit var progressStage: TextView
    private lateinit var progressModel: TextView
    private lateinit var circularProgress: com.google.android.material.progressindicator.CircularProgressIndicator
    private lateinit var resultContainer: LinearLayout
    private lateinit var btnStartBenchmark: Button
    private lateinit var btnBatchBenchmark: Button
    private lateinit var btnQuickBenchmark: Button
    private lateinit var modelCheckBoxContainer: LinearLayout
    private lateinit var downloadedCountText: TextView

    // Stage indicator views
    private lateinit var stageCheckDownload: TextView
    private lateinit var stageLoading: TextView
    private lateinit var stageGenerating: TextView
    private lateinit var stageUnloading: TextView

    private val modelCheckBoxes = mutableMapOf<String, CheckBox>()
    private var isRunning = false
    private val benchmarkResults = mutableListOf<LLMBenchmarkResult>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_benchmark)

        logger = MlLogger.getInstance(this)
        llmRunner = LLMBenchmarkRunner(this)

        initViews()
        populateModelList()

        // 检查是否需要自动选中模型
        val preselectModel = intent.getStringExtra(EXTRA_PRESELECT_MODEL)
        if (!preselectModel.isNullOrBlank()) {
            autoSelectModel(preselectModel)
        }
    }

    private fun autoSelectModel(modelName: String) {
        val checkBox = modelCheckBoxes[modelName]
        if (checkBox != null) {
            checkBox.isChecked = true
            Toast.makeText(this, "Model '$modelName' selected", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Model '$modelName' not found in catalog", Toast.LENGTH_SHORT).show()
        }
    }

    private fun initViews() {
        progressCard = findViewById(R.id.progressCard)
        progressBar = findViewById(R.id.benchmarkProgressBar)
        progressPercent = findViewById(R.id.progressPercent)
        progressStage = findViewById(R.id.progressStage)
        progressModel = findViewById(R.id.progressModel)
        circularProgress = findViewById(R.id.circularProgress)
        resultContainer = findViewById(R.id.resultContainer)
        modelCheckBoxContainer = findViewById(R.id.modelCheckBoxContainer)

        btnStartBenchmark = findViewById(R.id.btnStartBenchmark)
        btnBatchBenchmark = findViewById(R.id.btnBatchBenchmark)
        btnQuickBenchmark = findViewById(R.id.btnQuickBenchmark)
        downloadedCountText = findViewById(R.id.downloadedCountText)

        // Stage indicators
        stageCheckDownload = findViewById(R.id.stageCheckDownload)
        stageLoading = findViewById(R.id.stageLoading)
        stageGenerating = findViewById(R.id.stageGenerating)
        stageUnloading = findViewById(R.id.stageUnloading)

        btnStartBenchmark.setOnClickListener { startSingleBenchmark() }
        btnBatchBenchmark.setOnClickListener { startBatchBenchmark() }
        btnQuickBenchmark.setOnClickListener { startQuickBenchmark() }

        progressCard.visibility = View.GONE
    }

    private fun populateModelList() {
        modelCheckBoxContainer.removeAllViews()
        modelCheckBoxes.clear()

        // Check downloaded models
        val llmService = LocalLLMService.getInstance(this)
        val downloadedModels = LocalLLMService.AVAILABLE_MODELS.filter { llmService.isModelDownloaded(it.name) }

        // Display downloaded count
        val totalModels = LocalLLMService.AVAILABLE_MODELS.size
        downloadedCountText.text = "Built-in $totalModels models | Downloaded ${downloadedModels.size}"

        // LLM Models Section
        val llmHeader = createSectionHeader("Large Language Models", ContextCompat.getColor(this, R.color.primary))
        modelCheckBoxContainer.addView(llmHeader)

        LocalLLMService.AVAILABLE_MODELS.forEach { config ->
            val isDownloaded = llmService.isModelDownloaded(config.name)
            val downloadedTag = if (isDownloaded) "[Downloaded] " else "[Not downloaded] "
            val cb = CheckBox(this).apply {
                text = "$downloadedTag ${config.name} (${formatSize(config.size)})"
                isChecked = false
                tag = config.name
                textSize = 14f
                setTextColor(
                    if (isDownloaded) ContextCompat.getColor(context, R.color.success)
                    else ContextCompat.getColor(context, R.color.text_secondary)
                )
            }
            modelCheckBoxes[config.name] = cb
            modelCheckBoxContainer.addView(cb)
        }

        // Embedding Models Section
        val embedHeader = createSectionHeader("Embedding Models", ContextCompat.getColor(this, R.color.success))
        modelCheckBoxContainer.addView(embedHeader)

        LocalEmbeddingService.AVAILABLE_MODELS.forEach { config ->
            val cb = CheckBox(this).apply {
                text = "${config.name} (${formatSize(config.modelSize)})"
                isChecked = false
                tag = config.name
                textSize = 14f
            }
            modelCheckBoxes[config.name] = cb
            modelCheckBoxContainer.addView(cb)
        }
    }

    private fun createSectionHeader(title: String, color: Int): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            setPadding(0, dpToPx(12), 0, dpToPx(8))

            val colorBar = View(context).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(4), dpToPx(20)).apply {
                    marginEnd = dpToPx(8)
                }
                setBackgroundColor(color)
            }

            val titleView = TextView(context).apply {
                text = title
                textSize = 14f
                setTextColor(color)
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }

            addView(colorBar)
            addView(titleView)
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun startSingleBenchmark() {
        val selectedModel = modelCheckBoxes.entries.firstOrNull { it.value.isChecked }?.key
        if (selectedModel == null) {
            Toast.makeText(this, "Please select at least one model", Toast.LENGTH_SHORT).show()
            return
        }

        val config = LocalLLMService.AVAILABLE_MODELS.find { it.name == selectedModel }
        if (config == null) {
            Toast.makeText(this, "Model configuration not found", Toast.LENGTH_SHORT).show()
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
            Toast.makeText(this, "Please select at least one model", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            runBenchmarks(selectedModels)
        }
    }

    /** Quick test: auto-select all downloaded models */
    private fun startQuickBenchmark() {
        val llmService = LocalLLMService.getInstance(this)
        val downloadedConfigs = LocalLLMService.AVAILABLE_MODELS.filter { llmService.isModelDownloaded(it.name) }

        if (downloadedConfigs.isEmpty()) {
            Toast.makeText(this, "No downloaded models. Please download a model first.", Toast.LENGTH_LONG).show()
            return
        }

        // Select all downloaded models
        modelCheckBoxes.entries.forEach { (name, cb) ->
            cb.isChecked = downloadedConfigs.any { it.name == name }
        }

        Toast.makeText(this, "Selected ${downloadedConfigs.size} downloaded models. Tap 'Batch Test' to start.", Toast.LENGTH_SHORT).show()
    }

    private suspend fun runBenchmarks(models: List<LocalLLMService.ModelConfig>) {
        if (isRunning) return
        isRunning = true
        setButtonsEnabled(false)
        benchmarkResults.clear()

        logger.info("Benchmark", "Starting batch benchmark for ${models.size} models")

        resultContainer.removeAllViews()
        progressCard.visibility = View.VISIBLE
        resetStageIndicators()

        llmRunner.runBatchBenchmark(models)
            .catch { e ->
                logger.error("Benchmark", "Benchmark error", e)
                withContext(Dispatchers.Main) {
                    progressStage.text = "Error: ${e.message}"
                }
            }
            .collect { detailed ->
                withContext(Dispatchers.Main) {
                    updateProgress(detailed)
                    
                    // Add result card when model completes
                    if (detailed.lastResult != null) {
                        benchmarkResults.add(detailed.lastResult!!)
                        addResultCard(detailed.lastResult!!)
                    }
                    
                    // Show summary when all done
                    if (detailed.report != null) {
                        showSummary(detailed.report!!)
                    }
                }
            }

        isRunning = false
        setButtonsEnabled(true)
        progressStage.text = "Done"
    }

    private fun updateProgress(detailed: DetailedBenchmarkProgress) {
        val progress = detailed.overallProgress
        
        progressBar.max = 100
        progressBar.progress = progress.progressPercent
        circularProgress.progress = progress.progressPercent

        progressPercent.text = "${progress.progressPercent}%"
        progressModel.text = progress.currentModel?.name ?: "Preparing..."

        // Update main stage text
        progressStage.text = when (progress.state) {
            BenchmarkState.LOADING -> "Checking..."
            BenchmarkState.RUNNING -> "Running..."
            BenchmarkState.COMPLETED -> "Completed"
            BenchmarkState.FAILED -> "Failed"
            BenchmarkState.FINISHED -> "All Done"
        }

        // Update stage indicators
        updateStageIndicators(detailed.stage, detailed.stageProgress)

        logger.debug("Benchmark", "Progress: ${progress.currentIndex}/${progress.totalCount} - ${progress.currentModel?.name} - ${progress.state}")
    }

    private fun resetStageIndicators() {
        val inactiveColor = ContextCompat.getColor(this, R.color.text_tertiary)
        stageCheckDownload.setTextColor(inactiveColor)
        stageLoading.setTextColor(inactiveColor)
        stageGenerating.setTextColor(inactiveColor)
        stageUnloading.setTextColor(inactiveColor)
    }

    private fun updateStageIndicators(currentStage: BenchmarkStage, stageProgress: Int) {
        val activeColor = ContextCompat.getColor(this, R.color.primary)
        val inactiveColor = ContextCompat.getColor(this, R.color.text_tertiary)

        val stages = listOf(
            stageCheckDownload to BenchmarkStage.CHECK_DOWNLOAD,
            stageLoading to BenchmarkStage.LOADING,
            stageGenerating to BenchmarkStage.GENERATING,
            stageUnloading to BenchmarkStage.UNLOADING
        )

        var foundActive = false
        stages.forEach { (view, stage) ->
            val isCurrentOrPast = stage == currentStage || 
                (stages.indexOfFirst { it.first == view } < stages.indexOfFirst { it.second == currentStage })
            
            if (isCurrentOrPast && !foundActive) {
                view.setTextColor(activeColor)
                foundActive = true
            } else if (!foundActive) {
                view.setTextColor(activeColor)
            } else {
                view.setTextColor(inactiveColor)
            }
        }
    }

    private fun addResultCard(result: LLMBenchmarkResult) {
        val inflater = LayoutInflater.from(this)
        val cardView = inflater.inflate(R.layout.benchmark_result_card, resultContainer, false)

        // Model name and size
        cardView.findViewById<TextView>(R.id.modelName).text = result.modelName
        
        // Find model config for size
        val config = LocalLLMService.AVAILABLE_MODELS.find { it.name == result.modelName }
        cardView.findViewById<TextView>(R.id.modelSize).text = config?.let { formatSize(it.size) } ?: ""

        // Category indicator color
        val categoryColor = ContextCompat.getColor(this, R.color.primary)
        cardView.findViewById<View>(R.id.categoryIndicator).setBackgroundColor(categoryColor)

        // Status badge
        val statusBadge = cardView.findViewById<TextView>(R.id.statusBadge)
        val errorMessage = cardView.findViewById<TextView>(R.id.errorMessage)
        
        if (result.success) {
            statusBadge.text = "Success"
            statusBadge.setBackgroundResource(R.drawable.bg_badge_success)
            errorMessage.visibility = View.GONE
        } else {
            statusBadge.text = "Failed"
            statusBadge.setBackgroundResource(R.drawable.bg_badge_error)
            errorMessage.visibility = View.VISIBLE
            errorMessage.text = result.errorMessage ?: "Unknown error"
        }

        // Metrics
        cardView.findViewById<TextView>(R.id.throughputValue).text = 
            if (result.throughputTokensPerSec > 0) "%.1f".format(result.throughputTokensPerSec) else "--"
        cardView.findViewById<TextView>(R.id.loadTimeValue).text = 
            if (result.loadTimeMs > 0) formatDuration(result.loadTimeMs) else "--"
        cardView.findViewById<TextView>(R.id.tokensValue).text = 
            if (result.totalTokens > 0) result.totalTokens.toString() else "--"
        
        // Memory usage (assuming 1GB max for percentage)
        val memoryPercent = ((result.memoryUsageMB.toFloat() / 1024) * 100).toInt().coerceIn(0, 100)
        cardView.findViewById<com.google.android.material.progressindicator.LinearProgressIndicator>(R.id.memoryProgressBar)
            .progress = memoryPercent
        cardView.findViewById<TextView>(R.id.memoryValue).text = "${result.memoryUsageMB} MB"

        resultContainer.addView(cardView)
    }

    private fun showSummary(report: BatchBenchmarkReport) {
        val inflater = LayoutInflater.from(this)
        val summaryView = inflater.inflate(R.layout.benchmark_summary_card, resultContainer, false)

        // Total models badge
        summaryView.findViewById<TextView>(R.id.totalModelsBadge).text = 
            "${report.results.size} Models"

        // Best throughput
        val bestThroughput = report.getBestByThroughput()
        if (bestThroughput != null) {
            summaryView.findViewById<TextView>(R.id.bestThroughputValue).text = 
                "%.1f".format(bestThroughput.throughputTokensPerSec)
            summaryView.findViewById<TextView>(R.id.bestThroughputModel).text = bestThroughput.modelName
        } else {
            summaryView.findViewById<TextView>(R.id.bestThroughputValue).text = "--"
            summaryView.findViewById<TextView>(R.id.bestThroughputModel).text = "No data"
        }

        // Fastest load
        val fastestLoad = report.results.filter { it.success }.minByOrNull { it.loadTimeMs }
        if (fastestLoad != null) {
            summaryView.findViewById<TextView>(R.id.fastestLoadValue).text = formatDuration(fastestLoad.loadTimeMs)
            summaryView.findViewById<TextView>(R.id.fastestLoadModel).text = fastestLoad.modelName
        } else {
            summaryView.findViewById<TextView>(R.id.fastestLoadValue).text = "--"
            summaryView.findViewById<TextView>(R.id.fastestLoadModel).text = "No data"
        }

        // Total time
        summaryView.findViewById<TextView>(R.id.totalTimeValue).text = 
            "%.1f".format(report.totalDurationMs / 1000.0)

        // Device info
        summaryView.findViewById<TextView>(R.id.deviceInfo).text = report.deviceInfo

        resultContainer.addView(summaryView, 0) // Add at top

        logger.info("Benchmark", "Benchmark complete: ${report.results.size} models, ${report.totalDurationMs}ms total")
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        btnStartBenchmark.isEnabled = enabled
        btnBatchBenchmark.isEnabled = enabled
        btnQuickBenchmark.isEnabled = enabled
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1_000_000.0)
            else -> "%.0f KB".format(bytes / 1_000.0)
        }
    }

    private fun formatDuration(ms: Long): String {
        return when {
            ms >= 1000 -> "%.1fs".format(ms / 1000.0)
            else -> "${ms}ms"
        }
    }
}
