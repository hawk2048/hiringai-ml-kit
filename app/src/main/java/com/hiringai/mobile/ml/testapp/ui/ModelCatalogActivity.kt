package com.hiringai.mobile.ml.testapp.ui

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.hiringai.mobile.ml.LocalEmbeddingService
import com.hiringai.mobile.ml.LocalLLMService
import com.hiringai.mobile.ml.ModelManager
import com.hiringai.mobile.ml.catalog.ModelCatalogService
import com.hiringai.mobile.ml.catalog.CatalogModel
import com.hiringai.mobile.ml.logging.MlLogger
import com.hiringai.mobile.ml.testapp.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 模型目录界面
 *
 * 特性：
 * - 国内模型源 (ModelScope 等)
 * - 分类浏览 (LLM / 嵌入 / 语音 / 图像)
 * - 搜索过滤
 * - 本地 + 在线模型统一展示
 * - 嵌入模型选择
 */
class ModelCatalogActivity : AppCompatActivity() {

    private lateinit var logger: MlLogger
    private lateinit var modelManager: ModelManager

    private lateinit var searchInput: EditText
    private lateinit var categorySpinner: Spinner
    private lateinit var modelListContainer: LinearLayout
    private lateinit var loadingBar: ProgressBar
    private lateinit var btnRefresh: Button

    private var currentCategory = ModelManager.ModelCategory.ALL
    private var allCatalogModels: List<CatalogModel> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_catalog)

        logger = MlLogger.getInstance(this)
        modelManager = ModelManager.getInstance(this)

        initViews()
        loadModels()
    }

    private fun initViews() {
        searchInput = findViewById(R.id.searchInput)
        categorySpinner = findViewById(R.id.categorySpinner)
        modelListContainer = findViewById(R.id.modelListContainer)
        loadingBar = findViewById(R.id.loadingBar)
        btnRefresh = findViewById(R.id.btnRefresh)

        // 分类下拉
        val categories = listOf("全部", "LLM", "嵌入", "语音", "图像")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, categories)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        categorySpinner.adapter = adapter
        categorySpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                currentCategory = when (pos) {
                    1 -> ModelManager.ModelCategory.LLM
                    2 -> ModelManager.ModelCategory.EMBEDDING
                    3 -> ModelManager.ModelCategory.SPEECH
                    4 -> ModelManager.ModelCategory.IMAGE
                    else -> ModelManager.ModelCategory.ALL
                }
                filterAndDisplay()
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // 搜索
        searchInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                filterAndDisplay()
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        btnRefresh.setOnClickListener {
            loadModels(fetchOnline = true)
        }
    }

    private fun loadModels(fetchOnline: Boolean = false) {
        loadingBar.visibility = View.VISIBLE
        modelListContainer.removeAllViews()

        lifecycleScope.launch {
            try {
                // 加载本地内置模型
                val localModels = buildLocalCatalogModels()

                // 尝试加载在线模型目录
                val catalogService = ModelCatalogService.getInstance(this@ModelCatalogActivity)
                val onlineModels = if (fetchOnline) {
                    withContext(Dispatchers.IO) {
                        catalogService.fetchOnlineCatalog()
                    }
                } else {
                    withContext(Dispatchers.IO) {
                        catalogService.getCachedCatalog()
                    }
                }

                allCatalogModels = localModels + onlineModels
                logger.info("ModelCatalog", "加载模型: ${localModels.size} 本地 + ${onlineModels.size} 在线")

                withContext(Dispatchers.Main) {
                    filterAndDisplay()
                    loadingBar.visibility = View.GONE
                }
            } catch (e: Exception) {
                logger.error("ModelCatalog", "加载模型目录失败", e)
                withContext(Dispatchers.Main) {
                    loadingBar.visibility = View.GONE
                    Toast.makeText(this@ModelCatalogActivity, "加载失败: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun buildLocalCatalogModels(): List<CatalogModel> {
        val models = mutableListOf<CatalogModel>()

        LocalLLMService.AVAILABLE_MODELS.forEach { config ->
            models.add(CatalogModel(
                id = config.name,
                name = config.name,
                category = CatalogModel.ModelCategory.LLM,
                sizeBytes = config.size,
                description = config.description,
                source = "本地内置",
                isDownloaded = LocalLLMService.getInstance(this).isModelDownloaded(config.name),
                downloadUrl = config.url,
                recommendedRAM = config.requiredRAM
            ))
        }

        LocalEmbeddingService.AVAILABLE_MODELS.forEach { config ->
            models.add(CatalogModel(
                id = config.name,
                name = config.name,
                category = CatalogModel.ModelCategory.EMBEDDING,
                sizeBytes = config.modelSize,
                description = "${config.description}\n${config.recommendedFor}",
                source = "本地内置",
                isDownloaded = LocalEmbeddingService.getInstance(this).isModelDownloaded(config.name),
                downloadUrl = config.modelUrl,
                dimension = config.dimension,
                maxSeqLength = config.maxSeqLength
            ))
        }

        return models
    }

    private fun filterAndDisplay() {
        modelListContainer.removeAllViews()

        val query = searchInput.text.toString().lowercase().trim()
        val filtered = allCatalogModels.filter { model ->
            val categoryMatch = currentCategory == ModelManager.ModelCategory.ALL ||
                    model.category.name == currentCategory.name ||
                    (currentCategory == ModelManager.ModelCategory.EMBEDDING && model.category == CatalogModel.ModelCategory.EMBEDDING)

            val searchMatch = query.isEmpty() ||
                    model.name.lowercase().contains(query) ||
                    model.description.lowercase().contains(query)

            categoryMatch && searchMatch
        }

        // 按分类分组显示
        val grouped = filtered.groupBy { it.category }
        val categoryOrder = CatalogModel.ModelCategory.entries

        for (category in categoryOrder) {
            val group = grouped[category] ?: continue
            if (group.isEmpty()) continue

            // 分组标题
            val header = TextView(this).apply {
                text = when (category) {
                    CatalogModel.ModelCategory.LLM -> "🤖 大语言模型 (LLM)"
                    CatalogModel.ModelCategory.EMBEDDING -> "📐 嵌入模型 (Embedding)"
                    CatalogModel.ModelCategory.SPEECH -> "🎤 语音模型 (Speech)"
                    CatalogModel.ModelCategory.IMAGE -> "🖼️ 图像模型 (Image)"
                }
                setPadding(16, 24, 16, 8)
                setTextAppearance(android.R.style.TextAppearance_Medium)
            }
            modelListContainer.addView(header)

            // 模型卡片
            group.forEach { model ->
                val card = createModelCard(model)
                modelListContainer.addView(card)
            }
        }

        if (filtered.isEmpty()) {
            val emptyText = TextView(this).apply {
                text = "没有找到匹配的模型"
                setPadding(32, 32, 32, 32)
            }
            modelListContainer.addView(emptyText)
        }
    }

    private fun createModelCard(model: CatalogModel): View {
        return TextView(this).apply {
            text = buildString {
                append(if (model.isDownloaded) "✅ " else "⬇️ ")
                append(model.name)
                append(" (${formatSize(model.sizeBytes)})")
                if (model.dimension > 0) append(" [${model.dimension}维]")
                append("\n  ${model.description.take(80)}")
                append("\n  来源: ${model.source}")
            }
            setPadding(24, 12, 24, 12)
            setOnClickListener {
                showModelDetailDialog(model)
            }
        }
    }

    private fun showModelDetailDialog(model: CatalogModel) {
        val message = buildString {
            appendLine("名称: ${model.name}")
            appendLine("分类: ${model.category.label}")
            appendLine("大小: ${formatSize(model.sizeBytes)}")
            appendLine("来源: ${model.source}")
            appendLine("已下载: ${if (model.isDownloaded) "是" else "否"}")
            if (model.dimension > 0) appendLine("向量维度: ${model.dimension}")
            if (model.maxSeqLength > 0) appendLine("最大序列长度: ${model.maxSeqLength}")
            if (model.recommendedRAM > 0) appendLine("推荐内存: ${model.recommendedRAM} GB")
            appendLine()
            appendLine(model.description)
        }

        val builder = android.app.AlertDialog.Builder(this)
            .setTitle(model.name)
            .setMessage(message)

        if (!model.isDownloaded) {
            builder.setPositiveButton("下载") { _, _ ->
                downloadModel(model)
            }
        } else {
            builder.setPositiveButton("删除") { _, _ ->
                deleteModel(model)
            }
        }

        builder.setNegativeButton("关闭", null)
        builder.show()
    }

    private fun downloadModel(model: CatalogModel) {
        modelManager.downloadModel(model.name,
            onProgress = { progress ->
                runOnUiThread {
                    logger.debug("ModelCatalog", "下载 ${model.name}: $progress%")
                }
            },
            onComplete = { success ->
                runOnUiThread {
                    if (success) {
                        Toast.makeText(this, "${model.name} 下载完成", Toast.LENGTH_SHORT).show()
                        logger.info("ModelCatalog", "模型下载完成: ${model.name}")
                    } else {
                        Toast.makeText(this, "${model.name} 下载失败", Toast.LENGTH_SHORT).show()
                        logger.error("ModelCatalog", "模型下载失败: ${model.name}")
                    }
                    loadModels()
                }
            }
        )
    }

    private fun deleteModel(model: CatalogModel) {
        lifecycleScope.launch {
            val success = modelManager.deleteModel(model.name)
            withContext(Dispatchers.Main) {
                if (success) {
                    Toast.makeText(this@ModelCatalogActivity, "${model.name} 已删除", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@ModelCatalogActivity, "删除失败", Toast.LENGTH_SHORT).show()
                }
                loadModels()
            }
        }
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1_000_000.0)
            else -> "%.0f KB".format(bytes / 1_000.0)
        }
    }
}
