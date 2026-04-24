package com.hiringai.mobile.ml.catalog

import android.app.AlertDialog
import android.content.Context
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TabHost
import android.widget.TextView
import com.hiringai.mobile.ml.LocalEmbeddingService
import com.hiringai.mobile.ml.LocalLLMService
import com.hiringai.mobile.ml.ModelManager
import com.hiringai.mobile.ml.logging.MlLogger

/**
 * 模型选择弹窗
 *
 * 特性：
 * - 分类标签页 (LLM / 嵌入 / 语音 / 图像)
 * - 搜索过滤
 * - 单选模式 (选择一个模型)
 * - 多选模式 (批量选择)
 * - 嵌入模型选择支持
 *
 * 用法：
 * ```kotlin
 * // 单选 LLM 模型
 * ModelSelectionDialog(context)
 *     .setCategory(ModelManager.ModelCategory.LLM)
 *     .setSingleSelect(true)
 *     .setOnSelected { model -> ... }
 *     .show()
 *
 * // 多选批量测试
 * ModelSelectionDialog(context)
 *     .setMultiSelect(true)
 *     .setOnMultiSelected { models -> ... }
 *     .show()
 * ```
 */
class ModelSelectionDialog(private val context: Context) {

    private var category: ModelManager.ModelCategory = ModelManager.ModelCategory.ALL
    private var singleSelect: Boolean = true
    private var onSelected: ((CatalogModel) -> Unit)? = null
    private var onMultiSelected: ((List<CatalogModel>) -> Unit)? = null
    private var title: String = "选择模型"

    private val logger by lazy { MlLogger.getInstance(context) }

    fun setCategory(cat: ModelManager.ModelCategory): ModelSelectionDialog {
        category = cat
        return this
    }

    fun setSingleSelect(single: Boolean): ModelSelectionDialog {
        singleSelect = single
        return this
    }

    fun setMultiSelect(multi: Boolean): ModelSelectionDialog {
        singleSelect = !multi
        return this
    }

    fun setOnSelected(listener: (CatalogModel) -> Unit): ModelSelectionDialog {
        onSelected = listener
        return this
    }

    fun setOnMultiSelected(listener: (List<CatalogModel>) -> Unit): ModelSelectionDialog {
        onMultiSelected = listener
        return this
    }

    fun setTitle(t: String): ModelSelectionDialog {
        title = t
        return this
    }

    fun show() {
        val dialogView = createDialogView()
        val dialog = AlertDialog.Builder(context)
            .setTitle(title)
            .setView(dialogView)
            .setPositiveButton("确定") { _, _ ->
                collectSelection()
            }
            .setNegativeButton("取消", null)
            .create()

        dialog.show()
    }

    private val selectedModels = mutableSetOf<CatalogModel>()
    private lateinit var container: LinearLayout
    private lateinit var searchInput: EditText
    private lateinit var tabHost: TabHost
    private var allModels: List<CatalogModel> = emptyList()

    private fun createDialogView(): View {
        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 16, 24, 16)
        }

        // 搜索框
        searchInput = EditText(context).apply {
            hint = "搜索模型..."
            setSingleLine(true)
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                    filterModels()
                }
                override fun afterTextChanged(s: Editable?) {}
            })
        }
        root.addView(searchInput)

        // TabHost 分类标签
        tabHost = TabHost(context).apply {
            id = android.R.id.tabhost
        }
        val tabWidget = android.widget.TabWidget(context).apply { id = android.R.id.tabs }
        val frameLayout = android.widget.FrameLayout(context).apply { id = android.R.id.tabcontent }

        val tabContainer = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            addView(tabWidget)
            addView(frameLayout)
        }
        tabHost.addView(tabContainer)
        tabHost.setup()

        // 加载模型数据
        allModels = loadModels()

        // 创建标签页
        val tabs = mapOf(
            "全部" to ModelManager.ModelCategory.ALL,
            "LLM" to ModelManager.ModelCategory.LLM,
            "嵌入" to ModelManager.ModelCategory.EMBEDDING,
            "语音" to ModelManager.ModelCategory.SPEECH,
            "图像" to ModelManager.ModelCategory.IMAGE
        )

        tabs.forEach { (tabLabel, cat) ->
            val tabContent = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                id = View.generateViewId()
            }
            frameLayout.addView(tabContent)

            tabHost.addTab(tabHost.newTabSpec(cat.name).apply {
                setIndicator(tabLabel)
                setContent(tabContent.id)
            })
        }

        // 模型列表容器
        container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
        }
        root.addView(container)

        // 初始显示
        populateModels(allModels)

        // 切换标签页时过滤
        tabHost.setOnTabChangedListener { tabId ->
            filterModels()
        }

        return root
    }

    private fun loadModels(): List<CatalogModel> {
        val models = mutableListOf<CatalogModel>()

        // 加载 LLM 模型
        if (category == ModelManager.ModelCategory.ALL || category == ModelManager.ModelCategory.LLM) {
            LocalLLMService.AVAILABLE_MODELS.forEach { config ->
                models.add(CatalogModel(
                    id = config.name,
                    name = config.name,
                    category = CatalogModel.ModelCategory.LLM,
                    sizeBytes = config.size,
                    description = config.description,
                    source = "本地内置",
                    isDownloaded = LocalLLMService.getInstance(context).isModelDownloaded(config.name),
                    downloadUrl = config.url,
                    recommendedRAM = config.requiredRAM,
                    quantization = extractQuantization(config.name),
                    language = if (config.name.contains("qwen", ignoreCase = true)) "中文" else "多语言"
                ))
            }
        }

        // 加载嵌入模型
        if (category == ModelManager.ModelCategory.ALL || category == ModelManager.ModelCategory.EMBEDDING) {
            LocalEmbeddingService.AVAILABLE_MODELS.forEach { config ->
                models.add(CatalogModel(
                    id = config.name,
                    name = config.name,
                    category = CatalogModel.ModelCategory.EMBEDDING,
                    sizeBytes = config.modelSize,
                    description = "${config.description}\n用途: ${config.recommendedFor}",
                    source = "本地内置",
                    isDownloaded = LocalEmbeddingService.getInstance(context).isModelDownloaded(config.name),
                    downloadUrl = config.modelUrl,
                    dimension = config.dimension,
                    maxSeqLength = config.maxSeqLength
                ))
            }
        }

        // 尝试加载在线目录缓存
        try {
            val catalogService = ModelCatalogService.getInstance(context)
            val cachedModels = catalogService.getCachedCatalog()
            models.addAll(cachedModels)
        } catch (e: Exception) {
            logger.warn("ModelSelection", "在线目录缓存不可用: ${e.message}")
        }

        return models.distinctBy { it.name.lowercase() }
    }

    private fun filterModels() {
        val query = searchInput.text.toString().lowercase().trim()
        val currentTab = tabHost.currentTabTag

        val cat = try { ModelManager.ModelCategory.valueOf(currentTab ?: "ALL") } catch (_: Exception) { ModelManager.ModelCategory.ALL }

        val filtered = allModels.filter { model ->
            val categoryMatch = cat == ModelManager.ModelCategory.ALL ||
                    model.category.name == cat.name ||
                    (cat == ModelManager.ModelCategory.EMBEDDING && model.category == CatalogModel.ModelCategory.EMBEDDING)

            val searchMatch = query.isEmpty() ||
                    model.name.lowercase().contains(query) ||
                    model.description.lowercase().contains(query)

            categoryMatch && searchMatch
        }

        populateModels(filtered)
    }

    private fun populateModels(models: List<CatalogModel>) {
        container.removeAllViews()
        selectedModels.clear()

        models.forEach { model ->
            val itemView = if (singleSelect) {
                createSingleSelectItem(model)
            } else {
                createMultiSelectItem(model)
            }
            container.addView(itemView)
        }

        if (models.isEmpty()) {
            container.addView(TextView(context).apply {
                text = "没有可用的模型"
                setPadding(32, 32, 32, 32)
            })
        }
    }

    private fun createSingleSelectItem(model: CatalogModel): View {
        val radioGroup = container.findViewById<RadioGroup>(View.generateViewId())
            ?: RadioGroup(context)

        return RadioButton(context).apply {
            text = buildItemLabel(model)
            tag = model
            setOnClickListener {
                selectedModels.clear()
                selectedModels.add(model)
            }
        }
    }

    private fun createMultiSelectItem(model: CatalogModel): View {
        return android.widget.CheckBox(context).apply {
            text = buildItemLabel(model)
            tag = model
            setOnCheckedChangeListener { _, isChecked ->
                if (isChecked) {
                    selectedModels.add(model)
                } else {
                    selectedModels.remove(model)
                }
            }
        }
    }

    private fun buildItemLabel(model: CatalogModel): String {
        return buildString {
            append(if (model.isDownloaded) "✅ " else "⬇️ ")
            append(model.name)
            if (model.sizeBytes > 0) append(" (${model.formattedSize})")
            if (model.dimension > 0) append(" [${model.dimension}维]")
            if (model.recommendedRAM > 0) append(" [${model.recommendedRAM}GB RAM]")
            if (model.quantization.isNotEmpty()) append(" [${model.quantization}]")
        }
    }

    private fun collectSelection() {
        if (singleSelect && selectedModels.isNotEmpty()) {
            onSelected?.invoke(selectedModels.first())
        } else if (!singleSelect) {
            onMultiSelected?.invoke(selectedModels.toList())
        }
    }

    private fun extractQuantization(name: String): String {
        return when {
            name.contains("Q4_0", ignoreCase = true) -> "Q4_0"
            name.contains("Q4_K_M", ignoreCase = true) -> "Q4_K_M"
            name.contains("Q5_K_M", ignoreCase = true) -> "Q5_K_M"
            name.contains("Q8_0", ignoreCase = true) -> "Q8_0"
            else -> ""
        }
    }
}
