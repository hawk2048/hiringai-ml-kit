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
import android.widget.Button
import com.hiringai.mobile.ml.LocalEmbeddingService
import com.hiringai.mobile.ml.LocalLLMService
import com.hiringai.mobile.ml.ModelManager
import com.hiringai.mobile.ml.logging.MlLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

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
    private var sortByReleaseDate: Boolean = true  // 默认按发行时间排序

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

        // 刷新按钮行：刷新在线目录 + 排序切换
        val btnRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
        }

        val refreshBtn = Button(context).apply {
            text = "🔄 刷新模型目录"
            setOnClickListener { refreshOnlineCatalog() }
        }

        val sortBtn = Button(context).apply {
            fun updateSortLabel() {
                text = if (sortByReleaseDate) "📅 时间排序" else "⬇ 下载量排序"
            }
            updateSortLabel()
            setOnClickListener {
                sortByReleaseDate = !sortByReleaseDate
                updateSortLabel()
                // 重新排序后刷新列表
                val sortedModels = if (sortByReleaseDate) {
                    allModels.sortedWith(
                        compareByDescending<CatalogModel> { it.releaseDate }
                            .thenByDescending { it.downloadCount }
                    )
                } else {
                    allModels.sortedByDescending { it.downloadCount }
                }
                allModels = sortedModels
                filterModels()
            }
        }

        btnRow.addView(refreshBtn)
        btnRow.addView(sortBtn)
        root.addView(btnRow)

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

    /**
     * 异步刷新在线模型目录（HuggingFace / ModelScope / OpenXLab）
     * 刷新完成后自动更新列表并按发行时间排序
     */
    private fun refreshOnlineCatalog() {
        val catalogService = ModelCatalogService.getInstance(context)
        CoroutineScope(Dispatchers.Main).launch {
            val statusView = TextView(context).apply {
                text = "🔄 正在从 HuggingFace / ModelScope 获取最新模型..."
                setPadding(16, 8, 16, 8)
            }
            container.addView(statusView, 0)

            val freshModels = withContext(Dispatchers.IO) {
                catalogService.fetchOnlineCatalog(sortByReleaseDate = true, limit = 200)
            }

            // 移除状态文本
            container.removeViewAt(0)

            if (freshModels.isNotEmpty()) {
                // 合并到现有列表（去重）
                val merged = (allModels + freshModels).distinctBy { it.id.lowercase() }
                allModels = if (sortByReleaseDate) {
                    merged.sortedWith(
                        compareByDescending<CatalogModel> { it.releaseDate }
                            .thenByDescending { it.downloadCount }
                    )
                } else {
                    merged.sortedByDescending { it.downloadCount }
                }

                // 统计新增模型（30 天内）
                val newCount = allModels.count { it.isNew }
                val infoMsg = if (newCount > 0) {
                    "✅ 获取成功！共 ${allModels.size} 个模型（含 ${newCount} 个新模型 🆕）"
                } else {
                    "✅ 获取成功！共 ${allModels.size} 个模型"
                }

                val infoView = TextView(context).apply {
                    text = infoMsg
                    setPadding(16, 8, 16, 8)
                }
                container.addView(infoView, 0)

                // 2 秒后自动移除提示
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    if (container.childCount > 0) container.removeViewAt(0)
                }, 3000)

                filterModels()
            } else {
                val errView = TextView(context).apply {
                    text = "⚠️ 获取失败，请检查网络连接"
                    setPadding(16, 8, 16, 8)
                }
                container.addView(errView, 0)
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    if (container.childCount > 0) container.removeViewAt(0)
                }, 3000)
            }
        }
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
            // 状态图标：已下载 / 端侧可用
            append(if (model.isDownloaded) "✅ " else "⬇️ ")
            append(model.name)

            // 新模型徽章（最近 30 天发布）
            if (model.isNew) append(" 🆕")

            // 发行日期（在线模型）
            if (model.releaseDate > 0) {
                append(" [📅 ${model.formattedReleaseDate}]")
            }

            // 下载量（在线模型）
            if (model.downloadCount > 0) {
                append(" [⬇ ${model.formattedDownloads}]")
            }

            // 文件格式（ONNX / GGUF）
            if (model.modelFileFormat.isNotEmpty()) {
                append(" [${model.modelFileFormat.uppercase()}]")
            }

            // 量化类型
            if (model.quantizationBits.isNotEmpty()) {
                append(" [${model.quantizationBits}]")
            } else if (model.quantization.isNotEmpty()) {
                append(" [${model.quantization}]")
            }

            // 模型大小
            if (model.sizeBytes > 0) append(" (${model.formattedSize})")

            // 维度（嵌入模型）
            if (model.dimension > 0) append(" [${model.dimension}维]")

            // 内存需求
            if (model.recommendedRAM > 0) append(" [${model.recommendedRAM}GB RAM]")

            // 来源
            append(" — ${model.source}")
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
