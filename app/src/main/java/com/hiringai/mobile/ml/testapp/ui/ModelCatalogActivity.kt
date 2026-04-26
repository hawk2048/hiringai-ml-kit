package com.hiringai.mobile.ml.testapp.ui

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.work.*
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.hiringai.mobile.ml.LocalEmbeddingService
import com.hiringai.mobile.ml.LocalLLMService
import com.hiringai.mobile.ml.ModelManager
import com.hiringai.mobile.ml.catalog.ModelCatalogService
import com.hiringai.mobile.ml.catalog.CatalogModel
import com.hiringai.mobile.ml.download.ModelDownloadWorker
import com.hiringai.mobile.ml.logging.MlLogger
import com.hiringai.mobile.ml.testapp.R
import com.hiringai.mobile.ml.testapp.ui.BenchmarkActivity
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
 * - 下载进度实时显示
 */
class ModelCatalogActivity : AppCompatActivity() {

    private lateinit var logger: MlLogger
    private lateinit var modelManager: ModelManager

    private lateinit var searchInput: com.google.android.material.textfield.TextInputEditText
    private lateinit var modelListContainer: LinearLayout
    private lateinit var loadingState: View
    private lateinit var emptyState: View
    private lateinit var btnRefresh: com.google.android.material.button.MaterialButton
    private lateinit var categoryChipGroup: ChipGroup

    private var currentCategory = ModelManager.ModelCategory.ALL
    private var allCatalogModels: List<CatalogModel> = emptyList()

    // 下载状态追踪（每个下载任务有独立的 session ID）
    private val downloadStates = mutableMapOf<String, DownloadState>()
    // 追踪每个模型的当前下载 session（用于检测重复下载）
    private val downloadSessions = mutableMapOf<String, Long>()
    // Session ID 生成器
    private var sessionIdCounter = System.currentTimeMillis()

    // WorkManager 进度观察
    private lateinit var workManager: WorkManager
    private val workInfoLiveDataMap = mutableMapOf<String, androidx.lifecycle.LiveData<List<WorkInfo>>>()

    // 广播接收器 - 监听 WorkManager 下载进度
    private val downloadProgressReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                "com.hiringai.ml.download.progress" -> {
                    val modelName = intent.getStringExtra("model_name") ?: return
                    val progress = intent.getIntExtra("progress", 0)
                    val speed = intent.getStringExtra("speed") ?: ""
                    updateDownloadProgress(modelName, progress, speed)
                }
                "com.hiringai.ml.download.complete" -> {
                    val modelName = intent.getStringExtra("model_name") ?: return
                    val success = intent.getBooleanExtra("success", false)
                    onDownloadComplete(modelName, success)
                }
            }
        }
    }

    data class DownloadState(
        var sessionId: Long = 0L,      // 下载会话 ID，用于区分多次下载
        var progress: Int = 0,
        var isDownloading: Boolean = false,
        var progressBar: LinearProgressIndicator? = null,
        var progressText: TextView? = null,
        var speedText: TextView? = null,
        var statusIcon: TextView? = null,
        var cardView: com.google.android.material.card.MaterialCardView? = null,
        // 速度计算相关
        var lastUpdateTime: Long = 0L,
        var lastProgress: Int = 0,
        var speed: String = "" // 格式: "1.2 MB/s"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_catalog)

        logger = MlLogger.getInstance(this)
        modelManager = ModelManager.getInstance(this)
        workManager = WorkManager.getInstance(this)

        // 注册下载进度广播
        val filter = IntentFilter().apply {
            addAction("com.hiringai.ml.download.progress")
            addAction("com.hiringai.ml.download.complete")
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(downloadProgressReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(downloadProgressReceiver, filter)
        }

        initViews()
        // 启动时尝试加载本地缓存，同时在后台同步最新模型目录
        loadModels(fetchOnline = false, syncOnline = true)
    }

    override fun onResume() {
        super.onResume()
        // 刷新下载状态
        refreshDownloadStates()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            unregisterReceiver(downloadProgressReceiver)
        } catch (e: Exception) {
            // ignore
        }
    }

    /**
     * 刷新所有活跃下载的状态
     */
    private fun refreshDownloadStates() {
        downloadStates.forEach { (modelName, state) ->
            if (state.isDownloading) {
                observeDownloadWork(modelName)
            }
        }
    }

    private fun initViews() {
        searchInput = findViewById(R.id.searchInput)
        modelListContainer = findViewById(R.id.modelRecyclerView)
        loadingState = findViewById(R.id.loadingState)
        emptyState = findViewById(R.id.emptyState)
        btnRefresh = findViewById(R.id.btnRefreshBanner)
        categoryChipGroup = findViewById(R.id.categoryChipGroup)

        // 分类 Chip 选择
        categoryChipGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            if (checkedIds.isNotEmpty()) {
                currentCategory = when (checkedIds.first()) {
                    R.id.chipAll -> ModelManager.ModelCategory.ALL
                    R.id.chipLlm -> ModelManager.ModelCategory.LLM
                    R.id.chipEmbedding -> ModelManager.ModelCategory.EMBEDDING
                    R.id.chipImage -> ModelManager.ModelCategory.IMAGE
                    R.id.chipSpeech -> ModelManager.ModelCategory.SPEECH
                    else -> ModelManager.ModelCategory.ALL
                }
                filterAndDisplay()
            }
        }

        // 搜索输入监听
        searchInput.setOnEditorActionListener { _, actionId, _ ->
            // 回车键触发搜索
            if (actionId == EditorInfo.IME_ACTION_SEARCH || actionId == EditorInfo.IME_ACTION_DONE) {
                performSearch()
                true
            } else {
                false
            }
        }

        // 搜索按钮点击
        searchInput.setOnKeyListener { _, keyCode, event ->
            if (keyCode == KeyEvent.KEYCODE_ENTER && event.action == KeyEvent.ACTION_DOWN) {
                performSearch()
                true
            } else {
                false
            }
        }

        // 刷新按钮点击
        btnRefresh.setOnClickListener {
            loadModels(fetchOnline = true)
        }
    }

    /**
     * 执行搜索（支持本地 + 云端）
     */
    private fun performSearch() {
        val query = searchInput.text?.toString()?.trim() ?: ""
        
        if (query.isEmpty()) {
            // 空搜索词，显示全部
            loadModels(fetchOnline = false, syncOnline = true)
            return
        }

        loadingState.visibility = View.VISIBLE
        emptyState.visibility = View.GONE
        modelListContainer.removeAllViews()

        lifecycleScope.launch {
            try {
                val catalogService = ModelCatalogService.getInstance(this@ModelCatalogActivity)

                // 1. 先加载本地缓存匹配
                val localModels = buildLocalCatalogModels()
                val localMatches = localModels.filter { model ->
                    model.name.contains(query, ignoreCase = true) ||
                    model.description.contains(query, ignoreCase = true)
                }

                // 2. 同时发起云端搜索
                val onlineMatches = withContext(Dispatchers.IO) {
                    catalogService.searchOnlineModels(query, limit = 20)
                }

                // 3. 合并结果（本地优先）
                val searchResults = localMatches + onlineMatches

                withContext(Dispatchers.Main) {
                    loadingState.visibility = View.GONE
                    if (searchResults.isEmpty()) {
                        emptyState.visibility = View.VISIBLE
                    } else {
                        // 显示搜索结果（只显示搜索结果，不混合全部模型）
                        displaySearchResults(searchResults)
                    }
                }

                logger.info("ModelCatalog", "搜索 '$query': 本地 ${localMatches.size} + 云端 ${onlineMatches.size} = ${searchResults.size} 个")
            } catch (e: Exception) {
                logger.error("ModelCatalog", "搜索失败", e)
                withContext(Dispatchers.Main) {
                    loadingState.visibility = View.GONE
                    Toast.makeText(this@ModelCatalogActivity, "Search failed: ${e.message}", Toast.LENGTH_SHORT).show()
                    loadModels()
                }
            }
        }
    }

    /**
     * 显示搜索结果（不同于 filterAndDisplay 的全部展示）
     */
    private fun displaySearchResults(results: List<CatalogModel>) {
        modelListContainer.removeAllViews()

        // 搜索结果标题
        val header = TextView(this).apply {
            text = "Results (${results.size})"
            setPadding(20, 24, 20, 8)
            textSize = 13f
            setTextColor(resources.getColor(R.color.text_secondary, theme))
        }
        modelListContainer.addView(header)

        // 按分类分组显示
        val grouped = results.groupBy { it.category }
        for ((category, models) in grouped) {
            if (models.isEmpty()) continue

            // 分类小标题
            val categoryHeader = createSectionHeader(getCategoryDisplayName(category), models.size, getCategoryColor(category))
            modelListContainer.addView(categoryHeader)

            models.forEach { model ->
                modelListContainer.addView(createModelCard(model))
            }
        }

        emptyState.visibility = View.GONE
    }

    /**
     * 加载模型列表
     * @param fetchOnline 是否强制从在线获取（用户手动刷新）
     * @param syncOnline 是否在后台同步最新数据（启动时自动执行）
     *
     * 优化：先立即显示本地模型，后台异步加载在线数据，不阻塞 UI
     */
    private fun loadModels(fetchOnline: Boolean = false, syncOnline: Boolean = false) {
        // ⚠️ 先立即显示已有的数据，不等待网络
        lifecycleScope.launch {
            withContext(Dispatchers.Main) {
                // 保留正在下载的状态用于恢复
                val activeDownloadStates = mutableMapOf<String, DownloadState>()
                downloadStates.forEach { (name, state) ->
                    if (state.isDownloading) {
                        activeDownloadStates[name] = state
                    }
                }

                // 1. 先立即显示本地内置模型（不显示 loading，用户立即看到内容）
                val localModels = buildLocalCatalogModels()
                val cachedModels = withContext(Dispatchers.IO) {
                    ModelCatalogService.getInstance(this@ModelCatalogActivity).getCachedCatalog()
                }
                allCatalogModels = localModels + cachedModels
                logger.info("ModelCatalog", "立即显示: ${localModels.size} 本地 + ${cachedModels.size} 缓存")

                emptyState.visibility = if (allCatalogModels.isEmpty()) View.VISIBLE else View.GONE
                loadingState.visibility = View.GONE
                filterAndDisplay()

                // 恢复正在下载的模型状态
                activeDownloadStates.forEach { (name, _) ->
                    restoreDownloadUI(name)
                }
            }

            // 2. 后台加载在线模型（静默执行，不阻塞 UI）
            if (fetchOnline || syncOnline) {
                try {
                    val onlineModels = withContext(Dispatchers.IO) {
                        ModelCatalogService.getInstance(this@ModelCatalogActivity).fetchOnlineCatalog()
                    }

                    if (onlineModels.isNotEmpty()) {
                        withContext(Dispatchers.Main) {
                            val merged = (allCatalogModels + onlineModels).distinctBy { it.id.lowercase() }
                            allCatalogModels = merged
                            filterAndDisplay()

                            val newCount = onlineModels.size - (allCatalogModels.size - merged.size)
                            if (newCount > 0 && fetchOnline) {
                                Toast.makeText(this@ModelCatalogActivity, "Loaded $newCount new models", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                } catch (e: Exception) {
                    logger.warn("ModelCatalog", "在线模型加载失败: ${e.message}")
                    // 静默失败，不打扰用户
                }
            }
        }
    }

    /**
     * 恢复下载 UI 状态（列表重建后）
     */
    private fun restoreDownloadUI(modelName: String) {
        val state = downloadStates[modelName] ?: return
        if (!state.isDownloading) return

        // 查找对应的模型卡片视图
        for (i in 0 until modelListContainer.childCount) {
            val card = modelListContainer.getChildAt(i)
            if (card is com.google.android.material.card.MaterialCardView) {
                // 通过查找标题来定位
                val content = card.getChildAt(0) as? LinearLayout ?: continue
                for (j in 0 until content.childCount) {
                    val child = content.getChildAt(j)
                    if (child is TextView && child.text.toString().contains(modelName)) {
                        // 找到卡片，恢复下载状态
                        state.cardView = card
                        state.statusIcon = content.getChildAt(0) as? TextView
                        state.progressBar = content.findViewById(android.R.id.progress)
                        state.progressText = content.findViewById(android.R.id.text1)
                        state.speedText = content.findViewById(android.R.id.text2)

                        // 显示下载 UI
                        state.progressBar?.visibility = View.VISIBLE
                        state.progressText?.visibility = View.VISIBLE
                        state.speedText?.visibility = View.VISIBLE
                        state.statusIcon?.text = "⏳"
                        card.setCardBackgroundColor(resources.getColor(R.color.downloading_background, theme))
                        return
                    }
                }
            }
        }
    }

    /**
     * 在后台同步最新模型目录（静默执行，不显示加载状态）
     */
    private fun syncOnlineCatalogInBackground() {
        lifecycleScope.launch {
            try {
                val catalogService = ModelCatalogService.getInstance(this@ModelCatalogActivity)
                val freshModels = withContext(Dispatchers.IO) {
                    catalogService.fetchOnlineCatalog()
                }

                if (freshModels.isNotEmpty()) {
                    // 合并到现有列表（去重）
                    val merged = (allCatalogModels + freshModels).distinctBy { it.id.lowercase() }
                    allCatalogModels = merged

                    logger.info("ModelCatalog", "后台同步完成: ${freshModels.size} 个新模型")
                    withContext(Dispatchers.Main) {
                        filterAndDisplay()
                        // 显示简短提示
                        Toast.makeText(this@ModelCatalogActivity, "Synced ${freshModels.size} models", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                // 后台同步失败，静默处理
                logger.warn("ModelCatalog", "后台同步失败: ${e.message}")
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

        val query = searchInput.text?.toString()?.lowercase()?.trim() ?: ""
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
            val header = createSectionHeader(getCategoryDisplayName(category), group.size, getCategoryColor(category))
            modelListContainer.addView(header)

            // 模型卡片
            group.forEach { model ->
                val card = createModelCard(model)
                modelListContainer.addView(card)
            }
        }

        if (filtered.isEmpty()) {
            emptyState.visibility = View.VISIBLE
        } else {
            emptyState.visibility = View.GONE
        }
    }

    private fun createModelCard(model: CatalogModel): View {
        // 根据模型类别获取颜色
        val categoryColor = getCategoryColor(model.category)

        val cardView = com.google.android.material.card.MaterialCardView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                setMargins(0, 6, 0, 6)
            }
            radius = resources.getDimension(R.dimen.radius_lg)
            cardElevation = resources.getDimension(R.dimen.elevation_sm)
            setCardBackgroundColor(resources.getColor(R.color.card_background, theme))
            strokeWidth = 0
            setOnClickListener {
                showModelDetailDialog(model)
            }
        }

        // 保存卡片引用
        val downloadState = downloadStates.getOrPut(model.name) { DownloadState() }
        downloadState.cardView = cardView

        // 左侧分类色条
        val colorBar = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                resources.getDimensionPixelSize(R.dimen.color_bar_width),
                LinearLayout.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(categoryColor)
        }

        val contentLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 16, 20, 16)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        // 标题行
        val titleLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }

        val titleText = TextView(this).apply {
            text = model.name
            textSize = 15f
            setTextColor(resources.getColor(R.color.text_primary, theme))
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            )
        }

        val statusIcon = TextView(this).apply {
            text = getStatusIcon(model)
            textSize = 14f
            setTextColor(getStatusColor(model))
            setPadding(16, 4, 0, 4)
        }
        downloadState.statusIcon = statusIcon

        titleLayout.addView(titleText)
        titleLayout.addView(statusIcon)

        // 描述行
        val descText = TextView(this).apply {
            text = model.description.take(100).ifEmpty { "暂无描述" }
            textSize = 13f
            setTextColor(resources.getColor(R.color.text_tertiary, theme))
            setMargins(0, 6, 0, 0)
            maxLines = 2
        }

        // 底部信息行
        val bottomLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            setMargins(0, 10, 0, 0)
        }

        // 大小标签
        val sizeTag = createTagChip(formatSize(model.sizeBytes), R.color.surface_variant, R.color.text_secondary)

        // 来源标签
        val isLocal = model.source == "本地内置"
        val sourceTag = createTagChip(
            if (isLocal) "Local" else "HF",
            if (isLocal) R.color.success_light else R.color.surface_variant,
            if (isLocal) R.color.success else R.color.text_tertiary
        )

        // 分类标签
        val categoryTag = createTagChip(
            getCategoryShortName(model.category),
            lightenColor(categoryColor),
            categoryColor
        )

        bottomLayout.addView(sizeTag)
        bottomLayout.addView(sourceTag)
        bottomLayout.addView(categoryTag)

        // 添加空白占据剩余空间
        bottomLayout.addView(View(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 0, 1f)
        })

        contentLayout.addView(titleLayout)
        contentLayout.addView(descText)
        contentLayout.addView(bottomLayout)

        // 进度条（下载中显示）
        val progressBar = LinearProgressIndicator(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = 12
            }
            trackThickness = resources.getDimensionPixelSize(R.dimen.progress_height)
            trackCornerRadius = resources.getDimensionPixelSize(R.dimen.radius_sm)
            setIndicatorColor(categoryColor)
            trackColor = lightenColor(categoryColor)
            visibility = View.GONE
        }
        downloadState.progressBar = progressBar

        // 进度 + 速度
        val progressLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            setMargins(0, 6, 0, 0)
            visibility = View.GONE
        }

        val progressText = TextView(this).apply {
            text = "0%"
            textSize = 12f
            setTextColor(categoryColor)
        }
        downloadState.progressText = progressText

        val speedText = TextView(this).apply {
            text = ""
            textSize = 11f
            setTextColor(resources.getColor(R.color.text_tertiary, theme))
            setMargins(12, 0, 0, 0)
        }
        downloadState.speedText = speedText

        progressLayout.addView(progressText)
        progressLayout.addView(speedText)

        contentLayout.addView(progressBar)
        contentLayout.addView(progressLayout)

        // 组装卡片
        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        mainLayout.addView(colorBar)
        mainLayout.addView(contentLayout)

        cardView.addView(mainLayout)

        // 恢复下载状态
        if (downloadState.isDownloading) {
            progressBar.visibility = View.VISIBLE
            progressLayout.visibility = View.VISIBLE
            progressBar.progress = downloadState.progress
            progressText.text = "${downloadState.progress}%"
        }

        return cardView
    }

    /**
     * 创建标签小芯片
     */
    private fun createTagChip(text: String, bgColorRes: Int, textColorRes: Int): TextView {
        return TextView(this).apply {
            this.text = text
            textSize = 11f
            setTextColor(resources.getColor(textColorRes, theme))
            setPadding(24, 6, 24, 6)
            setBackgroundColor(resources.getColor(bgColorRes, theme))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                marginEnd = 8
            }
            // 圆角背景
            val drawable = GradientDrawable()
            drawable.cornerRadius = 12f
            drawable.setColor(resources.getColor(bgColorRes, theme))
            background = drawable
        }
    }

    /**
     * 创建分类标题栏
     */
    private fun createSectionHeader(title: String, count: Int, color: Int): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            setPadding(20, 28, 20, 12)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }.also { layout ->
            // 颜色指示点
            val dot = View(this@ModelCatalogActivity).apply {
                layoutParams = LinearLayout.LayoutParams(12, 12).apply {
                    marginEnd = 10
                }
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(color)
                }
            }
            layout.addView(dot)

            // 标题文字
            val titleText = TextView(this@ModelCatalogActivity).apply {
                text = title
                textSize = 14f
                setTextColor(resources.getColor(R.color.text_primary, theme))
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }
            layout.addView(titleText)

            // 数量
            val countText = TextView(this@ModelCatalogActivity).apply {
                text = "($count)"
                textSize = 12f
                setTextColor(resources.getColor(R.color.text_tertiary, theme))
                setPadding(8, 0, 0, 0)
            }
            layout.addView(countText)
        }
    }

    /**
     * 获取模型类别显示名称
     */
    private fun getCategoryDisplayName(category: CatalogModel.ModelCategory): String {
        return when (category) {
            CatalogModel.ModelCategory.LLM -> "Large Language Models"
            CatalogModel.ModelCategory.EMBEDDING -> "Embedding Models"
            CatalogModel.ModelCategory.SPEECH -> "Speech Models"
            CatalogModel.ModelCategory.IMAGE -> "Image Models"
        }
    }

    /**
     * 获取模型类别的颜色
     */
    private fun getCategoryColor(category: CatalogModel.ModelCategory): Int {
        return when (category) {
            CatalogModel.ModelCategory.LLM -> ContextCompat.getColor(this, R.color.primary)
            CatalogModel.ModelCategory.EMBEDDING -> ContextCompat.getColor(this, R.color.success)
            CatalogModel.ModelCategory.SPEECH -> ContextCompat.getColor(this, R.color.accent)
            CatalogModel.ModelCategory.IMAGE -> ContextCompat.getColor(this, R.color.info)
        }
    }

    /**
     * 获取模型类别简称
     */
    private fun getCategoryShortName(category: CatalogModel.ModelCategory): String {
        return when (category) {
            CatalogModel.ModelCategory.LLM -> "LLM"
            CatalogModel.ModelCategory.EMBEDDING -> "Embed"
            CatalogModel.ModelCategory.SPEECH -> "Speech"
            CatalogModel.ModelCategory.IMAGE -> "Image"
        }
    }

    /**
     * 浅化颜色
     */
    private fun lightenColor(color: Int): Int {
        val factor = 0.2f
        val r = ((Color.red(color) * (1 - factor) + 255 * factor)).toInt()
        val g = ((Color.green(color) * (1 - factor) + 255 * factor)).toInt()
        val b = ((Color.blue(color) * (1 - factor) + 255 * factor)).toInt()
        return android.graphics.Color.rgb(r, g, b)
    }

    /**
     * 获取状态图标
     */
    private fun getStatusIcon(model: CatalogModel): String {
        val state = downloadStates[model.name]
        return when {
            model.isDownloaded -> "Installed"
            state?.isDownloading == true -> "Downloading"
            else -> "Get"
        }
    }

    /**
     * 获取状态颜色
     */
    private fun getStatusColor(model: CatalogModel): Int {
        val state = downloadStates[model.name]
        return when {
            model.isDownloaded -> ContextCompat.getColor(this, R.color.success)
            state?.isDownloading == true -> ContextCompat.getColor(this, R.color.primary)
            else -> ContextCompat.getColor(this, R.color.text_tertiary)
        }
    }

    private fun LinearLayout.LayoutParams.setMargins(left: Int, top: Int, right: Int, bottom: Int) {
        this.setMargins(left, top, right, bottom)
    }

    private fun TextView.setMargins(left: Int, top: Int, right: Int, bottom: Int) {
        val params = layoutParams as? LinearLayout.LayoutParams
        params?.setMargins(left, top, right, bottom)
    }

    private fun LinearLayout.setMargins(left: Int, top: Int, right: Int, bottom: Int) {
        val params = layoutParams as? LinearLayout.LayoutParams
        params?.setMargins(left, top, right, bottom)
    }

    /**
     * 模型下载状态枚举 - 完整状态机
     */
    enum class ModelDownloadStatus {
        NOT_INSTALLED,    // 未安装
        DOWNLOADING,      // 下载中
        PAUSED,           // 已暂停（预留）
        INSTALLED         // 已安装
    }

    private fun showModelDetailDialog(model: CatalogModel) {
        val dialogView = layoutInflater.inflate(R.layout.dialog_model_detail, null)

        // 初始化视图
        val dialogIcon = dialogView.findViewById<TextView>(R.id.dialogIcon)
        val dialogTitle = dialogView.findViewById<TextView>(R.id.dialogTitle)
        val dialogCategory = dialogView.findViewById<TextView>(R.id.dialogCategory)
        val statusIcon = dialogView.findViewById<ImageView>(R.id.statusIcon)
        val statusText = dialogView.findViewById<TextView>(R.id.statusText)
        val sizeText = dialogView.findViewById<TextView>(R.id.sizeText)
        val downloadProgress = dialogView.findViewById<LinearProgressIndicator>(R.id.downloadProgress)
        val progressInfo = dialogView.findViewById<LinearLayout>(R.id.progressInfo)
        val progressPercent = dialogView.findViewById<TextView>(R.id.progressPercent)
        val speedText = dialogView.findViewById<TextView>(R.id.speedText)
        val detailText = dialogView.findViewById<TextView>(R.id.detailText)

        // 按钮
        val primaryButtonRow = dialogView.findViewById<LinearLayout>(R.id.primaryButtonRow)
        val downloadingButtonRow = dialogView.findViewById<LinearLayout>(R.id.downloadingButtonRow)
        val installedButtonRow = dialogView.findViewById<LinearLayout>(R.id.installedButtonRow)

        val btnDownload = dialogView.findViewById<MaterialButton>(R.id.btnDownload)
        val btnBackgroundDownload = dialogView.findViewById<MaterialButton>(R.id.btnBackgroundDownload)
        val btnPause = dialogView.findViewById<MaterialButton>(R.id.btnPause)
        val btnStop = dialogView.findViewById<MaterialButton>(R.id.btnStop)
        val btnStartUsing = dialogView.findViewById<MaterialButton>(R.id.btnStartUsing)
        val btnRemove = dialogView.findViewById<MaterialButton>(R.id.btnRemove)
        val btnViewSource = dialogView.findViewById<MaterialButton>(R.id.btnViewSource)

        // 设置基础信息
        dialogTitle.text = model.name
        dialogCategory.text = getCategoryDisplayName(model.category)
        sizeText.text = formatSize(model.sizeBytes)

        // 设置模型图标（使用分类简称）
        val categoryColor = getCategoryColor(model.category)
        dialogIcon.text = getCategoryShortName(model.category)
        dialogIcon.setTextColor(categoryColor)

        // 构建详情
        val details = buildString {
            // 来源
            if (model.source.isNotEmpty()) {
                appendLine("Source: ${model.source}")
            }

            // 量化类型
            if (model.quantization.isNotEmpty()) {
                appendLine("Quantization: ${model.quantization}")
            }

            // 内存需求
            if (model.recommendedRAM > 0) {
                appendLine("RAM Required: ${model.recommendedRAM} GB")
            }

            // 上下文长度
            if (model.maxSeqLength > 0) {
                appendLine("Context: ${model.maxSeqLength} tokens")
            }

            // 向量维度（Embedding 模型）
            if (model.dimension > 0) {
                appendLine("Dimension: ${model.dimension}")
            }

            // 下载量
            if (model.downloadCount > 0) {
                appendLine("Downloads: ${model.formattedDownloads}")
            }

            // 发布时间
            if (model.releaseDate > 0) {
                appendLine("Released: ${model.formattedReleaseDate}")
            }

            // 语言支持
            if (model.language.isNotEmpty()) {
                appendLine("Language: ${model.language}")
            }

            // 描述
            if (model.description.isNotEmpty()) {
                appendLine()
                appendLine("───────────────")
                append(model.description)
            }
        }
        detailText.text = details

        // 获取当前下载状态
        val currentState = downloadStates[model.name]

        /**
         * 更新按钮状态 - 完整状态机
         */
        fun updateButtonState(status: ModelDownloadStatus) {
            when (status) {
                ModelDownloadStatus.NOT_INSTALLED -> {
                    // 未安装：显示下载按钮
                    primaryButtonRow.visibility = View.VISIBLE
                    downloadingButtonRow.visibility = View.GONE
                    installedButtonRow.visibility = View.GONE
                    btnDownload.visibility = View.VISIBLE
                    btnBackgroundDownload.visibility = View.VISIBLE
                    btnDownload.text = "Download"
                    btnBackgroundDownload.text = "Background"
                }
                ModelDownloadStatus.DOWNLOADING -> {
                    // 下载中：显示暂停/停止按钮
                    primaryButtonRow.visibility = View.GONE
                    downloadingButtonRow.visibility = View.VISIBLE
                    installedButtonRow.visibility = View.GONE
                    // 暂停功能暂不可用，禁用按钮
                    btnPause.isEnabled = false
                    btnPause.alpha = 0.5f
                    btnPause.text = "Pause (N/A)"
                }
                ModelDownloadStatus.PAUSED -> {
                    // 已暂停：显示继续/停止按钮（预留）
                    primaryButtonRow.visibility = View.GONE
                    downloadingButtonRow.visibility = View.VISIBLE
                    installedButtonRow.visibility = View.GONE
                }
                ModelDownloadStatus.INSTALLED -> {
                    // 已安装：显示开始使用/删除按钮
                    primaryButtonRow.visibility = View.GONE
                    downloadingButtonRow.visibility = View.GONE
                    installedButtonRow.visibility = View.VISIBLE
                }
            }
        }

        /**
         * 更新状态显示
         */
        fun updateStatusUI() {
            val state = downloadStates[model.name]
            when {
                model.isDownloaded -> {
                    statusIcon.setImageResource(R.drawable.ic_check)
                    statusIcon.setColorFilter(ContextCompat.getColor(this, R.color.success))
                    statusText.text = "Ready to use"
                    statusText.setTextColor(ContextCompat.getColor(this, R.color.success))
                    downloadProgress.visibility = View.GONE
                    progressInfo.visibility = View.GONE
                    updateButtonState(ModelDownloadStatus.INSTALLED)
                }
                state?.isDownloading == true -> {
                    statusIcon.setImageResource(R.drawable.ic_download)
                    statusIcon.setColorFilter(ContextCompat.getColor(this, R.color.primary))
                    statusText.text = "Downloading..."
                    statusText.setTextColor(ContextCompat.getColor(this, R.color.primary))
                    downloadProgress.visibility = View.VISIBLE
                    progressInfo.visibility = View.VISIBLE
                    downloadProgress.progress = state.progress
                    progressPercent.text = "${state.progress}%"
                    speedText.text = state.speed
                    updateButtonState(ModelDownloadStatus.DOWNLOADING)
                }
                else -> {
                    statusIcon.setImageResource(R.drawable.ic_cloud)
                    statusIcon.setColorFilter(ContextCompat.getColor(this, R.color.text_secondary))
                    statusText.text = "Not installed"
                    statusText.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
                    downloadProgress.visibility = View.GONE
                    progressInfo.visibility = View.GONE
                    updateButtonState(ModelDownloadStatus.NOT_INSTALLED)
                }
            }
        }

        // 初始状态
        updateStatusUI()

        // 创建对话框
        val dialog = androidx.appcompat.app.AlertDialog.Builder(this)
            .setView(dialogView)
            .setCancelable(true)
            .create()

        // 设置按钮点击事件（在 dialog 创建后）
        // 下载（前台）
        btnDownload.setOnClickListener {
            downloadModel(model, backgroundMode = false)
            updateButtonState(ModelDownloadStatus.DOWNLOADING)
        }

        // 后台下载
        btnBackgroundDownload.setOnClickListener {
            downloadModel(model, backgroundMode = true)
            updateButtonState(ModelDownloadStatus.DOWNLOADING)
            // 关闭对话框，下载继续在后台进行
            dialog.dismiss()
        }

        // 停止下载
        btnStop.setOnClickListener {
            cancelDownload(model)
            updateButtonState(ModelDownloadStatus.NOT_INSTALLED)
        }

        // 开始使用（跳转到 Benchmark 界面）
        btnStartUsing.setOnClickListener {
            // 根据模型类型跳转到相应界面
            when (model.category) {
                CatalogModel.ModelCategory.LLM,
                CatalogModel.ModelCategory.EMBEDDING -> {
                    // 跳转到 BenchmarkActivity 并自动选中该模型
                    val intent = Intent(this, BenchmarkActivity::class.java).apply {
                        putExtra(BenchmarkActivity.EXTRA_PRESELECT_MODEL, model.name)
                    }
                    startActivity(intent)
                }
                else -> {
                    // 其他类型暂时显示提示
                    Toast.makeText(this, "${model.category.name} model - benchmark coming soon", Toast.LENGTH_SHORT).show()
                }
            }
            dialog.dismiss()
        }

        // 删除模型
        btnRemove.setOnClickListener {
            showDeleteConfirmation(model) {
                updateStatusUI()
            }
        }

        // 查看来源
        btnViewSource.setOnClickListener {
            openModelSource(model)
        }

        // 如果正在下载，持续更新进度
        if (downloadStates[model.name]?.isDownloading == true) {
            val handler = android.os.Handler(android.os.Looper.getMainLooper())
            val updateRunnable = object : Runnable {
                override fun run() {
                    if (downloadStates[model.name]?.isDownloading == true) {
                        updateStatusUI()
                        handler.postDelayed(this, 500)
                    } else {
                        // 下载完成或停止，关闭对话框
                        if (dialog.isShowing) {
                            dialog.dismiss()
                        }
                    }
                }
            }
            handler.post(updateRunnable)
            dialog.setOnDismissListener {
                handler.removeCallbacks(updateRunnable)
                // 重要：不要在这里调用 loadModels()，让下载继续在后台进行
                // 如果下载完成，会通过广播/WorkManager Observer 触发 UI 更新
            }
        } else {
            dialog.setOnDismissListener {
                // 未下载时关闭弹窗也要刷新列表
            }
        }

        dialog.show()
    }

    /**
     * 显示删除确认对话框
     */
    private fun showDeleteConfirmation(model: CatalogModel, onDeleted: () -> Unit) {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Remove ${model.name}?")
            .setMessage("This will delete the model from your device. You can download it again later.")
            .setPositiveButton("Remove") { _, _ ->
                deleteModel(model)
                onDeleted()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /**
     * 打开模型来源链接
     */
    private fun openModelSource(model: CatalogModel) {
        val url = model.downloadUrl?.substringBefore("/resolve/")?.let { baseUrl ->
            if (baseUrl.contains("huggingface.co")) {
                baseUrl
            } else if (baseUrl.contains("modelscope.cn")) {
                baseUrl
            } else {
                // 从模型 ID 构建 HuggingFace URL
                "https://huggingface.co/${model.id}"
            }
        } ?: "https://huggingface.co/${model.id}"

        try {
            val intent = android.content.Intent(android.content.Intent.ACTION_VIEW, android.net.Uri.parse(url))
            startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(this, "Cannot open URL: $url", Toast.LENGTH_SHORT).show()
        }
    }

    private fun cancelDownload(model: CatalogModel) {
        // 取消 WorkManager 下载
        ModelDownloadWorker.cancel(this, model.name)
        modelManager.cancelDownload(model.name)
        val state = downloadStates[model.name]
        state?.let {
            it.isDownloading = false
            it.progressBar?.visibility = View.GONE
            it.progressText?.visibility = View.GONE
            it.speedText?.visibility = View.GONE
            updateStatusIcon(it, false)
            it.cardView?.setCardBackgroundColor(resources.getColor(R.color.card_background, theme))
            downloadSessions.remove(model.name)
        }
        Toast.makeText(this, "Download cancelled", Toast.LENGTH_SHORT).show()
        loadModels()
    }

    /**
     * 开始下载 - 使用 WorkManager 后台下载
     *
     * @param backgroundMode 是否为后台模式（关闭弹窗继续下载）
     */
    private fun downloadModel(model: CatalogModel, backgroundMode: Boolean = false) {
        val state = downloadStates.getOrPut(model.name) { DownloadState() }

        // 生成新的下载会话 ID
        val newSessionId = ++sessionIdCounter
        state.sessionId = newSessionId
        downloadSessions[model.name] = newSessionId

        // 初始化下载状态
        state.isDownloading = true
        state.progress = 0
        state.lastUpdateTime = System.currentTimeMillis()
        state.lastProgress = 0
        state.speed = ""

        // 更新 UI
        updateDownloadUI(state, 0, "")

        // 使用 WorkManager 启动后台下载
        ModelDownloadWorker.enqueue(
            context = this,
            modelName = model.name,
            modelUrl = model.downloadUrl,
            modelSize = model.sizeBytes
        )

        // 观察下载进度
        observeDownloadWork(model.name)

        Toast.makeText(this, "Download started: ${model.name}", Toast.LENGTH_SHORT).show()
    }

    /**
     * 观察 WorkManager 下载进度
     */
    private fun observeDownloadWork(modelName: String) {
        val workName = "model_download_$modelName"

        // 避免重复观察
        if (workInfoLiveDataMap.containsKey(workName)) return

        val liveData = workManager.getWorkInfosForUniqueWorkLiveData(workName)
        workInfoLiveDataMap[workName] = liveData

        liveData.observe(this) { workInfoList ->
            workInfoList.firstOrNull()?.let { workInfo ->
                when (workInfo.state) {
                    WorkInfo.State.RUNNING -> {
                        val progress = workInfo.progress.getInt("progress", 0)
                        val speed = workInfo.progress.getString("speed") ?: ""
                        updateDownloadProgress(modelName, progress, speed)
                    }
                    WorkInfo.State.SUCCEEDED -> {
                        onDownloadComplete(modelName, true)
                    }
                    WorkInfo.State.FAILED -> {
                        onDownloadComplete(modelName, false)
                    }
                    WorkInfo.State.CANCELLED -> {
                        onDownloadComplete(modelName, false)
                    }
                    else -> {}
                }

                // 清理观察
                if (workInfo.state.isFinished) {
                    workInfoLiveDataMap.remove(workName)
                }
            }
        }
    }

    /**
     * 更新下载进度 UI
     */
    private fun updateDownloadProgress(modelName: String, progress: Int, speed: String) {
        val state = downloadStates[modelName] ?: return
        if (!state.isDownloading) return

        runOnUiThread {
            state.progress = progress
            state.progressBar?.progress = progress
            state.progressText?.text = "$progress%"
            state.speedText?.text = speed

            // 根据进度更新颜色
            val color = when {
                progress < 30 -> ContextCompat.getColor(this, R.color.download_progress_start)
                progress < 70 -> ContextCompat.getColor(this, R.color.download_progress_mid)
                else -> ContextCompat.getColor(this, R.color.download_progress_end)
            }
            state.progressBar?.setIndicatorColor(color)
        }
    }

    /**
     * 下载完成回调
     */
    private fun onDownloadComplete(modelName: String, success: Boolean) {
        val state = downloadStates[modelName] ?: return

        runOnUiThread {
            state.isDownloading = false
            state.progressBar?.visibility = View.GONE
            state.progressText?.visibility = View.GONE
            state.speedText?.visibility = View.GONE
            updateStatusIcon(state, success)
            state.cardView?.setCardBackgroundColor(ContextCompat.getColor(this, R.color.card_background))

            if (success) {
                Toast.makeText(this, "$modelName downloaded", Toast.LENGTH_SHORT).show()
                logger.info("ModelCatalog", "模型下载完成: $modelName")
            } else {
                Toast.makeText(this, "$modelName download failed", Toast.LENGTH_SHORT).show()
                logger.error("ModelCatalog", "模型下载失败: $modelName")
            }

            // 刷新列表
            loadModels()
        }
    }

    /**
     * 更新下载 UI 状态
     */
    private fun updateDownloadUI(state: DownloadState, progress: Int, speed: String) {
        state.progressBar?.progress = progress
        state.progressText?.text = "$progress%"
        state.speedText?.text = speed
        state.progressBar?.visibility = View.VISIBLE
        state.progressText?.visibility = View.VISIBLE
        state.speedText?.visibility = View.VISIBLE
        state.statusIcon?.text = "↓"
        state.cardView?.setCardBackgroundColor(ContextCompat.getColor(this, R.color.downloading_background))
    }

    /**
     * 更新状态图标（使用图标而非 emoji）
     */
    private fun updateStatusIcon(state: DownloadState, isSuccess: Boolean) {
        state.statusIcon?.text = if (isSuccess) "✓" else "✗"
        state.statusIcon?.setTextColor(
            ContextCompat.getColor(
                this,
                if (isSuccess) R.color.success else R.color.error
            )
        )
    }

    private fun deleteModel(model: CatalogModel) {
        lifecycleScope.launch {
            val success = modelManager.deleteModel(model.name)
            withContext(Dispatchers.Main) {
                if (success) {
                    Toast.makeText(this@ModelCatalogActivity, "${model.name} removed", Toast.LENGTH_SHORT).show()
                    // 清除下载状态
                    downloadStates.remove(model.name)
                } else {
                    Toast.makeText(this@ModelCatalogActivity, "Delete failed", Toast.LENGTH_SHORT).show()
                }
                loadModels()
            }
        }
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1_000_000.0)
            bytes >= 1_000 -> "%.0f KB".format(bytes / 1_000.0)
            else -> "$bytes B"
        }
    }

    /**
     * 格式化下载速度
     */
    private fun formatSpeed(bytesPerSecond: Long): String {
        return when {
            bytesPerSecond >= 1_000_000_000 -> "%.1f GB/s".format(bytesPerSecond / 1_000_000_000.0)
            bytesPerSecond >= 1_000_000 -> "%.1f MB/s".format(bytesPerSecond / 1_000_000.0)
            bytesPerSecond >= 1_000 -> "%.1f KB/s".format(bytesPerSecond / 1_000.0)
            else -> "$bytesPerSecond B/s"
        }
    }
}
