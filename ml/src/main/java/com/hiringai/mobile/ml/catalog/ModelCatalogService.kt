package com.hiringai.mobile.ml.catalog

import android.content.Context
import android.util.Log
import com.hiringai.mobile.ml.logging.MlLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * 模型目录服务
 *
 * 功能：
 * - 从国内权威模型源获取模型列表 (ModelScope, OpenXLab 等)
 * - 本地缓存 (JSON 文件)
 * - 模型分类与搜索
 * - 缓存有效期管理
 */
class ModelCatalogService private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ModelCatalog"
        private const val CACHE_DIR = "model_catalog"
        private const val CACHE_FILE = "catalog_cache.json"
        private const val CACHE_VALIDITY_MS = 24 * 60 * 60 * 1000L // 24小时

        // 国内模型源
        private const val MODELSCOPE_API = "https://modelscope.cn/api/v1/models"
        private const val OPENXLAB_API = "https://openxlab.org.cn/gw/algo-backend/api/v1/models"

        // ─── HuggingFace Hub API ────────────────────────────
        // HF 国内镜像（国内优先，速度更快）
        // hf-mirror.com 是国内最稳定的 HF 镜像站，API 路径与官方一致
        private const val HF_MIRROR_API = "https://hf-mirror.com/api"
        // 官方 HF API（镜像不可用时的备选）
        private const val HF_FALLBACK_API = "https://huggingface.co/api"

        @Volatile
        private var instance: ModelCatalogService? = null

        fun getInstance(context: Context): ModelCatalogService {
            return instance ?: synchronized(this) {
                instance ?: ModelCatalogService(context.applicationContext).also { instance = it }
            }
        }
    }

    private val logger by lazy { MlLogger.getInstance(context) }
    private val cacheDir by lazy { File(context.filesDir, CACHE_DIR).also { it.mkdirs() } }
    private val cacheFile by lazy { File(cacheDir, CACHE_FILE) }
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /**
     * 获取缓存的目录 (如果有效)
     */
    fun getCachedCatalog(): List<CatalogModel> {
        if (!cacheFile.exists()) return emptyList()

        return try {
            val cacheJson = cacheFile.readText()
            val cacheObj = JSONObject(cacheJson)

            // 检查缓存有效期
            val cacheTime = cacheObj.optLong("timestamp", 0)
            if (System.currentTimeMillis() - cacheTime > CACHE_VALIDITY_MS) {
                logger.info(TAG, "缓存已过期")
                return emptyList()
            }

            val modelsArray = cacheObj.optJSONArray("models") ?: return emptyList()
            parseModelArray(modelsArray)
        } catch (e: Exception) {
            logger.error(TAG, "读取缓存失败", e)
            emptyList()
        }
    }

    /**
     * 从所有在线源获取模型目录
     *
     * 优先级：HuggingFace（最新模型+发行时间）> ModelScope > OpenXLab
     *
     * @param sortByReleaseDate true=按发行时间倒序（最新优先），false=按下载量排序
     * @param limit 每个源最多返回多少个模型（默认 200）
     */
    suspend fun fetchOnlineCatalog(
        sortByReleaseDate: Boolean = true,
        limit: Int = 200
    ): List<CatalogModel> = withContext(Dispatchers.IO) {
        val models = mutableListOf<CatalogModel>()

        // 1. HuggingFace Hub（主源：最新模型 + 发行时间 + 下载量）
        try {
            logger.info(TAG, "从 HuggingFace Hub 获取最新模型...")
            val hfModels = fetchFromHuggingFace(sortByReleaseDate, limit)
            models.addAll(hfModels)
            logger.info(TAG, "HF Hub 获取到 ${hfModels.size} 个模型")
        } catch (e: Exception) {
            logger.error(TAG, "HF Hub 获取失败", e)
        }

        // 2. ModelScope（备用源）
        try {
            logger.info(TAG, "从 ModelScope 获取模型目录...")
            val msModels = fetchFromModelScope()
            models.addAll(msModels)
            logger.info(TAG, "ModelScope 获取到 ${msModels.size} 个模型")
        } catch (e: Exception) {
            logger.error(TAG, "ModelScope 获取失败", e)
        }

        // 3. OpenXLab（备用源）
        try {
            logger.info(TAG, "从 OpenXLab 获取模型目录...")
            val oxModels = fetchFromOpenXLab()
            models.addAll(oxModels)
            logger.info(TAG, "OpenXLab 获取到 ${oxModels.size} 个模型")
        } catch (e: Exception) {
            logger.error(TAG, "OpenXLab 获取失败", e)
        }

        // 4. 全局去重（按 id）
        val distinctModels = models.distinctBy { it.id.lowercase() }

        // 5. 按发行时间排序（最新优先），无发行时间的放最后
        val sorted = if (sortByReleaseDate) {
            distinctModels.sortedWith(
                compareByDescending<CatalogModel> { it.releaseDate }
                    .thenByDescending { it.downloadCount }
            )
        } else {
            distinctModels.sortedByDescending { it.downloadCount }
        }

        // 6. 缓存结果
        if (sorted.isNotEmpty()) {
            saveToCache(sorted)
        }

        sorted
    }

    // ─── HuggingFace Hub ───────────────────────────────────────────

    /**
     * 从 HuggingFace Hub API 获取模型列表
     *
     * HF Hub API 文档: https://huggingface.co/api/models
     * 支持字段：createdAt, lastModified, downloads, likes, pipeline_tag, tags
     *
     * @param sortByRelease true=按创建时间倒序（最新模型优先），false=按下载量
     * @param limit 每个任务类型最多取多少条（API 硬限制 100）
     */
    private fun fetchFromHuggingFace(
        sortByRelease: Boolean = true,
        limit: Int = 200
    ): List<CatalogModel> {
        val models = mutableListOf<CatalogModel>()

        val taskTypes = listOf(
            "text-generation" to CatalogModel.ModelCategory.LLM,
            "automatic-speech-recognition" to CatalogModel.ModelCategory.SPEECH,
            "image-classification" to CatalogModel.ModelCategory.IMAGE,
            "feature-extraction" to CatalogModel.ModelCategory.EMBEDDING,
            "text-to-image" to CatalogModel.ModelCategory.IMAGE,
            "fill-mask" to CatalogModel.ModelCategory.LLM
        )

        // ── 国内优先：先用镜像，失败则回退到官方 ──
        val baseApi = tryFetchWithMirror() ?: tryFetchWithFallback()
        if (baseApi == null) {
            logger.warn(TAG, "HF Hub (镜像 + 官方) 全部不可用，跳过 HF 数据源")
            return models
        }

        for ((task, category) in taskTypes) {
            try {
                val encodedTask = URLEncoder.encode(task, "UTF-8")
                val sortField = if (sortByRelease) "createdAt" else "downloads"
                // HF API: direction=1 表示降序（最大/最新在前）
                val url = "$baseApi/models?pipeline_tag=$encodedTask&sort=$sortField&direction=1&limit=100&full=true"

                val conn = URL(url).openConnection()
                conn.connectTimeout = 15000
                conn.readTimeout = 30000
                conn.setRequestProperty("Accept", "application/json")
                conn.setRequestProperty("User-Agent", "HiringAI-ML/2.1")

                val response = conn.getInputStream().bufferedReader().readText()
                val jsonArray = JSONArray(response)

                for (i in 0 until minOf(jsonArray.length(), 50)) {
                    val item = jsonArray.optJSONObject(i) ?: continue
                    val model = parseHuggingFaceItem(item, category, baseApi)
                    if (model != null) models.add(model)
                }
            } catch (e: Exception) {
                logger.warn(TAG, "HF Hub $task 获取失败: ${e.message}")
            }
        }

        // 单独获取 ONNX 分类下的模型
        try {
            val sortField = if (sortByRelease) "createdAt" else "downloads"
            val onnxUrl = "$baseApi/models?filter=onnx&sort=$sortField&direction=1&limit=50&full=true"
            val conn = URL(onnxUrl).openConnection()
            conn.connectTimeout = 15000
            conn.readTimeout = 30000
            conn.setRequestProperty("User-Agent", "HiringAI-ML/2.1")

            val resp = conn.getInputStream().bufferedReader().readText()
            val arr = JSONArray(resp)
            for (i in 0 until minOf(arr.length(), 30)) {
                val item = arr.optJSONObject(i) ?: continue
                val model = parseHuggingFaceItem(
                    item,
                    inferCategoryFromName(item.optString("id", "")),
                    baseApi
                )
                if (model != null && models.none { it.id == model.id }) {
                    models.add(model)
                }
            }
        } catch (e: Exception) {
            logger.warn(TAG, "HF Hub ONNX 分类获取失败: ${e.message}")
        }

        return models
    }

    /**
     * 优先尝试镜像 API，5 秒内响应即视为可用
     * @return 可用的 API 基础 URL，或 null（全部不可用）
     */
    private fun tryFetchWithMirror(): String? {
        return try {
            val url = URL("$HF_MIRROR_API/models?pipeline_tag=text-generation&sort=createdAt&direction=1&limit=1&full=true")
            val conn = url.openConnection() as java.net.HttpURLConnection
            conn.connectTimeout = 5000
            conn.readTimeout = 10000
            conn.requestMethod = "HEAD"
            val code = conn.responseCode
            conn.disconnect()
            if (code in 200..399) {
                logger.info(TAG, "HF 镜像可用: $HF_MIRROR_API")
                HF_MIRROR_API
            } else null
        } catch (e: Exception) {
            logger.warn(TAG, "HF 镜像不可用: ${e.message}")
            null
        }
    }

    /**
     * 官方 API 备选（镜像不可用时）
     */
    private fun tryFetchWithFallback(): String? {
        return try {
            val url = URL("$HF_FALLBACK_API/models?pipeline_tag=text-generation&sort=createdAt&direction=1&limit=1&full=true")
            val conn = url.openConnection() as java.net.HttpURLConnection
            conn.connectTimeout = 8000
            conn.readTimeout = 15000
            conn.requestMethod = "HEAD"
            val code = conn.responseCode
            conn.disconnect()
            if (code in 200..399) {
                logger.info(TAG, "使用官方 HF API: $HF_FALLBACK_API")
                HF_FALLBACK_API
            } else null
        } catch (e: Exception) {
            logger.warn(TAG, "官方 HF API 也不可用: ${e.message}")
            null
        }
    }

    /**
     * 解析 HF Hub API 返回的单个模型条目
     *
     * HF API 字段说明：
     * - id: "namespace/model-name" 格式
     * - createdAt: ISO 8601 创建时间（发行日期）
     * - lastModified: ISO 8601 最后修改时间
     * - downloads: 累计下载量
     * - likes: 点赞数
     * - pipeline_tag: 任务类型标签
     * - tags: 标签数组（含 "onnx" / "gguf" 标记）
     * - siblings: 文件列表（可判断 .onnx / .gguf / .safetensors 文件）
     *
     * @param baseApi 当前使用的 API 基础 URL（镜像或官方），用于生成正确的下载链接
     */
    private fun parseHuggingFaceItem(item: JSONObject, category: CatalogModel.ModelCategory, baseApi: String): CatalogModel? {
        return try {
            val id = item.optString("id", "")
            if (id.isEmpty()) return null

            val parts = id.split("/", limit = 2)
            val author = parts.getOrNull(0) ?: ""
            val name = parts.getOrNull(1) ?: id

            // 解析创建时间（ISO 8601 → epoch ms）
            val createdAt = parseIso8601(item.optString("createdAt", ""))
            val lastModified = parseIso8601(item.optString("lastModified", ""))

            // 解析文件列表，判断是否 ONNX / GGUF
            val siblings = item.optJSONArray("siblings") ?: JSONArray()
            var hasOnnx = false
            var hasGguf = false
            var modelFormat = ""
            var quantBits = ""

            for (i in 0 until siblings.length()) {
                val f = siblings.optJSONObject(i) ?: continue
                val rname = f.optString("rfilename", "")
                val lower = rname.lowercase()
                when {
                    lower.endsWith(".onnx") -> { hasOnnx = true; modelFormat = "onnx" }
                    lower.endsWith(".gguf") -> { hasGguf = true; modelFormat = "gguf" }
                    lower.endsWith(".safetensors") && modelFormat.isEmpty() -> { modelFormat = "safetensors" }
                    lower.endsWith(".pt") || lower.endsWith(".pth") -> {
                        if (modelFormat.isEmpty()) modelFormat = "pytorch"
                    }
                    // 量化位数字符串（如 Q4_K_M, Q8_0）
                    lower.contains("q4_") || lower.contains("q5_") ||
                    lower.contains("q6_") || lower.contains("q8_") -> {
                        val quant = lower.substringAfter(".gguf").substringBefore("_").takeIf { it.isNotEmpty() }
                            ?: run {
                                // 从文件名提取量化类型
                                val segs = rname.split("_", "-")
                                segs.find { s -> s.matches(Regex("Q[0-9]+.*")) } ?: ""
                            }
                        if (quant.isNotEmpty()) quantBits = quant
                    }
                }
            }

            // 标签
            val tagsArray = item.optJSONArray("tags") ?: JSONArray()
            val tags = (0 until tagsArray.length()).mapNotNull { tagsArray.optString(it) }

            // 下载量（HF API 有时返回 Int 有时返回 Long）
            val downloadsRaw = item.opt("downloads")
            val downloads = when (downloadsRaw) {
                is Number -> downloadsRaw.toLong()
                is String -> downloadsRaw.toLongOrNull() ?: 0L
                else -> 0L
            }

            // 点赞数
            val likes = item.optInt("likes", 0)

            // 下载 URL（根据实际使用的 API 源生成对应链接）
            val host = if (baseApi.contains("hf-mirror")) "hf-mirror.com" else "huggingface.co"
            val downloadUrl = "https://$host/$id"

            CatalogModel(
                id = id,
                name = name,
                category = category,
                sizeBytes = 0L,  // HF API 不直接返回文件大小，从 siblings 累加可在上传文件信息时做
                description = item.optString("modelId", ""),  // HF 没有 description，用 id 代替
                source = "HuggingFace",
                downloadUrl = downloadUrl,
                author = author,
                tags = tags,
                releaseDate = createdAt,
                lastModified = lastModified,
                downloadCount = downloads,
                likes = likes,
                pipelineTag = item.optString("pipeline_tag", ""),
                modelFileFormat = modelFormat,
                isGguf = hasGguf,
                quantizationBits = quantBits
            )
        } catch (e: Exception) {
            logger.warn(TAG, "HF 模型解析失败: ${e.message}")
            null
        }
    }

    /**
     * 解析 ISO 8601 时间字符串为 epoch milliseconds
     * 支持格式：2024-03-15T10:30:00Z, 2024-03-15T10:30:00.000Z
     */
    private fun parseIso8601(iso: String): Long {
        if (iso.isEmpty()) return 0L
        return try {
            val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
            sdf.timeZone = TimeZone.getTimeZone("UTC")
            sdf.parse(iso)?.time ?: 0L
        } catch (_: Exception) {
            try {
                val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
                sdf.timeZone = TimeZone.getTimeZone("UTC")
                sdf.parse(iso)?.time ?: 0L
            } catch (_: Exception) {
                0L
            }
        }
    }

    /**
     * 从 ModelScope API 获取模型列表
     *
     * ModelScope 是国内最大的 AI 模型社区 (阿里达摩院)
     * 支持按任务类型筛选: text-generation, feature-extraction, audio等
     */
    private fun fetchFromModelScope(): List<CatalogModel> {
        val models = mutableListOf<CatalogModel>()

        val taskTypes = mapOf(
            "text-generation" to CatalogModel.ModelCategory.LLM,
            "feature-extraction" to CatalogModel.ModelCategory.EMBEDDING,
            "automatic-speech-recognition" to CatalogModel.ModelCategory.SPEECH,
            "image-classification" to CatalogModel.ModelCategory.IMAGE
        )

        for ((taskType, category) in taskTypes) {
            try {
                val url = "$MODELSCOPE_API?task=$taskType&pageSize=50&sortBy=downloads"
                val connection = URL(url).openConnection()
                connection.connectTimeout = 15000
                connection.readTimeout = 30000

                val response = connection.getInputStream().bufferedReader().readText()
                val json = JSONObject(response)

                val data = json.optJSONObject("Data") ?: json
                val items = data.optJSONArray("Models") ?: data.optJSONArray("items") ?: continue

                for (i in 0 until items.length()) {
                    val item = items.optJSONObject(i) ?: continue
                    val model = parseModelScopeItem(item, category)
                    if (model != null) models.add(model)
                }
            } catch (e: Exception) {
                logger.warn(TAG, "ModelScope $taskType 类别获取失败: ${e.message}")
            }
        }

        return models
    }

    private fun parseModelScopeItem(item: JSONObject, category: CatalogModel.ModelCategory): CatalogModel? {
        return try {
            val name = item.optString("Name", item.optString("name", ""))
            val namespace = item.optString("Namespace", item.optString("namespace", ""))
            val fullId = if (namespace.isNotEmpty()) "$namespace/$name" else name

            CatalogModel(
                id = fullId,
                name = name,
                category = category,
                sizeBytes = parseSizeToBytes(item.optString("Size", "0")),
                description = item.optString("Description", item.optString("description", "")),
                source = "ModelScope",
                downloadUrl = "https://modelscope.cn/models/$fullId",
                author = namespace,
                tags = parseTags(item)
            )
        } catch (e: Exception) {
            null
        }
    }

    /**
     * 从 OpenXLab API 获取模型列表
     *
     * OpenXLab 是上海 AI Lab 开放的模型平台
     */
    private fun fetchFromOpenXLab(): List<CatalogModel> {
        val models = mutableListOf<CatalogModel>()

        try {
            val url = "$OPENXLAB_API?page=1&limit=50"
            val connection = URL(url).openConnection()
            connection.connectTimeout = 15000
            connection.readTimeout = 30000

            val response = connection.getInputStream().bufferedReader().readText()
            val json = JSONObject(response)

            val items = json.optJSONArray("data") ?: json.optJSONArray("items") ?: return models

            for (i in 0 until items.length()) {
                val item = items.optJSONObject(i) ?: continue
                val model = parseOpenXLabItem(item)
                if (model != null) models.add(model)
            }
        } catch (e: Exception) {
            logger.warn(TAG, "OpenXLab 获取失败: ${e.message}")
        }

        return models
    }

    private fun parseOpenXLabItem(item: JSONObject): CatalogModel? {
        return try {
            val name = item.optString("name", "")
            val category = inferCategoryFromName(name)

            CatalogModel(
                id = item.optString("id", name),
                name = name,
                category = category,
                sizeBytes = 0,
                description = item.optString("description", ""),
                source = "OpenXLab",
                downloadUrl = item.optString("url", ""),
                author = item.optString("author", "")
            )
        } catch (e: Exception) {
            null
        }
    }

    /**
     * 从模型名称推断分类
     */
    private fun inferCategoryFromName(name: String): CatalogModel.ModelCategory {
        val lower = name.lowercase()
        return when {
            lower.contains("whisper") || lower.contains("asr") || lower.contains("speech") ||
            lower.contains("paraformer") || lower.contains("tts") -> CatalogModel.ModelCategory.SPEECH

            lower.contains("embed") || lower.contains("bge-") || lower.contains("minilm") ||
            lower.contains("e5-") || lower.contains("sentence") -> CatalogModel.ModelCategory.EMBEDDING

            lower.contains("clip") || lower.contains("vit") || lower.contains("resnet") ||
            lower.contains("mobilenet") || lower.contains("efficientnet") ||
            lower.contains("ocr") || lower.contains("detection") -> CatalogModel.ModelCategory.IMAGE

            else -> CatalogModel.ModelCategory.LLM
        }
    }

    /**
     * 搜索模型 (本地缓存 + 内置模型)
     */
    fun searchModels(query: String): List<CatalogModel> {
        if (query.isBlank()) return getCachedCatalog()
        val lowerQuery = query.lowercase()
        return getCachedCatalog().filter { model ->
            model.name.lowercase().contains(lowerQuery) ||
                    model.description.lowercase().contains(lowerQuery) ||
                    model.tags.any { it.lowercase().contains(lowerQuery) } ||
                    model.author.lowercase().contains(lowerQuery)
        }
    }

    /**
     * 从 HuggingFace Hub 搜索模型（实时搜索，与缓存无关）
     * 
     * HF Hub API 支持的搜索参数：
     * - search: 搜索关键词（匹配模型名称/描述）
     * - pipeline_tag: 任务类型（text-generation, feature-extraction 等）
     * - filter: 标签过滤（onnx, gguf 等）
     * - sort: 排序字段（downloads, createdAt, likes）
     * - direction: 1=降序（最大/最新），-1=升序
     * - limit: 返回数量（API 硬限制 100）
     * 
     * @param query 搜索关键词
     * @param category 模型类别（用于筛选任务类型）
     * @param sortBy 排序方式（downloads/downloads_DESC/likes/createdAt）
     * @param limit 返回数量
     * @return 匹配的模型列表
     */
    suspend fun searchOnlineModels(
        query: String,
        category: CatalogModel.ModelCategory? = null,
        sortBy: String = "downloads",
        limit: Int = 30
    ): List<CatalogModel> = withContext(Dispatchers.IO) {
        val models = mutableListOf<CatalogModel>()

        // 决定 API 基础 URL
        val baseApi = tryFetchWithMirror() ?: tryFetchWithFallback() ?: run {
            logger.warn(TAG, "HF Hub 不可用，搜索失败")
            return@withContext models
        }

        // 根据类别确定 pipeline_tag
        val pipelineTag = when (category) {
            CatalogModel.ModelCategory.LLM -> "text-generation"
            CatalogModel.ModelCategory.EMBEDDING -> "feature-extraction"
            CatalogModel.ModelCategory.SPEECH -> "automatic-speech-recognition"
            CatalogModel.ModelCategory.IMAGE -> "image-classification"
            null -> null
        }

        try {
            // 构建搜索 URL
            val params = mutableListOf<String>()
            if (query.isNotBlank()) {
                params.add("search=${URLEncoder.encode(query, "UTF-8")}")
            }
            if (pipelineTag != null) {
                params.add("pipeline_tag=$pipelineTag")
            }
            // 排序：downloads 按下载量，createdAt 按时间
            when (sortBy) {
                "downloads", "downloads_DESC" -> {
                    params.add("sort=downloads")
                    params.add("direction=1")
                }
                "likes" -> {
                    params.add("sort=likes")
                    params.add("direction=1")
                }
                "createdAt" -> {
                    params.add("sort=createdAt")
                    params.add("direction=1")
                }
            }
            params.add("limit=$limit")
            params.add("full=true")

            val url = "$baseApi/models?${params.joinToString("&")}"
            logger.info(TAG, "搜索模型: $url")

            val conn = URL(url).openConnection()
            conn.connectTimeout = 15000
            conn.readTimeout = 30000
            conn.setRequestProperty("Accept", "application/json")
            conn.setRequestProperty("User-Agent", "HiringAI-ML/2.1")

            val response = conn.getInputStream().bufferedReader().readText()
            val jsonArray = JSONArray(response)

            for (i in 0 until jsonArray.length()) {
                val item = jsonArray.optJSONObject(i) ?: continue
                val inferredCategory = category ?: inferCategoryFromName(item.optString("id", ""))
                val model = parseHuggingFaceItem(item, inferredCategory, baseApi)
                if (model != null) {
                    models.add(model)
                }
            }

            logger.info(TAG, "搜索到 ${models.size} 个模型")
        } catch (e: Exception) {
            logger.error(TAG, "模型搜索失败: ${e.message}", e)
        }

        models
    }

    /**
     * 按分类获取模型
     */
    fun getModelsByCategory(category: CatalogModel.ModelCategory): List<CatalogModel> {
        return getCachedCatalog().filter { it.category == category }
    }

    /**
     * 保存到缓存
     */
    private fun saveToCache(models: List<CatalogModel>) {
        try {
            val cacheObj = JSONObject().apply {
                put("timestamp", System.currentTimeMillis())
                put("date", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date()))
                put("count", models.size)

                val modelsArray = JSONArray()
                models.forEach { model ->
                    modelsArray.put(modelToJsonObject(model))
                }
                put("models", modelsArray)
            }

            cacheFile.writeText(cacheObj.toString(2))
            logger.info(TAG, "模型目录已缓存: ${models.size} 个模型")
        } catch (e: Exception) {
            logger.error(TAG, "缓存写入失败", e)
        }
    }

    private fun modelToJsonObject(model: CatalogModel): JSONObject {
        return JSONObject().apply {
            put("id", model.id)
            put("name", model.name)
            put("category", model.category.name)
            put("sizeBytes", model.sizeBytes)
            put("description", model.description)
            put("source", model.source)
            put("downloadUrl", model.downloadUrl)
            put("author", model.author)
            put("quantization", model.quantization)
            put("language", model.language)
            put("license", model.license)
            put("dimension", model.dimension)
            put("maxSeqLength", model.maxSeqLength)
            put("recommendedRAM", model.recommendedRAM)
            // 发行信息
            put("releaseDate", model.releaseDate)
            put("lastModified", model.lastModified)
            put("downloadCount", model.downloadCount)
            put("likes", model.likes)
            put("pipelineTag", model.pipelineTag)
            put("modelFileFormat", model.modelFileFormat)
            put("isGguf", model.isGguf)
            put("quantizationBits", model.quantizationBits)

            val tagsArray = JSONArray()
            model.tags.forEach { tagsArray.put(it) }
            put("tags", tagsArray)
        }
    }

    private fun parseModelArray(array: JSONArray): List<CatalogModel> {
        val models = mutableListOf<CatalogModel>()
        for (i in 0 until array.length()) {
            try {
                val obj = array.getJSONObject(i)
                val tagsArray = obj.optJSONArray("tags")
                val tags = if (tagsArray != null) {
                    (0 until tagsArray.length()).mapNotNull { tagsArray.optString(it) }
                } else emptyList()

                models.add(CatalogModel(
                    id = obj.optString("id", ""),
                    name = obj.optString("name", ""),
                    category = CatalogModel.ModelCategory.fromString(obj.optString("category", "LLM")),
                    sizeBytes = obj.optLong("sizeBytes", 0),
                    description = obj.optString("description", ""),
                    source = obj.optString("source", "缓存"),
                    downloadUrl = obj.optString("downloadUrl", ""),
                    recommendedRAM = obj.optInt("recommendedRAM", 0),
                    dimension = obj.optInt("dimension", 0),
                    maxSeqLength = obj.optInt("maxSeqLength", 0),
                    quantization = obj.optString("quantization", ""),
                    language = obj.optString("language", ""),
                    author = obj.optString("author", ""),
                    license = obj.optString("license", ""),
                    tags = tags,
                    releaseDate = obj.optLong("releaseDate", 0L),
                    lastModified = obj.optLong("lastModified", 0L),
                    downloadCount = obj.optLong("downloadCount", 0L),
                    likes = obj.optInt("likes", 0),
                    pipelineTag = obj.optString("pipelineTag", ""),
                    modelFileFormat = obj.optString("modelFileFormat", ""),
                    isGguf = obj.optBoolean("isGguf", false),
                    quantizationBits = obj.optString("quantizationBits", "")
                ))
            } catch (_: Exception) {
            }
        }
        return models
    }

    private fun parseSizeToBytes(sizeStr: String): Long {
        if (sizeStr.isEmpty() || sizeStr == "0") return 0L
        return try {
            val lower = sizeStr.lowercase().trim()
            when {
                lower.endsWith("gb") -> (lower.removeSuffix("gb").trim().toDouble() * 1_000_000_000).toLong()
                lower.endsWith("mb") -> (lower.removeSuffix("mb").trim().toDouble() * 1_000_000).toLong()
                lower.endsWith("kb") -> (lower.removeSuffix("kb").trim().toDouble() * 1_000).toLong()
                else -> lower.toLongOrNull() ?: 0L
            }
        } catch (_: Exception) {
            0L
        }
    }

    private fun parseTags(item: JSONObject): List<String> {
        val tags = mutableListOf<String>()
        val tagsArray = item.optJSONArray("Tags") ?: item.optJSONArray("tags")
        if (tagsArray != null) {
            for (i in 0 until tagsArray.length()) {
                val tag = tagsArray.optString(i)
                if (tag.isNotEmpty()) tags.add(tag)
            }
        }
        return tags
    }

    /**
     * 清除缓存
     */
    fun clearCache() {
        cacheFile.delete()
        logger.info(TAG, "模型目录缓存已清除")
    }

    /**
     * 获取缓存信息
     */
    fun getCacheInfo(): String {
        if (!cacheFile.exists()) return "无缓存"
        val cacheJson = cacheFile.readText()
        val cacheObj = JSONObject(cacheJson)
        val timestamp = cacheObj.optLong("timestamp", 0)
        val count = cacheObj.optInt("count", 0)
        val date = cacheObj.optString("date", "未知")
        val isExpired = System.currentTimeMillis() - timestamp > CACHE_VALIDITY_MS

        return "缓存: $count 个模型, 日期: $date${if (isExpired) " (已过期)" else ""}"
    }
}
