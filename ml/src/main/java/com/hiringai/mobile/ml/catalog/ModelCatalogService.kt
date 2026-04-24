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
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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
     * 从在线源获取模型目录
     * 先尝试 ModelScope，再尝试 OpenXLab
     */
    suspend fun fetchOnlineCatalog(): List<CatalogModel> = withContext(Dispatchers.IO) {
        val models = mutableListOf<CatalogModel>()

        // 1. 从 ModelScope 获取 (主源)
        try {
            logger.info(TAG, "从 ModelScope 获取模型目录...")
            val msModels = fetchFromModelScope()
            models.addAll(msModels)
            logger.info(TAG, "ModelScope 获取到 ${msModels.size} 个模型")
        } catch (e: Exception) {
            logger.error(TAG, "ModelScope 获取失败", e)
        }

        // 2. 从 OpenXLab 获取 (备用源)
        try {
            logger.info(TAG, "从 OpenXLab 获取模型目录...")
            val oxModels = fetchFromOpenXLab()
            models.addAll(oxModels)
            logger.info(TAG, "OpenXLab 获取到 ${oxModels.size} 个模型")
        } catch (e: Exception) {
            logger.error(TAG, "OpenXLab 获取失败", e)
        }

        // 3. 去重 (按 name)
        val distinctModels = models.distinctBy { it.name.lowercase() }

        // 4. 缓存结果
        if (distinctModels.isNotEmpty()) {
            saveToCache(distinctModels)
        }

        distinctModels
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
                    tags = tags
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
