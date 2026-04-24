package com.hiringai.mobile.ml

import android.content.Context
import android.util.Log
import com.hiringai.mobile.ml.speech.LocalSpeechService
import com.hiringai.mobile.ml.speech.SpeechModelConfig
import com.hiringai.mobile.ml.speech.SpeechModelType
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * 统一模型管理服务
 *
 * 提供以下功能:
 * - 模型下载管理与进度跟踪
 * - 批量下载支持
 * - 下载取消支持
 * - 模型状态管理
 * - 存储空间管理
 */
class ModelManager private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"

        @Volatile
        private var instance: ModelManager? = null

        fun getInstance(context: Context): ModelManager {
            return instance ?: synchronized(this) {
                instance ?: ModelManager(context.applicationContext).also { instance = it }
            }
        }
    }

    /**
     * 模型类别
     */
    enum class ModelCategory {
        LLM,            // 大语言模型
        EMBEDDING,      // 嵌入模型
        IMAGE,          // 图像模型
        SPEECH,         // 语音模型
        ALL             // 所有类型
    }

    /**
     * 模型下载状态
     */
    sealed class DownloadState {
        object Idle : DownloadState()
        data class Downloading(val progress: Int, val speed: String = "") : DownloadState()
        data class Completed(val modelName: String) : DownloadState()
        data class Failed(val modelName: String, val error: String) : DownloadState()
        data class Cancelled(val modelName: String) : DownloadState()
    }

    /**
     * 模型信息
     */
    data class ModelInfo(
        val name: String,
        val category: ModelCategory,
        val sizeBytes: Long,
        val description: String,
        val isDownloaded: Boolean,
        val isLoaded: Boolean = false,
        val progress: Int = 0,
        val downloadState: DownloadState = DownloadState.Idle
    )

    /**
     * 批量下载任务
     */
    data class BatchDownloadTask(
        val id: String,
        val models: List<String>,
        val currentIndex: Int = 0,
        val overallProgress: Int = 0,
        val isCancelled: Boolean = false
    )

    // 下载状态流
    private val _downloadStates = MutableStateFlow<Map<String, DownloadState>>(emptyMap())
    val downloadStates: StateFlow<Map<String, DownloadState>> = _downloadStates.asStateFlow()

    // 活跃的下载任务
    private val activeDownloads = ConcurrentHashMap<String, Job>()

    // 批量下载任务
    private var batchDownloadJob: Job? = null

    /**
     * 获取所有可用模型列表
     */
    fun getAllModels(): List<ModelInfo> {
        val models = mutableListOf<ModelInfo>()

        // 语音模型
        LocalSpeechService.AVAILABLE_MODELS.forEach { config ->
            models.add(
                ModelInfo(
                    name = config.name,
                    category = ModelCategory.SPEECH,
                    sizeBytes = config.modelSize,
                    description = config.description,
                    isDownloaded = LocalSpeechService.getInstance(context).isModelDownloaded(config.name)
                )
            )
        }

        // 图像模型
        LocalImageModelService.AVAILABLE_MODELS.forEach { config ->
            models.add(
                ModelInfo(
                    name = config.name,
                    category = ModelCategory.IMAGE,
                    sizeBytes = config.modelSize,
                    description = config.description,
                    isDownloaded = LocalImageModelService.getInstance(context).isModelDownloaded(config.name)
                )
            )
        }

        // LLM 模型
        LocalLLMService.AVAILABLE_MODELS.forEach { config ->
            models.add(
                ModelInfo(
                    name = config.name,
                    category = ModelCategory.LLM,
                    sizeBytes = config.size,
                    description = config.description,
                    isDownloaded = LocalLLMService.getInstance(context).isModelDownloaded(config.name)
                )
            )
        }

        // 嵌入模型
        LocalEmbeddingService.AVAILABLE_MODELS.forEach { config ->
            models.add(
                ModelInfo(
                    name = config.name,
                    category = ModelCategory.EMBEDDING,
                    sizeBytes = config.modelSize,
                    description = config.description,
                    isDownloaded = LocalEmbeddingService.getInstance(context).isModelDownloaded(config.name)
                )
            )
        }

        return models
    }

    /**
     * 按类别获取模型列表
     */
    fun getModelsByCategory(category: ModelCategory): List<ModelInfo> {
        return if (category == ModelCategory.ALL) {
            getAllModels()
        } else {
            getAllModels().filter { it.category == category }
        }
    }

    /**
     * 获取已下载的模型列表
     */
    fun getDownloadedModels(): List<ModelInfo> {
        return getAllModels().filter { it.isDownloaded }
    }

    /**
     * 获取存储使用情况
     */
    fun getStorageUsage(): StorageInfo {
        val downloadedModels = getDownloadedModels()
        val totalSize = downloadedModels.sumOf { it.sizeBytes }

        // 获取模型目录大小
        val modelDirs = listOf(
            File(context.filesDir, "models"),
            File(context.filesDir, "speech_models"),
            File(context.filesDir, "image_models"),
            File(context.filesDir, "embedding_models")
        )

        var actualSize = 0L
        modelDirs.forEach { dir ->
            if (dir.exists()) {
                actualSize += getDirectorySize(dir)
            }
        }

        return StorageInfo(
            modelsCount = downloadedModels.size,
            totalModelSize = totalSize,
            actualStorageUsed = actualSize
        )
    }

    private fun getDirectorySize(dir: File): Long {
        var size = 0L
        dir.listFiles()?.forEach { file ->
            size += if (file.isDirectory) {
                getDirectorySize(file)
            } else {
                file.length()
            }
        }
        return size
    }

    data class StorageInfo(
        val modelsCount: Int,
        val totalModelSize: Long,
        val actualStorageUsed: Long
    ) {
        val formattedSize: String
            get() = formatSize(actualStorageUsed)

        private fun formatSize(bytes: Long): String {
            return when {
                bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1_000_000_000.0)
                bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1_000_000.0)
                bytes >= 1_000 -> "%.0f KB".format(bytes / 1_000.0)
                else -> "$bytes B"
            }
        }
    }

    /**
     * 下载单个模型
     */
    fun downloadModel(
        modelName: String,
        onProgress: (Int) -> Unit = {},
        onComplete: (Boolean) -> Unit = {}
    ): Job {
        // 取消已有的下载任务
        cancelDownload(modelName)

        val job = CoroutineScope(Dispatchers.IO).launch {
            try {
                updateDownloadState(modelName, DownloadState.Downloading(0))

                val success = downloadModelInternal(modelName) { progress ->
                    updateDownloadState(modelName, DownloadState.Downloading(progress))
                    onProgress(progress)
                }

                if (isActive) {
                    val state = if (success) {
                        DownloadState.Completed(modelName)
                    } else {
                        DownloadState.Failed(modelName, "下载失败")
                    }
                    updateDownloadState(modelName, state)
                    onComplete(success)
                }
            } catch (e: CancellationException) {
                updateDownloadState(modelName, DownloadState.Cancelled(modelName))
                Log.d(TAG, "Download cancelled: $modelName")
            } catch (e: Exception) {
                Log.e(TAG, "Download failed: $modelName", e)
                updateDownloadState(modelName, DownloadState.Failed(modelName, e.message ?: "未知错误"))
                onComplete(false)
            } finally {
                activeDownloads.remove(modelName)
            }
        }

        activeDownloads[modelName] = job
        return job
    }

    private suspend fun downloadModelInternal(modelName: String, onProgress: (Int) -> Unit): Boolean {
        // 查找模型配置并下载
        return when {
            // 语音模型
            LocalSpeechService.AVAILABLE_MODELS.any { it.name == modelName } -> {
                val config = LocalSpeechService.AVAILABLE_MODELS.find { it.name == modelName }!!
                LocalSpeechService.getInstance(context).downloadModel(config, onProgress)
            }
            // 图像模型
            LocalImageModelService.AVAILABLE_MODELS.any { it.name == modelName } -> {
                val config = LocalImageModelService.AVAILABLE_MODELS.find { it.name == modelName }!!
                LocalImageModelService.getInstance(context).downloadModel(config, onProgress)
            }
            // LLM 模型
            LocalLLMService.AVAILABLE_MODELS.any { it.name == modelName } -> {
                val config = LocalLLMService.AVAILABLE_MODELS.find { it.name == modelName }!!
                LocalLLMService.getInstance(context).downloadModel(config, onProgress)
            }
            // 嵌入模型
            LocalEmbeddingService.AVAILABLE_MODELS.any { it.name == modelName } -> {
                val config = LocalEmbeddingService.AVAILABLE_MODELS.find { it.name == modelName }!!
                LocalEmbeddingService.getInstance(context).downloadModel(config, onProgress)
            }
            else -> {
                Log.e(TAG, "Model not found: $modelName")
                false
            }
        }
    }

    /**
     * 批量下载模型
     */
    fun downloadModels(
        modelNames: List<String>,
        onProgress: (Int, String) -> Unit = { _, _ -> },
        onComplete: (Int, Int) -> Unit = { _, _ -> }
    ): Job {
        // 取消已有的批量下载
        cancelBatchDownload()

        val taskId = java.util.UUID.randomUUID().toString()

        batchDownloadJob = CoroutineScope(Dispatchers.IO).launch {
            var completed = 0
            var failed = 0

            modelNames.forEachIndexed { index, modelName ->
                if (!isActive) {
                    Log.d(TAG, "Batch download cancelled")
                    return@launch
                }

                val overallProgress = (index * 100) / modelNames.size
                onProgress(overallProgress, "正在下载: $modelName")

                val success = downloadModelInternal(modelName) { progress ->
                    val totalProgress = overallProgress + (progress / modelNames.size)
                    onProgress(totalProgress, "正在下载: $modelName ($progress%)")
                }

                if (success) {
                    completed++
                } else {
                    failed++
                }
            }

            onProgress(100, "下载完成")
            onComplete(completed, failed)
        }

        return batchDownloadJob!!
    }

    /**
     * 取消单个下载
     */
    fun cancelDownload(modelName: String) {
        activeDownloads[modelName]?.cancel()
        activeDownloads.remove(modelName)
        updateDownloadState(modelName, DownloadState.Cancelled(modelName))
    }

    /**
     * 取消批量下载
     */
    fun cancelBatchDownload() {
        batchDownloadJob?.cancel()
        batchDownloadJob = null
    }

    /**
     * 取消所有下载
     */
    fun cancelAllDownloads() {
        activeDownloads.values.forEach { it.cancel() }
        activeDownloads.clear()
        batchDownloadJob?.cancel()
        batchDownloadJob = null
    }

    /**
     * 删除已下载的模型
     */
    suspend fun deleteModel(modelName: String): Boolean = withContext(Dispatchers.IO) {
        try {
            var deleted = false

            // 删除语音模型
            val speechFile = File(LocalSpeechService.getModelsDir(context), "$modelName.onnx")
            if (speechFile.exists()) {
                deleted = speechFile.delete() || deleted
            }

            // 删除图像模型
            val imageFile = File(LocalImageModelService.getModelsDir(context), "$modelName.onnx")
            if (imageFile.exists()) {
                deleted = imageFile.delete() || deleted
            }

            // 删除 LLM 模型
            val llmFile = File(LocalLLMService.getModelsDir(context), "$modelName.gguf")
            if (llmFile.exists()) {
                deleted = llmFile.delete() || deleted
            }

            // 删除嵌入模型
            val embeddingFile = File(LocalLLMService.getEmbeddingModelDir(context), "$modelName.onnx")
            if (embeddingFile.exists()) {
                deleted = embeddingFile.delete() || deleted
            }

            if (deleted) {
                Log.i(TAG, "Model deleted: $modelName")
            }

            deleted
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete model: $modelName", e)
            false
        }
    }

    /**
     * 清理所有已下载的模型
     */
    suspend fun clearAllModels(): Int = withContext(Dispatchers.IO) {
        var count = 0

        val modelDirs = listOf(
            LocalSpeechService.getModelsDir(context),
            LocalImageModelService.getModelsDir(context),
            LocalLLMService.getModelsDir(context),
            LocalLLMService.getEmbeddingModelDir(context)
        )

        modelDirs.forEach { dir ->
            if (dir.exists()) {
                dir.listFiles()?.forEach { file ->
                    if (file.delete()) {
                        count++
                    }
                }
            }
        }

        Log.i(TAG, "Cleared $count models")
        count
    }

    /**
     * 更新下载状态
     */
    private fun updateDownloadState(modelName: String, state: DownloadState) {
        _downloadStates.update { current ->
            current + (modelName to state)
        }
    }

    /**
     * 获取单个模型的下载状态
     */
    fun getDownloadState(modelName: String): DownloadState {
        return _downloadStates.value[modelName] ?: DownloadState.Idle
    }

    /**
     * 检查是否有正在进行的下载
     */
    fun hasActiveDownloads(): Boolean {
        return activeDownloads.isNotEmpty() || batchDownloadJob?.isActive == true
    }

    /**
     * 获取推荐的模型组合
     */
    fun getRecommendedModels(): RecommendedModels {
        return RecommendedModels(
            llm = LocalLLMService.AVAILABLE_MODELS.firstOrNull {
                it.name.contains("qwen") || it.name.contains("phi")
            }?.name ?: LocalLLMService.AVAILABLE_MODELS.firstOrNull()?.name ?: "",
            embedding = "all-MiniLM-L6-v2",
            speech = "whisper-tiny",
            image = "mobilenet_v2_224"
        )
    }

    data class RecommendedModels(
        val llm: String,
        val embedding: String,
        val speech: String,
        val image: String
    )
}
