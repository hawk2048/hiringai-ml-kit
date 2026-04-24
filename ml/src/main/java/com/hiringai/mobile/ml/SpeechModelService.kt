package com.hiringai.mobile.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL

/**
 * 语音模型服务
 *
 * 支持的模型类型:
 * 1. Whisper - OpenAI 语音识别模型
 * 2. Paraformer - 阿里达摩院端到端语音识别
 * 3. Cam++ - 说话人识别/验证模型
 */
class SpeechModelService(private val context: Context) {

    enum class ModelType {
        ASR,        // 自动语音识别 (Whisper, Paraformer)
        SPEAKER,    // 说话人识别 (Cam++)
        TTS         // 语音合成
    }

    data class SpeechModelConfig(
        val name: String,
        val modelUrl: String,
        val modelSize: Long,
        val type: ModelType,
        val description: String = "",
        val requiredRAM: Int = 1,
        val sampleRate: Int = 16000,
        val language: String = "auto"
    )

    companion object {
        private const val TAG = "SpeechModelService"

        private const val HF_MIRROR = "https://hf-mirror.com"
        private const val MODELSCOPE_MIRROR = "https://modelscope.cn/models"

        val AVAILABLE_MODELS = listOf(
            // ========== Whisper ASR Models ==========
            SpeechModelConfig(
                name = "whisper-tiny",
                modelUrl = "$HF_MIRROR/openai/whisper-tiny/resolve/main/model.onnx",
                modelSize = 40_000_000,
                type = ModelType.ASR,
                description = "OpenAI Whisper Tiny - 最小版本，快速推理",
                requiredRAM = 1,
                sampleRate = 16000,
                language = "multi"
            ),
            SpeechModelConfig(
                name = "whisper-base",
                modelUrl = "$HF_MIRROR/openai/whisper-base/resolve/main/model.onnx",
                modelSize = 75_000_000,
                type = ModelType.ASR,
                description = "OpenAI Whisper Base - 基础版本，平衡速度和精度",
                requiredRAM = 1,
                sampleRate = 16000,
                language = "multi"
            ),
            SpeechModelConfig(
                name = "whisper-small",
                modelUrl = "$HF_MIRROR/openai/whisper-small/resolve/main/model.onnx",
                modelSize = 240_000_000,
                type = ModelType.ASR,
                description = "OpenAI Whisper Small - 小型版本，较好的识别精度",
                requiredRAM = 2,
                sampleRate = 16000,
                language = "multi"
            ),

            // ========== Paraformer ASR Models ==========
            SpeechModelConfig(
                name = "paraformer-zh-streaming",
                modelUrl = "$MODELSCOPE_MIRROR/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/resolve/master/model.onnx",
                modelSize = 220_000_000,
                type = ModelType.ASR,
                description = "阿里 Paraformer 中文流式语音识别 - 高精度中文ASR",
                requiredRAM = 2,
                sampleRate = 16000,
                language = "zh"
            ),
            SpeechModelConfig(
                name = "paraformer-multilingual",
                modelUrl = "$MODELSCOPE_MIRROR/damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020/resolve/master/model.onnx",
                modelSize = 230_000_000,
                type = ModelType.ASR,
                description = "Paraformer 多语言语音识别 - 支持中英混合",
                requiredRAM = 2,
                sampleRate = 16000,
                language = "multi"
            ),

            // ========== Cam++ Speaker Recognition ==========
            SpeechModelConfig(
                name = "cam++-speaker",
                modelUrl = "$MODELSCOPE_MIRROR/damo/speech_campplus_sv_zh-cn_16k-common/resolve/master/model.onnx",
                modelSize = 30_000_000,
                type = ModelType.SPEAKER,
                description = "Cam++ 说话人识别 - 中文说话人验证和识别",
                requiredRAM = 1,
                sampleRate = 16000,
                language = "zh"
            ),
            SpeechModelConfig(
                name = "ecapa-tdnn-speaker",
                modelUrl = "$MODELSCOPE_MIRROR/damo/speech_ecapa-tdnn_sv_zh-cn_16k-common/resolve/master/model.onnx",
                modelSize = 25_000_000,
                type = ModelType.SPEAKER,
                description = "ECAPA-TDNN 说话人识别 - 高精度说话人嵌入",
                requiredRAM = 1,
                sampleRate = 16000,
                language = "multi"
            )
        )

        @Volatile
        private var instance: SpeechModelService? = null

        fun getInstance(context: Context): SpeechModelService {
            return instance ?: synchronized(this) {
                instance ?: SpeechModelService(context.applicationContext).also { instance = it }
            }
        }

        fun getModelsDir(context: Context): File {
            val dir = File(context.filesDir, "speech_models")
            if (!dir.exists()) dir.mkdirs()
            return dir
        }
    }

    private var currentModelName: String = ""
    private var currentModelType: ModelType? = null

    val isModelLoaded: Boolean get() = currentModelType != null

    fun getLoadedModelName(): String = currentModelName

    fun getLoadedModelType(): ModelType? = currentModelType

    /**
     * 检查模型是否已下载
     */
    fun isModelDownloaded(modelName: String): Boolean {
        val modelFile = File(getModelsDir(context), "$modelName.onnx")
        return modelFile.exists() && modelFile.length() > 0
    }

    /**
     * 下载语音模型
     */
    suspend fun downloadModel(
        config: SpeechModelConfig,
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val targetFile = File(getModelsDir(context), "${config.name}.onnx")

            if (targetFile.exists() && targetFile.length() == config.modelSize) {
                onProgress(100)
                return@withContext true
            }

            if (targetFile.exists()) {
                targetFile.delete()
            }

            onProgress(0)

            val url = URL(config.modelUrl)
            val connection = url.openConnection()
            connection.connect()
            connection.readTimeout = 60000
            connection.connectTimeout = 30000
            val contentLength = connection.contentLengthLong

            val tempFile = File(targetFile.parent, "${config.name}.onnx.tmp")
            connection.getInputStream().use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Long = 0
                    var lastProgress = -1

                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        output.write(buffer, 0, read)
                        bytesRead += read

                        if (contentLength > 0) {
                            val progress = (bytesRead * 100 / contentLength).toInt()
                            if (progress != lastProgress) {
                                lastProgress = progress
                                onProgress(progress)
                            }
                        }
                    }
                }
            }

            if (tempFile.renameTo(targetFile)) {
                onProgress(100)
                Log.i(TAG, "Speech model downloaded: ${config.name} (${targetFile.length()} bytes)")
                true
            } else {
                tempFile.delete()
                Log.e(TAG, "Failed to rename temp file for ${config.name}")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download failed for ${config.name}", e)
            false
        }
    }

    /**
     * 加载模型到内存 (模拟实现)
     * 实际 ONNX Runtime 推理需要额外的依赖
     */
    suspend fun loadModel(config: SpeechModelConfig): Boolean = withContext(Dispatchers.IO) {
        try {
            val modelFile = File(getModelsDir(context), "${config.name}.onnx")
            if (!modelFile.exists()) {
                Log.e(TAG, "Model file not found: ${modelFile.absolutePath}")
                return@withContext false
            }

            // 标记为已加载（实际需要初始化 ONNX Session）
            currentModelName = config.name
            currentModelType = config.type

            Log.i(TAG, "Speech model loaded: ${config.name} (type: ${config.type})")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load speech model: ${config.name}", e)
            false
        }
    }

    /**
     * 卸载模型
     */
    fun unloadModel() {
        currentModelName = ""
        currentModelType = null
        Log.i(TAG, "Speech model unloaded")
    }

    /**
     * 获取所有已下载的模型
     */
    fun getDownloadedModels(): List<SpeechModelConfig> {
        return AVAILABLE_MODELS.filter { isModelDownloaded(it.name) }
    }

    /**
     * 根据类型获取模型
     */
    fun getModelsByType(type: ModelType): List<SpeechModelConfig> {
        return AVAILABLE_MODELS.filter { it.type == type }
    }
}
