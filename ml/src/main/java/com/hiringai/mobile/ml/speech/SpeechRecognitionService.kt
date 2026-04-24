package com.hiringai.mobile.ml.speech

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL

/**
 * 语音识别服务 - 支持 Whisper, Paraformer, Cam++ 等模型
 *
 * 支持的模型:
 * - Whisper (OpenAI): 多语言语音识别，支持 transcribe 和 translate
 * - Paraformer (阿里): 中文语音识别，高精度
 * - Cam++ (阿里): 说话人识别/声纹识别
 */
class SpeechRecognitionService(private val context: Context) {

    companion object {
        private const val TAG = "SpeechRecognition"

        private const val HF_MIRROR = "https://hf-mirror.com"

        /**
         * 可用的语音模型配置
         */
        val AVAILABLE_MODELS = listOf(
            // ========== Whisper Models (OpenAI) ==========
            SpeechModelConfig(
                name = "whisper-tiny",
                modelUrl = "$HF_MIRROR/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
                modelSize = 75_000_000,
                type = SpeechModelType.WHISPER,
                language = "en",
                description = "Whisper Tiny English - 最小英文模型，39M参数",
                requiredRAM = 1
            ),
            SpeechModelConfig(
                name = "whisper-tiny-multi",
                modelUrl = "$HF_MIRROR/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
                modelSize = 75_000_000,
                type = SpeechModelType.WHISPER,
                language = "multi",
                description = "Whisper Tiny 多语言 - 支持99种语言",
                requiredRAM = 1
            ),
            SpeechModelConfig(
                name = "whisper-base",
                modelUrl = "$HF_MIRROR/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
                modelSize = 142_000_000,
                type = SpeechModelType.WHISPER,
                language = "multi",
                description = "Whisper Base 多语言 - 平衡速度与精度",
                requiredRAM = 1
            ),
            SpeechModelConfig(
                name = "whisper-small",
                modelUrl = "$HF_MIRROR/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
                modelSize = 466_000_000,
                type = SpeechModelType.WHISPER,
                language = "multi",
                description = "Whisper Small 多语言 - 更高精度",
                requiredRAM = 2
            ),

            // ========== Paraformer Models (阿里达摩院) ==========
            SpeechModelConfig(
                name = "paraformer-zh-streaming",
                modelUrl = "$HF_MIRROR/funasr/paraformer-large-asr/resolve/main/model.onnx",
                modelSize = 220_000_000,
                type = SpeechModelType.PARAFORMER,
                language = "zh",
                description = "Paraformer 中文流式识别 - 实时语音输入",
                requiredRAM = 2
            ),
            SpeechModelConfig(
                name = "paraformer-zh-offline",
                modelUrl = "$HF_MIRROR/funasr/paraformer-large-asr/resolve/main/model.onnx",
                modelSize = 220_000_000,
                type = SpeechModelType.PARAFORMER,
                language = "zh",
                description = "Paraformer 中文离线识别 - 高精度",
                requiredRAM = 2
            ),

            // ========== Cam++ Speaker Recognition (声纹识别) ==========
            SpeechModelConfig(
                name = "camplusplus",
                modelUrl = "$HF_MIRROR/funasr/campplus/resolve/main/model.onnx",
                modelSize = 25_000_000,
                type = SpeechModelType.CAMPLUSPLUS,
                language = "multi",
                description = "Cam++ 说话人识别 - 声纹验证",
                requiredRAM = 1
            )
        )

        @Volatile
        private var instance: SpeechRecognitionService? = null

        fun getInstance(context: Context): SpeechRecognitionService {
            return instance ?: synchronized(this) {
                instance ?: SpeechRecognitionService(context.applicationContext).also { instance = it }
            }
        }

        fun getModelsDir(context: Context): File {
            val dir = File(context.filesDir, "speech_models")
            if (!dir.exists()) dir.mkdirs()
            return dir
        }
    }

    private var loadedModel: SpeechModelConfig? = null
    private var isLoaded = false

    data class SpeechModelConfig(
        val name: String,
        val modelUrl: String,
        val modelSize: Long,
        val type: SpeechModelType,
        val language: String,
        val description: String,
        val requiredRAM: Int = 1
    )

    enum class SpeechModelType {
        WHISPER,        // OpenAI Whisper - ASR
        PARAFORMER,     // 阿里 Paraformer - 中文ASR
        CAMPLUSPLUS     // Cam++ - 声纹识别
    }

    data class RecognitionResult(
        val text: String,
        val language: String? = null,
        val confidence: Float = 0f,
        val durationMs: Long = 0,
        val segments: List<Segment> = emptyList()
    )

    data class Segment(
        val text: String,
        val startMs: Long,
        val endMs: Long,
        val confidence: Float = 0f
    )

    data class SpeakerEmbedding(
        val embedding: FloatArray,
        val speakerId: String? = null
    )

    /**
     * 检查模型是否已下载
     */
    fun isModelDownloaded(modelName: String): Boolean {
        val modelFile = File(getModelsDir(context), getModelFileName(modelName))
        return modelFile.exists() && modelFile.length() > 0
    }

    private fun getModelFileName(modelName: String): String {
        return when {
            modelName.startsWith("whisper") -> "$modelName.bin"
            else -> "$modelName.onnx"
        }
    }

    /**
     * 下载语音模型
     */
    suspend fun downloadModel(
        config: SpeechModelConfig,
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val fileName = getModelFileName(config.name)
            val targetFile = File(getModelsDir(context), fileName)

            if (targetFile.exists() && targetFile.length() >= config.modelSize * 0.9) {
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

            val tempFile = File(targetFile.parent, "$fileName.tmp")
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
                Log.i(TAG, "Speech model downloaded: ${config.name}")
                true
            } else {
                tempFile.delete()
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download failed for ${config.name}", e)
            false
        }
    }

    /**
     * 加载模型到内存
     */
    suspend fun loadModel(config: SpeechModelConfig): Boolean = withContext(Dispatchers.IO) {
        try {
            val fileName = getModelFileName(config.name)
            val modelFile = File(getModelsDir(context), fileName)

            if (!modelFile.exists()) {
                Log.e(TAG, "Model file not found: ${modelFile.absolutePath}")
                return@withContext false
            }

            // Model loading simulation - actual implementation would use:
            // - whisper.cpp for Whisper models via JNI
            // - ONNX Runtime for Paraformer/Cam++

            loadedModel = config
            isLoaded = true

            Log.i(TAG, "Speech model loaded: ${config.name}")
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
        loadedModel = null
        isLoaded = false
        Log.i(TAG, "Speech model unloaded")
    }

    /**
     * 语音识别 (Whisper/Paraformer)
     */
    suspend fun transcribe(
        audioPath: String,
        language: String? = null
    ): RecognitionResult? = withContext(Dispatchers.IO) {
        if (!isLoaded || loadedModel == null) {
            Log.w(TAG, "No model loaded for transcription")
            return@withContext null
        }

        val startTime = System.currentTimeMillis()

        try {
            val config = loadedModel!!

            when (config.type) {
                SpeechModelType.WHISPER -> transcribeWithWhisper(audioPath, language)
                SpeechModelType.PARAFORMER -> transcribeWithParaformer(audioPath)
                SpeechModelType.CAMPLUSPLUS -> {
                    // Cam++ is for speaker recognition, not transcription
                    RecognitionResult(
                        text = "Cam++ model does not support transcription. Use for speaker recognition.",
                        durationMs = System.currentTimeMillis() - startTime
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Transcription failed", e)
            null
        }
    }

    private suspend fun transcribeWithWhisper(audioPath: String, language: String?): RecognitionResult {
        // Placeholder - actual implementation would call whisper.cpp via JNI
        val startTime = System.currentTimeMillis()

        // Simulate processing time based on model size
        val processingTime = when (loadedModel?.name) {
            "whisper-tiny", "whisper-tiny-multi" -> 500L
            "whisper-base" -> 1000L
            "whisper-small" -> 2000L
            else -> 1000L
        }
        kotlinx.coroutines.delay(processingTime)

        return RecognitionResult(
            text = "[Whisper transcription placeholder - actual implementation requires whisper.cpp JNI]",
            language = language ?: loadedModel?.language,
            confidence = 0.85f,
            durationMs = System.currentTimeMillis() - startTime
        )
    }

    private suspend fun transcribeWithParaformer(audioPath: String): RecognitionResult {
        // Placeholder - actual implementation would use ONNX Runtime
        val startTime = System.currentTimeMillis()

        kotlinx.coroutines.delay(800)

        return RecognitionResult(
            text = "[Paraformer 识别结果占位符 - 实际实现需要 ONNX Runtime]",
            language = "zh",
            confidence = 0.90f,
            durationMs = System.currentTimeMillis() - startTime
        )
    }

    /**
     * 声纹识别 (Cam++)
     */
    suspend fun extractSpeakerEmbedding(audioPath: String): SpeakerEmbedding? = withContext(Dispatchers.IO) {
        if (!isLoaded || loadedModel?.type != SpeechModelType.CAMPLUSPLUS) {
            Log.w(TAG, "Cam++ model not loaded for speaker embedding")
            return@withContext null
        }

        try {
            // Placeholder - actual implementation would use ONNX Runtime
            val embedding = FloatArray(192) { (Math.random() * 2 - 1).toFloat() }

            SpeakerEmbedding(embedding = embedding)
        } catch (e: Exception) {
            Log.e(TAG, "Speaker embedding extraction failed", e)
            null
        }
    }

    /**
     * 获取模型信息
     */
    fun getLoadedModelName(): String = loadedModel?.name ?: ""
    fun isModelLoaded(): Boolean = isLoaded
}
