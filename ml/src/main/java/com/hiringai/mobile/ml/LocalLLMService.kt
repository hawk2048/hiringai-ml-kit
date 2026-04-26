package com.hiringai.mobile.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.codeshipping.llamakotlin.LlamaModel
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.util.concurrent.TimeUnit

/**
 * 本地 LLM 推理服务 (单例模式)
 *
 * 支持两种推理方式：
 * 1. 设备端推理 — 通过 llama-kotlin-android (llama.cpp JNI) 加载 GGUF 模型
 * 2. 远程 Ollama — 通过 HTTP 调用远程 Ollama 服务器
 */
class LocalLLMService private constructor(private val context: Context) {

    private var model: LlamaModel? = null
    private var currentModelName: String = ""

    data class ModelConfig(
        val name: String,
        val url: String,
        val size: Long,
        val requiredRAM: Int,
        val contextSize: Int = 2048,
        val template: String = "chatml",
        val description: String = ""
    )

    companion object {
        private const val TAG = "LocalLLMService"

        @Volatile
        private var instance: LocalLLMService? = null

        private const val HF_MIRROR = "https://hf-mirror.com"

        val AVAILABLE_MODELS = listOf(
            // ========== DeepSeek-R1 蒸馏系列 (推理能力强!) ==========
            ModelConfig(
                name = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M",
                url = "$HF_MIRROR/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                size = 1_200_000_000,
                requiredRAM = 2,
                contextSize = 8192,
                template = "qwen2",
                description = "🧠 DeepSeek-R1 蒸馏版！1.5B参数，强大推理能力（数学/代码），推荐移动端"
            ),
            ModelConfig(
                name = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M",
                url = "$HF_MIRROR/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
                size = 4_680_000_000,
                requiredRAM = 6,
                contextSize = 8192,
                template = "qwen2",
                description = "🧠 DeepSeek-R1 蒸馏版！7B参数，更强推理，4.5GB内存"
            ),
            ModelConfig(
                name = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M",
                url = "$HF_MIRROR/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
                size = 4_700_000_000,
                requiredRAM = 6,
                contextSize = 8192,
                template = "llama",
                description = "🧠 DeepSeek-R1 蒸馏版！Llama架构8B，英文推理强"
            ),

            // ========== Qwen3.6 系列 (2026年4月最新!) ==========
            ModelConfig(
                name = "Qwen3.6-35B-A3B-UD-Q4_K_XL",
                url = "$HF_MIRROR/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                size = 22_000_000_000L,
                requiredRAM = 24,
                contextSize = 65536,
                template = "qwen35",
                description = "🔥 Qwen3.6 最新版！MoE架构，35B激活3B，256K上下文，支持201种语言，思维模式切换"
            ),

            // ========== Qwen3.5 系列 (2026年2月) ==========
            ModelConfig(
                name = "Qwen3.5-0.8B-UD-Q4_K_XL",
                url = "$HF_MIRROR/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
                size = 559_000_000,
                requiredRAM = 1,
                contextSize = 32768,
                template = "qwen35",
                description = "🆕 Qwen3.5 最强小模型！动态量化，3.5GB内存，中文能力大幅提升"
            ),
            ModelConfig(
                name = "Qwen3.5-2B-UD-Q4_K_XL",
                url = "$HF_MIRROR/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf",
                size = 1_100_000_000,
                requiredRAM = 2,
                contextSize = 32768,
                template = "qwen35",
                description = "🆕 Qwen3.5 均衡之选！2B参数，32K上下文，性价比最高"
            ),
            ModelConfig(
                name = "Qwen3.5-4B-UD-Q4_K_XL",
                url = "$HF_MIRROR/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q4_K_XL.gguf",
                size = 2_500_000_000,
                requiredRAM = 3,
                contextSize = 32768,
                template = "qwen35",
                description = "🆕 Qwen3.5 中小规模！4B参数，更强推理能力"
            ),
            ModelConfig(
                name = "Qwen3.5-9B-UD-Q4_K_XL",
                url = "$HF_MIRROR/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf",
                size = 5_500_000_000,
                requiredRAM = 6,
                contextSize = 32768,
                template = "qwen35",
                description = "🆕 Qwen3.5 高性能版！9B参数，接近大模型表现"
            ),

            // ========== Qwen3 系列 (2025年4月) ==========
            ModelConfig(
                name = "Qwen3-0.6B-Q4_0",
                url = "$HF_MIRROR/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf",
                size = 382_000_000,
                requiredRAM = 1,
                contextSize = 32768,
                template = "qwen3",
                description = "Qwen3 入门版！0.6B参数，支持思维/非思维模式切换"
            ),

            // ========== Qwen2.5 系列 ==========
            ModelConfig(
                name = "Qwen2.5-0.5B-Instruct-Q4_0",
                url = "$HF_MIRROR/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
                size = 394_774_816,
                requiredRAM = 1,
                contextSize = 2048,
                template = "chatml",
                description = "超轻量级，中文优化，推荐入门"
            ),
            ModelConfig(
                name = "Qwen2-0.5B-Instruct-Q4_0",
                url = "$HF_MIRROR/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0.5b-instruct-q4_0.gguf",
                size = 420_000_000,
                requiredRAM = 2,
                contextSize = 2048,
                template = "chatml",
                description = "Qwen2 基础版，中文能力出色"
            ),

            // ========== 其他轻量级模型 ==========
            ModelConfig(
                name = "Phi-2-Q4_0",
                url = "$HF_MIRROR/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf",
                size = 494_000_000,
                requiredRAM = 1,
                contextSize = 2048,
                template = "phi",
                description = "微软 Phi-2，极小体积，英文推理优秀"
            ),
            ModelConfig(
                name = "SmolLM2-1.7B-Instruct-Q4_0",
                url = "$HF_MIRROR/TheBloke/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct.q4_0.gguf",
                size = 1_000_000_000,
                requiredRAM = 1,
                contextSize = 2048,
                template = "chatml",
                description = "HuggingFace SmolLM2-1.7B，性能均衡"
            ),

            // ========== 中量级模型 (1-2GB RAM) ==========
            ModelConfig(
                name = "TinyLlama-1.1B-Chat-Q4_K_M",
                url = "$HF_MIRROR/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                size = 667_825_984,
                requiredRAM = 2,
                contextSize = 2048,
                template = "llama",
                description = "TinyLlama 1.1B，生态丰富，社区活跃"
            ),
            ModelConfig(
                name = "Gemma-2B-Q4_K_M",
                url = "$HF_MIRROR/TheBloke/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q4_k_m.gguf",
                size = 1_600_000_000,
                requiredRAM = 2,
                contextSize = 4096,
                template = "gemma",
                description = "Google Gemma-2B，指令遵循强"
            ),
            ModelConfig(
                name = "StableLM-3B-Q4_K_M",
                url = "$HF_MIRROR/TheBloke/StableLM-3B-4e1t-GGUF/resolve/main/stablelm-3b-4e1t-q4_k_m.gguf",
                size = 1_900_000_000,
                requiredRAM = 2,
                contextSize = 4096,
                template = "stablelm",
                description = "Stability AI StableLM-3B，长上下文"
            ),

            // ========== Gemma 4 系列 ==========
            ModelConfig(
                name = "gemma-4-e2b-q4_0",
                url = "$HF_MIRROR/google/gemma-4-e2b-it-GGUF/resolve/main/gemma-4-e2b-it-q4_0.gguf",
                size = 2_200_000_000,
                requiredRAM = 3,
                contextSize = 8192,
                template = "gemma",
                description = "Google Gemma 4 E2B - 高效小模型，指令遵循强"
            ),
            ModelConfig(
                name = "gemma-4-e4b-q4_0",
                url = "$HF_MIRROR/google/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-q4_0.gguf",
                size = 4_500_000_000,
                requiredRAM = 5,
                contextSize = 8192,
                template = "gemma",
                description = "Google Gemma 4 E4B - 中等规模，综合能力强"
            ),
            ModelConfig(
                name = "gemma-4-e4b-q5_k_m",
                url = "$HF_MIRROR/google/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-q5_k_m.gguf",
                size = 5_200_000_000,
                requiredRAM = 6,
                contextSize = 8192,
                template = "gemma",
                description = "Google Gemma 4 E4B Q5 - 更高精度量化版本"
            )
        )

        fun getInstance(context: Context): LocalLLMService {
            return instance ?: synchronized(this) {
                instance ?: LocalLLMService(context.applicationContext).also { instance = it }
            }
        }

        fun getModelsDir(context: Context): File = ModelStorage.getLLMDir(context)

        fun getEmbeddingModelDir(context: Context): File = ModelStorage.getEmbeddingDir(context)

        // 保存加载状态到SharedPreferences
        private const val PREFS_NAME = "ml_model_state"
        private const val KEY_LOADED_MODEL = "loaded_llm_model"

        fun saveLoadedModelName(context: Context, modelName: String) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(KEY_LOADED_MODEL, modelName)
                .apply()
        }

        fun getLoadedModelNameFromPrefs(context: Context): String? {
            return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .getString(KEY_LOADED_MODEL, null)
        }

        fun clearLoadedModelName(context: Context) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .remove(KEY_LOADED_MODEL)
                .apply()
        }

        /**
         * 静态版 token 估算（供 companion object 内部和外部使用）
         */
        fun estimateTokenCountStatic(text: String): Int {
            if (text.isEmpty()) return 0
            var cjkCount = 0
            var asciiWordCount = 0
            var punctCount = 0
            var inWord = false

            for (c in text) {
                val cp = c.code
                when {
                    cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF ||
                    cp in 0xF900..0xFAFF -> {
                        cjkCount++
                        if (inWord) { asciiWordCount++; inWord = false }
                    }
                    c.isLetter() -> { inWord = true }
                    c.isWhitespace() -> {
                        if (inWord) { asciiWordCount++; inWord = false }
                    }
                    else -> {
                        if (inWord) { asciiWordCount++; inWord = false }
                        punctCount++
                    }
                }
            }
            if (inWord) asciiWordCount++

            return ((cjkCount * 1.5) + (asciiWordCount * 1.3) + (punctCount * 0.1))
                .toInt().coerceAtLeast(1)
        }
    }

    val isModelLoaded: Boolean get() = model != null

    // ─── 流式指标数据类 ───────────────────────────────────

    /**
     * 流式 token 事件（兼容旧版接口）
     */
    data class StreamingToken(
        val token: String,
        val tokenIndex: Int,
        val timestampMs: Long
    )

    /**
     * 流式过程事件（实时 TTFT 捕获）
     */
    data class StreamingMetrics(
        val token: String,
        val charCount: Int,
        val tokenCount: Int,
        val firstTokenTimestampMs: Long,
        val isFirstToken: Boolean,
        val timestampMs: Long
    )

    /**
     * 完整流式推理结果（供 Benchmark 使用）
     */
    data class StreamingMetricsResult(
        val fullText: String,
        val totalDurationMs: Long,
        val ttftMs: Long,              // Time to First Token（真实值，非估算）
        val prefillTokenCount: Int,
        val prefillTps: Double,          // prompt tokens / TTFT(s)
        val outputTokenCount: Int,
        val decodeTps: Double,           // output tokens / decode(s)
        val avgTokenLatencyMs: Double
    )

    fun getLoadedModelName(): String = currentModelName

    fun isModelDownloaded(modelName: String): Boolean {
        val file = File(getModelsDir(context), "$modelName.gguf")
        return file.exists() && file.length() > 0
    }

    suspend fun downloadModel(
        config: ModelConfig,
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val targetFile = File(getModelsDir(context), "${config.name}.gguf")

            if (targetFile.exists() && targetFile.length() == config.size) {
                onProgress(100)
                return@withContext true
            }

            if (targetFile.exists()) {
                targetFile.delete()
            }

            onProgress(0)

            val url = URL(config.url)
            val connection = url.openConnection()
            connection.connect()
            connection.readTimeout = 60000
            connection.connectTimeout = 30000
            val contentLength = connection.contentLengthLong

            val tempFile = File(targetFile.parent, "${config.name}.gguf.tmp")
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
                Log.i(TAG, "Model downloaded: ${config.name} (${targetFile.length()} bytes)")
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

    suspend fun loadModel(
        config: ModelConfig,
        threads: Int = 4,
        gpuLayers: Int = 0
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            if (!SafeNativeLoader.loadLibrary("llama-android")) {
                Log.e(TAG, "llama.cpp native library not available")
                return@withContext false
            }

            unloadModel()

            val modelFile = File(getModelsDir(context), "${config.name}.gguf")
            if (!modelFile.exists()) {
                Log.e(TAG, "Model file not found: ${modelFile.absolutePath}")
                return@withContext false
            }

            model = LlamaModel.load(modelFile.absolutePath) {
                this.contextSize = config.contextSize
                this.threads = threads
                this.temperature = 0.7f
                this.topP = 0.9f
                this.maxTokens = 512
                this.gpuLayers = gpuLayers
            }
            currentModelName = config.name

            // 保存加载状态
            saveLoadedModelName(context, config.name)

            Log.i(TAG, "Model loaded: ${config.name}")
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "llama.cpp native library not found or incompatible", e)
            SafeNativeLoader.markCrashed("llama-android", context)
            model = null
            currentModelName = ""
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${config.name}", e)
            model = null
            currentModelName = ""
            false
        }
    }

    fun unloadModel() {
        try {
            model?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing model", e)
        }
        model = null
        currentModelName = ""

        // 清除加载状态
        clearLoadedModelName(context)
    }

    suspend fun generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ): String? = withContext(Dispatchers.IO) {
        val m = model
        if (m == null) {
            Log.w(TAG, "No model loaded, cannot generate locally")
            return@withContext null
        }

        try {
            val result = m.generate(prompt)
            Log.d(TAG, "Generated ${result.length} chars")
            result
        } catch (e: Exception) {
            Log.e(TAG, "Generation failed", e)
            null
        }
    }

    fun generateStream(prompt: String): Flow<String>? {
        val m = model
        if (m == null) {
            Log.w(TAG, "No model loaded, cannot stream generate")
            return null
        }
        return try {
            m.generateStream(prompt)
        } catch (e: Exception) {
            Log.e(TAG, "Stream generation failed", e)
            null
        }
    }

    /**
     * 流式生成，并实时测量 TTFT 和吞吐指标（第一个 token 到达即打时间戳）
     *
     * @return StreamingResult 包含 TTFT、prefill TPS、decode TPS、完整文本、token 计数
     */
    fun generateStreamingWithMetrics(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ): Flow<StreamingMetrics>? {
        val m = model
        if (m == null) {
            Log.w(TAG, "No model loaded, cannot stream generate")
            return null
        }
        return try {
            measureStreamingMetrics(prompt, maxTokens)
        } catch (e: Exception) {
            Log.e(TAG, "Stream generation failed", e)
            null
        }
    }

    /**
     * 真实 TTFT + 吞吐测量（非挂起，直接在 Flow 中记录时间戳）
     *
     * 使用方式：
     * ```kotlin
     * val result = llmService.measureStreamingMetrics(prompt, maxTokens)
     * result.collect { token ->
     *     if (result.firstTokenTimestampMs > 0 && token.isNotEmpty()) {
     *         val ttft = System.currentTimeMillis() - result.firstTokenTimestampMs
     *     }
     * }
     * ```
     */
    fun measureStreamingMetrics(
        prompt: String,
        maxTokens: Int = 512
    ): Flow<StreamingMetrics> = flow {
        val m = model
        if (m == null) {
            Log.w(TAG, "No model loaded")
            return@flow
        }

        val firstTokenTimestamp = AtomicLong(0L)
        val totalChars = AtomicInteger(0)
        val tokenCount = AtomicInteger(0)
        val prefillDone = AtomicBoolean(false)

        try {
            m.generateStream(prompt).collect { token ->
                // ── TTFT 捕获：第一个非空 token 到达时记录时间戳 ──
                if (firstTokenTimestamp.get() == 0L && token.isNotEmpty()) {
                    firstTokenTimestamp.set(System.currentTimeMillis())
                }

                totalChars.addAndGet(token.length)
                tokenCount.incrementAndGet()

                emit(
                    StreamingMetrics(
                        token = token,
                        charCount = totalChars.get(),
                        tokenCount = tokenCount.get(),
                        firstTokenTimestampMs = firstTokenTimestamp.get(),
                        isFirstToken = firstTokenTimestamp.get() != 0L && tokenCount.get() == 1,
                        timestampMs = System.currentTimeMillis()
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Streaming failed", e)
        }
    }.flowOn(Dispatchers.IO)

    /**
     * 完整 TTFT + 吞吐测量（挂起直到生成完成，返回汇总指标）
     *
     * 推荐用于 Benchmark Runner，因为它需要完整结果。
     */
    suspend fun measureFullStreamingMetrics(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ): StreamingMetricsResult = withContext(Dispatchers.IO) {
        val sb = StringBuilder()
        var firstTokenMs = 0L
        val start = System.currentTimeMillis()

        try {
            measureStreamingMetrics(prompt, maxTokens).collect { event ->
                sb.append(event.token)
                if (event.isFirstToken) {
                    firstTokenMs = event.timestampMs - start
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Full streaming failed", e)
        }

        val totalMs = System.currentTimeMillis() - start
        val text = sb.toString()
        val tokens = estimateTokenCountStatic(text)
        val ttft = firstTokenMs
        val decodeMs = (totalMs - ttft).coerceAtLeast(1)
        val decodeTokens = (tokens - estimateTokenCountStatic(prompt)).coerceAtLeast(1)
        val prefillTokens = estimateTokenCountStatic(prompt)

        StreamingMetricsResult(
            fullText = text,
            totalDurationMs = totalMs,
            ttftMs = ttft,
            prefillTokenCount = prefillTokens,
            prefillTps = if (ttft > 0) prefillTokens * 1000.0 / ttft else 0.0,
            outputTokenCount = decodeTokens,
            decodeTps = if (decodeMs > 0) decodeTokens * 1000.0 / decodeMs else 0.0,
            avgTokenLatencyMs = if (tokens > 0) totalMs.toDouble() / tokens else 0.0
        )
    }

    /**
     * 通过 Ollama API 生成文本
     */
    suspend fun generateViaOllama(
        ollamaUrl: String,
        model: String,
        prompt: String,
        maxTokens: Int = 512
    ): String? = withContext(Dispatchers.IO) {
        try {
            val jsonBody = """{"model":"$model","prompt":${escapeJson(prompt)},"stream":false,"options":{"num_predict":$maxTokens,"temperature":0.7}}"""

            val mediaType = "application/json; charset=utf-8".toMediaType()
            val requestBody = jsonBody.toRequestBody(mediaType)

            val client = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .build()

            val request = Request.Builder()
                .url("$ollamaUrl/api/generate")
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string()

            if (!response.isSuccessful) {
                Log.e(TAG, "Ollama request failed: ${response.code} - $body")
                return@withContext null
            }

            body?.let { parseOllamaResponse(it) }
        } catch (e: Exception) {
            Log.e(TAG, "Ollama call failed", e)
            null
        }
    }

    private fun parseOllamaResponse(json: String): String? {
        val key = "\"response\""
        val keyIndex = json.indexOf(key)
        if (keyIndex < 0) return null
        val colonIndex = json.indexOf(':', keyIndex + key.length)
        if (colonIndex < 0) return null
        val valueStart = json.indexOf('"', colonIndex) + 1
        val valueEnd = json.indexOf('"', valueStart)
        if (valueStart <= 0 || valueEnd <= valueStart) return null
        return json.substring(valueStart, valueEnd)
            .replace("\\n", "\n")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    }

    private fun escapeJson(str: String): String {
        val sb = StringBuilder("\"")
        for (ch in str) {
            when (ch) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> sb.append(ch)
            }
        }
        sb.append("\"")
        return sb.toString()
    }

    /**
     * 生成职位画像
     *
     * 使用 LLM 根据职位名称和要求生成结构化的职位画像
     *
     * @param job 职位实体
     * @return JSON 格式的职位画像字符串，包含 summary, keySkills, experience, education, salaryRange, highlights
     */
    suspend fun generateJobProfile(job: com.hiringai.mobile.ml.bridge.JobInfo): String {
        val prompt = buildJobProfilePrompt(job)
        val result: String?

        // 优先尝试本地模型，其次远程 Ollama
        result = if (isModelLoaded) {
            generate(prompt, maxTokens = 1024, temperature = 0.7f)
        } else {
            // 尝试从设置中获取 Ollama 配置
            val prefs = context.getSharedPreferences("ml_settings", Context.MODE_PRIVATE)
            val ollamaUrl = prefs.getString("ollama_url", "http://localhost:11434") ?: "http://localhost:11434"
            val ollamaModel = prefs.getString("ollama_model", "qwen2.5:0.5b") ?: "qwen2.5:0.5b"
            generateViaOllama(ollamaUrl, ollamaModel, prompt, maxTokens = 1024)
        }

        return result ?: buildEmptyProfile()
    }

    private fun buildJobProfilePrompt(job: com.hiringai.mobile.ml.bridge.JobInfo): String {
        return """
请根据以下职位信息，生成一个结构化的职位画像（JSON格式）。

职位名称: ${job.title}
职位要求: ${job.requirements}

请返回以下格式的JSON（必须严格是有效的JSON，不要包含其他内容）：
{
  "summary": "职位概述（1-2句话）",
  "keySkills": ["技能1", "技能2", "技能3"],
  "experience": "经验要求描述",
  "education": "学历要求",
  "salaryRange": "薪资范围（如：15K-25K/月）",
  "highlights": ["亮点1", "亮点2", "亮点3"]
}

请只返回JSON，不要包含其他解释文字。
        """.trimIndent()
    }

    private fun buildEmptyProfile(): String {
        return """
{
  "summary": "职位画像生成失败",
  "keySkills": [],
  "experience": "未提供",
  "education": "未提供",
  "salaryRange": "面议",
  "highlights": []
}
        """.trimIndent()
    }

    /**
     * 生成候选人画像
     *
     * 使用 LLM 根据候选人简历生成结构化的候选人画像
     *
     * @param candidate 候选人实体
     * @return JSON 格式的候选人画像字符串，包含 summary, skills, experience, education, strengths, weaknesses, highlights
     */
    suspend fun generateCandidateProfile(candidate: com.hiringai.mobile.ml.bridge.CandidateInfo): String {
        val prompt = buildCandidateProfilePrompt(candidate)
        val result: String?

        // 优先尝试本地模型，其次远程 Ollama
        result = if (isModelLoaded) {
            generate(prompt, maxTokens = 1024, temperature = 0.7f)
        } else {
            // 尝试从设置中获取 Ollama 配置
            val prefs = context.getSharedPreferences("ml_settings", Context.MODE_PRIVATE)
            val ollamaUrl = prefs.getString("ollama_url", "http://localhost:11434") ?: "http://localhost:11434"
            val ollamaModel = prefs.getString("ollama_model", "qwen2.5:0.5b") ?: "qwen2.5:0.5b"
            generateViaOllama(ollamaUrl, ollamaModel, prompt, maxTokens = 1024)
        }

        return result ?: buildEmptyCandidateProfile()
    }

    private fun buildCandidateProfilePrompt(candidate: com.hiringai.mobile.ml.bridge.CandidateInfo): String {
        val resumeContent = candidate.resume.ifEmpty { "无简历内容" }
        return """
请根据以下候选人简历信息，生成一个结构化的候选人画像（JSON格式）。

候选人姓名: ${candidate.name}
候选人邮箱: ${candidate.email.ifEmpty { "未提供" }}
候选人电话: ${candidate.phone.ifEmpty { "未提供" }}
简历内容:
$resumeContent

请返回以下格式的JSON（必须严格是有效的JSON，不要包含其他内容）：
{
  "summary": "候选人概述（1-2句话，概括候选人背景和特点）",
  "skills": ["技能1", "技能2", "技能3"],
  "experience": "工作经验描述（工作年限、主要经历）",
  "education": "教育背景（学校、学历、专业）",
  "strengths": ["优势1", "优势2", "优势3"],
  "weaknesses": ["不足1", "不足2"],
  "highlights": ["亮点1", "亮点2", "亮点3"]
}

请只返回JSON，不要包含其他解释文字。
        """.trimIndent()
    }

    private fun buildEmptyCandidateProfile(): String {
        return """
{
  "summary": "候选人画像生成失败",
  "skills": [],
  "experience": "未提供",
  "education": "未提供",
  "strengths": [],
  "weaknesses": [],
  "highlights": []
}
        """.trimIndent()
    }

    /**
     * 检查设置中是否配置了 Ollama
     */
    fun isOllamaConfigured(): Boolean {
        val prefs = context.getSharedPreferences("ml_settings", Context.MODE_PRIVATE)
        val url = prefs.getString("ollama_url", null)
        return !url.isNullOrEmpty()
    }

    /**
     * 获取配置的 Ollama URL
     */
    fun getOllamaUrl(): String {
        val prefs = context.getSharedPreferences("ml_settings", Context.MODE_PRIVATE)
        return prefs.getString("ollama_url", "http://localhost:11434") ?: "http://localhost:11434"
    }

    /**
     * 获取配置的 Ollama 模型
     */
    fun getOllamaModel(): String {
        val prefs = context.getSharedPreferences("ml_settings", Context.MODE_PRIVATE)
        return prefs.getString("ollama_model", "qwen2.5:0.5b") ?: "qwen2.5:0.5b"
    }
}