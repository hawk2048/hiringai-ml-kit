package com.hiringai.mobile.ml

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.hiringai.mobile.ml.acceleration.AccelerationConfig
import com.hiringai.mobile.ml.acceleration.AcceleratorDetector
import com.hiringai.mobile.ml.acceleration.GPUDelegateManager
import com.hiringai.mobile.ml.acceleration.NNAPIManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * 本地 Embedding 服务
 * 使用 ONNX Runtime 运行 sentence-transformers 模型
 *
 * 工作流程：
 * 1. 下载 ONNX 模型 + tokenizer 词汇表到设备
 * 2. 加载模型到 OrtSession
 * 3. 对输入文本做简单 tokenization（WordPiece/BPE 分词）
 * 4. 运行推理，取 [CLS] token 的 hidden state 作为句子向量
 * 5. L2 归一化后返回
 */
class LocalEmbeddingService(private val context: Context) {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isModelLoaded: Boolean = false
    private var vocab: Map<String, Int> = emptyMap()

    // Acceleration configuration
    private var accelerationConfig: AccelerationConfig = AccelerationConfig.load(context)
    private var currentBackend: AccelerationConfig.Backend = AccelerationConfig.Backend.CPU
    private val acceleratorDetector: AcceleratorDetector by lazy { AcceleratorDetector(context) }

    data class EmbeddingModelConfig(
        val name: String,
        val modelUrl: String,       // 模型文件 URL（.onnx 或 .bin）
        val vocabUrl: String,       // tokenizer vocab URL
        val modelSize: Long,
        val dimension: Int,
        val maxSeqLength: Int = 256,
        val description: String = "",       // 模型描述/适用场景
        val recommendedFor: String = "",    // 推荐用途
        /** 模型文件扩展名，默认 .onnx；PyTorch 模型（如 bge-base-zh-v1.5）使用 .bin */
        val modelFileExtension: String = ".onnx"
    )

    companion object {
        private const val TAG = "LocalEmbedding"

        // 使用 hf-mirror.com 国内镜像解决 huggingface.co 被墙问题
        private const val HF_MIRROR = "https://hf-mirror.com"

        /**
         * 所有可用的本地 Embedding 模型
         *
         * ONNX 文件名规律（sentence-transformers 官方导出标准）：
         *   - model.onnx          — 原始 FP32 精度（最大）
         *   - model_O4.onnx       — O4 优化（体积减半，精度接近 FP32）⭐ 推荐
         *   - model_qint8_*.onnx — INT8 量化（平台特定加速指令）
         *
         * vocab 文件类型：
         *   - vocab.txt                  — BERT WordPiece 词表（~232kB）
         *   - sentencepiece.bpe.model    — SentencePiece BPE 词表（~5MB）
         *
         * ⚠️ bge-base-zh-v1.5 和 bge-large-zh-v1.5 暂无 ONNX 导出，
         *    当前使用 PyTorch bin（需更大内存和 ONNX Runtime PyTorch 后端）。
         *    建议关注 https://huggingface.co/BAAI/bge-base-zh-v1.5/discussions
         */
        val AVAILABLE_MODELS = listOf(
            // ── 英文 Embedding（超轻量，适合移动端）──────────────────────────
            EmbeddingModelConfig(
                name = "all-MiniLM-L6-v2",
                modelUrl = "$HF_MIRROR/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
                vocabUrl = "$HF_MIRROR/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
                modelSize = 91_000_000,
                dimension = 384,
                maxSeqLength = 256,
                description = "轻量级英文语义匹配模型，6层Transformer，ONNX FP32（90MB）",
                recommendedFor = "英文文本相似度计算、语义匹配（资源受限设备首选）"
            ),
            EmbeddingModelConfig(
                name = "all-MiniLM-L12-v2",
                modelUrl = "$HF_MIRROR/sentence-transformers/all-MiniLM-L12-v2/resolve/main/onnx/model.onnx",
                vocabUrl = "$HF_MIRROR/sentence-transformers/all-MiniLM-L12-v2/resolve/main/vocab.txt",
                modelSize = 133_000_000,
                dimension = 384,
                maxSeqLength = 256,
                description = "英文语义匹配模型，12层Transformer，更高精度（133MB）",
                recommendedFor = "英文文本相似度计算、语义匹配（需要更高精度时）"
            ),

            // ── 多语言 Embedding（中英双语，支持 50+ 语言）───────────────────
            EmbeddingModelConfig(
                name = "paraphrase-multilingual-MiniLM-L12-v2",
                modelUrl = "$HF_MIRROR/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model.onnx",
                vocabUrl = "$HF_MIRROR/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/vocab.txt",
                modelSize = 471_000_000,
                dimension = 768,
                maxSeqLength = 128,
                description = "中英双语语义模型，支持50+语言，12层Transformer，性价比最高（470MB）",
                recommendedFor = "中英双语语义搜索、跨语言检索、问答匹配"
            ),

            // ── BGE 英文系列（智源开源，MTEB 榜单高分）─────────────────────
            EmbeddingModelConfig(
                name = "bge-small-en-v1.5",
                modelUrl = "$HF_MIRROR/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
                vocabUrl = "$HF_MIRROR/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt",
                modelSize = 134_000_000,
                dimension = 384,
                maxSeqLength = 512,
                description = "BGE 英文小模型 v1.5，召回效果好，512最长序列（133MB）",
                recommendedFor = "英文文本召回、检索（高召回率场景）"
            ),
            EmbeddingModelConfig(
                name = "bge-base-en-v1.5",
                modelUrl = "$HF_MIRROR/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx",
                vocabUrl = "$HF_MIRROR/BAAI/bge-base-en-v1.5/resolve/main/vocab.txt",
                modelSize = 439_000_000,
                dimension = 768,
                maxSeqLength = 512,
                description = "BGE 英文基础模型 v1.5，768维向量，精度更高（438MB）",
                recommendedFor = "英文高精度语义搜索、英文文档摘要"
            ),

            // ── BGE 中文系列（智源开源，主流中文 Embedding）────────────────
            // ⚠️ 注意：bge-base-zh-v1.5 目前无 ONNX 导出，使用 PyTorch 模型
            EmbeddingModelConfig(
                name = "bge-base-zh-v1.5",
                modelUrl = "$HF_MIRROR/BAAI/bge-base-zh-v1.5/resolve/main/pytorch_model.bin",
                vocabUrl = "$HF_MIRROR/BAAI/bge-base-zh-v1.5/resolve/main/vocab.txt",
                modelSize = 410_000_000,
                dimension = 768,
                maxSeqLength = 512,
                description = "BGE 中文基础模型，暂无 ONNX，使用 PyTorch bin（409MB，需更大内存）",
                recommendedFor = "中文文本召回、语义匹配（推荐中文场景）",
                modelFileExtension = ".bin"
            ),

            // ── 多语言 E5 系列（支持 94 种语言，含中文）────────────────────
            EmbeddingModelConfig(
                name = "multilingual-e5-small",
                modelUrl = "$HF_MIRROR/intfloat/multilingual-e5-small/resolve/main/onnx/model_O4.onnx",
                vocabUrl = "$HF_MIRROR/intfloat/multilingual-e5-small/resolve/main/sentencepiece.bpe.model",
                modelSize = 219_000_000,
                dimension = 384,
                maxSeqLength = 512,
                description = "多语言 E5 小模型，O4优化版，支持94种语言，中英双语（218MB）",
                recommendedFor = "多语言语义搜索、跨语言检索、中英双语"
            ),

            // ── Nomic Embed（开源长文本 Embedding，8192 上下文）────────────
            EmbeddingModelConfig(
                name = "nomic-embed-text-v1.5",
                modelUrl = "$HF_MIRROR/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_O4.onnx",
                vocabUrl = "$HF_MIRROR/nomic-ai/nomic-embed-text-v1.5/resolve/main/vocab.txt",
                modelSize = 548_000_000,
                dimension = 768,
                maxSeqLength = 8192,
                description = "开源长文本 Embedding，支持8192超长上下文，O4优化版（547MB）",
                recommendedFor = "英文长文档嵌入、论文检索、长文本语义搜索"
            )
        )

        @Volatile
        private var instance: LocalEmbeddingService? = null

        fun getInstance(context: Context): LocalEmbeddingService {
            return instance ?: synchronized(this) {
                instance ?: LocalEmbeddingService(context.applicationContext).also { instance = it }
            }
        }

        // 保存加载状态到SharedPreferences
        private const val PREFS_NAME = "ml_embedding_state"
        private const val KEY_LOADED_MODEL = "loaded_embedding_model"

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

        // Special tokens for BERT-style models
        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"
        private const val UNK_TOKEN = "[UNK]"
        private const val PAD_TOKEN_ID = 0
        private const val CLS_TOKEN_ID = 101
        private const val SEP_TOKEN_ID = 102
    }

    val loaded: Boolean get() = isModelLoaded

    /**
     * 检查模型是否已下载
     */
    /**
     * 检查模型是否已下载（通过 config）
     */
    fun isModelDownloaded(config: EmbeddingModelConfig): Boolean {
        val modelFile = File(LocalLLMService.getEmbeddingModelDir(context), "${config.name}${config.modelFileExtension}")
        val vocabFile = File(LocalLLMService.getEmbeddingModelDir(context), "${config.name}.vocab.txt")
        return modelFile.exists() && vocabFile.exists()
    }

    /**
     * 检查模型是否已下载（通过 name 字符串，兼容旧调用）
     */
    fun isModelDownloaded(modelName: String): Boolean {
        val config = AVAILABLE_MODELS.find { it.name == modelName }
            ?: return false
        return isModelDownloaded(config)
    }

    /**
     * 下载 Embedding 模型和词汇表
     */
    suspend fun downloadModel(
        config: EmbeddingModelConfig,
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val dir = LocalLLMService.getEmbeddingModelDir(context)
            val modelFile = File(dir, "${config.name}${config.modelFileExtension}")
            val vocabFile = File(dir, "${config.name}.vocab.txt")

            // Download vocab first (small file)
            if (!vocabFile.exists()) {
                onProgress(0)
                downloadFile(config.vocabUrl, vocabFile)
                onProgress(20)
            }

            // Download model (large file)
            if (!modelFile.exists()) {
                downloadFileWithProgress(config.modelUrl, modelFile) { pct ->
                    // Map 20-100% range for model download
                    onProgress(20 + (pct * 80 / 100))
                }
            }

            onProgress(100)
            Log.i(TAG, "Embedding model downloaded: ${config.name}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Download failed for ${config.name}", e)
            false
        }
    }

    /**
     * 加载模型到内存
     * 
     * 安全策略（v1.1.5 起）：
     * 1. 使用 SafeNativeLoader 延迟加载 ONNX Runtime native 库
     * 2. 使用 AccelerationConfig 选择加速后端
     * 3. 自动回退：GPU -> NNAPI -> XNNPACK -> CPU
     * 4. 设备兼容性检测通过后才尝试加载
     * 5. 加载失败会记录 crash marker，下次启动自动跳过
     */
    suspend fun loadModel(config: EmbeddingModelConfig): Boolean = withContext(Dispatchers.IO) {
        try {
            val dir = LocalLLMService.getEmbeddingModelDir(context)
            val modelFile = File(dir, "${config.name}${config.modelFileExtension}")
            val vocabFile = File(dir, "${config.name}.vocab.txt")

            if (!modelFile.exists() || !vocabFile.exists()) {
                Log.e(TAG, "Model or vocab file not found")
                return@withContext false
            }

            // Step 1: Safe-load ONNX Runtime native library (lazy, guarded)
            if (!SafeNativeLoader.loadLibrary("onnxruntime")) {
                Log.e(TAG, "ONNX Runtime native library not available — ML features disabled")
                isModelLoaded = false
                return@withContext false
            }

            // Step 2: Load vocab
            vocab = loadVocab(vocabFile)
            Log.i(TAG, "Vocab loaded: ${vocab.size} tokens")

            // Step 3: Create ONNX Runtime session with acceleration
            env = OrtEnvironment.getEnvironment()
            val sessionOptions = createSessionOptionsWithAcceleration()

            session = env?.createSession(modelFile.absolutePath, sessionOptions)
            isModelLoaded = true

            // 保存加载状态
            saveLoadedModelName(context, config.name)

            Log.i(TAG, "Embedding model loaded: ${config.name} with backend: $currentBackend")
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "ONNX Runtime native library not found or incompatible", e)
            SafeNativeLoader.markCrashed("onnxruntime", context)
            isModelLoaded = false
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load embedding model", e)
            isModelLoaded = false
            false
        }
    }

    /**
     * Create ONNX Runtime session options with hardware acceleration
     * Implements fallback chain: GPU -> NNAPI -> XNNPACK -> CPU
     */
    private fun createSessionOptionsWithAcceleration(): OrtSession.SessionOptions {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

        // Get effective fallback chain from config
        val fallbackChain = accelerationConfig.getEffectiveFallbackChain()

        for (backend in fallbackChain) {
            when (backend) {
                AccelerationConfig.Backend.GPU -> {
                    val gpuResult = GPUDelegateManager.createSessionOptions(context, enableXNNPACKFallback = false)
                    if (gpuResult.usedGPU) {
                        currentBackend = AccelerationConfig.Backend.GPU
                        Log.i(TAG, "Using GPU acceleration")
                        return gpuResult.sessionOptions ?: sessionOptions
                    }
                    Log.d(TAG, "GPU not available, trying next backend")
                }
                AccelerationConfig.Backend.NNAPI -> {
                    if (NNAPIManager.isNNAPISafe(context)) {
                        val nnapiOptions = NNAPIManager.createSafeSessionOptions(context)
                        if (nnapiOptions != null) {
                            currentBackend = AccelerationConfig.Backend.NNAPI
                            Log.i(TAG, "Using NNAPI acceleration")
                            return nnapiOptions
                        }
                    }
                    Log.d(TAG, "NNAPI not safe, trying next backend")
                }
                AccelerationConfig.Backend.XNNPACK -> {
                    // XNNPACK is enabled by default in ONNX Runtime Android
                    currentBackend = AccelerationConfig.Backend.XNNPACK
                    sessionOptions.setIntraOpNumThreads(accelerationConfig.globalSettings.defaultThreads)
                    Log.i(TAG, "Using XNNPACK CPU optimization")
                    return sessionOptions
                }
                AccelerationConfig.Backend.CPU -> {
                    currentBackend = AccelerationConfig.Backend.CPU
                    sessionOptions.setIntraOpNumThreads(accelerationConfig.globalSettings.defaultThreads)
                    Log.i(TAG, "Using CPU-only mode")
                    return sessionOptions
                }
            }
        }

        // Fallback to CPU
        currentBackend = AccelerationConfig.Backend.CPU
        sessionOptions.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
        Log.i(TAG, "Using CPU fallback")
        return sessionOptions
    }

    /**
     * Get current acceleration backend
     */
    fun getCurrentBackend(): AccelerationConfig.Backend = currentBackend

    /**
     * Set acceleration configuration
     */
    fun setAccelerationConfig(config: AccelerationConfig) {
        this.accelerationConfig = config
        AccelerationConfig.save(context, config)
    }

    /**
     * Get acceleration configuration
     */
    fun getAccelerationConfig(): AccelerationConfig = accelerationConfig

    /**
     * 释放模型资源
     */
    fun unloadModel() {
        try {
            session?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing session", e)
        }
        // Note: OrtEnvironment is a singleton, don't close it
        session = null
        isModelLoaded = false
        vocab = emptyMap()

        // 清除加载状态
        clearLoadedModelName(context)
    }

    /**
     * 计算文本 Embedding
     * 返回 L2 归一化后的向量
     */
    suspend fun encode(text: String): FloatArray? = withContext(Dispatchers.IO) {
        if (!isModelLoaded || session == null || env == null) {
            Log.w(TAG, "Model not loaded, cannot encode")
            return@withContext null
        }

        try {
            val maxLen = AVAILABLE_MODELS.firstOrNull()?.maxSeqLength ?: 256

            // Tokenize
            val tokenIds = tokenize(text, maxLen)
            val inputIds = tokenIds.map { it.toLong() }.toLongArray()
            val attentionMask = tokenIds.map { if (it != PAD_TOKEN_ID) 1L else 0L }.toLongArray()
            val tokenTypeIds = LongArray(inputIds.size) { 0L }

            val seqLen = inputIds.size

            // Create direct buffers for ONNX Runtime (requires native-order direct buffers)
            val inputIdsBuf = ByteBuffer.allocateDirect(seqLen * 8).order(ByteOrder.nativeOrder()).asLongBuffer()
            inputIdsBuf.put(inputIds)
            inputIdsBuf.rewind()

            val attentionMaskBuf = ByteBuffer.allocateDirect(seqLen * 8).order(ByteOrder.nativeOrder()).asLongBuffer()
            attentionMaskBuf.put(attentionMask)
            attentionMaskBuf.rewind()

            val tokenTypeIdsBuf = ByteBuffer.allocateDirect(seqLen * 8).order(ByteOrder.nativeOrder()).asLongBuffer()
            tokenTypeIdsBuf.put(tokenTypeIds)
            tokenTypeIdsBuf.rewind()

            // Create ONNX tensors
            val inputIdsTensor = OnnxTensor.createTensor(
                env!!,
                inputIdsBuf,
                longArrayOf(1, seqLen.toLong())
            )
            val attentionMaskTensor = OnnxTensor.createTensor(
                env!!,
                attentionMaskBuf,
                longArrayOf(1, seqLen.toLong())
            )
            val tokenTypeIdsTensor = OnnxTensor.createTensor(
                env!!,
                tokenTypeIdsBuf,
                longArrayOf(1, seqLen.toLong())
            )

            // Run inference
            val inputNames = session?.inputNames ?: return@withContext null
            val inputs = mutableMapOf<String, OnnxTensor>()
            for (name in inputNames) {
                when {
                    name.contains("input_ids", ignoreCase = true) -> inputs[name] = inputIdsTensor
                    name.contains("attention_mask", ignoreCase = true) -> inputs[name] = attentionMaskTensor
                    name.contains("token_type_ids", ignoreCase = true) -> inputs[name] = tokenTypeIdsTensor
                }
            }

            val output = session?.run(inputs) ?: return@withContext null

            // Extract [CLS] token embedding (first token, index 0)
            @Suppress("UNCHECKED_CAST")
            val lastHiddenState = output.get(0).value as Array<Array<FloatArray>>
            val clsEmbedding = lastHiddenState[0][0]  // [1, seqLen, dim] → [0][0] = CLS

            // L2 normalize
            val normalized = l2Normalize(clsEmbedding)

            // Clean up tensors
            inputIdsTensor.close()
            attentionMaskTensor.close()
            tokenTypeIdsTensor.close()
            output.close()

            normalized
        } catch (e: Exception) {
            Log.e(TAG, "Encoding failed", e)
            null
        }
    }

    /**
     * 简单的 BERT WordPiece tokenizer
     * 对中文按字分词，对英文按词+子词分词
     */
    private fun tokenize(text: String, maxLen: Int): List<Int> {
        val tokens = mutableListOf<Int>()

        // [CLS] at start
        tokens.add(CLS_TOKEN_ID)

        // Basic tokenization: split by whitespace, then per-character for CJK
        val words = text.trim().split(Regex("\\s+"))
        for (word in words) {
            if (tokens.size >= maxLen - 1) break

            // Check if whole word is in vocab
            val wholeWordId = vocab[word.lowercase()]
            if (wholeWordId != null) {
                tokens.add(wholeWordId)
            } else {
                // Try to find subword tokens (simple greedy matching)
                var remaining = word.lowercase()
                while (remaining.isNotEmpty() && tokens.size < maxLen - 1) {
                    var matched = false
                    for (len in remaining.length downTo 1) {
                        val sub = if (remaining == word.lowercase()) remaining.substring(0, len)
                                  else "##${remaining.substring(0, len)}"
                        val subId = vocab[sub]
                        if (subId != null) {
                            tokens.add(subId)
                            remaining = remaining.substring(len)
                            matched = true
                            break
                        }
                    }
                    if (!matched) {
                        // Fallback: character-level tokenization for CJK or unknown chars
                        for (ch in remaining) {
                            if (tokens.size >= maxLen - 1) break
                            val charId = vocab[ch.toString()] ?: vocab["##$ch"] ?: vocab[UNK_TOKEN] ?: 100
                            tokens.add(charId)
                        }
                        break
                    }
                }
            }
        }

        // [SEP] at end
        tokens.add(SEP_TOKEN_ID)

        // Pad to fixed length
        while (tokens.size < maxLen) {
            tokens.add(PAD_TOKEN_ID)
        }

        return tokens.take(maxLen)
    }

    /**
     * 加载 vocab.txt 文件
     * 格式: 每行一个 token，行号 = token ID
     */
    private fun loadVocab(file: File): Map<String, Int> {
        val map = mutableMapOf<String, Int>()
        file.bufferedReader().useLines { lines ->
            lines.forEachIndexed { index, line ->
                val token = line.trim()
                if (token.isNotEmpty()) {
                    map[token] = index
                }
            }
        }
        return map
    }

    /**
     * L2 归一化向量
     */
    private fun l2Normalize(vec: FloatArray): FloatArray {
        var normSq = 0f
        for (v in vec) normSq += v * v
        val norm = Math.sqrt(normSq.toDouble()).toFloat()
        return if (norm > 0f) {
            FloatArray(vec.size) { i -> vec[i] / norm }
        } else {
            vec
        }
    }

    /**
     * 计算两个向量的余弦相似度
     */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f

        var dotProduct = 0f
        var normA = 0f
        var normB = 0f

        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        return if (normA > 0 && normB > 0) {
            dotProduct / (Math.sqrt(normA.toDouble()) * Math.sqrt(normB.toDouble())).toFloat()
        } else {
            0f
        }
    }

    private suspend fun downloadFile(urlStr: String, target: File): Boolean =
        withContext(Dispatchers.IO) {
            try {
                // 检查是否已取消
                if (!isActive) {
                    Log.d(TAG, "Download cancelled: $urlStr")
                    return@withContext false
                }

                val url = URL(urlStr)
                url.openStream().use { input ->
                    FileOutputStream(target).use { output ->
                        val buffer = ByteArray(8192)
                        while (currentCoroutineContext().isActive) {
                            val read = input.read(buffer)
                            if (read == -1) break
                            output.write(buffer, 0, read)
                        }
                    }
                }
                // 如果被取消，删除不完整的文件
                if (!currentCoroutineContext().isActive) {
                    target.delete()
                    Log.d(TAG, "Download cancelled: $urlStr")
                    return@withContext false
                }
                true
            } catch (e: java.io.IOException) {
                Log.d(TAG, "Download interrupted: $urlStr - ${e.message}")
                target.delete()
                false
            } catch (e: Exception) {
                Log.e(TAG, "Download failed: $urlStr", e)
                target.delete()
                false
            }
        }

    private suspend fun downloadFileWithProgress(
        urlStr: String,
        target: File,
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            // 检查是否已取消
            if (!isActive) {
                Log.d(TAG, "Download cancelled before start: $urlStr")
                return@withContext false
            }

            val temp = File(target.parent, "${target.name}.tmp")
            val url = URL(urlStr)
            val conn = url.openConnection()
            conn.connect()
            val contentLen = conn.contentLengthLong

            conn.getInputStream().use { input ->
                FileOutputStream(temp).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead = 0L
                    var lastPct = -1

                    while (currentCoroutineContext().isActive) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        output.write(buffer, 0, read)
                        bytesRead += read
                        if (contentLen > 0) {
                            val pct = (bytesRead * 100 / contentLen).toInt()
                            if (pct != lastPct) {
                                lastPct = pct
                                onProgress(pct)
                            }
                        }
                    }
                }
            }

            // 如果被取消，删除不完整的文件
            if (!currentCoroutineContext().isActive) {
                temp.delete()
                Log.d(TAG, "Download cancelled during transfer: $urlStr")
                return@withContext false
            }

            temp.renameTo(target)
            true
        } catch (e: java.io.IOException) {
            Log.d(TAG, "Download interrupted: $urlStr - ${e.message}")
            false
        } catch (e: Exception) {
            Log.e(TAG, "Download with progress failed: $urlStr", e)
            false
        }
    }
}
