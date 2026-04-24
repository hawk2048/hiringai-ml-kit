package com.hiringai.mobile.ml.catalog

/**
 * 目录模型数据类
 *
 * 统一的模型描述，涵盖本地内置模型和在线获取的模型
 */
data class CatalogModel(
    val id: String,
    val name: String,
    val category: ModelCategory,
    val sizeBytes: Long,
    val description: String,
    val source: String = "在线",
    val isDownloaded: Boolean = false,
    val downloadUrl: String = "",
    val recommendedRAM: Int = 0,
    val dimension: Int = 0,
    val maxSeqLength: Int = 0,
    val quantization: String = "",
    val language: String = "",
    val author: String = "",
    val license: String = "",
    val tags: List<String> = emptyList(),
    val localModelPath: String = ""
) {
    enum class ModelCategory(val label: String) {
        LLM("大语言模型"),
        EMBEDDING("嵌入模型"),
        SPEECH("语音模型"),
        IMAGE("图像模型");

        companion object {
            fun fromString(s: String): ModelCategory {
                return entries.find {
                    it.name.equals(s, ignoreCase = true) || it.label == s
                } ?: LLM
            }
        }
    }

    val formattedSize: String
        get() = when {
            sizeBytes >= 1_000_000_000 -> "%.1f GB".format(sizeBytes / 1_000_000_000.0)
            sizeBytes >= 1_000_000 -> "%.0f MB".format(sizeBytes / 1_000_000.0)
            else -> "%.0f KB".format(sizeBytes / 1_000.0)
        }
}
