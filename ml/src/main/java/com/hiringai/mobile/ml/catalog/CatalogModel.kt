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
    val localModelPath: String = "",

    // ─── 发行信息 ────────────────────────────────
    /** 模型在 HF/ModelScope 的创建时间（用于排序和新模型标识） */
    val releaseDate: Long = 0L,
    /** 最后修改时间（模型有更新时变化） */
    val lastModified: Long = 0L,
    /** 下载次数 */
    val downloadCount: Long = 0L,
    /** 点赞数 */
    val likes: Int = 0,
    /** 模型对应的任务类型标签（text-generation / image-classification 等） */
    val pipelineTag: String = "",
    /** 模型文件格式（onnx / pytorch / safetensors） */
    val modelFileFormat: String = "",
    /** 是否是 GGUF 量化模型 */
    val isGguf: Boolean = false,
    /** 量化位数（Q4_K_M 等，GGUF 模型专用） */
    val quantizationBits: String = ""
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

    /** 发行日期的中文格式化显示 */
    val formattedReleaseDate: String
        get() = if (releaseDate <= 0) "未知"
        else {
            val diffDays = (System.currentTimeMillis() - releaseDate) / (1000 * 60 * 60 * 24)
            when {
                diffDays == 0L -> "今天"
                diffDays == 1L -> "昨天"
                diffDays < 7 -> "${diffDays}天前"
                diffDays < 30 -> "${diffDays / 7}周前"
                diffDays < 365 -> "${diffDays / 30}个月前"
                else -> "${diffDays / 365}年前"
            }
        }

    /** 精确发行日期（用于排序） */
    val preciseReleaseDate: String
        get() = if (releaseDate <= 0) ""
        else {
            val sdf = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
            sdf.format(java.util.Date(releaseDate))
        }

    /**
     * 是否是新模型（最近 30 天内发布）
     * 用于在列表中显示"新"徽章
     */
    val isNew: Boolean
        get() = releaseDate > 0 && (System.currentTimeMillis() - releaseDate) <= 30 * 24 * 60 * 60 * 1000L

    /**
     * 是否是最近更新（最近 7 天内）
     */
    val isRecentlyUpdated: Boolean
        get() = lastModified > 0 && (System.currentTimeMillis() - lastModified) <= 7 * 24 * 60 * 60 * 1000L

    /** 下载量的格式化显示 */
    val formattedDownloads: String
        get() = when {
            downloadCount >= 1_000_000 -> "%.1fM".format(downloadCount / 1_000_000.0)
            downloadCount >= 1_000 -> "%.1fK".format(downloadCount / 1_000.0)
            downloadCount > 0 -> downloadCount.toString()
            else -> "-"
        }
}
