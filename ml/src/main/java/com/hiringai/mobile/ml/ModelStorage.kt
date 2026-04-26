package com.hiringai.mobile.ml

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File

/**
 * 统一模型存储管理
 *
 * 使用 app-specific external storage（不受 Android 10+ Scoped Storage 限制）
 * 路径: Android/data/<package>/files/Downloads/MLModels/
 *
 * 优势:
 * - App 更新不丢模型
 * - 卸载重装同包名可恢复（Play Store/ADB 安装场景）
 * - 无需 Storage 权限
 *
 * 劣势:
 * - 通过设置卸载 App 会删除此目录
 * - 其他 App 无法访问
 */
object ModelStorage {

    private const val TAG = "ModelStorage"

    /** 各模型类型的子目录名 */
    private const val SUBDIR_LLM = "llm"
    private const val SUBDIR_EMBEDDING = "embedding"
    private const val SUBDIR_IMAGE = "image"
    private const val SUBDIR_SPEECH = "speech"

    /**
     * 获取模型根目录
     * Android/data/<package>/files/Downloads/MLModels/
     */
    fun getBaseDir(context: Context): File {
        val dir = File(
            context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),
            "MLModels"
        )
        if (!dir.exists()) {
            val created = dir.mkdirs()
            if (!created) {
                Log.e(TAG, "Failed to create base dir: ${dir.absolutePath}")
            } else {
                Log.i(TAG, "Created model base dir: ${dir.absolutePath}")
            }
        }
        return dir
    }

    /** LLM 模型目录 (GGUF) */
    fun getLLMDir(context: Context): File {
        val dir = File(getBaseDir(context), SUBDIR_LLM)
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /** Embedding 模型目录 (ONNX) */
    fun getEmbeddingDir(context: Context): File {
        val dir = File(getBaseDir(context), SUBDIR_EMBEDDING)
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /** 图像模型目录 (ONNX) */
    fun getImageDir(context: Context): File {
        val dir = File(getBaseDir(context), SUBDIR_IMAGE)
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /** 语音模型目录 (ONNX) */
    fun getSpeechDir(context: Context): File {
        val dir = File(getBaseDir(context), SUBDIR_SPEECH)
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * 获取总存储使用量
     */
    fun getTotalSize(context: Context): Long {
        return getDirSize(getBaseDir(context))
    }

    /**
     * 获取目录大小（递归）
     */
    private fun getDirSize(dir: File): Long {
        if (!dir.exists()) return 0L
        var size = 0L
        dir.listFiles()?.forEach { file ->
            size += if (file.isDirectory) getDirSize(file) else file.length()
        }
        return size
    }

    /**
     * 获取各目录的存储信息
     */
    fun getStorageInfo(context: Context): StorageInfo {
        return StorageInfo(
            llm = getDirSize(getLLMDir(context)),
            embedding = getDirSize(getEmbeddingDir(context)),
            image = getDirSize(getImageDir(context)),
            speech = getDirSize(getSpeechDir(context)),
            total = getTotalSize(context)
        )
    }

    /**
     * 清理所有模型
     */
    fun clearAll(context: Context): Int {
        var count = 0
        getBaseDir(context).listFiles()?.forEach { subdir ->
            subdir.listFiles()?.forEach { file ->
                if (file.delete()) count++
            }
        }
        Log.i(TAG, "Cleared $count model files")
        return count
    }

    data class StorageInfo(
        val llm: Long,
        val embedding: Long,
        val image: Long,
        val speech: Long,
        val total: Long
    ) {
        fun format(bytes: Long): String {
            return when {
                bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1_000_000_000.0)
                bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1_000_000.0)
                bytes >= 1_000 -> "%.0f KB".format(bytes / 1_000.0)
                else -> "$bytes B"
            }
        }
    }
}
