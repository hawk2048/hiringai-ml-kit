package com.hiringai.mobile.ml.benchmark.datasets

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.zip.ZipInputStream

/**
 * 数据集管理器
 * 单例模式，负责基准测试数据集的下载、缓存和管理
 */
object DatasetManager {

    private const val DATASET_DIR = "benchmark_datasets"
    private const val MANIFEST_FILE = "manifest.json"
    private const val BUFFER_SIZE = 8192
    private const val HUGGINGFACE_MIRROR = "https://huggingface.co"

    /**
     * 数据集清单信息
     */
    data class DatasetManifest(
        val id: String,
        val name: String,
        val description: String,
        val version: String,
        val url: String,
        val size: Long,              // 字节
        val checksumSha256: String,  // SHA256 校验和
        val format: String,          // zip, tar.gz, etc.
        val classes: Int,            // 类别数
        val samples: Int,            // 样本数
        val labels: List<String>     // 类别标签
    )

    /**
     * 下载进度回调
     */
    interface DownloadCallback {
        fun onProgress(downloadedBytes: Long, totalBytes: Long, progressPercent: Int)
        fun onStatusUpdate(status: String)
        fun onComplete(file: File)
        fun onError(error: Throwable)
    }

    /**
     * 数据集状态
     */
    enum class DatasetStatus {
        NOT_DOWNLOADED,
        DOWNLOADING,
        DOWNLOADED,
        EXTRACTED,
        CORRUPTED,
        ERROR
    }

    /**
     * 数据集信息（包含状态）
     */
    data class DatasetInfo(
        val manifest: DatasetManifest,
        val status: DatasetStatus,
        val localPath: File?,
        val downloadedBytes: Long = 0
    )

    // 下载任务缓存
    private val downloadTasks = ConcurrentHashMap<String, Boolean>()

    // 可用数据集清单
    private val availableDatasets = listOf(
        DatasetManifest(
            id = "mini_imagenet",
            name = "Mini-ImageNet",
            description = "ImageNet 子集，100类，每类500张图片",
            version = "1.0.0",
            url = "$HUGGINGFACE_MIRROR/datasets/mini-imagenet/resolve/main/mini-imagenet.zip",
            size = 52_428_800,  // ~50MB
            checksumSha256 = "abc123def456...", // 实际使用时替换为真实校验和
            format = "zip",
            classes = 100,
            samples = 5000,
            labels = generateMiniImageNetLabels()
        ),
        DatasetManifest(
            id = "cifar10_test",
            name = "CIFAR-10 Test",
            description = "CIFAR-10 测试集，10类，10000张图片",
            version = "1.0.0",
            url = "$HUGGINGFACE_MIRROR/datasets/cifar10/resolve/main/cifar-10-test.zip",
            size = 18_874_321,  // ~18MB
            checksumSha256 = "def789ghi012...",
            format = "zip",
            classes = 10,
            samples = 10_000,
            labels = listOf("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        ),
        DatasetManifest(
            id = "speech_commands",
            name = "Speech Commands v2",
            description = "语音命令数据集，35个命令词",
            version = "2.0.0",
            url = "$HUGGINGFACE_MIRROR/datasets/speech_commands/resolve/main/speech_commands_v2.zip",
            size = 104_857_600,  // ~100MB
            checksumSha256 = "jkl345mno678...",
            format = "zip",
            classes = 35,
            samples = 105_829,
            labels = generateSpeechCommandLabels()
        )
    )

    /**
     * 获取数据集目录
     */
    fun getDatasetDir(context: Context): File {
        val dir = File(context.filesDir, DATASET_DIR)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    /**
     * 获取所有可用数据集
     */
    fun getAvailableDatasets(): List<DatasetManifest> {
        return availableDatasets
    }

    /**
     * 获取数据集信息（包含状态）
     */
    fun getDatasetInfo(context: Context, datasetId: String): DatasetInfo? {
        val manifest = availableDatasets.find { it.id == datasetId } ?: return null
        val localPath = File(getDatasetDir(context), datasetId)
        val status = checkDatasetStatus(context, manifest)

        return DatasetInfo(
            manifest = manifest,
            status = status,
            localPath = if (status == DatasetStatus.EXTRACTED) localPath else null
        )
    }

    /**
     * 检查数据集状态
     */
    private fun checkDatasetStatus(context: Context, manifest: DatasetManifest): DatasetStatus {
        val datasetDir = File(getDatasetDir(context), manifest.id)
        val archiveFile = File(getDatasetDir(context), "${manifest.id}.${manifest.format}")

        return when {
            // 检查是否已解压
            datasetDir.exists() && datasetDir.isDirectory -> {
                // 验证完整性
                if (validateExtractedDataset(datasetDir, manifest)) {
                    DatasetStatus.EXTRACTED
                } else {
                    DatasetStatus.CORRUPTED
                }
            }
            // 检查是否已下载
            archiveFile.exists() -> {
                // 验证校验和
                if (validateChecksum(archiveFile, manifest.checksumSha256)) {
                    DatasetStatus.DOWNLOADED
                } else {
                    DatasetStatus.CORRUPTED
                }
            }
            // 检查是否正在下载
            downloadTasks.containsKey(manifest.id) -> DatasetStatus.DOWNLOADING
            // 未下载
            else -> DatasetStatus.NOT_DOWNLOADED
        }
    }

    /**
     * 下载数据集
     */
    suspend fun downloadDataset(
        context: Context,
        datasetId: String,
        callback: DownloadCallback? = null
    ): Result<File> = withContext(Dispatchers.IO) {
        val manifest = availableDatasets.find { it.id == datasetId }
            ?: return@withContext Result.failure(IllegalArgumentException("Unknown dataset: $datasetId"))

        // 检查是否已存在
        val status = checkDatasetStatus(context, manifest)
        if (status == DatasetStatus.EXTRACTED) {
            val dir = File(getDatasetDir(context), datasetId)
            callback?.onComplete(dir)
            return@withContext Result.success(dir)
        }

        // 检查是否正在下载
        if (downloadTasks.putIfAbsent(datasetId, true) != null) {
            return@withContext Result.failure(IllegalStateException("Download already in progress"))
        }

        try {
            callback?.onStatusUpdate("准备下载 ${manifest.name}...")

            val archiveFile = File(getDatasetDir(context), "${manifest.id}.${manifest.format}")

            // 执行下载
            downloadFile(manifest.url, archiveFile, manifest.size, callback)

            // 验证校验和
            callback?.onStatusUpdate("验证文件完整性...")
            if (!validateChecksum(archiveFile, manifest.checksumSha256)) {
                archiveFile.delete()
                return@withContext Result.failure(SecurityException("Checksum verification failed"))
            }

            // 解压
            callback?.onStatusUpdate("解压文件...")
            val extractDir = File(getDatasetDir(context), manifest.id)
            extractArchive(archiveFile, extractDir)

            // 删除压缩包以节省空间
            archiveFile.delete()

            callback?.onStatusUpdate("下载完成")
            callback?.onComplete(extractDir)

            Result.success(extractDir)
        } catch (e: Exception) {
            callback?.onError(e)
            Result.failure(e)
        } finally {
            downloadTasks.remove(datasetId)
        }
    }

    /**
     * 下载文件（带进度回调）
     */
    private fun downloadFile(
        url: String,
        destination: File,
        expectedSize: Long,
        callback: DownloadCallback?
    ) {
        val connection = URL(url).openConnection()
        connection.connectTimeout = 30000
        connection.readTimeout = 60000

        FileOutputStream(destination).use { output ->
            connection.getInputStream().use { input ->
                val buffer = ByteArray(BUFFER_SIZE)
                var totalRead = 0L
                var read: Int

                while (input.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                    totalRead += read

                    val progress = if (expectedSize > 0) {
                        (totalRead * 100 / expectedSize).toInt().coerceIn(0, 100)
                    } else {
                        -1
                    }
                    callback?.onProgress(totalRead, expectedSize, progress)
                }
            }
        }
    }

    /**
     * 解压文件
     */
    private fun extractArchive(archiveFile: File, destination: File) {
        destination.mkdirs()

        when (archiveFile.extension) {
            "zip" -> extractZip(archiveFile, destination)
            "gz", "tgz" -> extractTarGz(archiveFile, destination)
            else -> throw UnsupportedOperationException("Unsupported archive format: ${archiveFile.extension}")
        }
    }

    /**
     * 解压 ZIP 文件
     */
    private fun extractZip(zipFile: File, destination: File) {
        ZipInputStream(FileInputStream(zipFile)).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                val file = File(destination, entry.name)
                if (entry.isDirectory) {
                    file.mkdirs()
                } else {
                    file.parentFile?.mkdirs()
                    FileOutputStream(file).use { fos ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        var len: Int
                        while (zis.read(buffer).also { len = it } > 0) {
                            fos.write(buffer, 0, len)
                        }
                    }
                }
                zis.closeEntry()
                entry = zis.nextEntry
            }
        }
    }

    /**
     * 解压 tar.gz 文件
     */
    private fun extractTarGz(tarGzFile: File, destination: File) {
        // 简化实现，实际需要 tar 解析库
        // 可以使用 Apache Commons Compress 或 similar
        throw UnsupportedOperationException("tar.gz extraction requires additional library")
    }

    /**
     * 验证 SHA256 校验和
     */
    private fun validateChecksum(file: File, expectedChecksum: String): Boolean {
        if (!file.exists()) return false

        // 如果校验和未设置（占位符），跳过验证
        if (expectedChecksum.endsWith("...") || expectedChecksum.length != 64) {
            return true
        }

        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { fis ->
            val buffer = ByteArray(BUFFER_SIZE)
            var read: Int
            while (fis.read(buffer).also { read = it } > 0) {
                digest.update(buffer, 0, read)
            }
        }

        val checksum = digest.digest().joinToString("") { "%02x".format(it) }
        return checksum.equals(expectedChecksum, ignoreCase = true)
    }

    /**
     * 验证解压后的数据集
     */
    private fun validateExtractedDataset(datasetDir: File, manifest: DatasetManifest): Boolean {
        // 基本验证：检查目录是否存在且非空
        if (!datasetDir.exists() || !datasetDir.isDirectory) {
            return false
        }

        // 检查是否有文件
        val files = datasetDir.listFiles()
        return files != null && files.isNotEmpty()
    }

    /**
     * 删除数据集
     */
    fun deleteDataset(context: Context, datasetId: String): Boolean {
        val datasetDir = File(getDatasetDir(context), datasetId)
        val archiveFile = File(getDatasetDir(context), "$datasetId.zip")

        var success = true

        if (datasetDir.exists()) {
            success = deleteRecursively(datasetDir) && success
        }

        if (archiveFile.exists()) {
            success = archiveFile.delete() && success
        }

        return success
    }

    /**
     * 递归删除目录
     */
    private fun deleteRecursively(dir: File): Boolean {
        if (dir.isDirectory) {
            dir.listFiles()?.forEach { child ->
                deleteRecursively(child)
            }
        }
        return dir.delete()
    }

    /**
     * 获取缓存大小
     */
    fun getCacheSize(context: Context): Long {
        val datasetDir = getDatasetDir(context)
        return calculateDirectorySize(datasetDir)
    }

    /**
     * 计算目录大小
     */
    private fun calculateDirectorySize(dir: File): Long {
        if (!dir.exists()) return 0

        var size = 0L
        dir.listFiles()?.forEach { file ->
            size += if (file.isDirectory) {
                calculateDirectorySize(file)
            } else {
                file.length()
            }
        }
        return size
    }

    /**
     * 清理所有缓存
     */
    fun clearAllCache(context: Context): Boolean {
        return deleteRecursively(getDatasetDir(context))
    }

    /**
     * 格式化文件大小
     */
    fun formatSize(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> String.format("%.1f KB", bytes / 1024.0)
            bytes < 1024 * 1024 * 1024 -> String.format("%.1f MB", bytes / (1024.0 * 1024))
            else -> String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024))
        }
    }

    /**
     * 生成 Mini-ImageNet 标签（示例）
     */
    private fun generateMiniImageNetLabels(): List<String> {
        // Mini-ImageNet 有 100 个类别
        // 这里列出部分示例，实际使用时替换为完整列表
        return (1..100).map { "class_$it" }
    }

    /**
     * 生成语音命令标签
     */
    private fun generateSpeechCommandLabels(): List<String> {
        return listOf(
            "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
            "backward", "forward", "follow", "learn", "visual"
        )
    }
}
