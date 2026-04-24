package com.hiringai.mobile.ml.benchmark.datasets

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.io.FileInputStream

/**
 * Mini-ImageNet 数据集加载器
 * 加载压缩的 ImageNet 子集，支持懒加载以节省内存
 */
class MiniImageNetLoader(private val context: Context) {

    companion object {
        const val DATASET_ID = "mini_imagenet"
        const val NUM_CLASSES = 100
        const val IMAGES_PER_CLASS = 50
        const val TOTAL_IMAGES = NUM_CLASSES * IMAGES_PER_CLASS  // 5000
        const val DEFAULT_IMAGE_SIZE = 84  // Mini-ImageNet 标准尺寸

        /**
         * 创建加载器实例
         */
        fun create(context: Context): MiniImageNetLoader {
            return MiniImageNetLoader(context)
        }
    }

    /**
     * 图像样本
     * @param bitmap 图像数据
     * @param label 类别标签索引 (0-99)
     * @param labelName 类别名称
     */
    data class ImageSample(
        val bitmap: Bitmap,
        val label: Int,
        val labelName: String
    )

    /**
     * 批量图像样本（不含 Bitmap，用于列表展示）
     */
    data class ImageInfo(
        val filePath: String,
        val label: Int,
        val labelName: String,
        val width: Int,
        val height: Int
    )

    /**
     * 加载配置
     */
    data class LoadConfig(
        val targetWidth: Int = DEFAULT_IMAGE_SIZE,
        val targetHeight: Int = DEFAULT_IMAGE_SIZE,
        val maxSamples: Int = TOTAL_IMAGES,
        val classes: List<Int>? = null,  // null 表示加载所有类别
        val lazyLoad: Boolean = true      // 是否懒加载
    )

    /**
     * 加载进度
     */
    data class LoadProgress(
        val loadedCount: Int,
        val totalCount: Int,
        val currentClass: String?,
        val progressPercent: Int
    )

    private val datasetDir: File by lazy {
        File(DatasetManager.getDatasetDir(context), DATASET_ID)
    }

    private val _labels: List<String> by lazy {
        DatasetManager.getAvailableDatasets()
            .find { it.id == DATASET_ID }
            ?.labels ?: (0 until NUM_CLASSES).map { "class_$it" }
    }

    /**
     * 检查数据集是否可用
     */
    fun isAvailable(): Boolean {
        val info = DatasetManager.getDatasetInfo(context, DATASET_ID)
        return info?.status == DatasetManager.DatasetStatus.EXTRACTED
    }

    /**
     * 获取类别数量
     */
    fun getClassCount(): Int = NUM_CLASSES

    /**
     * 获取类别标签列表
     */
    fun getClassLabels(): List<String> = _labels

    /**
     * 获取类别名称
     */
    fun getLabelName(labelIndex: Int): String {
        return _labels.getOrNull(labelIndex) ?: "unknown"
    }

    /**
     * 加载所有图像（懒加载模式）
     * 返回图像信息列表，不加载实际图像数据
     */
    fun loadImageInfos(): List<ImageInfo> {
        if (!isAvailable()) {
            return emptyList()
        }

        val imageInfos = mutableListOf<ImageInfo>()

        for (classIndex in 0 until NUM_CLASSES) {
            val classDir = File(datasetDir, "class_$classIndex")
            if (!classDir.exists() || !classDir.isDirectory) continue

            classDir.listFiles()
                ?.filter { it.extension.lowercase() in listOf("jpg", "jpeg", "png") }
                ?.forEach { imageFile ->
                    // 获取图像尺寸（不解码完整图像）
                    val options = BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }
                    BitmapFactory.decodeFile(imageFile.absolutePath, options)

                    imageInfos.add(
                        ImageInfo(
                            filePath = imageFile.absolutePath,
                            label = classIndex,
                            labelName = getLabelName(classIndex),
                            width = options.outWidth,
                            height = options.outHeight
                        )
                    )
                }
        }

        return imageInfos
    }

    /**
     * 加载单个图像
     */
    fun loadImage(imageInfo: ImageInfo, config: LoadConfig = LoadConfig()): ImageSample? {
        return loadImageFromPath(
            path = imageInfo.filePath,
            label = imageInfo.label,
            config = config
        )
    }

    /**
     * 从路径加载图像
     */
    private fun loadImageFromPath(
        path: String,
        label: Int,
        config: LoadConfig
    ): ImageSample? {
        val options = BitmapFactory.Options().apply {
            // 计算采样率以优化内存
            inSampleSize = calculateSampleSize(path, config.targetWidth, config.targetHeight)
        }

        val bitmap = BitmapFactory.decodeFile(path, options) ?: return null

        // 如果需要缩放
        val scaledBitmap = if (config.targetWidth > 0 && config.targetHeight > 0) {
            if (bitmap.width != config.targetWidth || bitmap.height != config.targetHeight) {
                val scaled = Bitmap.createScaledBitmap(bitmap, config.targetWidth, config.targetHeight, true)
                if (scaled != bitmap) bitmap.recycle()
                scaled
            } else {
                bitmap
            }
        } else {
            bitmap
        }

        return ImageSample(
            bitmap = scaledBitmap,
            label = label,
            labelName = getLabelName(label)
        )
    }

    /**
     * 计算采样率
     */
    private fun calculateSampleSize(path: String, targetWidth: Int, targetHeight: Int): Int {
        if (targetWidth <= 0 || targetHeight <= 0) return 1

        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeFile(path, options)

        val width = options.outWidth
        val height = options.outHeight
        var inSampleSize = 1

        if (height > targetHeight || width > targetWidth) {
            val halfHeight = height / 2
            val halfWidth = width / 2

            while (halfHeight / inSampleSize >= targetHeight && halfWidth / inSampleSize >= targetWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }

    /**
     * 批量加载图像（Flow 方式，支持进度回调）
     */
    fun loadImagesFlow(
        config: LoadConfig = LoadConfig()
    ): Flow<LoadProgress> = flow {
        if (!isAvailable()) {
            emit(LoadProgress(0, 0, null, 0))
            return@flow
        }

        val targetClasses = config.classes ?: (0 until NUM_CLASSES).toList()
        val imagesPerClass = config.maxSamples.coerceAtMost(IMAGES_PER_CLASS)

        var loadedCount = 0
        val totalCount = targetClasses.size * imagesPerClass

        // 这里只发出进度，不返回图像（避免内存问题）
        for (classIndex in targetClasses) {
            val classDir = File(datasetDir, "class_$classIndex")
            if (!classDir.exists()) continue

            val imageFiles = classDir.listFiles()
                ?.filter { it.extension.lowercase() in listOf("jpg", "jpeg", "png") }
                ?.take(imagesPerClass) ?: continue

            for (imageFile in imageFiles) {
                loadedCount++
                val progress = (loadedCount * 100 / totalCount).coerceIn(0, 100)
                emit(
                    LoadProgress(
                        loadedCount = loadedCount,
                        totalCount = totalCount,
                        currentClass = getLabelName(classIndex),
                        progressPercent = progress
                    )
                )
            }
        }
    }.flowOn(Dispatchers.IO)

    /**
     * 懒加载迭代器
     * 逐个加载图像，节省内存
     */
    fun imageSequence(config: LoadConfig = LoadConfig()): Sequence<ImageSample> {
        if (!isAvailable()) {
            return emptySequence()
        }

        val targetClasses = config.classes ?: (0 until NUM_CLASSES).toList()
        val imagesPerClass = config.maxSamples.coerceAtMost(IMAGES_PER_CLASS)

        return sequence {
            for (classIndex in targetClasses) {
                val classDir = File(datasetDir, "class_$classIndex")
                if (!classDir.exists() || !classDir.isDirectory) continue

                val imageFiles = classDir.listFiles()
                    ?.filter { it.extension.lowercase() in listOf("jpg", "jpeg", "png") }
                    ?.take(imagesPerClass) ?: continue

                for (imageFile in imageFiles) {
                    loadImageFromPath(imageFile.absolutePath, classIndex, config)?.let {
                        yield(it)
                    }
                }
            }
        }
    }

    /**
     * 加载指定类别的图像
     */
    fun loadClassImages(
        classIndex: Int,
        config: LoadConfig = LoadConfig()
    ): List<ImageSample> {
        if (!isAvailable() || classIndex !in 0 until NUM_CLASSES) {
            return emptyList()
        }

        val classDir = File(datasetDir, "class_$classIndex")
        if (!classDir.exists() || !classDir.isDirectory) {
            return emptyList()
        }

        return classDir.listFiles()
            ?.filter { it.extension.lowercase() in listOf("jpg", "jpeg", "png") }
            ?.take(config.maxSamples)
            ?.mapNotNull { imageFile ->
                loadImageFromPath(imageFile.absolutePath, classIndex, config)
            }
            ?: emptyList()
    }

    /**
     * 加载指定数量的随机样本
     */
    fun loadRandomSamples(
        count: Int,
        config: LoadConfig = LoadConfig()
    ): List<ImageSample> {
        if (!isAvailable()) return emptyList()

        val allInfos = loadImageInfos().shuffled().take(count)
        return allInfos.mapNotNull { loadImage(it, config) }
    }

    /**
     * 获取数据集统计信息
     */
    fun getStatistics(): DatasetStatistics {
        if (!isAvailable()) {
            return DatasetStatistics(
                totalImages = 0,
                classCount = 0,
                imagesPerClass = emptyMap(),
                avgWidth = 0,
                avgHeight = 0
            )
        }

        val imagesPerClass = mutableMapOf<Int, Int>()
        var totalWidth = 0
        var totalHeight = 0
        var sampleCount = 0

        for (classIndex in 0 until NUM_CLASSES) {
            val classDir = File(datasetDir, "class_$classIndex")
            if (!classDir.exists()) continue

            val images = classDir.listFiles()
                ?.filter { it.extension.lowercase() in listOf("jpg", "jpeg", "png") }
                ?.toList() ?: emptyList()

            imagesPerClass[classIndex] = images.size

            // 采样部分图像获取尺寸
            images.take(5).forEach { imageFile ->
                val options = BitmapFactory.Options().apply {
                    inJustDecodeBounds = true
                }
                BitmapFactory.decodeFile(imageFile.absolutePath, options)
                totalWidth += options.outWidth
                totalHeight += options.outHeight
                sampleCount++
            }
        }

        return DatasetStatistics(
            totalImages = imagesPerClass.values.sum(),
            classCount = imagesPerClass.size,
            imagesPerClass = imagesPerClass,
            avgWidth = if (sampleCount > 0) totalWidth / sampleCount else 0,
            avgHeight = if (sampleCount > 0) totalHeight / sampleCount else 0
        )
    }

    /**
     * 数据集统计信息
     */
    data class DatasetStatistics(
        val totalImages: Int,
        val classCount: Int,
        val imagesPerClass: Map<Int, Int>,
        val avgWidth: Int,
        val avgHeight: Int
    ) {
        fun toSummary(): String {
            return buildString {
                appendLine("📊 Mini-ImageNet 数据集统计:")
                appendLine("   总图像数: $totalImages")
                appendLine("   类别数: $classCount")
                appendLine("   平均尺寸: ${avgWidth}x${avgHeight}")
                val minImages = imagesPerClass.values.minOrNull() ?: 0
                val maxImages = imagesPerClass.values.maxOrNull() ?: 0
                appendLine("   每类图像数: $minImages - $maxImages")
            }
        }
    }

    /**
     * 释放资源
     */
    fun release() {
        // 清理缓存（如果有）
    }

    /**
     * 获取数据集目录大小
     */
    fun getDatasetSize(): Long {
        if (!datasetDir.exists()) return 0
        return calculateDirectorySize(datasetDir)
    }

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
}
