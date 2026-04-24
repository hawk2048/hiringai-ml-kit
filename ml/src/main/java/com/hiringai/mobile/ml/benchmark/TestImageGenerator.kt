package com.hiringai.mobile.ml.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import java.io.File
import java.io.FileOutputStream

/**
 * 测试图像生成器
 * 生成用于基准测试的标准测试图像
 */
object TestImageGenerator {

    /**
     * 测试图像类型
     */
    enum class TestImageType {
        CLASSIFICATION_224,  // 224x224 分类测试图
        CLASSIFICATION_512,  // 512x512 分类测试图
        OCR_TEXT,           // OCR 文字识别测试图
        GRADIENT,           // 渐变测试图
        NOISE,              // 噪声测试图
        OBJECTS             // 物体检测测试图
    }

    /**
     * 生成标准测试图像
     */
    fun generateTestImage(type: TestImageType, width: Int, height: Int): Bitmap {
        return when (type) {
            TestImageType.CLASSIFICATION_224,
            TestImageType.CLASSIFICATION_512 -> generateClassificationTestImage(width, height)
            TestImageType.OCR_TEXT -> generateOCRTestImage(width, height)
            TestImageType.GRADIENT -> generateGradientImage(width, height)
            TestImageType.NOISE -> generateNoiseImage(width, height)
            TestImageType.OBJECTS -> generateObjectsTestImage(width, height)
        }
    }

    /**
     * 生成分类测试图像
     * 包含彩色方块和形状，用于测试图像分类模型
     */
    private fun generateClassificationTestImage(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // 背景
        canvas.drawColor(Color.rgb(240, 240, 245))

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        // 绘制彩色圆形
        paint.color = Color.rgb(65, 105, 225) // Royal Blue
        canvas.drawCircle(width * 0.25f, height * 0.3f, width * 0.15f, paint)

        // 绘制红色矩形
        paint.color = Color.rgb(220, 53, 69) // Red
        canvas.drawRect(
            width * 0.5f, height * 0.2f,
            width * 0.85f, height * 0.5f,
            paint
        )

        // 绘制绿色三角形
        paint.color = Color.rgb(40, 167, 69) // Green
        val path = android.graphics.Path()
        path.moveTo(width * 0.5f, height * 0.9f)
        path.lineTo(width * 0.3f, height * 0.55f)
        path.lineTo(width * 0.7f, height * 0.55f)
        path.close()
        canvas.drawPath(path, paint)

        // 绘制黄色椭圆形
        paint.color = Color.rgb(255, 193, 7) // Yellow
        canvas.drawOval(
            width * 0.1f, height * 0.6f,
            width * 0.35f, height * 0.9f,
            paint
        )

        return bitmap
    }

    /**
     * 生成 OCR 测试图像
     * 包含多种字体和大小的文字
     */
    private fun generateOCRTestImage(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // 白色背景
        canvas.drawColor(Color.WHITE)

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.BLACK

        // 标题文字
        paint.textSize = height * 0.12f
        paint.typeface = Typeface.DEFAULT_BOLD
        paint.textAlign = Paint.Align.CENTER
        canvas.drawText("TEST SAMPLE", width / 2f, height * 0.2f, paint)

        // 中等文字
        paint.textSize = height * 0.08f
        paint.typeface = Typeface.DEFAULT
        canvas.drawText("12345 ABCDE", width / 2f, height * 0.4f, paint)

        // 小文字
        paint.textSize = height * 0.06f
        canvas.drawText("The quick brown fox", width / 2f, height * 0.55f, paint)
        canvas.drawText("jumps over lazy dog", width / 2f, height * 0.68f, paint)

        // 中文测试
        paint.textSize = height * 0.1f
        canvas.drawText("测试文字识别", width / 2f, height * 0.85f, paint)

        return bitmap
    }

    /**
     * 生成渐变测试图像
     */
    private fun generateGradientImage(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = (x * 255 / width)
                val g = (y * 255 / height)
                val b = 128
                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }

        return bitmap
    }

    /**
     * 生成噪声测试图像
     */
    private fun generateNoiseImage(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val random = java.util.Random(42) // Fixed seed for reproducibility

        for (y in 0 until height) {
            for (x in 0 until width) {
                val gray = random.nextInt(256)
                bitmap.setPixel(x, y, Color.rgb(gray, gray, gray))
            }
        }

        return bitmap
    }

    /**
     * 生成物体检测测试图像
     */
    private fun generateObjectsTestImage(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // 浅灰背景
        canvas.drawColor(Color.rgb(245, 245, 250))

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        // 绘制多个"物体"
        val objects = listOf(
            Triple(0.15f, 0.15f, Color.rgb(255, 87, 87)),    // 红色圆形
            Triple(0.45f, 0.25f, Color.rgb(87, 255, 87)),    // 绿色圆形
            Triple(0.75f, 0.2f, Color.rgb(87, 87, 255)),     // 蓝色圆形
            Triple(0.25f, 0.55f, Color.rgb(255, 255, 87)),   // 黄色圆形
            Triple(0.55f, 0.6f, Color.rgb(255, 87, 255)),    // 紫色圆形
            Triple(0.85f, 0.65f, Color.rgb(87, 255, 255))    // 青色圆形
        )

        objects.forEach { (xRatio, yRatio, color) ->
            paint.color = color
            canvas.drawCircle(
                width * xRatio,
                height * yRatio,
                width * 0.08f,
                paint
            )
        }

        // 添加一些矩形"物体"
        paint.color = Color.rgb(100, 100, 100)
        canvas.drawRect(width * 0.1f, height * 0.75f, width * 0.35f, height * 0.9f, paint)
        canvas.drawRect(width * 0.4f, height * 0.8f, width * 0.6f, height * 0.92f, paint)
        canvas.drawRect(width * 0.65f, height * 0.75f, width * 0.9f, height * 0.88f, paint)

        return bitmap
    }

    /**
     * 保存测试图像到文件
     */
    fun saveTestImage(bitmap: Bitmap, file: File): Boolean {
        return try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * 初始化测试图像到指定目录
     */
    fun initTestImages(context: Context): List<File> {
        val testDir = File(context.filesDir, "test_images")
        if (!testDir.exists()) testDir.mkdirs()

        val testImages = listOf(
            "test_classification_224.png" to TestImageType.CLASSIFICATION_224,
            "test_classification_512.png" to TestImageType.CLASSIFICATION_512,
            "test_ocr_text.png" to TestImageType.OCR_TEXT,
            "test_gradient.png" to TestImageType.GRADIENT,
            "test_noise.png" to TestImageType.NOISE,
            "test_objects.png" to TestImageType.OBJECTS
        )

        val files = mutableListOf<File>()

        testImages.forEach { (filename, type) ->
            val file = File(testDir, filename)
            if (!file.exists()) {
                val sizes = when (type) {
                    TestImageType.CLASSIFICATION_224 -> 224 to 224
                    TestImageType.CLASSIFICATION_512 -> 512 to 512
                    TestImageType.OCR_TEXT -> 384 to 64
                    TestImageType.GRADIENT -> 256 to 256
                    TestImageType.NOISE -> 256 to 256
                    TestImageType.OBJECTS -> 416 to 416
                }
                val bitmap = generateTestImage(type, sizes.first, sizes.second)
                saveTestImage(bitmap, file)
                bitmap.recycle()
            }
            files.add(file)
        }

        return files
    }

    /**
     * 获取测试图像目录
     */
    fun getTestImagesDir(context: Context): File {
        val dir = File(context.filesDir, "test_images")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * 获取所有可用测试图像
     */
    fun getAvailableTestImages(context: Context): List<TestImageInfo> {
        val dir = getTestImagesDir(context)
        return listOf(
            TestImageInfo(
                filename = "test_classification_224.png",
                displayName = "分类测试图 224x224",
                description = "包含彩色形状的标准分类测试图",
                type = TestImageType.CLASSIFICATION_224,
                width = 224,
                height = 224
            ),
            TestImageInfo(
                filename = "test_classification_512.png",
                displayName = "分类测试图 512x512",
                description = "高分辨率分类测试图",
                type = TestImageType.CLASSIFICATION_512,
                width = 512,
                height = 512
            ),
            TestImageInfo(
                filename = "test_ocr_text.png",
                displayName = "OCR 文字测试图",
                description = "包含中英文混合文字的OCR测试图",
                type = TestImageType.OCR_TEXT,
                width = 384,
                height = 64
            ),
            TestImageInfo(
                filename = "test_gradient.png",
                displayName = "渐变测试图",
                description = "彩色渐变背景图",
                type = TestImageType.GRADIENT,
                width = 256,
                height = 256
            ),
            TestImageInfo(
                filename = "test_noise.png",
                displayName = "噪声测试图",
                description = "灰度随机噪声图",
                type = TestImageType.NOISE,
                width = 256,
                height = 256
            ),
            TestImageInfo(
                filename = "test_objects.png",
                displayName = "物体检测测试图",
                description = "包含多个彩色物体的测试图",
                type = TestImageType.OBJECTS,
                width = 416,
                height = 416
            )
        )
    }

    /**
     * 加载测试图像
     */
    fun loadTestImage(context: Context, info: TestImageInfo): Bitmap? {
        val dir = getTestImagesDir(context)
        val file = File(dir, info.filename)

        // 如果文件不存在，先生成
        if (!file.exists()) {
            val bitmap = generateTestImage(info.type, info.width, info.height)
            saveTestImage(bitmap, file)
            return bitmap
        }

        return try {
            android.graphics.BitmapFactory.decodeFile(file.absolutePath)
        } catch (e: Exception) {
            // 如果加载失败，重新生成
            val bitmap = generateTestImage(info.type, info.width, info.height)
            saveTestImage(bitmap, file)
            bitmap
        }
    }
}

/**
 * 测试图像信息
 */
data class TestImageInfo(
    val filename: String,
    val displayName: String,
    val description: String,
    val type: TestImageGenerator.TestImageType,
    val width: Int,
    val height: Int
)
