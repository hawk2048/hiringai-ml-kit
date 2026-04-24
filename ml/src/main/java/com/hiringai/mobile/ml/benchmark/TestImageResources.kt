package com.hiringai.mobile.ml.benchmark

import android.content.Context
import android.graphics.*
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * 测试图像资源管理器
 *
 * 提供用于基准测试的标准化测试图像:
 * - OCR测试图像: 包含各种字体、大小的文字
 * - 分类测试图像: 包含常见物体类别
 * - VLM测试图像: 包含复杂场景描述
 */
object TestImageResources {

    private const val TAG = "TestImageResources"

    /**
     * 测试图像类型
     */
    enum class TestImageType {
        OCR_TEXT,           // OCR文字识别测试
        OCR_HANDWRITING,    // 手写体OCR测试
        CLASSIFICATION,     // 图像分类测试
        VLM_SCENE,          // VLM场景理解测试
        FACE_DETECTION,     // 人脸检测测试
        DOCUMENT            // 文档扫描测试
    }

    /**
     * 测试图像配置
     */
    data class TestImageConfig(
        val name: String,
        val type: TestImageType,
        val width: Int,
        val height: Int,
        val description: String
    )

    /**
     * 预定义的测试图像配置列表
     */
    val TEST_IMAGES = listOf(
        // OCR测试图像
        TestImageConfig(
            name = "ocr_english_basic",
            type = TestImageType.OCR_TEXT,
            width = 640,
            height = 480,
            description = "英文基础文字识别测试 - 标准字体"
        ),
        TestImageConfig(
            name = "ocr_chinese_basic",
            type = TestImageType.OCR_TEXT,
            width = 640,
            height = 480,
            description = "中文基础文字识别测试 - 宋体/黑体"
        ),
        TestImageConfig(
            name = "ocr_mixed_languages",
            type = TestImageType.OCR_TEXT,
            width = 800,
            height = 600,
            description = "中英混合文字测试 - 多语言场景"
        ),
        TestImageConfig(
            name = "ocr_receipt",
            type = TestImageType.OCR_TEXT,
            width = 480,
            height = 640,
            description = "收据/发票OCR测试 - 实际应用场景"
        ),
        TestImageConfig(
            name = "ocr_id_card",
            type = TestImageType.OCR_TEXT,
            width = 480,
            height = 300,
            description = "证件OCR测试 - 身份证格式"
        ),
        TestImageConfig(
            name = "ocr_handwriting_basic",
            type = TestImageType.OCR_HANDWRITING,
            width = 640,
            height = 480,
            description = "手写体识别测试 - 中文手写"
        ),

        // 图像分类测试图像
        TestImageConfig(
            name = "classify_cat",
            type = TestImageType.CLASSIFICATION,
            width = 224,
            height = 224,
            description = "猫 - ImageNet分类测试"
        ),
        TestImageConfig(
            name = "classify_dog",
            type = TestImageType.CLASSIFICATION,
            width = 224,
            height = 224,
            description = "狗 - ImageNet分类测试"
        ),
        TestImageConfig(
            name = "classify_car",
            type = TestImageType.CLASSIFICATION,
            width = 224,
            height = 224,
            description = "汽车 - ImageNet分类测试"
        ),
        TestImageConfig(
            name = "classify_flower",
            type = TestImageType.CLASSIFICATION,
            width = 224,
            height = 224,
            description = "花朵 - ImageNet分类测试"
        ),
        TestImageConfig(
            name = "classify_food",
            type = TestImageType.CLASSIFICATION,
            width = 224,
            height = 224,
            description = "食物 - ImageNet分类测试"
        ),
        TestImageConfig(
            name = "classify_nature",
            type = TestImageType.CLASSIFICATION,
            width = 224,
            height = 224,
            description = "自然风景 - ImageNet分类测试"
        ),

        // VLM场景理解测试图像
        TestImageConfig(
            name = "vlm_office_scene",
            type = TestImageType.VLM_SCENE,
            width = 512,
            height = 384,
            description = "办公室场景 - VLM描述测试"
        ),
        TestImageConfig(
            name = "vlm_street_scene",
            type = TestImageType.VLM_SCENE,
            width = 512,
            height = 384,
            description = "街道场景 - VLM描述测试"
        ),
        TestImageConfig(
            name = "vlm_room_scene",
            type = TestImageType.VLM_SCENE,
            width = 512,
            height = 384,
            description = "房间场景 - VLM描述测试"
        ),

        // 文档测试图像
        TestImageConfig(
            name = "document_a4_text",
            type = TestImageType.DOCUMENT,
            width = 595,
            height = 842,
            description = "A4文档测试 - 纯文本"
        ),
        TestImageConfig(
            name = "document_resume",
            type = TestImageType.DOCUMENT,
            width = 595,
            height = 842,
            description = "简历文档测试 - 混合布局"
        )
    )

    /**
     * 获取测试图像目录
     */
    fun getTestImagesDir(context: Context): File {
        val dir = File(context.filesDir, "test_images")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * 检查测试图像是否已存在
     */
    fun isTestImageGenerated(context: Context, config: TestImageConfig): Boolean {
        val file = File(getTestImagesDir(context), "${config.name}.png")
        return file.exists() && file.length() > 0
    }

    /**
     * 生成所有测试图像
     */
    suspend fun generateAllTestImages(
        context: Context,
        onProgress: (Int, String) -> Unit
    ): Int = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
        var generated = 0
        val total = TEST_IMAGES.size

        TEST_IMAGES.forEachIndexed { index, config ->
            try {
                generateTestImage(context, config)
                generated++
                val progress = ((index + 1) * 100 / total)
                onProgress(progress, "已生成: ${config.name}")
                Log.i(TAG, "Generated test image: ${config.name}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate test image: ${config.name}", e)
            }
        }

        generated
    }

    /**
     * 生成单个测试图像
     */
    fun generateTestImage(context: Context, config: TestImageConfig): File {
        val file = File(getTestImagesDir(context), "${config.name}.png")

        if (file.exists()) {
            return file
        }

        val bitmap = when (config.type) {
            TestImageType.OCR_TEXT -> generateOCRTestImage(config)
            TestImageType.OCR_HANDWRITING -> generateHandwritingTestImage(config)
            TestImageType.CLASSIFICATION -> generateClassificationTestImage(config)
            TestImageType.VLM_SCENE -> generateVLMTestImage(config)
            TestImageType.FACE_DETECTION -> generateFaceTestImage(config)
            TestImageType.DOCUMENT -> generateDocumentTestImage(config)
        }

        FileOutputStream(file).use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }
        bitmap.recycle()

        return file
    }

    /**
     * 生成OCR文字识别测试图像
     */
    private fun generateOCRTestImage(config: TestImageConfig): Bitmap {
        val bitmap = Bitmap.createBitmap(config.width, config.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // 白色背景
        canvas.drawColor(Color.WHITE)

        val paint = Paint().apply {
            color = Color.BLACK
            isAntiAlias = true
        }

        when (config.name) {
            "ocr_english_basic" -> {
                // 英文文字测试
                paint.textSize = 32f
                paint.typeface = Typeface.MONOSPACE
                canvas.drawText("The quick brown fox jumps over the lazy dog.", 50f, 80f, paint)
                paint.textSize = 24f
                canvas.drawText("HELLO WORLD 12345", 50f, 140f, paint)
                canvas.drawText("Testing OCR Recognition", 50f, 200f, paint)
                paint.textSize = 18f
                canvas.drawText("Smaller text for detailed testing", 50f, 260f, paint)
                canvas.drawText("Mixed Case: AbCdEfGhIjKlMnOp", 50f, 300f, paint)
            }
            "ocr_chinese_basic" -> {
                // 中文文字测试
                paint.textSize = 36f
                paint.typeface = Typeface.DEFAULT
                canvas.drawText("中文文字识别测试", 50f, 80f, paint)
                paint.textSize = 28f
                canvas.drawText("欢迎使用智能招聘系统", 50f, 150f, paint)
                paint.textSize = 24f
                canvas.drawText("候选人姓名：张三", 50f, 220f, paint)
                canvas.drawText("应聘职位：高级工程师", 50f, 270f, paint)
                canvas.drawText("联系电话：138-1234-5678", 50f, 320f, paint)
                paint.textSize = 20f
                canvas.drawText("电子邮件：test@example.com", 50f, 370f, paint)
            }
            "ocr_mixed_languages" -> {
                // 中英混合测试
                paint.textSize = 28f
                canvas.drawText("HiringAI 智能招聘系统 v2.0", 50f, 60f, paint)
                paint.textSize = 22f
                canvas.drawText("Position: 高级Android开发工程师", 50f, 120f, paint)
                canvas.drawText("Requirements: 本科及以上学历", 50f, 160f, paint)
                canvas.drawText("Skills: Kotlin, Java, Android SDK", 50f, 200f, paint)
                canvas.drawText("薪资范围: 30K-50K / month", 50f, 240f, paint)
                paint.textSize = 18f
                canvas.drawText("简历投递: hr@company.com | 截止日期: 2024-12-31", 50f, 300f, paint)
            }
            "ocr_receipt" -> {
                // 收据格式测试
                paint.textSize = 20f
                paint.typeface = Typeface.MONOSPACE
                canvas.drawText("================================", 50f, 60f, paint)
                canvas.drawText("      INVOICE / 发票", 50f, 90f, paint)
                canvas.drawText("================================", 50f, 120f, paint)
                canvas.drawText("Date: 2024-01-15", 50f, 160f, paint)
                canvas.drawText("No: INV-2024-001234", 50f, 190f, paint)
                canvas.drawText("-------------------------------", 50f, 220f, paint)
                canvas.drawText("Item 1      x2    $99.00", 50f, 260f, paint)
                canvas.drawText("Item 2      x1    $49.50", 50f, 290f, paint)
                canvas.drawText("Item 3      x3    $29.97", 50f, 320f, paint)
                canvas.drawText("-------------------------------", 50f, 360f, paint)
                canvas.drawText("TOTAL:           $178.47", 50f, 400f, paint)
                canvas.drawText("================================", 50f, 440f, paint)
            }
            "ocr_id_card" -> {
                // 证件格式测试
                paint.textSize = 18f
                paint.strokeWidth = 2f
                paint.style = Paint.Style.STROKE
                canvas.drawRect(40f, 30f, config.width - 40f, config.height - 30f, paint)
                paint.style = Paint.Style.FILL
                paint.textSize = 22f
                canvas.drawText("居民身份证", 180f, 70f, paint)
                paint.textSize = 16f
                canvas.drawText("姓名：测试用户", 60f, 120f, paint)
                canvas.drawText("性别：男    民族：汉", 60f, 150f, paint)
                canvas.drawText("出生：1990年01月01日", 60f, 180f, paint)
                canvas.drawText("住址：北京市朝阳区测试路123号", 60f, 210f, paint)
                canvas.drawText("公民身份号码：110105199001011234", 60f, 260f, paint)
            }
            else -> {
                canvas.drawText("Test Image: ${config.name}", 50f, config.height / 2f, paint)
            }
        }

        return bitmap
    }

    /**
     * 生成手写体测试图像
     */
    private fun generateHandwritingTestImage(config: TestImageConfig): Bitmap {
        val bitmap = Bitmap.createBitmap(config.width, config.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.WHITE)

        val paint = Paint().apply {
            color = Color.BLACK
            isAntiAlias = true
            textSize = 28f
            strokeWidth = 2f
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }

        // 模拟手写体效果（使用路径绘制）
        val path = Path()

        // 绘制模拟手写文字
        canvas.drawText("手写体识别测试", 80f, 100f, paint)
        paint.textSize = 24f
        canvas.drawText("这是一段模拟的手写文字", 80f, 180f, paint)
        canvas.drawText("用于测试手写体OCR能力", 80f, 250f, paint)

        // 添加一些模拟的笔迹波动
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f

        // 绘制一些模拟的手写笔画
        path.moveTo(100f, 320f)
        path.cubicTo(120f, 310f, 180f, 330f, 200f, 320f)
        path.cubicTo(220f, 310f, 280f, 340f, 300f, 330f)
        canvas.drawPath(path, paint)

        return bitmap
    }

    /**
     * 生成图像分类测试图像
     */
    private fun generateClassificationTestImage(config: TestImageConfig): Bitmap {
        val bitmap = Bitmap.createBitmap(config.width, config.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        when (config.name) {
            "classify_cat" -> {
                // 绘制模拟猫的形状
                canvas.drawColor(Color.parseColor("#FFF3E0")) // 浅橙背景
                val paint = Paint().apply { color = Color.parseColor("#FF9800"); isAntiAlias = true }
                // 身体
                canvas.drawOval(40f, 100f, 180f, 200f, paint)
                // 头
                canvas.drawCircle(180f, 90f, 40f, paint)
                // 耳朵
                val earPath = Path()
                earPath.moveTo(155f, 55f)
                earPath.lineTo(165f, 30f)
                earPath.lineTo(175f, 55f)
                canvas.drawPath(earPath, paint)
                earPath.reset()
                earPath.moveTo(185f, 55f)
                earPath.lineTo(195f, 30f)
                earPath.lineTo(205f, 55f)
                canvas.drawPath(earPath, paint)
                // 尾巴
                paint.strokeWidth = 8f
                paint.style = Paint.Style.STROKE
                val tailPath = Path()
                tailPath.moveTo(40f, 150f)
                tailPath.quadTo(10f, 120f, 30f, 80f)
                canvas.drawPath(tailPath, paint)
                // 眼睛
                paint.style = Paint.Style.FILL
                paint.color = Color.GREEN
                canvas.drawCircle(170f, 85f, 6f, paint)
                canvas.drawCircle(190f, 85f, 6f, paint)
            }
            "classify_dog" -> {
                // 绘制模拟狗的形状
                canvas.drawColor(Color.parseColor("#E8D4B8")) // 浅棕背景
                val paint = Paint().apply { color = Color.parseColor("#8B4513"); isAntiAlias = true }
                // 身体
                canvas.drawOval(30f, 90f, 190f, 180f, paint)
                // 头
                canvas.drawCircle(170f, 70f, 50f, paint)
                // 耳朵
                canvas.drawOval(130f, 20f, 155f, 60f, paint)
                canvas.drawOval(185f, 20f, 210f, 60f, paint)
                // 鼻子
                paint.color = Color.BLACK
                canvas.drawOval(200f, 70f, 215f, 80f, paint)
                // 眼睛
                paint.color = Color.WHITE
                canvas.drawCircle(160f, 60f, 8f, paint)
                paint.color = Color.BLACK
                canvas.drawCircle(162f, 60f, 4f, paint)
                // 尾巴
                paint.color = Color.parseColor("#8B4513")
                paint.strokeWidth = 10f
                paint.style = Paint.Style.STROKE
                val tailPath = Path()
                tailPath.moveTo(30f, 120f)
                tailPath.quadTo(5f, 100f, 20f, 60f)
                canvas.drawPath(tailPath, paint)
            }
            "classify_car" -> {
                // 绘制模拟汽车形状
                canvas.drawColor(Color.parseColor("#E3F2FD")) // 浅蓝背景
                val paint = Paint().apply { color = Color.parseColor("#1976D2"); isAntiAlias = true }
                // 车身
                canvas.drawRoundRect(20f, 100f, 200f, 160f, 10f, 10f, paint)
                // 车顶
                canvas.drawRoundRect(50f, 60f, 170f, 100f, 10f, 10f, paint)
                // 窗户
                paint.color = Color.parseColor("#BBDEFB")
                canvas.drawRoundRect(60f, 70f, 110f, 95f, 5f, 5f, paint)
                canvas.drawRoundRect(115f, 70f, 160f, 95f, 5f, 5f, paint)
                // 轮子
                paint.color = Color.BLACK
                canvas.drawCircle(60f, 165f, 20f, paint)
                canvas.drawCircle(160f, 165f, 20f, paint)
                // 轮毂
                paint.color = Color.GRAY
                canvas.drawCircle(60f, 165f, 10f, paint)
                canvas.drawCircle(160f, 165f, 10f, paint)
            }
            "classify_flower" -> {
                // 绘制模拟花朵
                canvas.drawColor(Color.parseColor("#E8F5E9")) // 浅绿背景
                val paint = Paint().apply { isAntiAlias = true }
                // 茎
                paint.color = Color.parseColor("#4CAF50")
                paint.strokeWidth = 8f
                paint.style = Paint.Style.STROKE
                val stemPath = Path()
                stemPath.moveTo(112f, 200f)
                stemPath.lineTo(112f, 100f)
                canvas.drawPath(stemPath, paint)
                // 叶子
                paint.style = Paint.Style.FILL
                canvas.drawOval(80f, 150f, 112f, 170f, paint)
                canvas.drawOval(112f, 130f, 144f, 150f, paint)
                // 花瓣
                paint.color = Color.parseColor("#E91E63")
                for (angle in 0 until 360 step 45) {
                    canvas.save()
                    canvas.rotate(angle.toFloat(), 112f, 80f)
                    canvas.drawOval(102f, 40f, 122f, 80f, paint)
                    canvas.restore()
                }
                // 花心
                paint.color = Color.YELLOW
                canvas.drawCircle(112f, 80f, 15f, paint)
            }
            "classify_food" -> {
                // 绘制模拟食物（汉堡）
                canvas.drawColor(Color.parseColor("#FFF8E1")) // 浅黄背景
                val paint = Paint().apply { isAntiAlias = true }
                // 上面包
                paint.color = Color.parseColor("#D2691E")
                canvas.drawRoundRect(30f, 50f, 190f, 100f, 30f, 30f, paint)
                // 芝麻
                paint.color = Color.WHITE
                canvas.drawCircle(60f, 65f, 5f, paint)
                canvas.drawCircle(90f, 60f, 5f, paint)
                canvas.drawCircle(130f, 62f, 5f, paint)
                canvas.drawCircle(160f, 68f, 5f, paint)
                // 生菜
                paint.color = Color.parseColor("#4CAF50")
                val lettucePath = Path()
                lettucePath.moveTo(30f, 100f)
                lettucePath.cubicTo(50f, 90f, 70f, 110f, 90f, 100f)
                lettucePath.cubicTo(110f, 90f, 130f, 110f, 150f, 100f)
                lettucePath.cubicTo(170f, 90f, 185f, 105f, 190f, 100f)
                lettucePath.lineTo(190f, 115f)
                lettucePath.lineTo(30f, 115f)
                canvas.drawPath(lettucePath, paint)
                // 肉饼
                paint.color = Color.parseColor("#8B4513")
                canvas.drawRoundRect(35f, 115f, 185f, 145f, 5f, 5f, paint)
                // 番茄
                paint.color = Color.RED
                canvas.drawRoundRect(50f, 145f, 170f, 160f, 5f, 5f, paint)
                // 下面包
                paint.color = Color.parseColor("#D2691E")
                canvas.drawRoundRect(30f, 160f, 190f, 200f, 15f, 15f, paint)
            }
            "classify_nature" -> {
                // 绘制模拟自然风景
                // 天空渐变
                val skyGradient = LinearGradient(0f, 0f, 0f, 120f,
                    Color.parseColor("#87CEEB"), Color.parseColor("#E0F7FA"), Shader.TileMode.CLAMP)
                val skyPaint = Paint().apply { shader = skyGradient }
                canvas.drawRect(0f, 0f, config.width.toFloat(), 120f, skyPaint)
                // 太阳
                val sunPaint = Paint().apply { color = Color.YELLOW; isAntiAlias = true }
                canvas.drawCircle(180f, 40f, 25f, sunPaint)
                // 山
                val mountainPaint = Paint().apply { color = Color.parseColor("#4CAF50"); isAntiAlias = true }
                val mountainPath = Path()
                mountainPath.moveTo(0f, 150f)
                mountainPath.lineTo(70f, 60f)
                mountainPath.lineTo(140f, 150f)
                mountainPath.close()
                canvas.drawPath(mountainPath, mountainPaint)
                mountainPath.reset()
                mountainPath.moveTo(100f, 150f)
                mountainPath.lineTo(170f, 80f)
                mountainPath.lineTo(224f, 150f)
                mountainPath.close()
                canvas.drawPath(mountainPath, mountainPaint)
                // 草地
                val grassPaint = Paint().apply { color = Color.parseColor("#81C784") }
                canvas.drawRect(0f, 150f, config.width.toFloat(), config.height.toFloat(), grassPaint)
                // 树
                val trunkPaint = Paint().apply { color = Color.parseColor("#795548") }
                canvas.drawRect(45f, 130f, 55f, 180f, trunkPaint)
                val leavesPaint = Paint().apply { color = Color.parseColor("#388E3C"); isAntiAlias = true }
                canvas.drawCircle(50f, 115f, 25f, leavesPaint)
                canvas.drawCircle(35f, 125f, 18f, leavesPaint)
                canvas.drawCircle(65f, 125f, 18f, leavesPaint)
            }
            else -> {
                // 默认测试图像
                val gradient = LinearGradient(0f, 0f, config.width.toFloat(), config.height.toFloat(),
                    Color.BLUE, Color.GREEN, Shader.TileMode.CLAMP)
                val defaultPaint = Paint().apply { shader = gradient }
                canvas.drawRect(0f, 0f, config.width.toFloat(), config.height.toFloat(), defaultPaint)
            }
        }

        return bitmap
    }

    /**
     * 生成VLM场景理解测试图像
     */
    private fun generateVLMTestImage(config: TestImageConfig): Bitmap {
        val bitmap = Bitmap.createBitmap(config.width, config.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        when (config.name) {
            "vlm_office_scene" -> {
                // 办公室场景
                // 地板
                canvas.drawColor(Color.parseColor("#F5F5DC"))
                val floorPaint = Paint().apply { color = Color.parseColor("#DEB887") }
                canvas.drawRect(0f, 300f, config.width.toFloat(), config.height.toFloat(), floorPaint)

                // 墙壁
                val wallPaint = Paint().apply { color = Color.parseColor("#E8E8E8") }
                canvas.drawRect(0f, 0f, config.width.toFloat(), 300f, wallPaint)

                // 窗户
                val windowPaint = Paint().apply { color = Color.parseColor("#87CEEB") }
                canvas.drawRect(50f, 50f, 150f, 200f, windowPaint)
                val framePaint = Paint().apply { color = Color.WHITE; strokeWidth = 4f; style = Paint.Style.STROKE }
                canvas.drawRect(50f, 50f, 150f, 200f, framePaint)
                canvas.drawLine(100f, 50f, 100f, 200f, framePaint)
                canvas.drawLine(50f, 125f, 150f, 125f, framePaint)

                // 办公桌
                val deskPaint = Paint().apply { color = Color.parseColor("#8B4513") }
                canvas.drawRect(180f, 220f, 480f, 280f, deskPaint)
                canvas.drawRect(190f, 280f, 220f, 350f, deskPaint) // 桌腿
                canvas.drawRect(440f, 280f, 470f, 350f, deskPaint) // 桌腿

                // 电脑显示器
                val monitorPaint = Paint().apply { color = Color.BLACK }
                canvas.drawRect(280f, 140f, 380f, 210f, monitorPaint)
                val screenPaint = Paint().apply { color = Color.parseColor("#4FC3F7") }
                canvas.drawRect(285f, 145f, 375f, 205f, screenPaint)
                canvas.drawRect(320f, 210f, 340f, 230f, monitorPaint) // 支架

                // 椅子
                val chairPaint = Paint().apply { color = Color.parseColor("#2196F3") }
                canvas.drawRect(260f, 250f, 340f, 300f, chairPaint)
                canvas.drawRect(260f, 200f, 340f, 250f, chairPaint) // 椅背

                // 书架
                val shelfPaint = Paint().apply { color = Color.parseColor("#A0522D") }
                canvas.drawRect(400f, 50f, 500f, 250f, shelfPaint)
                canvas.drawLine(400f, 100f, 500f, 100f, Paint().apply { color = Color.parseColor("#8B4513"); strokeWidth = 3f })
                canvas.drawLine(400f, 150f, 500f, 150f, Paint().apply { color = Color.parseColor("#8B4513"); strokeWidth = 3f })
                canvas.drawLine(400f, 200f, 500f, 200f, Paint().apply { color = Color.parseColor("#8B4513"); strokeWidth = 3f })

                // 书本
                val bookPaint = Paint().apply { isAntiAlias = true }
                bookPaint.color = Color.RED
                canvas.drawRect(410f, 70f, 420f, 95f, bookPaint)
                bookPaint.color = Color.BLUE
                canvas.drawRect(420f, 75f, 430f, 95f, bookPaint)
                bookPaint.color = Color.GREEN
                canvas.drawRect(430f, 68f, 440f, 95f, bookPaint)
            }
            "vlm_street_scene" -> {
                // 街道场景
                // 天空
                val skyGradient = LinearGradient(0f, 0f, 0f, 180f,
                    Color.parseColor("#87CEEB"), Color.parseColor("#B3E5FC"), Shader.TileMode.CLAMP)
                canvas.drawRect(0f, 0f, config.width.toFloat(), 180f, Paint().apply { shader = skyGradient })

                // 道路
                val roadPaint = Paint().apply { color = Color.parseColor("#424242") }
                canvas.drawRect(0f, 250f, config.width.toFloat(), config.height.toFloat(), roadPaint)

                // 斑马线
                val zebraPaint = Paint().apply { color = Color.WHITE }
                for (i in 0 until 6) {
                    canvas.drawRect(200f + i * 25, 280f, 215f + i * 25, 340f, zebraPaint)
                }

                // 建筑
                val buildingPaint = Paint().apply { color = Color.parseColor("#90A4AE") }
                canvas.drawRect(20f, 80f, 120f, 250f, buildingPaint)
                canvas.drawRect(140f, 50f, 200f, 250f, buildingPaint)
                canvas.drawRect(350f, 100f, 450f, 250f, buildingPaint)
                canvas.drawRect(460f, 60f, 510f, 250f, buildingPaint)

                // 窗户
                val windowPaint = Paint().apply { color = Color.parseColor("#B3E5FC") }
                for (y in intArrayOf(100, 150, 200)) {
                    canvas.drawRect(35f, y.toFloat(), 55f, (y + 30).toFloat(), windowPaint)
                    canvas.drawRect(70f, y.toFloat(), 90f, (y + 30).toFloat(), windowPaint)
                }

                // 树
                val trunkPaint = Paint().apply { color = Color.parseColor("#795548") }
                canvas.drawRect(280f, 200f, 295f, 250f, trunkPaint)
                val leavesPaint = Paint().apply { color = Color.parseColor("#4CAF50"); isAntiAlias = true }
                canvas.drawCircle(287f, 185f, 30f, leavesPaint)

                // 汽车
                val carPaint = Paint().apply { color = Color.RED }
                canvas.drawRoundRect(320f, 260f, 420f, 290f, 10f, 10f, carPaint)
                canvas.drawCircle(340f, 295f, 12f, Paint().apply { color = Color.BLACK })
                canvas.drawCircle(400f, 295f, 12f, Paint().apply { color = Color.BLACK })

                // 红绿灯
                val polePaint = Paint().apply { color = Color.DKGRAY }
                canvas.drawRect(180f, 100f, 195f, 250f, polePaint)
                val lightPaint = Paint().apply { color = Color.parseColor("#37474F") }
                canvas.drawRect(170f, 80f, 205f, 140f, lightPaint)
                canvas.drawCircle(187f, 95f, 8f, Paint().apply { color = Color.RED })
                canvas.drawCircle(187f, 115f, 8f, Paint().apply { color = Color.parseColor("#4CAF50") })
            }
            "vlm_room_scene" -> {
                // 房间场景
                canvas.drawColor(Color.parseColor("#FFF8E1"))

                // 地板
                val floorPaint = Paint().apply { color = Color.parseColor("#D7CCC8") }
                canvas.drawRect(0f, 280f, config.width.toFloat(), config.height.toFloat(), floorPaint)

                // 床
                val bedPaint = Paint().apply { color = Color.parseColor("#7986CB") }
                canvas.drawRect(20f, 250f, 200f, 340f, bedPaint)
                val pillowPaint = Paint().apply { color = Color.WHITE }
                canvas.drawRoundRect(30f, 260f, 90f, 290f, 10f, 10f, pillowPaint)
                val blanketPaint = Paint().apply { color = Color.parseColor("#5C6BC0") }
                canvas.drawRect(20f, 290f, 200f, 340f, blanketPaint)

                // 床头柜
                val nightstandPaint = Paint().apply { color = Color.parseColor("#8D6E63") }
                canvas.drawRect(210f, 280f, 260f, 340f, nightstandPaint)
                // 台灯
                val lampShadePaint = Paint().apply { color = Color.parseColor("#FFEB3B") }
                canvas.drawOval(220f, 250f, 250f, 280f, lampShadePaint)
                canvas.drawRect(230f, 280f, 240f, 300f, Paint().apply { color = Color.parseColor("#757575") })

                // 衣柜
                val wardrobePaint = Paint().apply { color = Color.parseColor("#A1887F") }
                canvas.drawRect(350f, 100f, 500f, 340f, wardrobePaint)
                canvas.drawLine(425f, 100f, 425f, 340f, Paint().apply { color = Color.parseColor("#8D6E63"); strokeWidth = 3f })
                // 把手
                val handlePaint = Paint().apply { color = Color.parseColor("#FFD54F") }
                canvas.drawCircle(410f, 220f, 5f, handlePaint)
                canvas.drawCircle(440f, 220f, 5f, handlePaint)

                // 窗户
                val windowPaint = Paint().apply { color = Color.parseColor("#B3E5FC") }
                canvas.drawRect(260f, 80f, 340f, 200f, windowPaint)
                val curtainPaint = Paint().apply { color = Color.parseColor("#F48FB1") }
                canvas.drawRect(250f, 70f, 270f, 210f, curtainPaint)
                canvas.drawRect(330f, 70f, 350f, 210f, curtainPaint)

                // 地毯
                val rugPaint = Paint().apply { color = Color.parseColor("#CE93D8") }
                canvas.drawOval(100f, 300f, 300f, 380f, rugPaint)
            }
            else -> {
                // 默认场景
                canvas.drawColor(Color.LTGRAY)
                val paint = Paint().apply { color = Color.DKGRAY; textSize = 24f; isAntiAlias = true }
                canvas.drawText("Test Scene: ${config.name}", 50f, config.height / 2f, paint)
            }
        }

        return bitmap
    }

    /**
     * 生成人脸检测测试图像
     */
    private fun generateFaceTestImage(config: TestImageConfig): Bitmap {
        val bitmap = Bitmap.createBitmap(config.width, config.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.parseColor("#E3F2FD"))

        val paint = Paint().apply { isAntiAlias = true }

        // 绘制多个人脸
        val faces = listOf(
            Triple(100f, 100f, 40f),  // x, y, radius
            Triple(200f, 120f, 35f),
            Triple(300f, 100f, 38f),
            Triple(150f, 220f, 30f),
            Triple(250f, 200f, 32f)
        )

        faces.forEach { (x, y, r) ->
            // 脸
            paint.color = Color.parseColor("#FFCC80")
            canvas.drawOval(x - r, y - r * 1.2f, x + r, y + r, paint)
            // 眼睛
            paint.color = Color.WHITE
            canvas.drawCircle(x - r * 0.4f, y - r * 0.3f, r * 0.2f, paint)
            canvas.drawCircle(x + r * 0.4f, y - r * 0.3f, r * 0.2f, paint)
            paint.color = Color.BLACK
            canvas.drawCircle(x - r * 0.4f, y - r * 0.3f, r * 0.1f, paint)
            canvas.drawCircle(x + r * 0.4f, y - r * 0.3f, r * 0.1f, paint)
            // 嘴
            paint.color = Color.parseColor("#E57373")
            canvas.drawArc(x - r * 0.3f, y + r * 0.2f, x + r * 0.3f, y + r * 0.5f, 0f, 180f, false, paint)
        }

        return bitmap
    }

    /**
     * 生成文档测试图像
     */
    private fun generateDocumentTestImage(config: TestImageConfig): Bitmap {
        val bitmap = Bitmap.createBitmap(config.width, config.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.WHITE)

        val paint = Paint().apply {
            color = Color.BLACK
            isAntiAlias = true
            textSize = 14f
        }

        when (config.name) {
            "document_a4_text" -> {
                // A4文档格式
                paint.textSize = 24f
                paint.isFakeBoldText = true
                canvas.drawText("技术文档", 250f, 60f, paint)

                paint.textSize = 14f
                paint.isFakeBoldText = false

                var y = 100f
                val lines = listOf(
                    "1. 概述",
                    "",
                    "本文档描述了系统的整体架构设计。系统采用模块化设计，",
                    "支持多平台部署，具有良好的扩展性和可维护性。",
                    "",
                    "2. 系统架构",
                    "",
                    "2.1 客户端层",
                    "客户端采用MVVM架构，使用Jetpack组件实现。",
                    "主要包含以下模块:",
                    "- UI层: Fragment, ViewModel",
                    "- 数据层: Repository, Room Database",
                    "- 网络层: Retrofit, OkHttp",
                    "",
                    "2.2 服务端层",
                    "服务端采用微服务架构，提供RESTful API接口。",
                    "",
                    "3. 数据流程",
                    "",
                    "数据从客户端发起请求，经过网络层传输到服务端，",
                    "服务端处理后将结果返回给客户端。",
                    "",
                    "4. 安全机制",
                    "",
                    "系统实现了完整的安全机制，包括:",
                    "- 用户认证: JWT Token",
                    "- 数据加密: AES-256",
                    "- 通信安全: HTTPS/TLS 1.3"
                )

                lines.forEach { line ->
                    canvas.drawText(line, 50f, y, paint)
                    y += 22f
                }
            }
            "document_resume" -> {
                // 简历文档格式
                paint.textSize = 22f
                paint.isFakeBoldText = true
                canvas.drawText("个人简历", 250f, 50f, paint)

                // 分隔线
                paint.strokeWidth = 2f
                paint.style = Paint.Style.STROKE
                canvas.drawLine(50f, 70f, 545f, 70f, paint)
                paint.style = Paint.Style.FILL

                paint.textSize = 12f
                paint.isFakeBoldText = false

                var y = 100f

                // 基本信息
                paint.isFakeBoldText = true
                canvas.drawText("基本信息", 50f, y, paint)
                paint.isFakeBoldText = false
                y += 25f

                canvas.drawText("姓名: 张三                  性别: 男                  年龄: 28", 50f, y, paint)
                y += 20f
                canvas.drawText("电话: 138-1234-5678        邮箱: zhangsan@email.com", 50f, y, paint)
                y += 30f

                // 工作经历
                paint.isFakeBoldText = true
                canvas.drawText("工作经历", 50f, y, paint)
                paint.isFakeBoldText = false
                y += 25f

                canvas.drawText("2020.07 - 至今    ABC科技有限公司    高级Android开发工程师", 50f, y, paint)
                y += 20f
                canvas.drawText("• 负责公司核心App的开发与维护，用户量达500万+", 60f, y, paint)
                y += 18f
                canvas.drawText("• 主导架构重构，提升应用启动速度40%", 60f, y, paint)
                y += 18f
                canvas.drawText("• 引入Jetpack组件，代码可维护性显著提升", 60f, y, paint)
                y += 30f

                // 教育背景
                paint.isFakeBoldText = true
                canvas.drawText("教育背景", 50f, y, paint)
                paint.isFakeBoldText = false
                y += 25f

                canvas.drawText("2016.09 - 2020.06    XX大学    计算机科学与技术    本科", 50f, y, paint)
                y += 30f

                // 技能特长
                paint.isFakeBoldText = true
                canvas.drawText("技能特长", 50f, y, paint)
                paint.isFakeBoldText = false
                y += 25f

                canvas.drawText("编程语言: Kotlin, Java, Python", 50f, y, paint)
                y += 18f
                canvas.drawText("框架技术: Android SDK, Jetpack, Coroutines, Retrofit", 50f, y, paint)
                y += 18f
                canvas.drawText("开发工具: Android Studio, Git, Jenkins", 50f, y, paint)
            }
            else -> {
                canvas.drawText("Document: ${config.name}", 50f, config.height / 2f, paint)
            }
        }

        return bitmap
    }

    /**
     * 加载测试图像
     */
    fun loadTestImage(context: Context, config: TestImageConfig): Bitmap? {
        return try {
            val file = File(getTestImagesDir(context), "${config.name}.png")
            if (!file.exists()) {
                generateTestImage(context, config)
            }
            BitmapFactory.decodeFile(file.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load test image: ${config.name}", e)
            null
        }
    }

    /**
     * 获取所有测试图像文件
     */
    fun getAllTestImageFiles(context: Context): List<File> {
        val dir = getTestImagesDir(context)
        return dir.listFiles()?.filter { it.extension == "png" }?.toList() ?: emptyList()
    }

    /**
     * 清除所有测试图像
     */
    fun clearAllTestImages(context: Context) {
        getTestImagesDir(context).deleteRecursively()
    }
}
