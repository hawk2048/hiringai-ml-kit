# HiringAI-ML-Kit 白盒测试用例文档

> 版本: v1.0.0  
> 日期: 2026-04-26  
> 类型: 白盒测试 / 单元测试 + 集成测试

---

## 📋 测试概述

### 1.1 测试范围

| 模块 | 测试重点 | 测试类型 |
|------|----------|----------|
| `LocalLLMService` | Token 计数、模型加载/卸载、生成、Ollama 集成 | 白盒单元 |
| `LocalEmbeddingService` | Tokenization、向量归一化、相似度计算 | 白盒单元 |
| `LocalImageModelService` | 图像预处理、推理流程、加速后端选择 | 白盒单元 |
| `ModelStorage` | 路径管理、存储统计、清理 | 白盒单元 |
| `LLMBenchmarkRunner` | 进度计算、内存测量、批量测试流程 | 白盒单元 |
| `SpeechModelBenchmark` | 音频生成、RTF 计算、模型类型分支 | 白盒单元 |

### 1.2 测试环境

- **设备**: Android 模拟器 (x86_64) / 物理设备 (arm64-v8a)
- **JDK**: 17
- **Gradle**: 8.9.1
- **APK**: `app-debug.apk` (77MB)

---

## 🧪 测试用例详细规格

---

## 模块 1: LocalLLMService

### TC-LLM-001: Token 计数（静态方法）

**目的**: 验证 `estimateTokenCountStatic()` 的字符统计逻辑

**输入/预期输出**:

| 输入文本 | 预期最小值 | 验证点 |
|----------|------------|--------|
| `""` (空字符串) | 0 | 空字符串返回 0 |
| `"hello"` | 1 | 纯英文单词 |
| `"你好"` | 1 | 单个 CJK 字符 |
| `"hello world 你好 世界"` | 4 | 混合文本 |
| `"123!@#$%^&*()"` | 1 | 纯标点符号 |

**白盒验证点**:
- `kotlin if (text.isEmpty()) return 0` — 空字符串分支
- `kotlin when { cp in 0x4E00..0x9FFF ... }` — CJK 范围检测
- `kotlin cjkCount * 1.5 + asciiWordCount * 1.3 + punctCount * 0.1` — 加权计算公式
- `kotlin coerceAtLeast(1)` — 返回值最小为 1

**代码覆盖目标**: 覆盖所有 `when` 分支

---

### TC-LLM-002: 模型下载完整性检查

**目的**: 验证 `isModelDownloaded()` 文件存在性检查

**前置条件**: 模型文件可能存在/不存在/空文件

| 场景 | 文件状态 | 预期结果 |
|------|----------|----------|
| 文件存在且非空 | `test.gguf` (size > 0) | `true` |
| 文件存在但为空 | `empty.gguf` (size = 0) | `false` |
| 文件不存在 | — | `false` |

**白盒验证点**:
```kotlin
// LocalLLMService.kt:261-264
val file = File(getModelsDir(context), "$modelName.gguf")
return file.exists() && file.length() > 0
```

**边界值**: `file.length() == 0` 必须返回 `false`

---

### TC-LLM-003: 模型下载进度回调

**目的**: 验证下载进度计算的正确性

**输入**: 文件大小 1000 字节

| 已下载字节 | 预期进度 | 验证条件 |
|-----------|----------|----------|
| 0 | 0% | `bytesRead * 100 / contentLength` |
| 500 | 50% | 中点 |
| 999 | 99% | 不超过 100% |
| 1000 | 100% | 完整下载 |

**白盒验证点**:
```kotlin
// LocalLLMService.kt:304-309
val progress = (bytesRead * 100 / contentLength).toInt()
if (progress != lastProgress) {
    lastProgress = progress
    onProgress(progress)
}
```

**边界条件**: 
- `contentLength = 0` (服务器未返回大小) → 进度不更新
- `lastProgress` 避免重复回调

---

### TC-LLM-004: 模型加载/卸载状态管理

**目的**: 验证单例模式和状态转换

**操作序列**:

```
1. getInstance(context) → instance == null? → 创建新实例
2. loadModel(config) → model != null, currentModelName = config.name
3. unloadModel() → model = null, currentModelName = ""
4. loadModel(config) → 重新加载，替换旧模型
```

**白盒验证点**:
- `@Volatile private var instance` — 线程安全单例
- `unloadModel()` 总是调用 `clearLoadedModelName()`
- 加载新模型前自动调用 `unloadModel()` (第 341 行)

---

### TC-LLM-005: Ollama API JSON 解析

**目的**: 验证 `parseOllamaResponse()` 正确提取 JSON 中的 `response` 字段

**输入 JSON**:
```json
{"model":"qwen2.5:0.5b","response":"Hello! 我是通义千问。","done":true}
```

**预期输出**: `"Hello! 我是通义千问。"`

**白盒验证点**:
```kotlin
// LocalLLMService.kt:588-601
val key = "\"response\""
val keyIndex = json.indexOf(key)
// ... 查找冒号、提取引号内容
return json.substring(valueStart, valueEnd)
    .replace("\\n", "\n")
    .replace("\\\"", "\"")
    .replace("\\\\", "\\")
```

**边界情况**:
- 嵌套引号 `\"` → 转换为 `"`
- 转义反斜杠 `\\` → 转换为 `\`
- 字段不存在 → 返回 `null`

---

### TC-LLM-006: Ollama JSON 转义

**目的**: 验证 `escapeJson()` 正确处理特殊字符

| 输入 | 预期输出 |
|------|----------|
| `"hello"` | `"hello"` |
| `"he\"llo"` | `"he\\\"llo"` |
| `"line1\nline2"` | `"line1\\nline2"` |
| `"tab\there"` | `"tab\\there"` |
| `"back\\slash"` | `"back\\\\slash"` |

---

## 模块 2: LocalEmbeddingService

### TC-EMB-001: Vocab 加载

**目的**: 验证 `loadVocab()` 正确解析 vocab.txt

**输入文件格式**:
```
[CLS]
[SEP]
[UNK]
hello
world
你
好
```

**预期输出**: `Map("[CLS]" → 0, "[SEP]" → 1, "[UNK]" → 2, "hello" → 3, ...)`

**白盒验证点**:
```kotlin
// LocalEmbeddingService.kt:574-585
file.bufferedReader().useLines { lines ->
    lines.forEachIndexed { index, line ->
        val token = line.trim()
        if (token.isNotEmpty()) {
            map[token] = index
        }
    }
}
```

---

### TC-EMB-002: Tokenization 逻辑

**目的**: 验证 `tokenize()` 正确处理 [CLS]、WordPiece、[SEP]

**输入**: `"hello world 你好"`

**预期 token 序列**: `[CLS_ID, "hello", "world", "你", "好", SEP_ID, PAD..., PAD...]`

**白盒验证点**:
- 第 519 行: `tokens.add(CLS_TOKEN_ID)` — 开头添加 [CLS]
- 第 560 行: `tokens.add(SEP_TOKEN_ID)` — 结尾添加 [SEP]
- 第 524 行: `if (tokens.size >= maxLen - 1) break` — 长度限制
- 第 550 行: 字符级 fallback

---

### TC-EMB-003: L2 归一化

**目的**: 验证 `l2Normalize()` 数学正确性

**输入向量**: `[3.0f, 4.0f]` (L2 = 5)

**预期输出**: `[0.6f, 0.8f]` (除以 L2 范数)

**数学验证**:
```
||v|| = sqrt(3² + 4²) = 5
v_normalized[i] = v[i] / ||v||

验证: sqrt(0.6² + 0.8²) = sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0 ✓
```

**边界情况**: 零向量 `[0, 0]` → 返回原向量 (避免除零)

---

### TC-EMB-004: 余弦相似度

**目的**: 验证 `cosineSimilarity()` 数学正确性

**测试向量**:
- `a = [1.0, 0.0]` (沿 x 轴)
- `b = [1.0, 0.0]` (同向) → 预期 1.0
- `c = [0.0, 1.0]` (垂直) → 预期 0.0
- `d = [-1.0, 0.0]` (反向) → 预期 -1.0

**数学公式**: `cos(θ) = (a·b) / (||a|| × ||b||)`

---

### TC-EMB-005: ONNX Tensor 创建

**目的**: 验证 `encode()` 正确创建 ONNX 输入张量

**白盒验证点**:
```kotlin
// LocalEmbeddingService.kt:448-458
val inputIdsBuf = ByteBuffer.allocateDirect(seqLen * 8)
    .order(ByteOrder.nativeOrder())
    .asLongBuffer()

val inputIdsTensor = OnnxTensor.createTensor(
    env!!,
    inputIdsBuf,
    longArrayOf(1, seqLen.toLong())  // shape: [batch=1, seq_len]
)
```

**验证点**:
- Direct buffer (native memory)
- Native byte order
- Shape `[1, seqLen]`

---

### TC-EMB-006: 下载协程取消

**目的**: 验证 `isActive` 检查在下载循环中生效

**白盒验证点**:
```kotlin
// LocalEmbeddingService.kt:637
while (currentCoroutineContext().isActive) {
    val read = input.read(buffer)
    if (read == -1) break
    // ...
}

// 取消后清理
if (!currentCoroutineContext().isActive) {
    target.delete()  // 删除不完整文件
    return@withContext false
}
```

---

## 模块 3: LocalImageModelService

### TC-IMG-001: 图像预处理 NCHW 格式

**目的**: 验证 `preprocessImage()` 正确转换为 NCHW 并归一化到 [-1, 1]

**输入 Bitmap 像素** (ARGB):
```
像素 (0,0) = 0xFFFF0000 (红色, R=255, G=0, B=0)
像素 (0,1) = 0xFF00FF00 (绿色, R=0, G=255, B=0)
```

**预期 FloatArray** (假设 2x2 图像):
```
Channel 0 (R): [1.0, 1.0, ...]  // 红色归一化
Channel 1 (G): [-1.0, -1.0, ...] // 绿色归一化
Channel 2 (B): [-1.0, -1.0, ...] // 蓝色归一化
```

**归一化公式**: `value = pixel / 255.0 * 2 - 1`  
`(255/255)*2-1 = 1.0`  
`(0/255)*2-1 = -1.0`

**白盒验证点**:
```kotlin
// LocalImageModelService.kt:711-721
for (c in 0 until channels) {
    for (y in 0 until height) {
        for (x in 0 until width) {
            val pixel = pixels[y * width + x]
            val value = when (c) {
                0 -> ((pixel shr 16) and 0xFF) / 255.0f * 2 - 1 // R
                1 -> ((pixel shr 8) and 0xFF) / 255.0f * 2 - 1  // G
                else -> (pixel and 0xFF) / 255.0f * 2 - 1       // B
            }
            inputData[c * height * width + y * width + x] = value
        }
    }
}
```

---

### TC-IMG-002: 分类结果提取

**目的**: 验证 `extractClassificationResult()` 正确找到最大概率类别

**输入** (模型输出): `Array<FloatArray>([[0.1, 0.2, 0.5, 0.2]])`

**预期输出**: `Pair("class_2", 0.5f)` (索引 2 最大概率 0.5)

**白盒验证点**:
```kotlin
// LocalImageModelService.kt:736-743
var maxIdx = 0
var maxProb = probabilities[0]
for (i in probabilities.indices) {
    if (probabilities[i] > maxProb) {
        maxProb = probabilities[i]
        maxIdx = i
    }
}
```

---

### TC-IMG-003: 输入尺寸选择

**目的**: 验证 `getCurrentModelInputSize()` 根据模型类型返回正确尺寸

| 模型类型 | 预期尺寸 |
|----------|----------|
| `CLASSIFICATION` | `224 to 224` |
| `OCR` | `320 to 32` |
| `VLM` | `336 to 336` |

---

### TC-IMG-004: 加速后端回退链

**目的**: 验证 `createSessionOptionsWithAcceleration()` 正确回退

**调用顺序**: GPU → NNAPI → XNNPACK → CPU

**白盒验证点**:
```kotlin
// LocalImageModelService.kt:343-387
val fallbackChain = accelerationConfig.getEffectiveFallbackChain()
for (backend in fallbackChain) {
    when (backend) {
        GPU -> { if (usedGPU) return sessionOptions }
        NNAPI -> { if (isNNAPISafe) return sessionOptions }
        XNNPACK -> { return sessionOptions } // 默认启用
        CPU -> { return sessionOptions } // 最终回退
    }
}
```

---

## 模块 4: ModelStorage

### TC-STR-001: 目录结构创建

**目的**: 验证 `getBaseDir()` 和子目录正确创建

**预期路径**: 
```
Android/data/com.hiringai.mobile.ml.testapp/files/Download/MLModels/
├── llm/
├── embedding/
├── image/
└── speech/
```

**白盒验证点**:
```kotlin
// ModelStorage.kt:37-51
fun getBaseDir(context: Context): File {
    val dir = File(
        context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),
        "MLModels"
    )
    if (!dir.exists()) {
        val created = dir.mkdirs()
        // ...
    }
    return dir
}
```

---

### TC-STR-002: 存储统计计算

**目的**: 验证 `getStorageInfo()` 正确汇总各目录大小

**测试场景**:
```
llm/        → 394 MB (Qwen2.5-0.5B.gguf)
embedding/ → 91 MB  (all-MiniLM-L6-v2.onnx)
image/      → 14 MB  (mobilenet_v2_224.onnx)
speech/     → 0 MB
───────────────────────────
total       → 499 MB
```

**预期输出**: `StorageInfo(llm=394MB, embedding=91MB, image=14MB, speech=0MB, total=499MB)`

---

### TC-STR-003: 格式化为可读字符串

**目的**: 验证 `StorageInfo.format()` 正确转换字节数

| 字节数 | 预期输出 |
|--------|----------|
| `500` | `"500 B"` |
| `50_000` | `"50 KB"` |
| `50_000_000` | `"50 MB"` |
| `5_000_000_000` | `"4.7 GB"` |

---

### TC-STR-004: 清理所有模型

**目的**: 验证 `clearAll()` 删除所有文件但保留目录结构

**前置条件**: 目录中存在多个模型文件

**操作**: `ModelStorage.clearAll(context)`

**预期结果**: 
- 所有 `.gguf`、`.onnx` 等模型文件被删除
- 目录结构 (`llm/`, `embedding/` 等) 保留
- 返回删除的文件数量

---

## 模块 5: LLMBenchmarkRunner

### TC-BEN-001: 进度阶段权重

**目的**: 验证 `BenchmarkStage` 权重分配合理

**预期权重分布**:
| 阶段 | 权重 | 说明 |
|------|------|------|
| `CHECK_DOWNLOAD` | 5% | 快速检查 |
| `LOADING` | 30% | 模型加载 |
| `GENERATING` | 55% | 推理生成 |
| `UNLOADING` | 10% | 资源清理 |

**验证**: `0.05 + 0.30 + 0.55 + 0.10 = 1.0`

---

### TC-BEN-002: 总体进度计算

**目的**: 验证批量测试总体进度计算公式

**场景**: 3 个模型，测试到第 2 个模型 (索引 1)，GENERATING 阶段 50%

**公式**:
```
baseProgress = index = 1
stageWeight = 0.55 (GENERATING)
stageProgress = 50

overallPercent = ((1 + (50/100 * 0.55)) / 3 * 100)
              = ((1 + 0.275) / 3 * 100)
              = (1.275 / 3 * 100)
              = 42.5%
```

---

### TC-BEN-003: 内存使用测量

**目的**: 验证 `getCurrentMemoryUsageMB()` 计算准确

**计算方式**:
```kotlin
// ActivityManager.MemoryInfo
val memInfo = ActivityManager.MemoryInfo()
activityManager.getMemoryInfo(memInfo)
val usedMemory = memInfo.totalMem - memInfo.availMem
return usedMemory / (1024 * 1024)  // 转换为 MB
```

---

### TC-BEN-004: Token 吞吐量计算

**目的**: 验证 throughput 计算正确

**场景**:
- `generatedText.length = 150` 字符
- `generateTimeMs = 3000` 毫秒
- Token 估算: `150 / 1.5 = 100` tokens

**计算**:
```
throughput = (tokens / time) * 1000
           = (100 / 3000) * 1000
           = 33.33 tokens/s
```

---

### TC-BEN-005: 快速基准测试（无加载）

**目的**: 验证 `quickBenchmark()` 在模型已加载时直接测试

**前置条件**: `llmService.isModelLoaded == true`

**流程**: 只执行 GENERATING 阶段，跳过 LOADING

---

## 模块 6: SpeechModelBenchmark

### TC-SPE-001: 测试音频生成

**目的**: 验证 `generateTestAudio()` 生成正确采样率

**输入**: `durationSeconds = 5.0, sampleRate = 16000`

**预期**:
- `FloatArray` 长度 = `5.0 * 16000 = 80000`
- 值域: `[-0.5, 0.5]` (正弦波)

**公式**: `sample[i] = sin(2π × 440 × t) × 0.5`  
其中 `t = i / sampleRate`

---

### TC-SPE-002: RTF 计算

**目的**: 验证 Real-Time Factor 计算

**场景**:
- 音频时长: `testDurationSeconds = 5.0` 秒
- 推理耗时: `inferenceTimeMs = 2500` 毫秒

**计算**:
```
RTF = inferenceTimeMs / 1000 / audioDuration
    = 2500 / 1000 / 5.0
    = 0.5

实时倍数 = 1 / RTF = 1 / 0.5 = 2.0x
```

**预期**: RTF < 1.0 表示实时或超实时

---

### TC-SPE-003: 模型类型分支覆盖

**目的**: 验证 `benchmarkModel()` 正确路由不同模型类型

| 模型类型 | 调用方法 |
|----------|----------|
| `WHISPER` | `speechService.transcribe()` |
| `WHISPER_ENCODER_DECODER` | `speechService.transcribe()` |
| `PARAFORMER` | `speechService.transcribe()` |
| `CAM_PLUS` / `VAD` | `speechService.detectVoiceActivity()` |
| `TTS` | `speechService.synthesize()` |

**白盒验证点**:
```kotlin
// SpeechModelBenchmark.kt:183-193
when (config.type) {
    WHISPER, WHISPER_ENCODER_DECODER, PARAFORMER -> {
        speechService.transcribe(testAudio, config.sampleRate)
    }
    CAM_PLUS, VAD -> {
        speechService.detectVoiceActivity(testAudio, config.sampleRate)
    }
    TTS -> {
        speechService.synthesize("测试语音合成")
    }
}
```

---

## 模块 7: TestImageGenerator

### TC-TST-001: 分类测试图生成

**目的**: 验证 `generateClassificationTestImage()` 包含预期形状

**预期形状**:
- 蓝色圆形 (`Color.rgb(65, 105, 225)`)
- 红色矩形
- 绿色三角形
- 黄色椭圆形

---

### TC-TST-002: OCR 测试图包含中文

**目的**: 验证 `generateOCRTestImage()` 包含中英文混合文本

**预期内容**:
- `"TEST SAMPLE"` (标题)
- `"12345 ABCDE"` (数字+大写)
- `"The quick brown fox jumps over lazy dog"` (英文句子)
- `"测试文字识别"` (中文)

---

### TC-TST-003: 噪声图可复现性

**目的**: 验证 `generateNoiseImage()` 使用固定种子产生可复现结果

**白盒验证点**:
```kotlin
// TestImageGenerator.kt:149
val random = java.util.Random(42) // 固定种子
```

**验证**: 相同尺寸调用两次产生完全相同的像素值

---

### TC-TST-004: 测试图像保存/加载

**目的**: 验证 `saveTestImage()` 和 `loadTestImage()` 往返一致性

**往返测试**:
```
generateTestImage() → bitmap1
saveTestImage(bitmap1, file)
loadTestImage(file) → bitmap2
bitmap1 == bitmap2?
```

---

## 🔄 自动化测试执行

### 测试脚本

参见项目根目录 `auto_test.py`

```bash
# 运行所有测试
python auto_test.py

# 仅测试主界面
python auto_test.py --main

# 仅测试模型目录
python auto_test.py --models

# 重新安装后测试
python auto_test.py --reinstall
```

---

## 📊 测试报告模板

```
========================================
HiringAI-ML-Kit 白盒测试报告
========================================
日期: {timestamp}
设备: {device_info}
APK版本: {app_version}

【模块测试结果】
┌─────────────────────┬────────┬────────┬────────┐
│ 模块                 │ 通过   │ 失败   │ 总计   │
├─────────────────────┼────────┼────────┼────────┤
│ LocalLLMService     │   6    │   0    │   6    │
│ LocalEmbedding      │   6    │   0    │   6    │
│ LocalImageModel     │   4    │   0    │   4    │
│ ModelStorage        │   4    │   0    │   4    │
│ LLMBenchmark       │   5    │   0    │   5    │
│ SpeechBenchmark     │   3    │   0    │   3    │
│ TestImageGenerator  │   4    │   0    │   4    │
├─────────────────────┼────────┼────────┼────────┤
│ 总计                │  32    │   0    │  32    │
└─────────────────────┴────────┴────────┴────────┘

【失败测试详情】
(无)

【代码覆盖率】
(需要 JaCoCo/Android Studio Profiler)

【性能基准】
- 冷启动时间: {startup_time}ms
- 模型加载时间: {load_time}ms
- 推理延迟: {inference_latency}ms
========================================
```

---

## 📝 附录

### A. 测试数据生成

```kotlin
// 测试用 LLM 模型配置
val testModel = LocalLLMService.ModelConfig(
    name = "Qwen2.5-0.5B-Instruct-Q4_0",
    url = "https://hf-mirror.com/Qwen/...",
    size = 394_774_816,
    requiredRAM = 1
)

// 测试用 Embedding 模型配置
val testEmbedding = LocalEmbeddingService.EmbeddingModelConfig(
    name = "all-MiniLM-L6-v2",
    modelUrl = "https://hf-mirror.com/sentence-transformers/...",
    modelSize = 91_000_000,
    dimension = 384
)
```

### B. Mock 对象策略

由于 Android 环境的复杂性，以下组件需要 Mock:

| 组件 | Mock 策略 |
|------|-----------|
| `OrtEnvironment` | 使用 ONNX Runtime Test 包 |
| `LlamaModel` | 使用 llama.cpp 测试桩 |
| `Context.getExternalFilesDir()` | 使用临时目录替代 |

### C. 测试执行顺序

```
1. 单元测试 (ml 模块内部)
   ├── ModelStorageTest
   ├── LocalEmbeddingTest (tokenization, L2 normalize)
   └── LocalLLMTest (token count, JSON parse)

2. 集成测试 (app 模块)
   ├── BenchmarkActivityTest
   └── ModelCatalogActivityTest

3. UI 自动化测试
   └── auto_test.py (基于 uiautomator)
```
