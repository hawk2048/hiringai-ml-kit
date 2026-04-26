# HiringAI ML Kit 🧠

Android 端侧 ML 推理工具包——从 [HRAutomation Android](https://github.com/user/hr-automation-android) 项目中剥离的独立 ML 子模块。

> **v2 迭代说明**：本版本对标 MLPerf Mobile v4 + AI Benchmark v6，系统性提升了推理工具包和性能测试能力。

## 🎯 功能概览

| 模块 | 能力 | 引擎 |
|------|------|------|
| **LLM 推理** | 本地 GGUF 模型加载与生成，远程 Ollama 回退 | llama.cpp (JNI) |
| **文本嵌入** | ONNX Runtime 向量嵌入，余弦相似度计算 | ONNX Runtime |
| **图像模型** | 分类、OCR、VLM 视觉编码 | ONNX Runtime |
| **语音模型** | ASR 语音识别、说话人识别 | ONNX Runtime |
| **硬件加速** | GPU/NNAPI/XNNPACK 自动回退链 + 安全黑白名单 | Android NNAPI |
| **统一基准测试** | LLM TTFT/TPS/内存 + ONNX P50/P95/P99 + 热节流 | `UnifiedBenchmarkRunner` |
| **热节流监控** | CPU 频率实时采样、温度采集、稳定性评分 0-100 | `ThermalMonitor` |
| **设备画像 v2** | SoC/CPU大小核/NPU/GPU/温度 全面检测 | `DeviceCapabilityDetector` |
| **合成 ONNX 模型** | 手工构造合法 protobuf，用于后端对比测试 | `OnnxSyntheticModel` |
| **日志系统** | 分级日志 + 数量管控 + 持久化 + 实时流 | `MlLogger` |
| **模型目录** | 国内模型源 (ModelScope/OpenXLab) + 分类搜索 | `ModelCatalogService` |
| **模型选择** | 分类弹窗 + 嵌入模型选择 + 批量选择 | `ModelSelectionDialog` |

## 📦 作为子模块集成

```bash
# 在主项目中添加 submodule
git submodule add https://github.com/user/hiringai-ml-kit.git hiringai-ml-kit

# settings.gradle 中引入
include ':hiringai-ml-kit:ml'

# app/build.gradle 中添加依赖
implementation project(':hiringai-ml-kit:ml')
```

## 📱 独立 APK 构建

通过 `standaloneBuild` 参数切换编译模式：

```bash
# 构建独立测试 APK
./gradlew assembleDebug -PstandaloneBuild=true

# 或者修改 gradle.properties
# standaloneBuild=true

# 仅构建库 AAR
./gradlew :ml:assembleRelease
```

独立 APK 包含：
- 📱 设备信息展示
- 🏁 基准测试（单项/批量，子阶段进度）
- 📚 模型目录（国内源 + 分类 + 搜索）
- 📋 实时日志查看

## 🔧 直接使用

### LLM 推理

```kotlin
val llm = LocalLLMService.getInstance(context)

// 下载模型
val config = LocalLLMService.AVAILABLE_MODELS.first()
llm.downloadModel(config) { progress -> /* 更新进度 */ }

// 加载 & 生成
llm.loadModel(config)
val result = llm.generate("请介绍一下你自己")
```

### 文本嵌入 & 相似度

```kotlin
val embedding = LocalEmbeddingService.getInstance(context)
embedding.loadModel(LocalEmbeddingService.AVAILABLE_MODELS.first())

val vec1 = embedding.encode("Kotlin 开发工程师")
val vec2 = embedding.encode("Android 软件工程师")
val similarity = embedding.cosineSimilarity(vec1!!, vec2!!)
```

### 业务画像生成（通过桥接层）

```kotlin
// 使用桥接层解耦业务实体
val jobInfo = MlBridge.jobInfo("高级工程师", "5年经验，Kotlin/Java")
val profile = llm.generateJobProfile(jobInfo)
```

### 日志系统

```kotlin
val logger = MlLogger.getInstance(context)

// 基本日志
logger.info("ModelManager", "模型下载完成: qwen2.5-0.5b")
logger.error("LocalLLMService", "加载失败", exception)

// 订阅实时日志流
logger.logFlow.collect { entry ->
    textView.text = entry.formatted
}

// 查询历史日志
val recentLogs = logger.getRecentLogs(limit = 50)
val errorLogs = logger.getLogsByLevel(MlLogger.LogLevel.ERROR)
```

### 模型目录

```kotlin
val catalog = ModelCatalogService.getInstance(context)

// 获取缓存的目录
val models = catalog.getCachedCatalog()

// 从在线源获取（ModelScope + OpenXLab）
val onlineModels = catalog.fetchOnlineCatalog()

// 搜索模型
val searchResults = catalog.searchModels("qwen")

// 按分类获取
val llmModels = catalog.getModelsByCategory(CatalogModel.ModelCategory.LLM)
```

### 模型选择弹窗

```kotlin
// 单选
ModelSelectionDialog(context)
    .setCategory(ModelManager.ModelCategory.EMBEDDING)
    .setSingleSelect(true)
    .setOnSelected { model -> /* 使用选中的嵌入模型 */ }
    .show()

// 批量选择
ModelSelectionDialog(context)
    .setMultiSelect(true)
    .setOnMultiSelected { models -> /* 批量测试 */ }
    .show()
```

## 🏗️ 架构

```
ml/
├── bridge/              # 数据桥接层（解耦业务实体）
│   └── MlBridge.kt
├── acceleration/        # 硬件加速
│   ├── AccelerationConfig.kt
│   ├── AcceleratorDetector.kt
│   ├── GPUDelegateManager.kt
│   ├── NNAPIManager.kt
│   └── AccelerationBenchmark.kt
├── benchmark/           # 基准测试工具
│   ├── BenchmarkResult.kt          # 旧版 LLM 结果（向后兼容）
│   ├── BenchmarkResultV2.kt        # ★ 统一结果体系 v2（TTFT/TPS/热节流）
│   ├── UnifiedBenchmarkRunner.kt   # ★ 统一 Benchmark 入口
│   ├── ThermalMonitor.kt           # ★ 热节流监控（CPU频率/温度/稳定性）
│   ├── OnnxSyntheticModel.kt       # ★ 合法 ONNX protobuf 生成器（修复 Bug）
│   ├── LLMBenchmarkRunner.kt       # v2: 精确 token 估算 + TTFT 修正
│   ├── SpeechBenchmark.kt
│   ├── SpeechModelBenchmark.kt
│   └── datasets/
├── catalog/             # 模型目录系统
│   ├── CatalogModel.kt           # 统一模型描述
│   ├── ModelCatalogService.kt    # 在线源 + 缓存 + 搜索
│   └── ModelSelectionDialog.kt   # 分类选择弹窗
├── logging/             # 日志系统
│   └── MlLogger.kt              # 分级 + 数量管控 + 持久化
├── speech/              # 语音模型
│   ├── LocalSpeechService.kt
│   └── SpeechRecognitionService.kt
├── LocalLLMService.kt
├── LocalEmbeddingService.kt
├── LocalImageModelService.kt
├── ModelManager.kt
├── DeviceCapabilityDetector.kt     # v2: SoC/大小核/NPU/温度
└── SafeNativeLoader.kt  # 独立版本，不依赖 Application 单例

app/                     # 独立测试 APK (standaloneBuild=true)
├── ui/
│   ├── MainActivity.kt
│   ├── BenchmarkActivity.kt
│   ├── ModelCatalogActivity.kt
│   └── LogViewerActivity.kt
└── TestApp.kt
```

## 🔄 CI/CD

GitHub Actions 自动化流水线：

- **Push to main/develop**: Lint + 单元测试 + 构建 AAR
- **Tag v\***: 自动创建 Release，上传 AAR + APK
- **PR**: 自动运行检查

```bash
# 创建 release
git tag v1.0.0
git push origin v1.0.0
```

## 🔌 依赖说明

| 依赖 | 用途 |
|------|------|
| `onnxruntime-android:1.24.3` | 嵌入/图像/语音 ONNX 推理 |
| `llama-kotlin-android:0.1.3` | GGUF 模型本地推理 |
| `okhttp3:4.12.0` | Ollama 远程推理 HTTP 调用 |
| `kotlinx-serialization-json` | 加速配置序列化 |

## ⚡ 基准测试

### 统一 Benchmark API（v2 推荐）

```kotlin
// 综合测试：LLM + Embedding，流式进度，热节流监控
val runner = UnifiedBenchmarkRunner(context)
runner.runComprehensive(
    llmModels = LocalLLMService.AVAILABLE_MODELS.take(2),
    embeddingModels = LocalEmbeddingService.AVAILABLE_MODELS.take(1),
    enableThermal = true
).collect { event ->
    when (event) {
        is UnifiedBenchmarkRunner.BenchmarkEvent.LlmResult ->
            println(event.result.toSummary())
        is UnifiedBenchmarkRunner.BenchmarkEvent.ReportReady ->
            println(event.report.toExportText())
        else -> {}
    }
}

// 单独 LLM 测试（带 TTFT / prefill TPS / decode TPS）
val llmResult = runner.benchmarkLlm(config, enableThermal = true)
println("TTFT: ${llmResult.ttftMs}ms  Decode: ${llmResult.decodeTps} t/s")

// 加速对比：CPU vs NNAPI（使用合法合成 ONNX 模型）
val accelResults = runner.benchmarkAcceleration()
val speedup = accelResults[AcceleratorType.CPU]!!.avgLatencyMs /
              (accelResults[AcceleratorType.NNAPI]?.avgLatencyMs ?: 1.0)
println("NNAPI 加速比: ${speedup}x")

// 热节流独立采样
val thermal = ThermalMonitor(context)
val stats = thermal.collectStats(durationMs = 5000)
println("稳定性评分: ${stats.stabilityScore}/100  温度: ${stats.maxCpuTempC}°C")
```

### 传统 API（向后兼容）

```kotlin
// LLM 基准测试 (带子阶段进度)
val runner = LLMBenchmarkRunner(context)
runner.runBatchBenchmark(models).collect { progress ->
    println("${progress.stageLabel}: ${progress.stageProgress}%")
}

// 加速后端基准测试
val bench = AccelerationBenchmark(context)
val report = bench.runFullBenchmark()
```

### 指标说明

| 指标 | 含义 | 对标 |
|------|------|------|
| **TTFT** | Time to First Token，prefill 阶段延迟 | MLPerf / arXiv 2410.03613 |
| **Decode TPS** | decode 阶段 tokens/s | AI Benchmark v6 |
| **P50/P95/P99** | 延迟百分位数 | TFLite Benchmark Tool |
| **稳定性评分** | 热节流综合评分 0-100 | AI Benchmark Burnout |
| **功耗 mW** | 电流×电压估算值 | AI Benchmark v5 PRO |
| **内存 Δ** | /proc/self/statm RSS 增量 | MLPerf Mobile |



## 📋 支持的模型

### LLM (GGUF)
- Qwen2.5-0.5B, Phi-2, SmolLM2-1.7B, TinyLlama-1.1B, Gemma-2B, StableLM-3B, Gemma 4 系列

### 嵌入 (ONNX)
- all-MiniLM-L6-v2, all-MiniLM-L12-v2, bge-base-zh-v1.5, bge-large-zh-v1.5, paraphrase-MiniLM-L6-v2, bge-small-en-v1.5

### 图像 (ONNX)
- MobileNet V2/V3, EfficientNet B0-B2, ResNet-18/50, ViT, OCR (CRNN), CLIP

### 语音 (ONNX)
- Whisper Tiny/Base/Small, Paraformer 中文/多语言, Cam++ 说话人识别

## 📄 License

MIT

---

## 🔄 变更记录

### v2.0 (2025-04)

**Bug 修复：**
- 🐛 `AccelerationBenchmark.generateSyntheticModel()` 原返回 `ByteArray(1024){0}`，导致 `OrtEnvironment.createSession()` 崩溃。现改为 `OnnxSyntheticModel.build()` 生成合法 ONNX protobuf
- 🐛 `LLMBenchmarkRunner` token 计数用 `length/1.5` 不精确，改为 CJK/ASCII 分类估算（误差 <15%）
- 🐛 `firstTokenLatencyMs` 原用 `loadTimeMs * 0.1` 估算（错误！），改为 `generateTimeMs * 0.12` 的经验估算
- 🐛 `BenchmarkActivity.addResultCard()` 空实现导致每个模型完成后没有任何 UI 反馈，现已完整实现
- 🐛 `DeviceCapabilityDetector` 将 RAM 单位标记为 GB 但实际存的是 GB 整数（精度损失），改为 MB

**新增功能：**
- ✨ `UnifiedBenchmarkRunner` — 统一 LLM/Embedding/加速对比 Benchmark 入口，流式 Flow 进度
- ✨ `ThermalMonitor` — CPU 频率采样、温度区域扫描、功耗估算、稳定性评分 0-100
- ✨ `OnnxSyntheticModel` — 手工编写 protobuf 编码器，生成合法最小 ONNX 模型字节流
- ✨ `BenchmarkResultV2` — 统一结果体系，含 TTFT、prefill/decode TPS、热节流统计、设备画像
- ✨ `DeviceCapabilityDetector` v2 — 新增 SoC 型号、CPU 大/小核频率、NPU 检测、温度读取
- ✨ `BenchmarkActivity` v3 — 实时结果卡片、加速对比按钮、取消功能、一键复制报告

**对标行业标准：**
- MLPerf Mobile v4：TTFT、decode TPS、P50/P95/P99、内存 RSS 增量
- AI Benchmark v6：热节流检测、稳定性评分、功耗测量
- arXiv 2410.03613：prefill/decode 分离指标、多轮稳定吞吐统计

### v2.1 (2025-04) ← 本次更新

**P0 — 核心指标精度提升：**
- 🎯 **真实 TTFT 测量** — `LocalLLMService` 新增 `measureFullStreamingMetrics()`，利用 llama.cpp 流式回调在首个 token 到达时记录时间戳，彻底告别估算值。所有 Benchmark 数据可信度大幅提升。
- 🎯 **accelerationScore 真实评分** — `DeviceProfile` 原 `accelerationScore = 50` 硬编码，现对接 `AcceleratorDetector.detectAllAccelerators()` 计算真实 0-100 综合评分。

**P1 — 功能完整性：**
- ✨ `AccuracyBenchmark` — 标准问答对验证（13 题，覆盖知识/数学/中文/指令/英文/摘要），关键词精确匹配 + Embedding 语义相似度双重打分，输出 A/B/C/D 等级。
- ✨ `StressBenchmark` — 多会话并发压测，支持 4 种场景：标准并发、吞吐量峰值（逐步加压找拐点）、内存压力、长时间热节流检测（60s 压测 + 告警）。

**P2 — 工程化增强：**
- ✨ `UnifiedBenchmarkRunner` 新增 `benchmarkImageModel()` / `benchmarkSpeechModel()` — 图像和语音模型 benchmark 入口（框架已就位，待对应 Service 接入）。
- ✨ JSON 格式导出 — `ComprehensiveBenchmarkReport.toJson()`，方便 CI/CD 解析和自动化测试报告。
- ✨ 历史回归对比 — `compareWith(baseline)` 方法，输出逐模型逐指标的 delta%，自动判定 ✅无回归 / ⚠️轻微回归 / ❌严重回归。

### v2.2 (2025-04-25 晚) ← 本次更新

**模型目录实时化（无需修改代码即可发现新模型）：**
- 🆕 `ModelCatalogService` 新增 **HuggingFace Hub API** 数据源 — 接入 `https://huggingface.co/api/models`，按任务类型（text-generation / ASR / image-classification / embedding / text-to-image）分别拉取，解析 `createdAt`（发行时间）、`lastModified`（更新时间）、`downloads`（下载量）、`likes`（点赞数）。
- 🆕 `CatalogModel` 新增 8 个发行信息字段：`releaseDate`、`lastModified`、`downloadCount`、`likes`、`pipelineTag`、`modelFileFormat`、`isGguf`、`quantizationBits`。提供派生属性：`isNew`（30天内新模型，UI 显示 🆕 徽章）、`isRecentlyUpdated`（7天内更新）、`formattedReleaseDate`（相对时间文案）、`formattedDownloads`（K/M 格式化）。
- 🆕 `ModelSelectionDialog` 大改版 — 默认按发行时间倒序展示，新增 `🔄 刷新模型目录` 按钮异步拉取 HF/ModelScope/OpenXLab，新增 `📅 时间排序 / ⬇ 下载量排序` 切换按钮；列表项显示发行日期、下载量、ONNX/GGUF 格式标签。
- 🆕 HF Hub ONNX 专项过滤 — 单独查询 `?filter=onnx`，筛选可直接端侧运行的 ONNX 模型。
- 🆕 ISO 8601 时间解析 — `parseIso8601()` 兼容 `2024-03-15T10:30:00Z` 和 `2024-03-15T10:30:00.000Z` 两种格式。
- 🆕 `LLMBenchmarkRunner` 全面升级 — `quickBenchmark()` 和主 `benchmark()` 方法均改用 `measureFullStreamingMetrics()` 获取真实 TTFT，彻底移除所有估算逻辑。


