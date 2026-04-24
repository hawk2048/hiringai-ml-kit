# HiringAI ML Kit 🧠

Android 端侧 ML 推理工具包——从 [HRAutomation Android](https://github.com/user/hr-automation-android) 项目中剥离的独立 ML 子模块。

## 🎯 功能概览

| 模块 | 能力 | 引擎 |
|------|------|------|
| **LLM 推理** | 本地 GGUF 模型加载与生成，远程 Ollama 回退 | llama.cpp (JNI) |
| **文本嵌入** | ONNX Runtime 向量嵌入，余弦相似度计算 | ONNX Runtime |
| **图像模型** | 分类、OCR、VLM 视觉编码 | ONNX Runtime |
| **语音模型** | ASR 语音识别、说话人识别 | ONNX Runtime |
| **硬件加速** | GPU/NNAPI/XNNPACK 自动回退链 | Android NNAPI |
| **基准测试** | LLM/语音/图像/加速后端全面性能基准测试 | 内置工具集 |
| **设备检测** | CPU/GPU/NPU 能力检测与模型推荐 | 系统属性 +  EGL |

## 📦 作为子模块集成

```bash
# 在主项目中添加 submodule
git submodule add https://github.com/user/hiringai-ml-kit.git hiringai-ml-kit

# settings.gradle 中引入
include ':hiringai-ml-kit:ml'

# app/build.gradle 中添加依赖
implementation project(':hiringai-ml-kit:ml')
```

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
│   ├── BenchmarkResult.kt
│   ├── LLMBenchmarkRunner.kt
│   ├── SpeechBenchmark.kt
│   ├── SpeechModelBenchmark.kt
│   ├── TestAudioGenerator.kt
│   ├── TestAudioResources.kt
│   ├── TestImageGenerator.kt
│   ├── TestImageResources.kt
│   └── datasets/
│       ├── DatasetManager.kt
│       └── MiniImageNetLoader.kt
├── speech/              # 语音模型
│   ├── LocalSpeechService.kt
│   └── SpeechRecognitionService.kt
├── LocalLLMService.kt
├── LocalEmbeddingService.kt
├── LocalImageModelService.kt
├── SpeechModelService.kt
├── ModelManager.kt
├── DeviceCapabilityDetector.kt
└── SafeNativeLoader.kt  # 独立版本，不依赖 Application 单例
```

## 🔌 依赖说明

| 依赖 | 用途 |
|------|------|
| `onnxruntime-android:1.24.3` | 嵌入/图像/语音 ONNX 推理 |
| `llama-kotlin-android:0.1.3` | GGUF 模型本地推理 |
| `okhttp3:4.12.0` | Ollama 远程推理 HTTP 调用 |
| `kotlinx-serialization-json` | 加速配置序列化 |

## ⚡ 基准测试

```kotlin
// LLM 基准测试
val runner = LLMBenchmarkRunner(context)
val result = runner.benchmarkModel(config)

// 加速后端基准测试
val bench = AccelerationBenchmark(context)
val report = bench.runFullBenchmark()

// 语音模型基准测试
val speechBench = SpeechModelBenchmark(context)
val speechReport = speechBench.runBatchBenchmark()
```

## 📋 支持的模型

### LLM (GGUF)
- Qwen2.5-0.5B, Phi-2, SmolLM2-1.7B, TinyLlama-1.1B, Gemma-2B, StableLM-3B, Gemma 4 系列

### 嵌入 (ONNX)
- all-MiniLM-L6-v2, all-MiniLM-L12-v2, bge-base-zh-v1.5, bge-large-zh-v1.5

### 图像 (ONNX)
- MobileNet V2/V3, EfficientNet B0-B2, ResNet-18/50, ViT, OCR (CRNN), CLIP

### 语音 (ONNX)
- Whisper Tiny/Base/Small, Paraformer 中文/多语言, Cam++ 说话人识别

## 📄 License

MIT
