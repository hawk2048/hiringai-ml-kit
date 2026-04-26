# hiringai-ml-kit 基准测试报告

**测试日期**：2026-04-26
**测试设备**：AVD Emulator (sdk_gphone16k_x86_64)
**APK 版本**：1.0.0 (build 1)

---

## 一、测试执行过程

### 1.1 UI 流程验证
| 步骤 | 操作 | 结果 |
|------|------|------|
| 启动 App | monkey 触发 LAUNCHER intent | ✅ 成功 |
| 主页显示 | 设备信息 + 存储状态 + 功能按钮 | ✅ 正常 |
| 点击"基准测试" | 进入 BenchmarkActivity | ✅ 正常 |
| 模型列表 | 显示 10 个 LLM + 8 个嵌入模型 | ✅ 正常 |
| 选中 Qwen2.5-0.5B | 点击 CheckBox | ✅ checked="true" |
| 点击"单项测试" | 启动基准测试 | ✅ 触发，5秒完成 |

### 1.2 设备环境信息
```
SoC        : Google_ranchu
CPU        : 4核 x86_64 (大核 0MHz · 小核 0MHz)
RAM        : 2971MB (可用 1439MB)
GPU        : Google sdk_gphone16k_x86_64
NPU        : ✗ 未检测到
Vulkan     : 支持
性能评分    : 93→98/100
```

### 1.3 可测试模型列表
**LLM 模型（10 个）**
| 模型 | 大小 | 建议 |
|------|------|------|
| Qwen2.5-0.5B-Instruct-Q4_0 | 395 MB | 🔴 最小，推荐先测 |
| Phi-2-Q4_0 | 494 MB | 🔴 次小 |
| Qwen2-0.5B-Instruct-Q4_0 | 420 MB | 🔴 |
| SmolLM2-1.7B-Instruct-Q4_0 | 1.0 GB | 🟡 |
| TinyLlama-1.1B-Chat-Q4_K_M | 668 MB | 🟡 |
| Gemma-2B-Q4_K_M | 1.6 GB | 🟡 |
| StableLM-3B-Q4_K_M | 1.9 GB | 🟠 |
| gemma-4-e2b-q4_0 | 2.2 GB | 🟠 |
| gemma-4-e4b-q4_0 | 4.5 GB | 🔴 最大 |
| gemma-4-e4b-q5_k_m | 5.2 GB | 🔴 |

**嵌入模型（8 个）**
| 模型 | 大小 |
|------|------|
| all-MiniLM-L6-v2 | 91 MB |
| bge-small-en-v1.5 | 134 MB |
| all-MiniLM-L12-v2 | 133 MB |
| multilingual-e5-small | 219 MB |
| bge-base-zh-v1.5 | 410 MB |
| bge-base-en-v1.5 | 439 MB |
| Phi-2-Q4_0 | 494 MB |
| nomic-embed-text-v1.5 | 548 MB |

---

## 二、代码逻辑验证

### 2.1 基准测试是真实推理（非 Mock）
通过代码审查确认：
- ✅ `benchmarkModel()` → `llmService.generate()` 调用真实 llama.cpp 推理
- ✅ 无随机数、无假数据、Token 计数基于实际输出文本
- ✅ 吞吐量 = 实际生成 token 数 / 实际耗时
- ✅ 4 阶段流程：CHECK_DOWNLOAD → LOADING → GENERATING → UNLOADING

### 2.2 测试 Prompt
```kotlin
"请用一句话介绍你自己"  // TEST_PROMPTS.first()
```

### 2.3 模型文件存储路径
```
/data/data/com.hiringai.mobile.ml.testapp/files/models/{modelName}.gguf
```

---

## 三、发现的阻塞问题

### ❌ 问题 1：JNI 库架构不匹配（严重）
**现象**：`llamaService.loadModel()` 失败，`SafeNativeLoader.loadLibrary("llama-android")` 返回 false

**根本原因**：
- APK 中的 `lib/` 目录只有 `arm64-v8a` 架构
- 模拟器是 `x86_64` 架构
- `llama-kotlin-android:0.1.3` AAR 未正确打包 native `.so` 库

**APK lib/ 目录内容**：
```
lib/arm64-v8a/libc++_shared.so
lib/arm64-v8a/libllama-android.so
lib/arm64-v8a/libonnxruntime.so
lib/arm64-v8a/libonnxruntime4j_jni.so
```

**影响**：模拟器无法运行任何 llama.cpp 或 ONNX Runtime 本地推理

### ❌ 问题 2：模型文件缺失
**现象**：`isModelDownloaded()` 返回 false → `benchmarkModel()` 直接返回失败

**状态**：`/data/data/.../files/models/` 目录为空

**原因**：基准测试模块不负责下载，只负责检查和运行

### ❌ 问题 3：物理设备离线
**设备**：小米 2201122C
**预期地址**：`192.168.1.5:37913`
**实际状态**：连接被拒绝 (10061)

---

## 四、结论

| 测试项 | 状态 | 说明 |
|--------|------|------|
| APK 编译 | ✅ 通过 | BUILD SUCCESSFUL |
| APK 安装 | ✅ 通过 | 成功推送到 emulator-5554 |
| App UI | ✅ 通过 | MainActivity / BenchmarkActivity 均正常 |
| 模型选择 | ✅ 通过 | CheckBox 交互正常 |
| 基准测试代码 | ✅ 真实 | llama.cpp 推理，无 mock |
| **实际推理运行** | ❌ 阻塞 | JNI 架构不匹配 + 无模型文件 |

**核心结论**：基准测试框架代码是真实可用的，但需要在 **物理设备（ARM）** 上运行才能执行真正的模型推理。当前模拟器环境因架构不兼容无法加载 native 库。

---

## 五、下一步行动

### 优先级 1：物理设备准备
1. 开启小米 2201122C 的 USB 调试或无线调试
2. 重新连接 `adb connect 192.168.1.5:37913`
3. 确认设备显示在 `adb devices` 中

### 优先级 2：模型下载
1. 在物理设备上打开 App
2. 找到模型下载入口（需确认是否有内置下载功能）
3. 或通过 ADB push 从电脑推送 `.gguf` 文件
   ```bash
   adb push qwen2.5-0.5b-instruct-q4_0.gguf /data/data/com.hiringai.mobile.ml.testapp/files/models/
   ```

### 优先级 3：推荐测试顺序
1. **all-MiniLM-L6-v2** (91 MB) — 最小，ONNX 模型，最快出结果
2. **Qwen2.5-0.5B** (395 MB) — 最小 LLM，验证 llama.cpp 推理
3. **Qwen2-0.5B** (420 MB) — 对比 Qwen2.5 vs Qwen2
4. **图像模型** (mobilenet_v2_224) — ONNX Runtime 验证

---

## 六、相关文件

- 基准测试代码：`ml/src/main/java/com/hiringai/mobile/ml/benchmark/LLMBenchmarkRunner.kt`
- UI 代码：`app/src/main/java/com/hiringai/mobile/ml/testapp/ui/BenchmarkActivity.kt`
- 本次截图：`d:\AI\AIModel\test-screenshots/`
- APK 文件：`d:\AI\AIModel\test-screenshots/testapp.apk` (38.9 MB)
