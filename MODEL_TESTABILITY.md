# 模型可测试性指南

> 本文档记录每个内置模型的可测试性和已知问题。
> 更新日期：2026-04-25
>
> 测试环境：Android AVD（内存 1-4GB），hf-mirror.com 镜像

---

## 1. Embedding 模型 ✅ 可测试（部分）

### ✅ 完全可用

| 模型 | 大小 | AVD 可测 | 备注 |
|------|------|----------|------|
| `all-MiniLM-L6-v2` | 91MB | ✅ 可测 | 最小最快，英文 |
| `all-MiniLM-L12-v2` | 133MB | ✅ 可测 | 英文，精度更高 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 471MB | ⚠️ 可测 | 中英双语，内存消耗大 |
| `bge-small-en-v1.5` | 134MB | ✅ 可测 | 英文，512 上下文 |
| `bge-base-en-v1.5` | 439MB | ⚠️ 可测 | 英文，内存消耗较大 |
| `multilingual-e5-small` | 219MB | ✅ 可测 | 多语言，94种语言 |
| `nomic-embed-text-v1.5` | 548MB | ⚠️ 可测 | 英文长文本，8192 上下文 |

### ⚠️ 可用但无 ONNX（使用 PyTorch）

| 模型 | 大小 | AVD 可测 | 备注 |
|------|------|----------|------|
| `bge-base-zh-v1.5` | 410MB | ⚠️ 可测 | 中文优先，无 ONNX 导出，使用 PyTorch bin |

### ❌ 暂不可用

| 模型 | 原因 |
|------|------|
| `bge-large-zh-v1.5` | 暂无 ONNX 导出（PyTorch 版本 1.3GB，AVD 内存不够） |

### ⚠️ Embedding 已知问题

1. **`bge-base-zh-v1.5` 使用 PyTorch bin**：`modelFileExtension = ".bin"`，
   需 ONNX Runtime 的 PyTorch 后端支持（`ai.onnxruntime.cxx.Dll_loader`）。
   加载时需额外处理 PyTorch → ONNX Runtime 的兼容性问题。

2. **`multilingual-e5-small` 使用 SentencePiece vocab**：需要下载
   `sentencepiece.bpe.model`（5.07MB）而非 `vocab.txt`，tokenizer 逻辑需相应调整。

3. **`nomic-embed-text-v1.5` 需要长上下文**：8192 上下文长度，
   对 AVD 内存压力较大，建议在实体设备上测试。

---

## 2. 图像分类模型 ⚠️ 大部分待验证

### ✅ 已验证可用

| 模型 | 大小 | AVD 可测 | 备注 |
|------|------|----------|------|
| `resnet18` | 47MB | ✅ 可测 | Xenova/resnet-18 ONNX 导出（确认存在） |
| `joytag` | 366MB | ⚠️ 可测 | 图像标签，有 `top_tags.txt` |
| `vit_base_nsfw` | 344MB | ⚠️ 可测 | NSFW 检测，多种量化版本 |

### ⚠️ 待验证路径（可能 404）

以下模型在 `hf-mirror.com/onnx-community/` 下的路径**已失效**，
需要手动确认正确路径后才能使用：

| 模型名 | 原路径 | 状态 |
|--------|--------|------|
| `mobilenet_v2_224` | `onnx-community/mobilenet_v2_100` | ❌ 404 |
| `mobilenet_v3_small` | `onnx-community/mobilenetv3_small-100` | ❌ 404 |
| `mobilenet_v3_large` | `onnx-community/mobilenetv3_large-100` | ❌ 404 |
| `efficientnet_b0` | `onnx-community/timm_efficientnet_b0_ns_1k_32px` | ❌ 404 |
| `efficientnet_b1` | `onnx-community/timm_efficientnet_b1_ns` | ❌ 404 |
| `efficientnet_b2` | `onnx-community/timm_efficientnet_b2_ns` | ❌ 404 |
| `squeezenet1_1` | `onnx-community/squeezenet1.1` | ❌ 404 |
| `shufflenet_v2` | `onnx-community/shufflenet_v2_x1.0` | ❌ 404 |
| `resnet50` | `onnx-community/resnet50` | ❌ 404 |
| `vit_small_patch16_224` | `onnx-community/vit-small-patch16-224` | ❌ 404 |
| `vit_base_patch16_224` | `onnx-community/vit-base-patch16-224` | ❌ 404 |
| `crnn_mobilenet_v3` | `TheMuppets/CRNN_ResNet18` | ❌ 404 |
| `crnn_vgg16` | `TheMuppets/crnn_vgg16` | ❌ 404 |

### ✅ OCR 已验证

| 模型 | 大小 | AVD 可测 | 备注 |
|------|------|----------|------|
| `chinese_ocr_db_crnn` | 100MB | ⚠️ 可测 | PaddleOCR 中文识别（路径已验证） |

### ❌ VLM / 图像生成暂不可用

| 模型 | 状态 |
|------|------|
| `mobilevlm_v2_1.7b` | ⚠️ 待验证（TheMuppets 路径 404） |
| `sd_turbo_onnx` | ⚠️ 待验证（ONNX 导出路径可能不存在） |
| `sdxl_turbo_onnx` | ⚠️ 待验证（同上） |

### ⚠️ 图像模型已知问题

1. **onnx-community 组织大部分路径失效**：2025-2026 年间 HuggingFace
   的 `onnx-community` 组织模型大量迁移或删除。建议在 hf-mirror.com
   手动搜索 `timm/mobilenet*` 或 `apple/coreml*` 找替代导出。

2. **labels.txt 缺失**：大部分 ONNX 导出模型没有 `labels.txt`，
   需要使用内置 ImageNet 标签文件或从 `preprocessor_config.json` 解析。

3. **joytag 模型**：366MB，AVD 内存压力大，建议在 4GB+ 设备上测试。

---

## 3. 语音模型 ❌ 全部待验证

### ❌ 所有语音模型均不可用

| 模型 | 原路径 | 状态 | 原因 |
|------|--------|------|------|
| `whisper-tiny` | `openai/whisper-tiny/model.onnx` | ⚠️ 架构不兼容 | 官方 Whisper 无单文件 ONNX（encoder/decoder 分离） |
| `whisper-base` | `openai/whisper-base/model.onnx` | ⚠️ 同上 | 同上 |
| `whisper-small` | `openai/whisper-small/model.onnx` | ⚠️ 同上 | 同上 |
| `paraformer-small` | `alibaba-damo/paraformer-small` | ❌ 404 | 组织名已变更（可能改为 FunAudioLLM） |
| `paraformer-large` | `alibaba-damo/paraformer-large` | ❌ 404 | 同上 |
| `cam-plus-vad` | `alibaba-damo/cam-plus-vad` | ❌ 404 | 同上 |
| `vits-small-zh` | `vits-models/vits-small-zh` | ❌ 401 | 需要认证或组织不存在 |

### ⚠️ Whisper 架构问题详解

当前代码 `transcribeWithWhisper()` 期望一个单文件 `model.onnx`，
内部处理 mel spectrogram + encoder + decoder + token decoding。

但 `openai/whisper-tiny` 的 ONNX 导出是**分离架构**：
```
Xenova/whisper-tiny/onnx/
├── encoder_model.onnx          # 32.9 MB — mel spectrogram → encoder hidden states
├── decoder_model.onnx          # 118 MB — decoder + LM head
├── decoder_model_fp16.onnx     # 59.3 MB — 半精度版
├── decoder_model_merged.onnx   # 119 MB — 合并版
└── ...
```

要支持 Whisper，需要重构 `LocalSpeechService`：
1. 分别下载 `encoder_model.onnx` 和 `decoder_model.onnx`
2. `transcribe()` 先调 encoder，再调 decoder，最后解码
3. 或者使用 `decoder_model_merged.onnx` 一次性处理

---

## 4. LLM 模型 ✅ 全部可用

所有 TheBloke GGUF 模型均使用 `hf-mirror.com/TheBloke/` 路径，
GGUF 格式在 hf-mirror 上维护良好，**暂未发现问题**。

### AVD 可测试性（按内存要求排序）

| 模型 | 大小 | 最低内存 | AVD 可测 |
|------|------|----------|----------|
| Qwen2.5-0.5B-Instruct-Q4_0 | 394MB | 1GB | ✅ 可测 |
| Phi-2-Q4_0 | 494MB | 1GB | ✅ 可测 |
| SmolLM2-1.7B-Instruct-Q4_0 | 1GB | 1GB | ⚠️ 可测 |
| Qwen2-0.5B-Instruct-Q4_0 | 420MB | 2GB | ⚠️ 可测 |
| TinyLlama-1.1B-Chat-Q4_K_M | 667MB | 2GB | ⚠️ 可测 |
| Gemma-2B-Q4_K_M | 1.6GB | 2GB | ⚠️ 可测 |
| StableLM-3B-Q4_K_M | 1.9GB | 2GB | ⚠️ 可测 |
| gemma-4-e2b-q4_0 | 2.2GB | 3GB | ⚠️ 高配 AVD |
| gemma-4-e4b-q4_0 | 4.5GB | 5GB | ❌ AVD 内存不够 |
| gemma-4-e4b-q5_k_m | 5.2GB | 6GB | ❌ AVD 内存不够 |

---

## 5. 模型目录 API（ModelCatalogService）

### ✅ HF 镜像已就绪

`ModelCatalogService` 已更新为：
- **优先使用** `https://hf-mirror.com/api`（国内镜像）
- **回退官方** `https://huggingface.co/api`（镜像不可用时）
- 探测超时：**镜像 5 秒**，**官方 8 秒**
- `downloadUrl` 随 API 源动态生成（镜像源 → hf-mirror.com，官方源 → huggingface.co）

### 📋 验证步骤

1. 启动 App，进入「模型目录」页面
2. 点击「🔄 刷新」
3. 观察 Logcat 过滤 `ModelCatalog`：
   - ✅ 镜像可用：`HF 镜像可用: https://hf-mirror.com/api`
   - ⚠️ 镜像超时：`HF 镜像不可用: connection timeout`
   - ✅ 回退成功：`使用官方 HF API: https://huggingface.co/api`
   - ❌ 全部失败：`HF Hub (镜像 + 官方) 全部不可用，跳过 HF 数据源`

---

## 6. 后续行动计划

### ✅ P0 — 已完成

- [x] **2026-04-25** 确认 `onnx-community` 图像模型的正确路径
  - ✅ 已验证：`onnxmodelzoo/mobilenetv2-12` → `mobilenetv2-12.onnx` (14MB)
  - ✅ 代码已更新：`LocalImageModelService.kt` 路径已修复

- [x] **2026-04-25** 搜索 `FunAudioLLM` Paraformer 模型
  - ❌ FunAudioLLM 组织 401 认证失败
  - ⚠️ Paraformer ONNX 暂无可靠来源，保持标记待验证

- [x] **2026-04-25** 修复 Whisper 架构（encoder/decoder 分离）
  - ✅ 发现 Xenova/whisper-tiny ONNX 导出可用
  - ✅ 发现 Distil-Whisper 系列 ONNX 导出
  - ✅ 代码已重构：`LocalSpeechService.kt` 支持 encoder/decoder 分离

### ✅ P0 发现总结（2026-04-25）

#### 图像模型新发现
| 模型 | 新路径 | 状态 |
|------|--------|------|
| `mobilenet_v2_224` | `onnxmodelzoo/mobilenetv2-12/mobilenetv2-12.onnx` | ✅ 已验证 |
| `efficientnet_lite4` | `onnxmodelzoo/efficientnet-lite4-11/` | ✅ 有 ONNX |

#### 语音模型新发现
| 模型 | 路径 | 状态 |
|------|------|------|
| `whisper-tiny` | `Xenova/whisper-tiny/onnx/encoder_model.onnx` + `decoder_model_merged.onnx` | ✅ 已验证架构 |
| `whisper-tiny-q4f16` | 同上，使用 q4f16 量化版本 | ✅ 推荐移动端 |
| `whisper-base/small` | `Xenova/whisper-*/onnx/` | ✅ 同架构 |
| `distil-whisper-large-v3` | `distil-whisper/distil-large-v3/` | ✅ 推荐精度 |

### P1 — 有空再补
- [ ] 找 `vits-small-zh` 替代模型（BricksDisplay/vits-cmn？）
- [ ] 确认 PaddleOCR 之外的其他 OCR 模型
- [ ] 补充 `bge-large-zh-v1.5` ONNX 导出（关注 BAAI 仓库 discussions）
- [ ] 验证 `multilingual-e5-small` SentencePiece tokenizer 兼容性
- [ ] 验证 `onnxmodelzoo/efficientnet-lite4-*` 文件路径
