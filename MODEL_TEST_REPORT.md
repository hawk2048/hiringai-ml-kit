# HiringAI-ML-Kit 模型测试报告

> 生成时间: 2026-04-26
> 应用版本: 1.0.0
> 测试设备: Pixel_9_Pro_Fold (Android 模拟器, SDK 17)

---

## 📊 模型总览

| 类别 | 模型数量 | 状态 |
|------|----------|------|
| 🤖 LLM 大语言模型 | 14 | 待测试 |
| 📐 Embedding 嵌入模型 | 8 | 待测试 |
| 🖼️ Image 图像模型 | 15+ | 待测试 |
| 🎤 Speech 语音模型 | 6 | 待测试 |

---

## 🤖 LLM 大语言模型

| 模型名称 | 大小 | 内存需求 | 上下文 | 描述 |
|----------|------|----------|--------|------|
| Qwen2.5-0.5B-Instruct-Q4_0 | 377MB | 1GB | 2048 | 超轻量级，中文优化，推荐入门 |
| Phi-2-Q4_0 | 471MB | 1GB | 2048 | 微软 Phi-2，极小体积，英文推理优秀 |
| SmolLM2-1.7B-Instruct-Q4_0 | 954MB | 1GB | 2048 | HuggingFace SmolLM2-1.7B，性能均衡 |
| Qwen2-0.5B-Instruct-Q4_0 | 401MB | 2GB | 2048 | Qwen2 基础版，中文能力出色 |
| TinyLlama-1.1B-Chat-Q4_K_M | 637MB | 2GB | 2048 | TinyLlama 1.1B，生态丰富 |
| Gemma-2B-Q4_K_M | 1.5GB | 2GB | 4096 | Google Gemma-2B，指令遵循强 |
| StableLM-3B-Q4_K_M | 1.8GB | 2GB | 4096 | Stability AI StableLM-3B，长上下文 |
| gemma-4-e2b-q4_0 | 2.1GB | 3GB | 8192 | Google Gemma 4 系列 |
| Qwen2.5-1.5B-Instruct-Q4_0 | 1.0GB | 2GB | 2048 | Qwen2.5 1.5B，中文优化 |
| Qwen2.5-3B-Instruct-Q4_0 | 1.9GB | 4GB | 2048 | Qwen2.5 3B，中文优化 |
| Llama-3.2-1B-Instruct-Q4_0 | 0.7GB | 2GB | 4096 | Meta Llama 3.2 1B |
| Llama-3.2-3B-Instruct-Q4_0 | 1.9GB | 4GB | 4096 | Meta Llama 3.2 3B |
| Phi-3.5-mini-Q4_0 | 2.3GB | 4GB | 4096 | 微软 Phi-3.5 mini |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.0GB | 2GB | 2048 | DeepSeek 蒸馏模型 |

**存储格式**: GGUF (llama.cpp)
**下载源**: hf-mirror.com

---

## 📐 Embedding 嵌入模型

| 模型名称 | 大小 | 维度 | 最大长度 | 描述 |
|----------|------|------|----------|------|
| all-MiniLM-L6-v2 | 87MB | 384 | 256 | 轻量级英文语义匹配，6层Transformer |
| all-MiniLM-L12-v2 | 127MB | 384 | 256 | 英文语义匹配，12层Transformer |
| paraphrase-multilingual-MiniLM-L12-v2 | 449MB | 768 | 128 | 中英双语，50+语言支持 |
| bge-small-en-v1.5 | 128MB | 384 | 512 | BGE 英文小模型，高召回率 |
| bge-base-en-v1.5 | 419MB | 768 | 512 | BGE 英文基础模型，高精度 |
| bge-base-zh-v1.5 | 391MB | 768 | 512 | ⚠️ BGE 中文基础（PyTorch格式） |
| multilingual-e5-small | 209MB | 384 | 512 | 多语言E5，支持94种语言 |
| nomic-embed-text-v1.5 | 523MB | 768 | 8192 | 开源长文本，支持8192上下文 |

**存储格式**: ONNX
**下载源**: hf-mirror.com

---

## 🖼️ Image 图像模型

### ✅ 已验证可用

| 模型名称 | 大小 | 内存 | 输入尺寸 | 描述 |
|----------|------|------|----------|------|
| resnet18 | 45MB | 2GB | 224x224 | 经典残差网络分类 |
| joytag | 349MB | 4GB | 224x224 | 图像标签，支持丰富标签 |
| vit_base_nsfw | 328MB | 4GB | 224x224 | ViT NSFW 检测器 |
| mobilenet_v2_224 | 13MB | 1GB | 224x224 | 轻量级图像分类 |

### ⚠️ 待验证

| 模型名称 | 大小 | 内存 | 状态 |
|----------|------|------|------|
| mobilenet_v3_small | 10MB | 1GB | ⚠️ 路径可能失效 |
| mobilenet_v3_large | 20MB | 1GB | ⚠️ 路径可能失效 |
| efficientnet_b0 | 19MB | 2GB | ⚠️ 路径可能失效 |
| efficientnet_b1 | 29MB | 2GB | ⚠️ 路径可能失效 |
| efficientnet_lite4 | ~50MB | 4GB | ⚠️ 待验证 |

**存储格式**: ONNX
**下载源**: hf-mirror.com

---

## 🎤 Speech 语音模型

| 模型名称 | 大小 | 内存 | 采样率 | 描述 |
|----------|------|------|--------|------|
| whisper-tiny | 113MB | 1GB | 16kHz | OpenAI Whisper Tiny |
| whisper-base | 290MB | 2GB | 16kHz | OpenAI Whisper Base |
| whisper-small | 967MB | 4GB | 16kHz | OpenAI Whisper Small |
| whisper-medium | 2.9GB | 6GB | 16kHz | OpenAI Whisper Medium |
| whisper-large-v3 | 7.8GB | 8GB | 16kHz | OpenAI Whisper Large v3 |
| distil-whisper-large-v3 | 3.2GB | 6GB | 16kHz | Distil-Whisper 推荐精度 |

**存储格式**: ONNX (encoder + decoder 分离)
**下载源**: Xenova/whisper-* on hf-mirror.com

---

## 📈 测试计划

### Phase 1: 基础功能测试
- [ ] 模型下载功能
- [ ] 模型加载功能
- [ ] 模型推理延迟

### Phase 2: 性能基准测试
- [ ] LLM 首 token 延迟 (TTFT)
- [ ] LLM 推理吞吐量 (tokens/s)
- [ ] Embedding 推理延迟
- [ ] 图像分类推理延迟
- [ ] 语音识别延迟

### Phase 3: 设备兼容性测试
- [ ] 不同 Android 版本测试
- [ ] 不同设备性能对比
- [ ] 内存占用监控

---

## 🔧 测试命令

```bash
# 启动模拟器
emulator -avd Pixel_9_Pro_Fold

# 安装APK
adb install -r app-debug.apk

# 运行自动化测试
python auto_test.py --reinstall

# 获取模型列表
adb shell pm list packages | grep hiring

# 查看应用日志
adb logcat -s ML
```

---

## 📝 备注

1. **模拟器限制**: x86_64 架构，部分原生库可能无法运行
2. **真机测试**: 推荐使用 ARM 设备进行完整测试
3. **模型下载**: 需要网络连接，hf-mirror.com 国内访问
4. **内存需求**: LLM 模型需要较大 RAM，建议 6GB+ 设备
