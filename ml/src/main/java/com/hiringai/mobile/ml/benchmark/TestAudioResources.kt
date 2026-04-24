package com.hiringai.mobile.ml.benchmark

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sin

/**
 * 测试音频资源管理器
 *
 * 提供用于语音识别基准测试的标准化测试音频:
 * - 中文语音样本
 * - 英文语音样本
 * - 多语言混合样本
 * - 静音/噪声样本
 */
object TestAudioResources {

    private const val TAG = "TestAudioResources"

    /**
     * 测试音频类型
     */
    enum class TestAudioType {
        CHINESE_SPEECH,     // 中文语音
        ENGLISH_SPEECH,     // 英文语音
        MIXED_SPEECH,       // 混合语言
        SILENCE,            // 静音
        NOISE,              // 噪声
        NUMBERS,            // 数字序列
        COMMANDS            // 命令词
    }

    /**
     * 测试音频配置
     */
    data class TestAudioConfig(
        val name: String,
        val type: TestAudioType,
        val durationSec: Float,
        val sampleRate: Int = 16000,
        val channels: Int = 1,
        val expectedText: String,
        val description: String
    )

    /**
     * WAV文件头信息
     */
    data class WavHeader(
        val sampleRate: Int,
        val channels: Int,
        val bitsPerSample: Int = 16,
        val dataSize: Long
    )

    /**
     * 预定义的测试音频配置列表
     */
    val TEST_AUDIOS = listOf(
        // 中文语音测试
        TestAudioConfig(
            name = "chinese_greeting",
            type = TestAudioType.CHINESE_SPEECH,
            durationSec = 3.0f,
            expectedText = "你好欢迎使用智能招聘系统",
            description = "中文问候语 - 测试基本中文识别"
        ),
        TestAudioConfig(
            name = "chinese_name",
            type = TestAudioType.CHINESE_SPEECH,
            durationSec = 2.5f,
            expectedText = "我叫张三应聘软件工程师职位",
            description = "姓名和职位 - 模拟面试场景"
        ),
        TestAudioConfig(
            name = "chinese_address",
            type = TestAudioType.CHINESE_SPEECH,
            durationSec = 4.0f,
            expectedText = "我的地址是北京市朝阳区建国路一百号",
            description = "地址信息 - 测试数字和地名识别"
        ),
        TestAudioConfig(
            name = "chinese_phone",
            type = TestAudioType.CHINESE_SPEECH,
            durationSec = 3.0f,
            expectedText = "电话号码是13812345678",
            description = "电话号码 - 测试数字序列"
        ),
        TestAudioConfig(
            name = "chinese_experience",
            type = TestAudioType.CHINESE_SPEECH,
            durationSec = 5.0f,
            expectedText = "我有五年Android开发经验熟练掌握Kotlin和Java",
            description = "工作经验 - 模拟简历朗读"
        ),

        // 英文语音测试
        TestAudioConfig(
            name = "english_greeting",
            type = TestAudioType.ENGLISH_SPEECH,
            durationSec = 3.0f,
            expectedText = "Hello welcome to the hiring system",
            description = "英文问候语 - 测试基本英文识别"
        ),
        TestAudioConfig(
            name = "english_name",
            type = TestAudioType.ENGLISH_SPEECH,
            durationSec = 2.5f,
            expectedText = "My name is John Smith I am applying for the software engineer position",
            description = "英文姓名和职位"
        ),
        TestAudioConfig(
            name = "english_skills",
            type = TestAudioType.ENGLISH_SPEECH,
            durationSec = 4.0f,
            expectedText = "I have experience with Python Java and machine learning",
            description = "技能描述 - 测试技术词汇"
        ),
        TestAudioConfig(
            name = "english_numbers",
            type = TestAudioType.ENGLISH_SPEECH,
            durationSec = 3.0f,
            expectedText = "My phone number is 555 123 4567",
            description = "英文数字序列"
        ),

        // 混合语言测试
        TestAudioConfig(
            name = "mixed_intro",
            type = TestAudioType.MIXED_SPEECH,
            durationSec = 4.0f,
            expectedText = "你好 my name is 李明 I have 5年开发经验",
            description = "中英混合 - 测试语言切换"
        ),
        TestAudioConfig(
            name = "mixed_tech",
            type = TestAudioType.MIXED_SPEECH,
            durationSec = 5.0f,
            expectedText = "熟练使用Android SDK Jetpack组件和Kotlin Coroutines",
            description = "技术术语混合"
        ),

        // 数字序列测试
        TestAudioConfig(
            name = "numbers_sequence",
            type = TestAudioType.NUMBERS,
            durationSec = 4.0f,
            expectedText = "一二三四五六七八九十",
            description = "中文数字1-10"
        ),
        TestAudioConfig(
            name = "numbers_phone",
            type = TestAudioType.NUMBERS,
            durationSec = 3.0f,
            expectedText = "138 1234 5678",
            description = "电话号码格式"
        ),
        TestAudioConfig(
            name = "numbers_id",
            type = TestAudioType.NUMBERS,
            durationSec = 5.0f,
            expectedText = "110105199001011234",
            description = "身份证号码"
        ),

        // 命令词测试
        TestAudioConfig(
            name = "command_start",
            type = TestAudioType.COMMANDS,
            durationSec = 1.5f,
            expectedText = "开始录音",
            description = "开始命令"
        ),
        TestAudioConfig(
            name = "command_stop",
            type = TestAudioType.COMMANDS,
            durationSec = 1.5f,
            expectedText = "停止",
            description = "停止命令"
        ),
        TestAudioConfig(
            name = "command_next",
            type = TestAudioType.COMMANDS,
            durationSec = 1.5f,
            expectedText = "下一题",
            description = "导航命令"
        ),

        // 特殊测试样本
        TestAudioConfig(
            name = "silence_3s",
            type = TestAudioType.SILENCE,
            durationSec = 3.0f,
            expectedText = "",
            description = "3秒静音 - 测试VAD"
        ),
        TestAudioConfig(
            name = "noise_3s",
            type = TestAudioType.NOISE,
            durationSec = 3.0f,
            expectedText = "",
            description = "3秒白噪声 - 测试降噪能力"
        )
    )

    /**
     * 获取测试音频目录
     */
    fun getTestAudiosDir(context: Context): File {
        val dir = File(context.filesDir, "test_audios")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * 检查测试音频是否已存在
     */
    fun isTestAudioGenerated(context: Context, config: TestAudioConfig): Boolean {
        val file = File(getTestAudiosDir(context), "${config.name}.wav")
        return file.exists() && file.length() > 0
    }

    /**
     * 生成所有测试音频
     */
    suspend fun generateAllTestAudios(
        context: Context,
        onProgress: (Int, String) -> Unit
    ): Int = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
        var generated = 0
        val total = TEST_AUDIOS.size

        TEST_AUDIOS.forEachIndexed { index, config ->
            try {
                generateTestAudio(context, config)
                generated++
                val progress = ((index + 1) * 100 / total)
                onProgress(progress, "已生成: ${config.name}")
                Log.i(TAG, "Generated test audio: ${config.name}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate test audio: ${config.name}", e)
            }
        }

        generated
    }

    /**
     * 生成单个测试音频
     */
    fun generateTestAudio(context: Context, config: TestAudioConfig): File {
        val file = File(getTestAudiosDir(context), "${config.name}.wav")

        if (file.exists()) {
            return file
        }

        // 生成音频数据
        val audioData = generateAudioData(config)

        // 写入WAV文件
        writeWavFile(file, audioData, config)

        return file
    }

    /**
     * 生成音频数据 (模拟语音波形)
     */
    private fun generateAudioData(config: TestAudioConfig): ShortArray {
        val totalSamples = (config.sampleRate * config.durationSec).toInt()
        val audioData = ShortArray(totalSamples)

        when (config.type) {
            TestAudioType.SILENCE -> {
                // 静音数据
                // 默认case会处理，这里可以留空或者返回全0
            }
            TestAudioType.NOISE -> {
                // 白噪声
                for (i in audioData.indices) {
                    audioData[i] = ((Math.random() * 2 - 1) * 3000).toInt().toShort()
                }
            }
            else -> {
                // 模拟语音波形
                generateSimulatedSpeech(audioData, config)
            }
        }

        return audioData
    }

    /**
     * 生成模拟语音波形
     * 使用多个正弦波叠加模拟语音的基本特性
     */
    private fun generateSimulatedSpeech(audioData: ShortArray, config: TestAudioConfig) {
        val sampleRate = config.sampleRate
        val expectedText = config.expectedText

        // 基于文本长度估算音节数
        val syllableCount = estimateSyllableCount(expectedText, config.type)
        val samplesPerSyllable = audioData.size / maxOf(syllableCount, 1)

        var currentIndex = 0

        for (syllable in 0 until syllableCount) {
            val startSample = syllable * samplesPerSyllable
            val endSample = minOf((syllable + 1) * samplesPerSyllable, audioData.size)

            // 生成一个音节的波形
            generateSyllable(
                audioData,
                startSample,
                endSample,
                sampleRate,
                syllable,
                config.type
            )

            currentIndex = endSample
        }

        // 应用包络使音频更自然
        applyEnvelope(audioData)

        // 添加一些变化使音频更像真实语音
        addVoiceCharacteristics(audioData, sampleRate)
    }

    /**
     * 估算音节数
     */
    private fun estimateSyllableCount(text: String, type: TestAudioType): Int {
        return when (type) {
            TestAudioType.CHINESE_SPEECH -> {
                // 中文每个汉字大约一个音节
                text.filter { it.code > 127 }.length + text.filter { it.code <= 127 }.length / 3
            }
            TestAudioType.ENGLISH_SPEECH -> {
                // 英文单词数 * 平均音节数
                text.split(" ").size * 2
            }
            TestAudioType.MIXED_SPEECH -> {
                // 混合语言估算
                val chinese = text.filter { it.code > 127 }.length
                val english = text.filter { it.code <= 127 && !it.isWhitespace() }.length
                chinese + english / 3
            }
            TestAudioType.NUMBERS -> {
                // 数字逐个读
                text.filter { it.isDigit() }.length
            }
            TestAudioType.COMMANDS -> {
                // 命令词较短
                text.length / 2
            }
            else -> text.length / 3
        }
    }

    /**
     * 生成单个音节的波形
     */
    private fun generateSyllable(
        audioData: ShortArray,
        startSample: Int,
        endSample: Int,
        sampleRate: Int,
        syllableIndex: Int,
        type: TestAudioType
    ) {
        // 基频根据语言类型调整
        val baseFreq = when (type) {
            TestAudioType.CHINESE_SPEECH -> 150f + (syllableIndex % 4) * 20f // 中文音调变化
            TestAudioType.ENGLISH_SPEECH -> 120f + (syllableIndex % 3) * 15f
            TestAudioType.MIXED_SPEECH -> 140f + (syllableIndex % 5) * 10f
            TestAudioType.NUMBERS -> 160f + (syllableIndex % 2) * 30f
            TestAudioType.COMMANDS -> 180f
            else -> 150f
        }

        val duration = endSample - startSample

        for (i in startSample until endSample) {
            val t = (i - startSample).toFloat() / sampleRate
            val progress = (i - startSample).toFloat() / duration

            // 音节包络
            val envelope = when {
                progress < 0.1f -> progress * 10f // 起音
                progress > 0.8f -> (1f - progress) * 5f // 收音
                else -> 1f
            }

            // 多个谐波叠加
            var sample = 0.0

            // 基频
            sample += sin(2.0 * Math.PI * baseFreq * t) * 0.5

            // 二次谐波
            sample += sin(2.0 * Math.PI * baseFreq * 2 * t) * 0.25

            // 三次谐波
            sample += sin(2.0 * Math.PI * baseFreq * 3 * t) * 0.15

            // 四次谐波
            sample += sin(2.0 * Math.PI * baseFreq * 4 * t) * 0.1

            // 添加共振峰
            val formant1 = when (syllableIndex % 5) {
                0 -> 500f
                1 -> 700f
                2 -> 900f
                3 -> 600f
                else -> 800f
            }
            sample += sin(2.0 * Math.PI * formant1 * t) * 0.15

            // 应用包络和振幅
            val amplitude = 8000 * envelope
            audioData[i] = (sample * amplitude).toInt().toShort()
        }
    }

    /**
     * 应用整体包络
     */
    private fun applyEnvelope(audioData: ShortArray) {
        val fadeLength = minOf(audioData.size / 20, 160) // 10ms fade

        // 起始淡入
        for (i in 0 until fadeLength) {
            val factor = i.toFloat() / fadeLength
            audioData[i] = (audioData[i] * factor).toInt().toShort()
        }

        // 结束淡出
        for (i in audioData.size - fadeLength until audioData.size) {
            val factor = (audioData.size - i).toFloat() / fadeLength
            audioData[i] = (audioData[i] * factor).toInt().toShort()
        }
    }

    /**
     * 添加语音特性
     */
    private fun addVoiceCharacteristics(audioData: ShortArray, sampleRate: Int) {
        // 添加轻微的颤音效果
        val vibratoRate = 5f // Hz
        val vibratoDepth = 0.02f

        for (i in audioData.indices) {
            val t = i.toFloat() / sampleRate
            val vibrato = sin(2.0 * Math.PI * vibratoRate * t) * vibratoDepth
            audioData[i] = (audioData[i] * (1 + vibrato)).toInt().toShort()
        }

        // 添加轻微噪声
        for (i in audioData.indices) {
            val noise = (Math.random() - 0.5) * 200
            audioData[i] = (audioData[i] + noise).toInt().toShort()
        }
    }

    /**
     * 写入WAV文件
     */
    private fun writeWavFile(file: File, audioData: ShortArray, config: TestAudioConfig) {
        FileOutputStream(file).use { output ->
            // WAV头大小
            val headerSize = 44
            val dataSize = audioData.size * 2 // 16-bit samples
            val fileSize = headerSize + dataSize - 8

            // 写入WAV头
            val buffer = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)

            // RIFF头
            buffer.put("RIFF".toByteArray())
            buffer.putInt(fileSize)
            buffer.put("WAVE".toByteArray())

            // fmt子块
            buffer.put("fmt ".toByteArray())
            buffer.putInt(16) // fmt块大小
            buffer.putShort(1) // 音频格式 (1 = PCM)
            buffer.putShort(config.channels.toShort()) // 声道数
            buffer.putInt(config.sampleRate) // 采样率
            buffer.putInt(config.sampleRate * config.channels * 2) // 字节率
            buffer.putShort((config.channels * 2).toShort()) // 块对齐
            buffer.putShort(16) // 位深度

            // data子块
            buffer.put("data".toByteArray())
            buffer.putInt(dataSize)

            output.write(buffer.array())

            // 写入音频数据
            val dataBuffer = ByteBuffer.allocate(audioData.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in audioData) {
                dataBuffer.putShort(sample)
            }
            output.write(dataBuffer.array())
        }
    }

    /**
     * 读取WAV文件
     */
    fun readWavFile(file: File): Pair<ShortArray, Int>? {
        return try {
            RandomAccessFile(file, "r").use { raf ->
                // 读取WAV头
                val headerBuffer = ByteArray(44)
                raf.readFully(headerBuffer)

                val buffer = ByteBuffer.wrap(headerBuffer).order(ByteOrder.LITTLE_ENDIAN)

                // 验证RIFF头
                val riff = String(headerBuffer, 0, 4)
                if (riff != "RIFF") return null

                // 跳过文件大小
                buffer.position(8)

                // 验证WAVE
                val wave = String(headerBuffer, 8, 4)
                if (wave != "WAVE") return null

                // 读取fmt块
                buffer.position(12)
                val fmt = String(headerBuffer, 12, 4)
                if (fmt != "fmt ") return null

                buffer.position(22)
                val channels = buffer.short.toInt()
                val sampleRate = buffer.int

                // 查找data块
                raf.seek(12)
                var foundData = false
                var dataSize = 0L

                while (!foundData && raf.filePointer < raf.length() - 8) {
                    val chunkId = ByteArray(4)
                    raf.readFully(chunkId)
                    val chunkSize = raf.readInt()

                    if (String(chunkId) == "data") {
                        foundData = true
                        dataSize = chunkSize.toLong() and 0xFFFFFFFFL
                    } else {
                        raf.skipBytes(chunkSize)
                    }
                }

                if (!foundData) return null

                // 读取音频数据
                val sampleCount = (dataSize / 2).toInt()
                val audioData = ShortArray(sampleCount)
                val dataBytes = ByteArray(dataSize.toInt())
                raf.readFully(dataBytes)

                val dataBuffer = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
                for (i in audioData.indices) {
                    audioData[i] = dataBuffer.short
                }

                Pair(audioData, sampleRate)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read WAV file: ${file.absolutePath}", e)
            null
        }
    }

    /**
     * 加载测试音频
     */
    fun loadTestAudio(context: Context, config: TestAudioConfig): ShortArray? {
        return try {
            val file = File(getTestAudiosDir(context), "${config.name}.wav")
            if (!file.exists()) {
                generateTestAudio(context, config)
            }
            readWavFile(file)?.first
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load test audio: ${config.name}", e)
            null
        }
    }

    /**
     * 获取所有测试音频文件
     */
    fun getAllTestAudioFiles(context: Context): List<File> {
        val dir = getTestAudiosDir(context)
        return dir.listFiles()?.filter { it.extension == "wav" }?.toList() ?: emptyList()
    }

    /**
     * 清除所有测试音频
     */
    fun clearAllTestAudios(context: Context) {
        getTestAudiosDir(context).deleteRecursively()
    }

    /**
     * 录制音频
     */
    fun recordAudio(
        context: Context,
        durationSec: Float,
        sampleRate: Int = 16000
    ): ShortArray? {
        // Check RECORD_AUDIO permission
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "RECORD_AUDIO permission not granted")
            return null
        }

        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

        if (bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Invalid buffer size for recording")
            return null
        }

        @Suppress("MissingPermission") // Permission checked above
        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize * 2
        )

        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized")
            return null
        }

        val totalSamples = (sampleRate * durationSec).toInt()
        val audioData = ShortArray(totalSamples)

        audioRecord.startRecording()

        var samplesRead = 0
        while (samplesRead < totalSamples) {
            val samplesToRead = minOf(bufferSize / 2, totalSamples - samplesRead)
            val read = audioRecord.read(audioData, samplesRead, samplesToRead)
            if (read < 0) {
                Log.e(TAG, "Error reading audio: $read")
                break
            }
            samplesRead += read
        }

        audioRecord.stop()
        audioRecord.release()

        return if (samplesRead == totalSamples) audioData else null
    }

    /**
     * 保存录制的音频到WAV文件
     */
    fun saveRecordedAudio(
        context: Context,
        audioData: ShortArray,
        sampleRate: Int = 16000,
        filename: String
    ): File? {
        return try {
            val file = File(getTestAudiosDir(context), "$filename.wav")
            writeWavFile(
                file,
                audioData,
                TestAudioConfig(
                    name = filename,
                    type = TestAudioType.CHINESE_SPEECH,
                    durationSec = audioData.size.toFloat() / sampleRate,
                    sampleRate = sampleRate,
                    expectedText = "",
                    description = "Recorded audio"
                )
            )
            file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save recorded audio", e)
            null
        }
    }
}
