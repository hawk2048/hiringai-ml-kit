package com.hiringai.mobile.ml.benchmark

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sin

/**
 * 测试音频生成器
 * 生成用于语音识别基准测试的标准测试音频
 */
object TestAudioGenerator {

    private const val TAG = "TestAudioGenerator"
    const val SAMPLE_RATE = 16000
    const val CHANNELS = 1
    const val BITS_PER_SAMPLE = 16

    /**
     * 测试音频类型
     */
    enum class TestAudioType {
        SILENCE,            // 静音
        TONE_440HZ,         // 440Hz 正弦波 (A音符)
        TONE_1000HZ,        // 1000Hz 正弦波
        SWEEP,              // 频率扫描
        CHIRP,              // 啁啾信号
        WHITE_NOISE,        // 白噪声
        SIMULATED_SPEECH    // 模拟语音信号
    }

    /**
     * 测试音频信息
     */
    data class TestAudioInfo(
        val filename: String,
        val displayName: String,
        val description: String,
        val type: TestAudioType,
        val durationSeconds: Float,
        val sampleRate: Int = SAMPLE_RATE
    )

    /**
     * 获取所有可用测试音频
     */
    fun getAvailableTestAudios(): List<TestAudioInfo> = listOf(
        TestAudioInfo(
            filename = "test_silence_1s.wav",
            displayName = "静音测试 (1秒)",
            description = "1秒静音，用于VAD测试",
            type = TestAudioType.SILENCE,
            durationSeconds = 1f
        ),
        TestAudioInfo(
            filename = "test_tone_440hz_5s.wav",
            displayName = "440Hz 正弦波 (5秒)",
            description = "标准A音符，用于音频处理测试",
            type = TestAudioType.TONE_440HZ,
            durationSeconds = 5f
        ),
        TestAudioInfo(
            filename = "test_tone_1000hz_5s.wav",
            displayName = "1000Hz 正弦波 (5秒)",
            description = "标准测试频率",
            type = TestAudioType.TONE_1000HZ,
            durationSeconds = 5f
        ),
        TestAudioInfo(
            filename = "test_sweep_5s.wav",
            displayName = "频率扫描 (5秒)",
            description = "从20Hz到20kHz的频率扫描",
            type = TestAudioType.SWEEP,
            durationSeconds = 5f
        ),
        TestAudioInfo(
            filename = "test_chirp_3s.wav",
            displayName = "啁啾信号 (3秒)",
            description = "非线性频率变化信号",
            type = TestAudioType.CHIRP,
            durationSeconds = 3f
        ),
        TestAudioInfo(
            filename = "test_white_noise_2s.wav",
            displayName = "白噪声 (2秒)",
            description = "均匀分布的白噪声",
            type = TestAudioType.WHITE_NOISE,
            durationSeconds = 2f
        ),
        TestAudioInfo(
            filename = "test_simulated_speech_5s.wav",
            displayName = "模拟语音 (5秒)",
            description = "模拟语音频谱特性的测试信号",
            type = TestAudioType.SIMULATED_SPEECH,
            durationSeconds = 5f
        )
    )

    /**
     * 生成测试音频数据 (FloatArray, -1.0 to 1.0)
     */
    fun generateAudioData(type: TestAudioType, durationSeconds: Float, sampleRate: Int = SAMPLE_RATE): FloatArray {
        val numSamples = (durationSeconds * sampleRate).toInt()

        return when (type) {
            TestAudioType.SILENCE -> FloatArray(numSamples)

            TestAudioType.TONE_440HZ -> generateTone(440f, numSamples, sampleRate)

            TestAudioType.TONE_1000HZ -> generateTone(1000f, numSamples, sampleRate)

            TestAudioType.SWEEP -> generateSweep(20f, 20000f, numSamples, sampleRate)

            TestAudioType.CHIRP -> generateChirp(numSamples, sampleRate)

            TestAudioType.WHITE_NOISE -> generateWhiteNoise(numSamples)

            TestAudioType.SIMULATED_SPEECH -> generateSimulatedSpeech(numSamples, sampleRate)
        }
    }

    /**
     * 生成正弦波
     */
    private fun generateTone(frequency: Float, numSamples: Int, sampleRate: Int): FloatArray {
        return FloatArray(numSamples) { i ->
            val t = i.toFloat() / sampleRate
            sin(2.0 * Math.PI * frequency * t).toFloat() * 0.8f
        }
    }

    /**
     * 生成频率扫描 (线性)
     */
    private fun generateSweep(startFreq: Float, endFreq: Float, numSamples: Int, sampleRate: Int): FloatArray {
        return FloatArray(numSamples) { i ->
            val t = i.toFloat() / numSamples
            val freq = startFreq + (endFreq - startFreq) * t
            val phase = 2.0 * Math.PI * freq * i / sampleRate
            sin(phase).toFloat() * 0.7f * (1f - t * 0.5f) // 衰减
        }
    }

    /**
     * 生成啁啾信号 (对数扫描)
     */
    private fun generateChirp(numSamples: Int, sampleRate: Int): FloatArray {
        val startFreq = 100f
        val endFreq = 5000f
        val ratio = endFreq / startFreq

        return FloatArray(numSamples) { i ->
            val t = i.toFloat() / numSamples
            val freq = startFreq * Math.pow(ratio.toDouble(), t.toDouble()).toFloat()
            val phase = 2.0 * Math.PI * freq * i / sampleRate
            sin(phase).toFloat() * 0.6f
        }
    }

    /**
     * 生成白噪声
     */
    private fun generateWhiteNoise(numSamples: Int): FloatArray {
        val random = java.util.Random(42) // 固定种子确保可重复
        return FloatArray(numSamples) {
            (random.nextFloat() * 2 - 1) * 0.5f
        }
    }

    /**
     * 生成模拟语音信号
     * 包含基频和谐波，模拟人声频谱特性
     */
    private fun generateSimulatedSpeech(numSamples: Int, sampleRate: Int): FloatArray {
        val result = FloatArray(numSamples)
        val random = java.util.Random(123)

        // 基频范围 (男性: 85-180Hz, 女性: 165-255Hz)
        val fundamentalFreq = 150f + random.nextFloat() * 50f

        // 生成多段"音节"
        val syllableDuration = sampleRate / 4 // 每个音节约250ms
        var currentSample = 0

        while (currentSample < numSamples) {
            val syllableLength = minOf(syllableDuration, numSamples - currentSample)
            val intensity = 0.5f + random.nextFloat() * 0.5f

            for (i in 0 until syllableLength) {
                val idx = currentSample + i
                if (idx >= numSamples) break

                // 添加基频和谐波
                var sample = 0f
                val t = idx.toFloat() / sampleRate

                // 基频
                sample += sin(2.0 * Math.PI * fundamentalFreq * t).toFloat() * intensity * 0.5f

                // 二次谐波
                sample += sin(2.0 * Math.PI * fundamentalFreq * 2 * t).toFloat() * intensity * 0.3f

                // 三次谐波
                sample += sin(2.0 * Math.PI * fundamentalFreq * 3 * t).toFloat() * intensity * 0.15f

                // 添加一些噪声
                sample += (random.nextFloat() * 2 - 1) * 0.05f

                result[idx] = sample
            }

            currentSample += syllableLength

            // 音节间可能有小停顿
            val pause = (random.nextFloat() * sampleRate * 0.05f).toInt()
            currentSample += pause
        }

        // 应用简单的包络
        applyEnvelope(result, 0.01f, 0.01f)

        return result
    }

    /**
     * 应用包络 (淡入淡出)
     */
    private fun applyEnvelope(samples: FloatArray, fadeInSec: Float, fadeOutSec: Float) {
        val fadeInSamples = (fadeInSec * SAMPLE_RATE).toInt()
        val fadeOutSamples = (fadeOutSec * SAMPLE_RATE).toInt()

        // 淡入
        for (i in 0 until minOf(fadeInSamples, samples.size)) {
            samples[i] *= i.toFloat() / fadeInSamples
        }

        // 淡出
        val fadeOutStart = samples.size - minOf(fadeOutSamples, samples.size)
        for (i in fadeOutStart until samples.size) {
            val fadeRatio = (samples.size - i).toFloat() / fadeOutSamples
            samples[i] *= fadeRatio
        }
    }

    /**
     * FloatArray 转 16-bit PCM ByteArray
     */
    fun floatToPcm16(samples: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(samples.size * 2)
        buffer.order(ByteOrder.LITTLE_ENDIAN)

        for (sample in samples) {
            val clamped = sample.coerceIn(-1f, 1f)
            val intSample = (clamped * 32767).toInt()
            buffer.putShort(intSample.toShort())
        }

        return buffer.array()
    }

    /**
     * 生成 WAV 文件头
     */
    fun createWavHeader(dataSize: Int, sampleRate: Int = SAMPLE_RATE): ByteArray {
        val header = ByteBuffer.allocate(44)
        header.order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        header.put("RIFF".toByteArray())
        header.putInt(36 + dataSize) // File size - 8
        header.put("WAVE".toByteArray())

        // fmt chunk
        header.put("fmt ".toByteArray())
        header.putInt(16) // Chunk size
        header.putShort(1) // Audio format (PCM)
        header.putShort(CHANNELS.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * CHANNELS * BITS_PER_SAMPLE / 8) // Byte rate
        header.putShort((CHANNELS * BITS_PER_SAMPLE / 8).toShort()) // Block align
        header.putShort(BITS_PER_SAMPLE.toShort())

        // data chunk
        header.put("data".toByteArray())
        header.putInt(dataSize)

        return header.array()
    }

    /**
     * 保存 WAV 文件
     */
    fun saveWavFile(audioData: FloatArray, file: File, sampleRate: Int = SAMPLE_RATE): Boolean {
        return try {
            val pcmData = floatToPcm16(audioData)
            val header = createWavHeader(pcmData.size, sampleRate)

            FileOutputStream(file).use { out ->
                out.write(header)
                out.write(pcmData)
            }

            Log.i(TAG, "Saved WAV file: ${file.absolutePath} (${audioData.size} samples)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save WAV file", e)
            false
        }
    }

    /**
     * 初始化测试音频文件
     */
    fun initTestAudios(context: Context): List<File> {
        val testDir = getTestAudiosDir(context)
        val files = mutableListOf<File>()

        getAvailableTestAudios().forEach { info ->
            val file = File(testDir, info.filename)
            if (!file.exists()) {
                val audioData = generateAudioData(info.type, info.durationSeconds, info.sampleRate)
                saveWavFile(audioData, file, info.sampleRate)
            }
            files.add(file)
        }

        return files
    }

    /**
     * 获取测试音频目录
     */
    fun getTestAudiosDir(context: Context): File {
        val dir = File(context.filesDir, "test_audios")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * 加载测试音频文件
     */
    fun loadTestAudio(context: Context, info: TestAudioInfo): FloatArray? {
        val dir = getTestAudiosDir(context)
        val file = File(dir, info.filename)

        // 如果文件不存在，先生成
        if (!file.exists()) {
            val audioData = generateAudioData(info.type, info.durationSeconds, info.sampleRate)
            saveWavFile(audioData, file, info.sampleRate)
            return audioData
        }

        return loadWavFile(file)
    }

    /**
     * 从 WAV 文件加载音频数据
     */
    fun loadWavFile(file: File): FloatArray? {
        return try {
            RandomAccessFile(file, "r").use { raf ->
                // 跳过 WAV 头
                raf.seek(44)

                val dataSize = (raf.length() - 44).toInt()
                val pcmData = ByteArray(dataSize)
                raf.readFully(pcmData)

                // 转换为 FloatArray
                val numSamples = dataSize / 2
                val samples = FloatArray(numSamples)
                val buffer = ByteBuffer.wrap(pcmData)
                buffer.order(ByteOrder.LITTLE_ENDIAN)

                for (i in 0 until numSamples) {
                    samples[i] = buffer.short.toFloat() / 32768f
                }

                samples
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load WAV file: ${file.absolutePath}", e)
            null
        }
    }

    /**
     * 录音器 - 用于录制自定义测试音频
     */
    class AudioRecorder(private val context: Context) {
        private var audioRecord: AudioRecord? = null
        private var isRecording = false
        private val recordedData = mutableListOf<Short>()

        /**
         * 开始录音
         */
        fun startRecording(): Boolean {
            val bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            try {
                audioRecord = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize * 2
                )

                if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                    Log.e(TAG, "AudioRecord not initialized")
                    return false
                }

                audioRecord?.startRecording()
                isRecording = true
                recordedData.clear()

                Thread {
                    val buffer = ShortArray(bufferSize / 2)
                    while (isRecording) {
                        val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                        if (read > 0) {
                            synchronized(recordedData) {
                                for (i in 0 until read) {
                                    recordedData.add(buffer[i])
                                }
                            }
                        }
                    }
                }.start()

                return true
            } catch (e: SecurityException) {
                Log.e(TAG, "Missing RECORD_AUDIO permission", e)
                return false
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start recording", e)
                return false
            }
        }

        /**
         * 停止录音并返回录音数据
         */
        fun stopRecording(): FloatArray? {
            isRecording = false
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            if (recordedData.isEmpty()) return null

            return synchronized(recordedData) {
                val samples = FloatArray(recordedData.size) { i ->
                    recordedData[i].toFloat() / 32768f
                }
                recordedData.clear()
                samples
            }
        }

        /**
         * 获取录音时长 (秒)
         */
        fun getRecordingDuration(): Float {
            return synchronized(recordedData) {
                recordedData.size.toFloat() / SAMPLE_RATE
            }
        }

        /**
         * 保存录音到文件
         */
        fun saveRecordingToFile(filename: String): File? {
            val samples = stopRecording() ?: return null
            val file = File(getTestAudiosDir(context), filename)
            return if (saveWavFile(samples, file)) file else null
        }
    }
}
