package com.hiringai.mobile.ml.logging

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.atomic.AtomicInteger

/**
 * HiringAI ML Kit 统一日志系统
 *
 * 功能：
 * - 分级日志 (VERBOSE / DEBUG / INFO / WARN / ERROR)
 * - 日志数量管控 (环形缓冲区，默认最多 500 条)
 * - 持久化到文件 (自动清理，默认最多保留 5MB)
 * - 实时日志流 (SharedFlow，供 UI 订阅)
 * - 按标签过滤
 * - 性能开销极低 (异步写入，非阻塞)
 *
 * 用法：
 * ```kotlin
 * val logger = MlLogger.getInstance(context)
 *
 * // 基本日志
 * logger.info("ModelManager", "模型下载完成: qwen2.5-0.5b")
 * logger.error("LocalLLMService", "加载失败", exception)
 *
 * // 订阅实时日志流
 * logger.logFlow.collect { entry ->
 *     textView.text = entry.formatted
 * }
 *
 * // 查询历史日志
 * val recentLogs = logger.getRecentLogs(limit = 50)
 * val errorLogs = logger.getLogsByLevel(LogLevel.ERROR)
 * ```
 */
class MlLogger private constructor(private val context: Context) {

    // ── 日志级别 ─────────────────────────────────────────────
    enum class LogLevel(val priority: Int, val label: String) {
        VERBOSE(2, "V"),
        DEBUG(3, "D"),
        INFO(4, "I"),
        WARN(5, "W"),
        ERROR(6, "E");

        companion object {
            fun fromPriority(priority: Int): LogLevel {
                return entries.find { it.priority == priority } ?: VERBOSE
            }
        }
    }

    // ── 日志条目 ─────────────────────────────────────────────
    data class LogEntry(
        val timestamp: Long = System.currentTimeMillis(),
        val level: LogLevel,
        val tag: String,
        val message: String,
        val throwable: String? = null
    ) {
        val formatted: String
            get() = buildString {
                append(DATE_FORMAT.format(Date(timestamp)))
                append(" ")
                append(level.label)
                append("/")
                append(tag.take(MAX_TAG_LEN))
                append(": ")
                append(message)
                if (throwable != null) {
                    append("\n")
                    append(throwable)
                }
            }

        val shortFormatted: String
            get() = buildString {
                append(level.label)
                append("/")
                append(tag.take(MAX_TAG_LEN))
                append(": ")
                append(message.take(120))
            }

        fun toJson(): JSONObject {
            return JSONObject().apply {
                put("ts", timestamp)
                put("level", level.name)
                put("tag", tag)
                put("msg", message)
                if (throwable != null) put("throwable", throwable)
            }
        }

        companion object {
            private val DATE_FORMAT = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.getDefault())
            private const val MAX_TAG_LEN = 23

            fun fromJson(json: JSONObject): LogEntry {
                return LogEntry(
                    timestamp = json.optLong("ts", 0),
                    level = try { LogLevel.valueOf(json.optString("level", "INFO")) } catch (_: Exception) { LogLevel.INFO },
                    tag = json.optString("tag", ""),
                    message = json.optString("msg", ""),
                    throwable = json.optString("throwable", "").ifEmpty { null }
                )
            }
        }
    }

    // ── 日志配置 ─────────────────────────────────────────────
    data class LogConfig(
        val maxInMemoryEntries: Int = DEFAULT_MAX_ENTRIES,
        val maxLogFileBytes: Long = DEFAULT_MAX_LOG_FILE_SIZE,
        val minLogLevel: LogLevel = LogLevel.VERBOSE,
        val enableFileLogging: Boolean = true,
        val enableAndroidLog: Boolean = true,
        val persistToDisk: Boolean = true
    )

    companion object {
        private const val TAG = "MlLogger"

        const val DEFAULT_MAX_ENTRIES = 500
        const val DEFAULT_MAX_LOG_FILE_SIZE = 5L * 1024 * 1024 // 5MB
        private const val LOG_DIR = "ml_logs"
        private const val LOG_FILE = "ml_kit.log"
        private const val LOG_ARCHIVE = "ml_kit_archive.log"

        @Volatile
        private var instance: MlLogger? = null

        fun getInstance(context: Context): MlLogger {
            return instance ?: synchronized(this) {
                instance ?: MlLogger(context.applicationContext).also { instance = it }
            }
        }
    }

    // ── 环形缓冲区 ───────────────────────────────────────────
    private val ringBuffer = ConcurrentLinkedDeque<LogEntry>()
    private val bufferSize = AtomicInteger(0)
    private var config = LogConfig()

    // ── 实时日志流 ───────────────────────────────────────────
    private val _logFlow = MutableSharedFlow<LogEntry>(extraBufferCapacity = 64)
    val logFlow: Flow<LogEntry> = _logFlow.asSharedFlow()

    // ── 日志文件 ─────────────────────────────────────────────
    private val logDir by lazy { File(context.filesDir, LOG_DIR).also { it.mkdirs() } }
    private val logFile by lazy { File(logDir, LOG_FILE) }
    private val archiveFile by lazy { File(logDir, LOG_ARCHIVE) }
    private var fileWriter: FileWriter? = null

    // ── 公开 API ─────────────────────────────────────────────

    fun verbose(tag: String, message: String) = log(LogLevel.VERBOSE, tag, message)
    fun debug(tag: String, message: String) = log(LogLevel.DEBUG, tag, message)
    fun info(tag: String, message: String) = log(LogLevel.INFO, tag, message)
    fun warn(tag: String, message: String) = log(LogLevel.WARN, tag, message)
    fun error(tag: String, message: String, throwable: Throwable? = null) {
        log(LogLevel.ERROR, tag, message, throwable?.stackTraceToString())
    }

    /**
     * 更新日志配置
     */
    fun updateConfig(block: LogConfig.() -> LogConfig) {
        config = config.block()
    }

    /**
     * 获取当前配置
     */
    fun getConfig(): LogConfig = config

    /**
     * 获取最近的日志 (从环形缓冲区)
     */
    fun getRecentLogs(limit: Int = DEFAULT_MAX_ENTRIES): List<LogEntry> {
        val all = ringBuffer.toList()
        return if (all.size <= limit) all else all.takeLast(limit)
    }

    /**
     * 按级别过滤日志
     */
    fun getLogsByLevel(level: LogLevel, limit: Int = DEFAULT_MAX_ENTRIES): List<LogEntry> {
        return ringBuffer.filter { it.level == level }.takeLast(limit)
    }

    /**
     * 按标签过滤日志
     */
    fun getLogsByTag(tag: String, limit: Int = DEFAULT_MAX_ENTRIES): List<LogEntry> {
        return ringBuffer.filter { it.tag == tag }.takeLast(limit)
    }

    /**
     * 搜索日志
     */
    fun searchLogs(query: String, limit: Int = 100): List<LogEntry> {
        if (query.isBlank()) return emptyList()
        val lowerQuery = query.lowercase()
        return ringBuffer.filter { entry ->
            entry.message.lowercase().contains(lowerQuery) ||
                    entry.tag.lowercase().contains(lowerQuery)
        }.takeLast(limit)
    }

    /**
     * 获取日志统计
     */
    fun getLogStats(): LogStats {
        val entries = ringBuffer.toList()
        return LogStats(
            totalEntries = entries.size,
            verboseCount = entries.count { it.level == LogLevel.VERBOSE },
            debugCount = entries.count { it.level == LogLevel.DEBUG },
            infoCount = entries.count { it.level == LogLevel.INFO },
            warnCount = entries.count { it.level == LogLevel.WARN },
            errorCount = entries.count { it.level == LogLevel.ERROR },
            logFileSize = if (logFile.exists()) logFile.length() else 0L,
            oldestEntry = entries.firstOrNull()?.timestamp,
            newestEntry = entries.lastOrNull()?.timestamp
        )
    }

    data class LogStats(
        val totalEntries: Int,
        val verboseCount: Int,
        val debugCount: Int,
        val infoCount: Int,
        val warnCount: Int,
        val errorCount: Int,
        val logFileSize: Long,
        val oldestEntry: Long?,
        val newestEntry: Long?
    )

    /**
     * 导出日志为文本
     */
    suspend fun exportLogs(): String = withContext(Dispatchers.IO) {
        val sb = StringBuilder()
        sb.appendLine("=== HiringAI ML Kit 日志导出 ===")
        sb.appendLine("导出时间: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())}")
        sb.appendLine("日志条数: ${ringBuffer.size}")
        sb.appendLine()

        ringBuffer.forEach { entry ->
            sb.appendLine(entry.formatted)
        }

        sb.toString()
    }

    /**
     * 导出日志为 JSON
     */
    suspend fun exportLogsAsJson(): String = withContext(Dispatchers.IO) {
        val array = JSONArray()
        ringBuffer.forEach { entry ->
            array.put(entry.toJson())
        }
        array.toString(2)
    }

    /**
     * 清除内存中的日志
     */
    fun clearMemoryLogs() {
        ringBuffer.clear()
        bufferSize.set(0)
    }

    /**
     * 清除日志文件
     */
    suspend fun clearLogFiles() = withContext(Dispatchers.IO) {
        closeFileWriter()
        logFile.delete()
        archiveFile.delete()
        logDir.listFiles()?.forEach { it.delete() }
    }

    /**
     * 清除所有日志 (内存 + 文件)
     */
    suspend fun clearAll() {
        clearMemoryLogs()
        clearLogFiles()
    }

    /**
     * 刷新日志到磁盘
     */
    suspend fun flush() = withContext(Dispatchers.IO) {
        fileWriter?.flush()
    }

    // ── 内部实现 ─────────────────────────────────────────────

    private fun log(level: LogLevel, tag: String, message: String, throwable: String? = null) {
        // 级别过滤
        if (level.priority < config.minLogLevel.priority) return

        val entry = LogEntry(
            level = level,
            tag = tag,
            message = message,
            throwable = throwable
        )

        // 1. 写入环形缓冲区 (数量管控)
        addToRingBuffer(entry)

        // 2. 发送到 Android Logcat
        if (config.enableAndroidLog) {
            writeToAndroidLog(entry)
        }

        // 3. 发送到实时流
        _logFlow.tryEmit(entry)

        // 4. 异步写入文件
        if (config.persistToDisk && config.enableFileLogging) {
            writeToFileAsync(entry)
        }
    }

    private fun addToRingBuffer(entry: LogEntry) {
        ringBuffer.addLast(entry)
        val size = bufferSize.incrementAndGet()

        // 超出容量时移除最旧的
        if (size > config.maxInMemoryEntries) {
            ringBuffer.pollFirst()
            bufferSize.decrementAndGet()
        }
    }

    private fun writeToAndroidLog(entry: LogEntry) {
        when (entry.level) {
            LogLevel.VERBOSE -> Log.v(entry.tag, entry.message)
            LogLevel.DEBUG -> Log.d(entry.tag, entry.message)
            LogLevel.INFO -> Log.i(entry.tag, entry.message)
            LogLevel.WARN -> Log.w(entry.tag, entry.message)
            LogLevel.ERROR -> {
                val t = entry.throwable?.let { Throwable(it) }
                Log.e(entry.tag, entry.message, t)
            }
        }
    }

    @Synchronized
    private fun writeToFileAsync(entry: LogEntry) {
        try {
            // 检查文件大小限制
            if (logFile.exists() && logFile.length() > config.maxLogFileBytes) {
                rotateLogFile()
            }

            // 追加写入
            if (fileWriter == null) {
                fileWriter = FileWriter(logFile, true)
            }
            fileWriter?.apply {
                write(entry.formatted)
                write("\n")
                flush()
            }
        } catch (e: Exception) {
            // 日志系统本身的错误不抛出，避免递归
            Log.e(TAG, "Failed to write log entry to file", e)
        }
    }

    @Synchronized
    private fun rotateLogFile() {
        try {
            closeFileWriter()
            // 归档旧日志
            if (archiveFile.exists()) {
                archiveFile.delete()
            }
            logFile.renameTo(archiveFile)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to rotate log file", e)
        }
    }

    @Synchronized
    private fun closeFileWriter() {
        try {
            fileWriter?.close()
        } catch (_: Exception) {
        }
        fileWriter = null
    }
}
