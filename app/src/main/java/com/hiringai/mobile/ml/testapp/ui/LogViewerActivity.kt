package com.hiringai.mobile.ml.testapp.ui

import android.os.Bundle
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.hiringai.mobile.ml.logging.MlLogger
import com.hiringai.mobile.ml.testapp.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 日志查看界面
 */
class LogViewerActivity : AppCompatActivity() {

    private lateinit var logger: MlLogger
    private lateinit var logContainer: LinearLayout
    private lateinit var logScrollView: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_log_viewer)

        logger = MlLogger.getInstance(this)
        logContainer = findViewById(R.id.logContainer)
        logScrollView = findViewById(R.id.logScrollView)

        loadRecentLogs()

        // 订阅实时日志
        lifecycleScope.launch {
            logger.logFlow.collect { entry ->
                withContext(Dispatchers.Main) {
                    addLogEntry(entry)
                    logScrollView.post { logScrollView.fullScroll(ScrollView.FOCUS_DOWN) }
                }
            }
        }
    }

    private fun loadRecentLogs() {
        val logs = logger.getRecentLogs(limit = 100)
        logContainer.removeAllViews()
        logs.forEach { entry -> addLogEntry(entry) }
        logScrollView.post { logScrollView.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    private fun addLogEntry(entry: MlLogger.LogEntry) {
        val tv = TextView(this).apply {
            text = entry.formatted
            setPadding(4, 2, 4, 2)
            textSize = 11f
            setTextColor(
                when (entry.level) {
                    MlLogger.LogLevel.ERROR -> 0xFFFF4444.toInt()
                    MlLogger.LogLevel.WARN -> 0xFFFFAA00.toInt()
                    MlLogger.LogLevel.INFO -> 0xFF44AA44.toInt()
                    MlLogger.LogLevel.DEBUG -> 0xFF6688CC.toInt()
                    MlLogger.LogLevel.VERBOSE -> 0xFF888888.toInt()
                }
            )
        }
        logContainer.addView(tv)

        // 限制内存中的 View 数量
        while (logContainer.childCount > 500) {
            logContainer.removeViewAt(0)
        }
    }
}
