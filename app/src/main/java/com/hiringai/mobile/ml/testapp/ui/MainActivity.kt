package com.hiringai.mobile.ml.testapp.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.hiringai.mobile.ml.ModelManager
import com.hiringai.mobile.ml.DeviceCapabilityDetector
import com.hiringai.mobile.ml.logging.MlLogger
import com.hiringai.mobile.ml.testapp.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 主界面 — 模型管理 + 功能入口
 */
class MainActivity : AppCompatActivity() {

    private lateinit var logger: MlLogger
    private lateinit var modelManager: ModelManager

    // Views
    private lateinit var deviceInfoText: TextView
    private lateinit var storageInfoText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        logger = MlLogger.getInstance(this)
        modelManager = ModelManager.getInstance(this)

        initViews()
        loadDeviceInfo()

        logger.info("MainActivity", "主界面创建完成")
    }

    private fun initViews() {
        deviceInfoText = findViewById(R.id.deviceInfoText)
        storageInfoText = findViewById(R.id.storageInfoText)
        progressBar = findViewById(R.id.progressBar)
        statusText = findViewById(R.id.statusText)

        // 功能按钮
        findViewById<View>(R.id.btnBenchmark).setOnClickListener {
            startActivity(Intent(this, BenchmarkActivity::class.java))
        }
        findViewById<View>(R.id.btnModelCatalog).setOnClickListener {
            startActivity(Intent(this, ModelCatalogActivity::class.java))
        }
        findViewById<View>(R.id.btnLogViewer).setOnClickListener {
            startActivity(Intent(this, LogViewerActivity::class.java))
        }
    }

    private fun loadDeviceInfo() {
        lifecycleScope.launch {
            val detector = DeviceCapabilityDetector.getInstance(this@MainActivity)
            val capability = withContext(Dispatchers.IO) { detector.detect() }
            val storage = modelManager.getStorageUsage()

            deviceInfoText.text = buildString {
                appendLine("📱 设备: ${capability.deviceName}")
                appendLine("🔧 CPU: ${capability.cpuCores}核, ABI: ${capability.supportedAbis.joinToString()}")
                appendLine("💾 RAM: ${capability.totalRamMB.toInt()} MB")
                appendLine("🎮 GPU: ${capability.gpuInfo}")
                appendLine("⚡ NPU: ${if (capability.hasNpu) "支持" else "不支持"}")
            }

            storageInfoText.text = buildString {
                appendLine("📦 已下载模型: ${storage.modelsCount} 个")
                appendLine("💽 存储占用: ${storage.formattedSize}")
            }
        }
    }

    override fun onResume() {
        super.onResume()
        loadDeviceInfo()
    }
}
