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
    private lateinit var apiConfigStatusText: TextView

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
        apiConfigStatusText = findViewById(R.id.apiConfigStatusText)

        // BenchmarkActivity
        findViewById<View>(R.id.btnBenchmark).setOnClickListener {
            startActivity(Intent(this, BenchmarkActivity::class.java))
        }
        findViewById<View>(R.id.btnModelCatalog).setOnClickListener {
            startActivity(Intent(this, ModelCatalogActivity::class.java))
        }
        findViewById<View>(R.id.btnLogViewer).setOnClickListener {
            startActivity(Intent(this, LogViewerActivity::class.java))
        }
        findViewById<View>(R.id.btnApiConfig).setOnClickListener {
            startActivity(Intent(this, ApiConfigActivity::class.java))
        }

        // 更新 API 配置状态显示
        updateApiConfigStatus()
    }

    private fun updateApiConfigStatus() {
        val prefs = getSharedPreferences(ApiConfigActivity.PREF_NAME, MODE_PRIVATE)
        val configured = when {
            prefs.getString(ApiConfigActivity.KEY_SILICONFLOW, "")?.isNotBlank() == true -> "✓ 硅基流动已配置"
            prefs.getString(ApiConfigActivity.KEY_ZHIPU, "")?.isNotBlank() == true -> "✓ 智谱AI已配置"
            prefs.getString(ApiConfigActivity.KEY_ALIYUN, "")?.isNotBlank() == true -> "✓ 阿里云百炼已配置"
            prefs.getString(ApiConfigActivity.KEY_GROQ, "")?.isNotBlank() == true -> "✓ Groq已配置"
            prefs.getString(ApiConfigActivity.KEY_OPENROUTER, "")?.isNotBlank() == true -> "✓ OpenRouter已配置"
            else -> "点击配置免费 API 密钥"
        }
        apiConfigStatusText.text = configured
    }

    private fun loadDeviceInfo() {
        lifecycleScope.launch {
            val detector = DeviceCapabilityDetector(this@MainActivity)
            val capability = withContext(Dispatchers.IO) { detector.detectCapabilities() }
            val storage = modelManager.getStorageUsage()

            deviceInfoText.text = buildString {
                appendLine("📱 SoC : ${capability.socModel}")
                appendLine("⚙️ CPU : ${capability.cpuCores}核 ${capability.cpuArchitecture}")
                appendLine("       大核 ${capability.cpuBigCoreMaxFreqMHz}MHz · 小核 ${capability.cpuLittleCoreMaxFreqMHz}MHz")
                appendLine("💾 RAM : ${capability.totalRAM}MB (可用 ${capability.availableRAM}MB)")
                appendLine("🎮 GPU : ${capability.gpuName}")
                appendLine("🧠 NPU : ${if (capability.npuAvailable) "✓ 可用" else "✗ 未检测到"}")
                appendLine("⚡ Vulkan: ${if (capability.hasVulkan) "支持" else "不支持"}")
                if (capability.cpuTempCelsius > 0) {
                    appendLine("🌡️ CPU 温度: ${"%.1f".format(capability.cpuTempCelsius)}°C")
                }
                appendLine("📊 性能评分: ${capability.benchmarkScore}/100")
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
        updateApiConfigStatus()
    }
}
