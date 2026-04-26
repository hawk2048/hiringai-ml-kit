package com.hiringai.mobile.ml.testapp.ui

import android.content.Context
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.chip.Chip
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.button.MaterialButton
import com.hiringai.mobile.ml.testapp.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * 在线 API 配置界面
 *
 * 支持的平台：
 * - 国内：硅基流动（推荐）、智谱AI、阿里云百炼
 * - 国际：Groq API、OpenRouter（需科学上网）
 *
 * 所有密钥仅本地存储，不会上传到任何服务器
 */
class ApiConfigActivity : AppCompatActivity() {

    companion object {
        const val PREF_NAME = "api_config"

        // 平台配置
        const val KEY_SILICONFLOW = "siliconflow_key"
        const val KEY_ZHIPU = "zhipu_key"
        const val KEY_ALIYUN = "aliyun_key"
        const val KEY_GROQ = "groq_key"
        const val KEY_OPENROUTER = "openrouter_key"

        // API 端点
        const val URL_SILICONFLOW = "https://api.siliconflow.cn/v1/chat/completions"
        const val URL_ZHIPU = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        const val URL_ALIYUN = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        const val URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
        const val URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
    }

    private val prefs by lazy { getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE) }

    // Views
    private lateinit var editSiliconflowKey: TextInputEditText
    private lateinit var chipSiliconflow: Chip
    private lateinit var btnSaveSiliconflow: MaterialButton
    private lateinit var editZhipuKey: TextInputEditText
    private lateinit var chipZhipu: Chip
    private lateinit var btnSaveZhipu: MaterialButton
    private lateinit var editAliyunKey: TextInputEditText
    private lateinit var chipAliyun: Chip
    private lateinit var btnSaveAliyun: MaterialButton
    private lateinit var editGroqKey: TextInputEditText
    private lateinit var chipGroq: Chip
    private lateinit var btnSaveGroq: MaterialButton
    private lateinit var editOpenrouterKey: TextInputEditText
    private lateinit var chipOpenrouter: Chip
    private lateinit var btnSaveOpenrouter: MaterialButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_api_config)

        initViews()
        loadSavedKeys()
        setupSaveButtons()
    }

    private fun initViews() {
        editSiliconflowKey = findViewById(R.id.editSiliconflowKey)
        chipSiliconflow = findViewById(R.id.chipSiliconflow)
        btnSaveSiliconflow = findViewById(R.id.btnSaveSiliconflow)
        editZhipuKey = findViewById(R.id.editZhipuKey)
        chipZhipu = findViewById(R.id.chipZhipu)
        btnSaveZhipu = findViewById(R.id.btnSaveZhipu)
        editAliyunKey = findViewById(R.id.editAliyunKey)
        chipAliyun = findViewById(R.id.chipAliyun)
        btnSaveAliyun = findViewById(R.id.btnSaveAliyun)
        editGroqKey = findViewById(R.id.editGroqKey)
        chipGroq = findViewById(R.id.chipGroq)
        btnSaveGroq = findViewById(R.id.btnSaveGroq)
        editOpenrouterKey = findViewById(R.id.editOpenrouterKey)
        chipOpenrouter = findViewById(R.id.chipOpenrouter)
        btnSaveOpenrouter = findViewById(R.id.btnSaveOpenrouter)
    }

    private fun loadSavedKeys() {
        val siliconflow = prefs.getString(KEY_SILICONFLOW, "") ?: ""
        val zhipu = prefs.getString(KEY_ZHIPU, "") ?: ""
        val aliyun = prefs.getString(KEY_ALIYUN, "") ?: ""
        val groq = prefs.getString(KEY_GROQ, "") ?: ""
        val openrouter = prefs.getString(KEY_OPENROUTER, "") ?: ""

        editSiliconflowKey.setText(siliconflow)
        editZhipuKey.setText(zhipu)
        editAliyunKey.setText(aliyun)
        editGroqKey.setText(groq)
        editOpenrouterKey.setText(openrouter)

        updateChipStatus(chipSiliconflow, siliconflow)
        updateChipStatus(chipZhipu, zhipu)
        updateChipStatus(chipAliyun, aliyun)
        updateChipStatus(chipGroq, groq)
        updateChipStatus(chipOpenrouter, openrouter)
    }

    private fun updateChipStatus(chip: Chip, key: String) {
        if (key.isNotBlank()) {
            chip.text = "已配置 ✓"
            chip.setChipBackgroundColorResource(R.color.success_light)
        } else {
            chip.text = "未配置"
            chip.setChipBackgroundColorResource(R.color.surface_variant)
        }
    }

    private fun setupSaveButtons() {
        btnSaveSiliconflow.setOnClickListener { saveAndTest(KEY_SILICONFLOW, editSiliconflowKey, chipSiliconflow, URL_SILICONFLOW, "硅基流动") }
        btnSaveZhipu.setOnClickListener { saveAndTest(KEY_ZHIPU, editZhipuKey, chipZhipu, URL_ZHIPU, "智谱AI") }
        btnSaveAliyun.setOnClickListener { saveAndTest(KEY_ALIYUN, editAliyunKey, chipAliyun, URL_ALIYUN, "阿里云百炼") }
        btnSaveGroq.setOnClickListener { saveAndTest(KEY_GROQ, editGroqKey, chipGroq, URL_GROQ, "Groq") }
        btnSaveOpenrouter.setOnClickListener { saveAndTest(KEY_OPENROUTER, editOpenrouterKey, chipOpenrouter, URL_OPENROUTER, "OpenRouter") }
    }

    private fun saveAndTest(keyName: String, editText: TextInputEditText, chip: Chip, apiUrl: String, platformName: String) {
        val apiKey = editText.text?.toString()?.trim() ?: ""
        if (apiKey.isBlank()) {
            Toast.makeText(this, "请输入 API Key", Toast.LENGTH_SHORT).show()
            return
        }

        prefs.edit().putString(keyName, apiKey).apply()
        setAllButtonsEnabled(false)

        lifecycleScope.launch {
            val success = withContext(Dispatchers.IO) {
                testApiConnection(apiKey, apiUrl)
            }

            withContext(Dispatchers.Main) {
                setAllButtonsEnabled(true)
                updateChipStatus(chip, apiKey)
                if (success) {
                    Toast.makeText(this@ApiConfigActivity, "$platformName 配置成功并已验证 ✓", Toast.LENGTH_SHORT).show()
                } else {
                    chip.text = "已保存(未验证)"
                    Toast.makeText(this@ApiConfigActivity, "$platformName 密钥已保存，但验证失败（请检查网络或 Key）", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun setAllButtonsEnabled(enabled: Boolean) {
        btnSaveSiliconflow.isEnabled = enabled
        btnSaveZhipu.isEnabled = enabled
        btnSaveAliyun.isEnabled = enabled
        btnSaveGroq.isEnabled = enabled
        btnSaveOpenrouter.isEnabled = enabled
    }

    private fun testApiConnection(apiKey: String, apiUrl: String): Boolean {
        return try {
            val url = URL(apiUrl)
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.connectTimeout = 15000
            conn.readTimeout = 30000
            conn.setRequestProperty("Authorization", "Bearer $apiKey")
            conn.setRequestProperty("Content-Type", "application/json")
            conn.doOutput = true

            val body = """{"model":"auto","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"""
            OutputStreamWriter(conn.outputStream).use { it.write(body) }

            val responseCode = conn.responseCode
            conn.disconnect()
            responseCode in 200..299
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    // ─── 公开方法（供其他组件调用）───────────────────────

    fun getConfiguredPlatform(): String? {
        return when {
            prefs.getString(KEY_SILICONFLOW, "")?.isNotBlank() == true -> "硅基流动"
            prefs.getString(KEY_ZHIPU, "")?.isNotBlank() == true -> "智谱AI"
            prefs.getString(KEY_ALIYUN, "")?.isNotBlank() == true -> "阿里云百炼"
            prefs.getString(KEY_GROQ, "")?.isNotBlank() == true -> "Groq"
            prefs.getString(KEY_OPENROUTER, "")?.isNotBlank() == true -> "OpenRouter"
            else -> null
        }
    }

    fun getSiliconflowKey(): String? = prefs.getString(KEY_SILICONFLOW, null)?.takeIf { it.isNotBlank() }
    fun getZhipuKey(): String? = prefs.getString(KEY_ZHIPU, null)?.takeIf { it.isNotBlank() }
    fun getAliyunKey(): String? = prefs.getString(KEY_ALIYUN, null)?.takeIf { it.isNotBlank() }
    fun getGroqKey(): String? = prefs.getString(KEY_GROQ, null)?.takeIf { it.isNotBlank() }
    fun getOpenrouterKey(): String? = prefs.getString(KEY_OPENROUTER, null)?.takeIf { it.isNotBlank() }
}
