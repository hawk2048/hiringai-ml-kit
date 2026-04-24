package com.hiringai.mobile.ml.testapp

import android.app.Application
import com.hiringai.mobile.ml.logging.MlLogger

class TestApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // 初始化日志系统
        val logger = MlLogger.getInstance(this)
        logger.info("TestApp", "HiringAI ML Kit Test App 启动")
    }
}
