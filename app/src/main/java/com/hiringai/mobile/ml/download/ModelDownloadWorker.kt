package com.hiringai.mobile.ml.download

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.*
import com.hiringai.mobile.ml.ModelManager
import com.hiringai.mobile.ml.testapp.R
import com.hiringai.mobile.ml.logging.MlLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

/**
 * 后台模型下载 Worker
 *
 * 使用 WorkManager 确保下载在后台继续进行，即使 Activity 已销毁。
 * 支持：
 * - 前台通知显示下载进度
 * - 下载完成后自动更新 UI（通过 LiveData/Broadcast）
 * - 网络恢复后自动重试
 */
class ModelDownloadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    private val logger = MlLogger.getInstance(context)
    private val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    companion object {
        const val KEY_MODEL_NAME = "model_name"
        const val KEY_MODEL_URL = "model_url"
        const val KEY_MODEL_SIZE = "model_size"
        const val KEY_PROGRESS = "progress"
        const val KEY_SPEED = "speed"
        const val KEY_COMPLETE = "complete"
        const val KEY_SUCCESS = "success"
        const val KEY_ERROR = "error"

        private const val CHANNEL_ID = "model_download"
        private const val NOTIFICATION_ID = 1001
        private const val WORK_TAG_PREFIX = "model_download_"

        /**
         * 启动后台下载
         */
        fun enqueue(
            context: Context,
            modelName: String,
            modelUrl: String,
            modelSize: Long
        ): String {
            val workRequest = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
                .setInputData(workDataOf(
                    KEY_MODEL_NAME to modelName,
                    KEY_MODEL_URL to modelUrl,
                    KEY_MODEL_SIZE to modelSize
                ))
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .build()
                )
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    30,
                    TimeUnit.SECONDS
                )
                .addTag("$WORK_TAG_PREFIX$modelName")
                .build()

            WorkManager.getInstance(context)
                .enqueueUniqueWork(
                    "$WORK_TAG_PREFIX$modelName",
                    ExistingWorkPolicy.KEEP,
                    workRequest
                )

            return "$WORK_TAG_PREFIX$modelName"
        }

        /**
         * 取消下载
         */
        fun cancel(context: Context, modelName: String) {
            WorkManager.getInstance(context)
                .cancelUniqueWork("$WORK_TAG_PREFIX$modelName")
        }

        /**
         * 取消所有下载
         */
        fun cancelAll(context: Context) {
            WorkManager.getInstance(context).cancelAllWorkByTag("model_download")
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val modelName = inputData.getString(KEY_MODEL_NAME) ?: return@withContext Result.failure()
        val modelUrl = inputData.getString(KEY_MODEL_URL) ?: return@withContext Result.failure()
        val modelSize = inputData.getLong(KEY_MODEL_SIZE, 0L)

        logger.info("DownloadWorker", "开始后台下载: $modelName")

        // 创建通知渠道
        createNotificationChannel()

        // 显示前台通知
        setForeground(createForegroundInfo(modelName, 0))

        try {
            val modelManager = ModelManager.getInstance(applicationContext)

            // 执行下载
            val success = modelManager.downloadModelSync(
                modelName = modelName,
                downloadUrl = modelUrl,
                modelSize = modelSize,
                onProgress = { progress, speed ->
                    // 更新通知进度
                    updateNotification(modelName, progress, speed)
                    // 同步到 WorkData
                    setProgressAsync(workDataOf(
                        KEY_PROGRESS to progress,
                        KEY_SPEED to speed
                    ))
                }
            )

            if (success) {
                logger.info("DownloadWorker", "下载完成: $modelName")
                showCompletionNotification(modelName, true)
                Result.success(workDataOf(
                    KEY_SUCCESS to true,
                    KEY_COMPLETE to true
                ))
            } else {
                logger.warn("DownloadWorker", "下载失败: $modelName")
                showCompletionNotification(modelName, false)
                Result.failure(workDataOf(
                    KEY_SUCCESS to false,
                    KEY_ERROR to "下载失败"
                ))
            }
        } catch (e: Exception) {
            logger.error("DownloadWorker", "下载异常: $modelName", e)
            showCompletionNotification(modelName, false)
            Result.failure(workDataOf(
                KEY_ERROR to (e.message ?: "未知错误")
            ))
        }
    }

    /**
     * 创建通知渠道 (Android 8.0+)
     */
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "模型下载",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "显示模型下载进度"
                setShowBadge(false)
            }
            notificationManager.createNotificationChannel(channel)
        }
    }

    /**
     * 创建前台通知信息
     */
    private fun createForegroundInfo(modelName: String, progress: Int): ForegroundInfo {
        val intent = applicationContext.packageManager
            .getLaunchIntentForPackage(applicationContext.packageName)
        val pendingIntent = PendingIntent.getActivity(
            applicationContext,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle("正在下载模型")
            .setContentText(modelName)
            .setSmallIcon(R.drawable.ic_download)
            .setOngoing(true)
            .setProgress(100, progress, progress == 0)
            .setContentIntent(pendingIntent)
            .setOnlyAlertOnce(true)
            .build()

        return ForegroundInfo(NOTIFICATION_ID, notification)
    }

    /**
     * 更新通知进度
     */
    private fun updateNotification(modelName: String, progress: Int, speed: String) {
        val intent = applicationContext.packageManager
            .getLaunchIntentForPackage(applicationContext.packageName)
        val pendingIntent = PendingIntent.getActivity(
            applicationContext,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle(modelName)
            .setContentText("$progress% · $speed")
            .setSmallIcon(R.drawable.ic_download)
            .setOngoing(true)
            .setProgress(100, progress, false)
            .setContentIntent(pendingIntent)
            .setOnlyAlertOnce(true)
            .build()

        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    /**
     * 显示下载完成通知
     */
    private fun showCompletionNotification(modelName: String, success: Boolean) {
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle(if (success) "下载完成" else "下载失败")
            .setContentText(modelName)
            .setSmallIcon(if (success) R.drawable.ic_check else R.drawable.ic_error)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()

        notificationManager.notify(NOTIFICATION_ID + 1, notification)
    }
}
