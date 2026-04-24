package com.hiringai.mobile.ml.bridge

/**
 * ML 模块的数据桥接层
 *
 * 这个文件定义了 ML 模块对外部数据实体的最小接口。
 * 宿主应用需要提供这些数据的实现（通过直接构造或映射），
 * 从而实现 ML 模块与业务数据层的完全解耦。
 */

/**
 * 职位信息（用于 LLM 生成职位画像）
 */
data class JobInfo(
    val title: String,
    val requirements: String
)

/**
 * 候选人信息（用于 LLM 生成候选人画像）
 */
data class CandidateInfo(
    val name: String,
    val email: String,
    val phone: String,
    val resume: String
)

/**
 * 桥接工具：从宿主应用的数据实体转换到 ML 模块的数据类
 */
object MlBridge {
    /**
     * 从任意对象提取职位信息
     * 宿主应用可调用此方法或直接构造 JobInfo
     */
    fun jobInfo(title: String, requirements: String): JobInfo {
        return JobInfo(title, requirements)
    }

    /**
     * 从任意对象提取候选人信息
     * 宿主应用可调用此方法或直接构造 CandidateInfo
     */
    fun candidateInfo(name: String, email: String, phone: String, resume: String): CandidateInfo {
        return CandidateInfo(name, email, phone, resume)
    }
}
