# HiringAI ML Kit Test App Proguard Rules

# Keep ML Kit public API
-keep class com.hiringai.mobile.ml.** { *; }
-keep class com.hiringai.mobile.ml.bridge.** { *; }
-keep class com.hiringai.mobile.ml.logging.** { *; }
-keep class com.hiringai.mobile.ml.catalog.** { *; }

# ONNX Runtime
-keep class ai.onnxruntime.** { *; }

# llama.cpp
-keep class org.codeshipping.llamakotlin.** { *; }
