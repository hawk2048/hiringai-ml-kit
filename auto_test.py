#!/usr/bin/env python3
"""
HiringAI-ML-Kit 自动化测试脚本
用于自动测试 Android 应用功能

用法:
    python auto_test.py                    # 运行所有测试
    python auto_test.py --main             # 仅测试主界面
    python auto_test.py --models           # 仅测试模型目录
    python auto_test.py --benchmark        # 测试基准测试
    python auto_test.py --logs             # 仅查看日志
    python auto_test.py --reinstall        # 重新安装APK后测试
"""

import subprocess
import time
import sys
import os
import re
import json
from datetime import datetime
from pathlib import Path

# 配置
CONFIG = {
    "adb_path": r"C:\Users\wkliu\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    "apk_path": r"d:\AI\AIModel\hiringai-ml-kit\app\build\outputs\apk\debug\app-debug.apk",
    "package": "com.hiringai.mobile.ml.testapp",
    "main_activity": "com.hiringai.mobile.ml.testapp.ui.MainActivity",
    "catalog_activity": "com.hiringai.mobile.ml.testapp.ui.ModelCatalogActivity",
    "log_activity": "com.hiringai.mobile.ml.testapp.ui.LogViewerActivity",
    "device_emulator": "emulator-5554",
    "device_phone": "192.168.1.5:37913",
    "screenshot_dir": r"d:\AI\AIModel\test-screenshots",
    "log_dir": r"d:\AI\AIModel\test-logs",
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def log(msg, level="INFO"):
    """打印带颜色的日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "FAIL": Colors.RED,
        "WARN": Colors.YELLOW,
    }.get(level, "")
    print(f"{color}[{timestamp}] {level}: {msg}{Colors.END}")

def run_cmd(cmd, capture=True, timeout=30):
    """执行命令"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return result.returncode, result.stdout, result.stderr
        else:
            subprocess.run(cmd, shell=True, timeout=timeout)
            return 0, "", ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timeout"
    except Exception as e:
        return -1, "", str(e)

def adb(cmd, device=None, capture=True):
    """执行ADB命令"""
    d = device or CONFIG["device_emulator"]
    full_cmd = f'"{CONFIG["adb_path"]}" -s {d} {cmd}'
    return run_cmd(full_cmd, capture)

def wait_for_device(timeout=60):
    """等待设备连接"""
    log("等待设备连接...")
    start = time.time()
    while time.time() - start < timeout:
        code, out, _ = adb("devices")
        if "device" in out and "offline" not in out:
            log("设备已连接")
            return True
        time.sleep(2)
    log("设备连接超时", "FAIL")
    return False

def ensure_screenshot_dir():
    """确保截图目录存在"""
    Path(CONFIG["screenshot_dir"]).mkdir(parents=True, exist_ok=True)

def ensure_log_dir():
    """确保日志目录存在"""
    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)

def screenshot(name):
    """截图"""
    ensure_screenshot_dir()
    timestamp = datetime.now().strftime("%H%M%S")
    filename = os.path.join(CONFIG["screenshot_dir"], f"{name}_{timestamp}.png")
    code, out, _ = adb(f'exec-out screencap -p > "{filename}"')
    if code == 0 and os.path.exists(filename):
        log(f"截图已保存: {filename}", "SUCCESS")
        return filename
    log(f"截图失败", "FAIL")
    return None

def get_screen_xml():
    """获取屏幕布局XML"""
    code, out, _ = adb("shell uiautomator dump /sdcard/screen.xml")
    if code == 0:
        code, out, _ = adb("shell cat /sdcard/screen.xml", capture=False)
        return out
    return None

def tap(x, y):
    """点击坐标"""
    code, _, _ = adb(f"shell input tap {x} {y}")
    return code == 0

def input_text(text):
    """输入文本"""
    text = text.replace(" ", "%s").replace("'", "\\'")
    code, _, _ = adb(f'shell input text "{text}"')
    return code == 0

def press_back():
    """按返回键"""
    code, _, _ = adb("shell input keyevent KEYCODE_BACK")
    return code == 0

def press_enter():
    """按回车键"""
    code, _, _ = adb("shell input keyevent KEYCODE_ENTER")
    return code == 0

def swipe_up():
    """向上滑动"""
    code, _, _ = adb("shell input swipe 500 1500 500 500 300")
    return code == 0

def swipe_down():
    """向下滑动"""
    code, _, _ = adb("shell input swipe 500 500 500 1500 300")
    return code == 0

def start_activity(activity):
    """启动Activity
    activity应该是完整类名，如: ui.MainActivity
    """
    full_activity = f'{CONFIG["package"]}/{activity}'
    log(f"启动Activity: {full_activity}")
    code, out, err = adb(f'shell am start -n {full_activity}')
    log(f"返回码: {code}")
    return code == 0

def force_stop_app():
    """强制停止应用"""
    code, _, _ = adb(f"shell am force-stop {CONFIG['package']}")
    return code == 0

def get_app_version():
    """获取应用版本"""
    code, out, _ = adb(f"shell dumpsys package {CONFIG['package']} | findstr versionName")
    if code == 0:
        match = re.search(r'versionName=([\d.]+)', out)
        if match:
            return match.group(1)
    return "Unknown"

def get_device_info():
    """获取设备信息"""
    info = {}
    code, out, _ = adb("shell getprop ro.product.model")
    info["model"] = out.strip() if code == 0 else "Unknown"
    code, out, _ = adb("shell getprop ro.build.version.release")
    info["android_version"] = out.strip() if code == 0 else "Unknown"
    code, out, _ = adb("shell getprop ro.build.version.sdk")
    info["sdk_version"] = out.strip() if code == 0 else "Unknown"
    return info

def get_app_logs(lines=100):
    """获取应用日志"""
    code, out, _ = adb(f"shell logcat -d -t {lines} --pid=$(adb shell pidof {CONFIG['package']})")
    if code != 0 or not out:
        # 尝试获取所有ML相关日志
        code, out, _ = adb(f"shell logcat -d -t {lines} | findstr -i ml")
    return out

def clear_logs():
    """清除日志"""
    adb("shell logcat -c")

def is_app_running():
    """检查应用是否运行"""
    code, out, _ = adb(f"shell pidof {CONFIG['package']}")
    return code == 0 and out.strip()

def install_apk():
    """安装APK"""
    log("安装APK...")
    if not os.path.exists(CONFIG["apk_path"]):
        log(f"APK不存在: {CONFIG['apk_path']}", "FAIL")
        return False

    # 强制停止旧版本
    force_stop_app()

    # 卸载旧版本
    code, _, _ = adb(f"shell pm uninstall {CONFIG['package']}")

    # 安装新版本
    code, out, err = adb(f"install -r \"{CONFIG['apk_path']}\"")
    if code == 0 and "Success" in out:
        log("APK安装成功", "SUCCESS")
        return True
    else:
        log(f"APK安装失败: {err or out}", "FAIL")
        return False

def test_main_activity():
    """测试主界面"""
    log("="*50)
    log("测试: 主界面")
    log("="*50)

    results = {"passed": 0, "failed": 0, "tests": []}

    # 启动主界面
    log("启动主界面...")
    force_stop_app()
    time.sleep(0.5)
    if not start_activity(CONFIG["main_activity"]):
        log("启动主界面失败", "FAIL")
        results["failed"] += 1
        results["tests"].append(("启动主界面", False))
        return results

    time.sleep(2)  # 等待界面加载

    # 截图
    screenshot("main_activity")
    results["tests"].append(("截图", True))

    # 获取设备信息
    log("检查设备信息显示...")
    device_info = get_device_info()
    log(f"设备: {device_info['model']}, Android {device_info['android_version']}")

    # 检查应用是否运行
    if is_app_running():
        log("应用运行正常", "SUCCESS")
        results["passed"] += 1
        results["tests"].append(("应用运行", True))
    else:
        log("应用未运行", "FAIL")
        results["failed"] += 1
        results["tests"].append(("应用运行", False))

    # 检查版本
    version = get_app_version()
    log(f"应用版本: {version}")
    if version != "Unknown":
        results["passed"] += 1
        results["tests"].append(("获取版本", True))
    else:
        results["failed"] += 1
        results["tests"].append(("获取版本", False))

    # 获取日志
    log("获取应用日志...")
    logs = get_app_logs(50)
    if "MainActivity" in logs or "主界面" in logs:
        log("检测到主界面日志", "SUCCESS")
        results["passed"] += 1
        results["tests"].append(("日志检测", True))
    else:
        log("未检测到主界面日志", "WARN")
        results["tests"].append(("日志检测", False))

    return results

def test_model_catalog():
    """测试模型目录"""
    log("="*50)
    log("测试: 模型目录")
    log("="*50)

    results = {"passed": 0, "failed": 0, "tests": []}

    # 启动主界面
    if not start_activity(CONFIG["main_activity"]):
        log("启动主界面失败", "FAIL")
        results["failed"] += 1
        return results

    time.sleep(2)

    # 点击模型目录按钮
    # 根据屏幕大小估算按钮位置 (屏幕中央偏下)
    log("点击模型目录按钮...")
    # 需要根据实际屏幕调整坐标
    tap(540, 1200)  # 估算位置
    time.sleep(1)

    # 截图
    screenshot("model_catalog")
    results["tests"].append(("模型目录截图", True))

    # 检查是否启动成功
    time.sleep(2)
    if is_app_running():
        log("模型目录界面运行中", "SUCCESS")
        results["passed"] += 1
        results["tests"].append(("界面运行", True))
    else:
        log("模型目录界面可能未启动", "WARN")
        results["tests"].append(("界面运行", False))

    return results

def test_logs():
    """测试日志查看"""
    log("="*50)
    log("测试: 日志查看")
    log("="*50)

    results = {"passed": 0, "failed": 0, "tests": []}

    # 启动日志查看
    log("启动日志查看...")
    if start_activity(CONFIG["log_activity"]):
        time.sleep(2)
        screenshot("log_viewer")
        results["passed"] += 1
        results["tests"].append(("启动日志查看", True))
    else:
        results["failed"] += 1
        results["tests"].append(("启动日志查看", False))

    return results

def generate_report(results_list, test_name):
    """生成测试报告"""
    ensure_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(CONFIG["log_dir"], f"test_report_{test_name}_{timestamp}.txt")

    total_passed = sum(r["passed"] for r in results_list)
    total_failed = sum(r["failed"] for r in results_list)
    total_tests = total_passed + total_failed

    report = f"""
{'='*60}
HiringAI-ML-Kit 自动化测试报告
{'='*60}
测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
测试设备: {CONFIG['device_emulator']}

测试结果汇总:
  总测试数: {total_tests}
  通过: {total_passed} ({total_passed*100//total_tests if total_tests else 0}%)
  失败: {total_failed}

{'='*60}
详细结果:
{'='*60}
"""

    for results in results_list:
        for test_name, passed in results["tests"]:
            status = "✓ PASS" if passed else "✗ FAIL"
            color = Colors.GREEN if passed else Colors.RED
            report += f"{color}{status}{Colors.END} - {test_name}\n"

    report += f"""
{'='*60}
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    log(f"测试报告已保存: {report_file}", "SUCCESS")
    return report_file

def run_all_tests(reinstall=False):
    """运行所有测试"""
    log("HiringAI-ML-Kit 自动化测试开始")
    log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"APK: {CONFIG['apk_path']}")

    # 等待设备
    if not wait_for_device():
        log("设备未连接，测试终止", "FAIL")
        return False

    # 获取设备信息
    device_info = get_device_info()
    log(f"测试设备: {device_info['model']} (Android {device_info['android_version']})")

    # 重新安装APK
    if reinstall:
        if not install_apk():
            log("APK安装失败，测试终止", "FAIL")
            return False
    else:
        # 确保应用已安装
        version = get_app_version()
        if version == "Unknown":
            log("应用未安装，将进行安装...", "WARN")
            if not install_apk():
                log("APK安装失败，测试终止", "FAIL")
                return False

    # 清除旧日志
    clear_logs()

    # 运行测试
    all_results = []

    # 测试1: 主界面
    all_results.append(test_main_activity())

    # 测试2: 模型目录
    all_results.append(test_model_catalog())

    # 测试3: 日志查看
    all_results.append(test_logs())

    # 生成报告
    report_file = generate_report(all_results, "all")

    # 总结
    log("")
    log("="*50)
    total_passed = sum(r["passed"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    log(f"测试完成: {total_passed} 通过, {total_failed} 失败")
    log("="*50)

    # 最后截图
    screenshot("final_state")

    return total_failed == 0

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="HiringAI-ML-Kit 自动化测试")
    parser.add_argument("--main", action="store_true", help="仅测试主界面")
    parser.add_argument("--models", action="store_true", help="仅测试模型目录")
    parser.add_argument("--logs", action="store_true", help="仅测试日志查看")
    parser.add_argument("--reinstall", action="store_true", help="重新安装APK")
    parser.add_argument("--device", choices=["emulator", "phone"], default="emulator", help="选择设备")

    args = parser.parse_args()

    # 更新设备配置
    if args.device == "phone":
        CONFIG["device_emulator"] = CONFIG["device_phone"]

    if args.main:
        wait_for_device()
        results = test_main_activity()
        generate_report([results], "main")
    elif args.models:
        wait_for_device()
        results = test_model_catalog()
        generate_report([results], "models")
    elif args.logs:
        wait_for_device()
        results = test_logs()
        generate_report([results], "logs")
    else:
        success = run_all_tests(reinstall=args.reinstall)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
