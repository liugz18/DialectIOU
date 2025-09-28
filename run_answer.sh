#!/bin/bash
# 生成带时间戳的日志文件名（格式：YYYY-MM-DD_HH-MM-SS.log）
logfile="Answer_$(date +'%Y-%m-%d_%H-%M-%S').log"

# 执行 Python 脚本，并捕获所有输出（标准输出 + 标准错误）
{
    echo "======== 开始执行: $(date +'%Y-%m-%d %H:%M:%S') ========"
    python answer.py  # 运行你的 Python 脚本
    echo "======== 结束执行: $(date +'%Y-%m-%d %H:%M:%S') ========"
} 2>&1 | tee "$logfile"  # 同时输出到终端和日志文件

# 提示日志路径
echo "日志已保存至: $PWD/$logfile"