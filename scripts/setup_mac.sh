#!/bin/bash
# QubitvaleTrading — Mac 一键部署每日任务（R5）
# 用法：在 Mac 终端运行  bash scripts/setup_mac.sh
# 做三件事：1) 建仓库内 .venv 并装依赖（不依赖系统 python3 是否有 pandas）
#          2) 安装 launchd 定时任务（每天本地 08:10，睡眠错过会在唤醒后补跑）
#          3) 立即试跑一次并显示结果
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
echo "repo: $REPO"

# 1) venv
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt
mkdir -p logs

# 2) launchd
PLIST="$HOME/Library/LaunchAgents/com.qubitvale.paperdaily.plist"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.qubitvale.paperdaily</string>
  <key>ProgramArguments</key><array>
    <string>${REPO}/.venv/bin/python</string>
    <string>-m</string><string>scripts.run_paper_daily</string>
  </array>
  <key>WorkingDirectory</key><string>${REPO}</string>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>8</integer><key>Minute</key><integer>10</integer>
  </dict>
  <key>StandardOutPath</key><string>${REPO}/logs/paper.log</string>
  <key>StandardErrorPath</key><string>${REPO}/logs/paper.log</string>
</dict></plist>
EOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "launchd installed: com.qubitvale.paperdaily (daily 08:10 local; runs on wake if missed)"

# 3) 冻结期望带基准（只在首次；已冻结则跳过）——6 周 gate 的口径自此固定
./.venv/bin/python -m scripts.freeze_baseline || true

# 4) 试跑
./.venv/bin/python -m scripts.run_paper_daily || {
  echo "!! first run failed — check logs/paper.log"; exit 1; }
echo "OK. 每日日志: $REPO/logs/paper.log ；状态: ./.venv/bin/python -m scripts.paper_status"
