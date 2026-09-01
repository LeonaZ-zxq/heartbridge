# 原始素材放这里

把博主的视频 / 音频 / 录屏文件直接拖进这个文件夹。

- 支持 mp4 / mov / m4a / mp3 / wav / aac（其它格式 ffmpeg 认识的也行）
- **这个目录已被 .gitignore**，不会进 Git、不会上传、不会出现在公开仓库里
- 文件名随意，但建议带上博主名，方便之后核对来源：
  `某某某_如何陪伴抑郁的人.mp4`

放好之后跑：

    source .venv/bin/activate
    python scripts/ingest.py --dir data/raw_media

它会自己做转写（本地 Whisper），你**不需要**先手动跑 whisper。
