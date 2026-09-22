# 音频文件替换与播放模式说明

本文档说明如何在本项目中管理和切换音频文件（MP3, WAV, FLAC）。

## 1. 目前的播放机制

在 `index.html` 中，每个演示（Demo）的音频播放器部分都配置了三种格式的音频源。浏览器会按照代码中的顺序依次尝试加载，直到找到支持且存在的格式。

默认顺序如下：
1. **MP3** (`audio/mpeg`) - 最优先
2. **WAV** (`audio/wav`) - 备选
3. **FLAC** (`audio/flac`) - 最后尝试

代码示例：
```html
<audio id="demo_p1_multitrack-audio">
  <!-- 优先尝试播放 MP3 -->
  <source src="demo/p1/multitrack.mp3" type="audio/mpeg">
  <!-- 如果 MP3 无法播放，尝试 WAV -->
  <source src="demo/p1/multitrack.wav" type="audio/wav">
  <!-- 最后尝试 FLAC -->
  <source src="demo/p1/multitrack.flac" type="audio/flac">
  Your browser does not support the audio tag.
</audio>
```

## 2. 如何切换音频文件

### 方法 A：直接替换文件（推荐）
你不需要修改代码。只需要将你的音频文件（例如 `.wav` 或 `.flac`）放入对应的文件夹（如 `demo/p1/`），并确保文件名一致（例如 `multitrack.wav`）。

- 如果你只想用 WAV：确保文件夹里有 `multitrack.wav`，并且**删除**或**重命名**同名的 `multitrack.mp3`。浏览器找不到 MP3 时，会自动播放 WAV。
- 如果你同时有 MP3 和 WAV：浏览器会优先播放 MP3。

### 方法 B：修改代码优先级
如果你希望在 MP3 存在的情况下，依然强制优先播放 WAV，请修改 `index.html` 中 `<audio>` 标签内的顺序，将 WAV 放在第一位。

修改后：
```html
<audio id="demo_p1_multitrack-audio">
  <!-- 此时优先播放 WAV -->
  <source src="demo/p1/multitrack.wav" type="audio/wav">
  <source src="demo/p1/multitrack.mp3" type="audio/mpeg">
  <source src="demo/p1/multitrack.flac" type="audio/flac">
</audio>
```

## 3. 常见问题
- **文件路径**：请确保文件位于 `demo/pX/` (例如 `demo/p1/`, `demo/p2/`) 目录下。
- **文件名**：文件名必须与 HTML 中引用的名字一致（如 `multitrack`, `piano`, `leadsheet`）。
