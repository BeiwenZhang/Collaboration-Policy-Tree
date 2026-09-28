# 第三方代码与模型

- `src1/test_Overcooked/` 的部分评估接口和基线实现基于 [ProAgent](https://github.com/PKU-Alignment/ProAgent)，其许可为 MIT，Copyright (c) 2023 CeyaoZhang。原始许可见 [ProAgent LICENSE](https://github.com/PKU-Alignment/ProAgent/blob/main/LICENSE)。
- `lib/overcooked_ai/` 基于 [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai)，并包含 Co-π-tree 所需的少量修改；许可和版权声明保存在 [`lib/overcooked_ai/LICENSE`](lib/overcooked_ai/LICENSE)。
- `human_ai_demo/` 的服务器、浏览器界面和图形素材基于 [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) demo，并为 Co-π-tree 人机实验作了适配；原项目采用 MIT 许可，Copyright (c) 2019 Center for Human-Compatible AI。
- `human_ai_demo/server/static/vendor/` 收录固定版本的 Bootstrap 4.3.1、Popper.js 1.14.7、Socket.IO Client 2.3.0 和 Phaser 3.23.0，供实验界面离线加载。它们均采用 MIT 许可；对应版权与许可全文保存在该目录的 `licenses/` 子目录。
- `lib/stable-baselines/` 基于 [Stable-Baselines](https://github.com/hill-a/stable-baselines)，并作为本项目加载旧版预训练搭档模型的本地依赖；许可和版权声明保存在 [`lib/stable-baselines/LICENSE`](lib/stable-baselines/LICENSE)。
- AI 搭档的预训练权重由 [COLE-Platform 模型文件夹](https://drive.google.com/drive/folders/1s88a_muyG6pVlfcKDKop6R1Fhxr8dcGH) 单独提供，不包含在本仓库；下载者应遵守模型来源处的使用条款。

仓库根目录的 [`LICENSE`](LICENSE) 适用于 Co-π-tree 项目代码；第三方文件保留各自的版权与许可声明。

## ProAgent MIT 许可声明

Copyright (c) 2023 CeyaoZhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
