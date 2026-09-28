# 本地运行库

该目录保存 Co-π-tree 运行时使用的两个本地库：

- `overcooked_ai/`：基于 Overcooked-AI，并包含本项目所需的布局、规划接口、兼容性调整和目标选择改进。
- `stable-baselines/`：用于加载和运行 SP、PBT、FCP、MEP、COLE、BC 等预训练搭档模型的本地依赖。

这里不是 ProAgent 与 CausalPlan 两个基线方法的代码集合。ProAgent 的适配与调用代码位于 `src1/test_Overcooked/proagent/`；本仓库没有提供 CausalPlan 的可执行实现。图表数据中出现这两个名称，只表示论文比较对象。

两个本地库保留各自的上游许可证。项目对库文件做过少量修改，因此运行本项目时应安装本目录中的版本，避免替换为同名的 PyPI 或其他仓库版本：

```bash
pip install -e ./lib/overcooked_ai
pip install -e ./lib/stable-baselines
```
