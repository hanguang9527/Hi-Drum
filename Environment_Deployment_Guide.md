# Hi_Drum 项目环境一键部署指南

本文档用于指导如何在全新的 Ubuntu 服务器上，快速、无损地还原与原开发环境完全一致的 `drum` 虚拟环境。

## 1. 前置要求

在目标服务器上运行自动化脚本前，请确保满足以下基础条件：
*   **操作系统**：Ubuntu 20.04/22.04（或兼容的 Linux 发行版）
*   **基础环境**：已安装 Anaconda 或 Miniconda
*   **网络**：服务器需具备外网访问权限（脚本已配置清华源加速国内网络）

## 2. 部署文件清单

请确保将以下两个核心文件一同拷贝到新服务器的同一目录下：
1.  `requirements_ubuntu.txt`：锁定了原 `drum` 环境（Python 3.10.19）所有依赖的精确版本。
2.  `deploy_env.sh`：自动化部署 Bash 脚本。

## 3. 一键部署步骤

在新服务器上，打开终端并按照以下步骤操作：

**步骤一：进入文件所在目录**
```bash
cd /你的/项目/路径
```

**步骤二：赋予脚本执行权限**
```bash
chmod +x deploy_env.sh
```

**步骤三：执行自动化部署脚本**
```bash
./deploy_env.sh
```

> **注意**：脚本执行期间会自动创建名为 `drum` 的 Conda 虚拟环境，并自动安装依赖库，整个过程可能需要几分钟（视网络情况而定）。请耐心等待终端输出 `✅ 环境部署完成！` 的提示。

## 4. 激活与验证

脚本运行完毕后，在终端执行以下命令激活环境：

```bash
conda activate drum
```

（可选）你可以运行以下命令验证核心依赖是否安装成功：
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```
如果输出正常且不报错，则代表 `drum` 环境已完美复刻。

## 5. 常见问题 (FAQ)

**Q1：提示 `conda: command not found` 怎么办？**
A：说明新服务器尚未安装 Conda 或者未将 Conda 加入环境变量。请先安装 Miniconda 或执行 `source ~/.bashrc` 刷新环境变量。

**Q2：如果我想改一个环境名字怎么办？**
A：使用文本编辑器（如 `vim` 或 `nano`）打开 `deploy_env.sh`，将第 8 行的 `ENV_NAME="drum"` 修改为你想要的名称（如 `ENV_NAME="drum_v2"`），然后再执行脚本即可。

**Q3：下载依赖包速度极慢或超时怎么办？**
A：脚本默认已开启清华镜像源加速。如果目标服务器在海外，请打开 `deploy_env.sh`，将第 38 行的 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 参数删除即可。
