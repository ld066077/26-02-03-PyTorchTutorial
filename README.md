深度学习实践课程学习代码
https://www.bilibili.com/video/BV1Y7411d7Ys

## 本地运行环境

这是一个 PyTorch 学习教程项目。下载代码到本地之后，建议先创建独立的 Python 虚拟环境，再安装 `requirements.txt` 中锁定的依赖包。

### 1. 进入项目目录

```bash
cd 26-02-03-PyTorchTutorial
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
```

### 3. 激活虚拟环境

Linux / macOS:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
venv\Scripts\activate.bat
```

### 4. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

当前 `requirements.txt` 锁定的是 `torch==1.12.0+cu102`，也就是 CUDA 10.2 版本的 PyTorch。没有 GPU 的机器也可以安装并使用 CPU 运行；如果机器有 NVIDIA GPU，并且驱动兼容 CUDA 10.2，则可以使用 GPU 加速。

### 5. 运行示例代码

激活虚拟环境后，可以进入对应章节目录运行示例，例如：

```bash
python 06-Logistic-Regression/train.py
python 08-Using-DataLoader/titanic-exercise/train_titanic.py
```

如果运行 Titanic 数据下载脚本，需要能访问 KaggleHub 相关资源：

```bash
python 08-Using-DataLoader/titanic-exercise/download_titanic.py
```

### 6. 后续维护依赖

如果在虚拟环境中安装了新的库，并希望把当前环境完整记录到 `requirements.txt`，可以运行：

```bash
venv/bin/python -m pip freeze > requirements.txt
```

Windows 可以使用：

```powershell
.\venv\Scripts\python -m pip freeze > requirements.txt
```
