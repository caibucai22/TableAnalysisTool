<div align="center">
<img src=https://img.shields.io/badge/License-MIT-blue.svg>
    <img src=https://img.shields.io/github/stars/caibucai22/TableAnalysisTool.svg>
<img src=https://img.shields.io/badge/Python-3.8%2B-green.svg>
<img src=https://img.shields.io/badge/Release-preparing-brightgreen.svg >
</div>



# TableAnalysisTool 

智能表格分析工具 

<img src="resources/app.png" alt="UI Demo" style="zoom:42%;" />

## :sparkles: 核心特性
- **多算法融合**: 集成PPOcr、PPStructure、Table-Transformer、Cycle-Centernet等前沿模型
- **高效统分引擎**: 支持对勾识别、数字统计等定制化功能
- **跨平台支持**: 提供Windows/Linux双平台CPU/GPU推理方案
- **可视化交互**: 基于PyQt5的友好用户界面
- **配置分离**: APP配置、模型配置、自定义业务配置

## :rocket: 快速开始
### 环境配置

```bash
conda create -n tableAnalysis python=3.8 -y
conda activate tableAnalysis

# 一定要按照以下顺序安装相关工具
pip install pyqt5-sip==12.11.0
pip install pyqt5==5.15.9
pip install pyqt5-tools

# torch-gpu
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# paddle-gpu (windows)
python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
# paddle-gpu (linux)
python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 版本一致
pip install transformers==4.46.3
pip install accelerate==1.0.1
pip install hf-transfer

pip install numpy==1.24.4 opencv-python==4.6.0.66 pillow==10.4.0 pandas==2.0.2
pip install openvino==2024.4.0
pip install onnxruntime==1.19.2

# 安装其他依赖
pip install -r requirements.txt
```

### 启动运行

```python
python app_v2.py
```



## :hugs: 模型矩阵

### 1. 表格定位 (Table Detection)

<div style="width:100%">

| 模型名称          | 算法框架         | 功能说明                           | 权重来源                          |
|--------------------|------------------|----------------------------------|---------------------------------------|
| PP-Structure       | PaddleDetection  | 仅开放表格定位功能进行表格区域检测 |  |
| Table-Transformer  | DETR             | 表格区域检测                |   |
| yolov8m-table-extraction | Yolov8 | 表格区域检测                       |  |

</div>

### 2. 表格结构识别 (Structure Recognition)

<div style="width:100%">

| 模型名称          | 算法框架         | 功能说明                           | 权重来源                            |
|--------------------|------------------|----------------------------------|---------------------------------------|
| Table-Transformer | DETR | 支持行/列/表头/跨行单元格/检测 |  |
| Cycle-CenterNet    | CenterNet        | 支持有线/无线表格单元格检测 |          |
| RT-DETR-Cell_det | RT-DETR          | 表格单元格检测            |                          |

</div>

### 3. 表格文字识别 (Text Recognition)

<div style="width:100%">

| 模型名称          | 算法框架         | 功能说明                           | 权重来源                         |
|--------------------|------------------|----------------------------------|---------------------------------------|
| PP-OCRv4           | CRNN+SVTR        | 仅启用文字识别        |                          |

</div>

### 4. 其他模型

<div style="width:100%">

| 模型名称          | 算法框架         | 功能说明                           | 权重来源                            |
|--------------------|------------------|----------------------------------|---------------------------------------|
| YOLOv8-cls/det   | YOLO             | 特殊符号分类/检测(√/×/○)            |          |

</div>

## :page_facing_up:文档资源

- [OCR实践—PaddleOCR](https://blog.csdn.net/csy1021/article/details/144518451?spm=1001.2014.3001.5502)
- [OCR实践-Table-Transformer](https://blog.csdn.net/csy1021/article/details/144742974?spm=1001.2014.3001.5502)
- [OCR实践-问卷表格统计](https://blog.csdn.net/csy1021/article/details/144777615?spm=1001.2014.3001.5501)



## :earth_asia: 联系方式

| 平台          | 联系方式                                                     |
| :------------ | :----------------------------------------------------------- |
| GitHub        | [提交issue](https://github.com/caibucai22/TableAnalysisTool/issues) |
| 社群 :penguin: | 392784757                                                    |
| 邮箱 :mailbox: | caibucai22@gmail.com                                         |

