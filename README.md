# CPath-Visualization
针对病理图像的热图可视化软件

下载压缩包即可使用

launcher.py和main.py搭建运行GUI，visualization_core.py含有核心程序generate_heatmap和generate_cam，目前有生成基于注意力的MIL可视化注意力热图和CAM肿瘤预测概率图两个功能，支持多头注意力模型。

需要提供tif格式原片，xml文件和h5文件的相关标注信息。

Simply download the zip file to get started.

launcher.py and main.py are used to set up and run the GUI, whilst visualization_core.py contains the core functions generate_heatmap and generate_cam. Currently, there are two features available: generating attention-based MIL visualisations (attention heatmaps) and CAM tumour prediction probability maps, with support for multi-head attention models.

You will need to provide the original slides in TIF format, along with the relevant annotation information in XML and H5 files.
