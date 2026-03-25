import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QSpinBox, QDoubleSpinBox, QFileDialog,
                             QTabWidget, QGroupBox, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
import matplotlib
matplotlib.use('Qt5Agg')

import torch

from visualization_core import generate_heatmap, generate_cam

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')


def save_config(config, merge=True):

    try:
        if merge:
            # 加载现有配置
            existing_config = load_config()
            # 合并新配置（新配置覆盖旧配置）
            existing_config.update(config)
            config_to_save = existing_config
        else:
            config_to_save = config

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            for key, value in config_to_save.items():
                f.write(f"{key}={value}\n")
    except Exception as e:
        print(f"保存配置失败: {e}")


def load_config():
    config = {}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
    except Exception as e:
        print(f"加载配置失败: {e}")
    return config


# 设置现代化暗色主题
def set_dark_theme(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.WindowText, QColor(232, 234, 255))
    palette.setColor(QPalette.Base, QColor(42, 42, 58))
    palette.setColor(QPalette.AlternateBase, QColor(48, 48, 64))
    palette.setColor(QPalette.ToolTipBase, QColor(42, 42, 58))
    palette.setColor(QPalette.ToolTipText, QColor(232, 234, 255))
    palette.setColor(QPalette.Text, QColor(232, 234, 255))
    palette.setColor(QPalette.Button, QColor(42, 42, 58))
    palette.setColor(QPalette.ButtonText, QColor(232, 234, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 180, 100))
    palette.setColor(QPalette.Link, QColor(77, 166, 255))
    palette.setColor(QPalette.Highlight, QColor(77, 166, 255))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)
    app.setStyle('Fusion')


class BatchProcessingThread(QThread):
    progress = pyqtSignal(int, str, str)  # current, total, filename
    finished = pyqtSignal(str, int, int)  # output_dir, success_count, failed_count
    error = pyqtSignal(str)

    def __init__(self, mode, file_groups, output_dir, params):
        super().__init__()
        self.mode = mode
        self.file_groups = file_groups
        self.output_dir = output_dir
        self.params = params

    def run(self):
        try:
            print(f"[调试] BatchProcessingThread 开始，模式: {self.mode}")
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)

            total = len(self.file_groups)
            success_count = 0
            failed_count = 0
            print(f"[调试] 共 {total} 个文件待处理")

            for idx, group in enumerate(self.file_groups, 1):
                filename = group.get('base_name', os.path.basename(group['tif']))
                self.progress.emit(idx, str(total), filename)

                try:
                    if self.mode == 'heatmap':
                        result, roi = generate_heatmap(
                            tif_path=group['tif'],
                            xml_path=group['xml'],
                            h5_path=group['h5'],
                            attention_path=group['attn'],
                            vis_level=self.params['vis_level'],
                            alpha=self.params['alpha'],
                            margin_percentage=self.params['margin_percentage'],
                            filter_thr=self.params['filter_thr'],
                            rel_roi_idx=self.params['rel_roi_idx'],
                            patch_size=512
                        )

                        if result is not None:
                            base_name = group.get('base_name', os.path.splitext(os.path.basename(group['tif']))[0])
                            output_path = os.path.join(self.output_dir, f'{base_name}_heatmap.png')
                            from PIL import Image
                            img = Image.fromarray(result)
                            img.save(output_path)
                            success_count += 1
                        else:
                            failed_count += 1

                    elif self.mode == 'cam':
                        print(f"[调试] 处理 CAM 文件 {idx}/{total}: {filename}")
                        if 'feat' not in group or not group['feat']:
                            print(f"[错误] 文件 {filename} 缺少特征文件")
                            failed_count += 1
                            continue

                        if 'cls' not in group or not group['cls']:
                            print(f"[错误] 文件 {filename} 缺少分类器文件")
                            failed_count += 1
                            continue

                        norm_method = self.params.get('cam_norm_method', 'Softmax')
                        class_idx = self.params.get('cam_class_idx', 0)
                        print(f"[调试] 调用 generate_cam，归一化方式: {norm_method}")
                        print(f"[调试] 类别索引: {class_idx}")
                        print(f"[调试] feat_path: {group['feat']}")
                        print(f"[调试] classifier_path: {group['cls']}")
                        print(f"[调试] attention_path: {group['attn']}")

                        result, roi = generate_cam(
                            tif_path=group['tif'],
                            xml_path=group['xml'],
                            h5_path=group['h5'],
                            attention_path=group['attn'],
                            feat_path=group['feat'],
                            classifier_path=group['cls'],
                            vis_level=self.params['vis_level'],
                            alpha=self.params['alpha'],
                            margin_percentage=self.params['margin_percentage'],
                            filter_thr=self.params['filter_thr'],
                            filter_thr_cam=self.params['filter_thr_cam'],
                            rel_roi_idx=self.params['rel_roi_idx'],
                            patch_size=512,
                            cam_norm_method=norm_method,
                            cam_class_idx=class_idx
                        )

                        if result is not None:
                            base_name = group.get('base_name', os.path.splitext(os.path.basename(group['tif']))[0])
                            output_path = os.path.join(self.output_dir, f'{base_name}_cam.png')
                            from PIL import Image
                            img = Image.fromarray(result)
                            img.save(output_path)
                            success_count += 1
                            print(f"[调试] 文件 {filename} 处理成功")
                        else:
                            failed_count += 1
                            print(f"[调试] 文件 {filename} 处理返回 None")

                except Exception as e:
                    print(f"[错误] 处理文件 {filename} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    failed_count += 1

            self.finished.emit(self.output_dir, success_count, failed_count)

        except Exception as e:
            print(f"[错误] BatchProcessingThread 运行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f'批量处理错误: {str(e)}')


def remove_separator_chars(filename, remove_suffixes=None):
    base_name, ext = os.path.splitext(filename)

    if remove_suffixes:
        for suffix in remove_suffixes:
            if base_name.lower().endswith(suffix.lower()):
                base_name = base_name[:-len(suffix)]
                break

    clean_name = base_name.replace('-', '').replace('_', '')

    return clean_name


def find_matching_files_separate_paths(tif_dir, xml_dir, h5_dir, attn_dir, feat_dir=None, cls_dir=None):
    file_groups = []

    required_dirs = {
        'TIF': tif_dir,
        'XML': xml_dir,
        'H5': h5_dir,
        'Attention': attn_dir
    }

    for dir_name, dir_path in required_dirs.items():
        if not os.path.exists(dir_path):
            print(f"[错误] {dir_name}目录不存在: {dir_path}")
            return []

    def get_files_by_extension(directory, extensions):
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) if f.lower().endswith(extensions)]

    print("[调试] 开始扫描文件...")
    tif_files = get_files_by_extension(tif_dir, ('.tif', '.tiff'))
    print(f"[调试] TIF文件: {len(tif_files)} 个")

    xml_files = get_files_by_extension(xml_dir, '.xml')
    print(f"[调试] XML文件: {len(xml_files)} 个")

    h5_files = get_files_by_extension(h5_dir, '.h5')
    print(f"[调试] H5文件: {len(h5_files)} 个")

    attn_files = get_files_by_extension(attn_dir, '.pt')
    print(f"[调试] Attention文件: {len(attn_files)} 个")

    feat_files = []
    if feat_dir:
        feat_files = get_files_by_extension(feat_dir, '.pt')
        print(f"[调试] Feature文件: {len(feat_files)} 个")

    cls_files = []
    if cls_dir:
        cls_files = get_files_by_extension(cls_dir, '.pt')
        print(f"[调试] Classifier文件: {len(cls_files)} 个")

    # 定义辅助函数：在文件列表中查找清理后文件名匹配的文件
    def find_matching_file(files_list, attn_clean_name, file_type="", remove_suffixes=None):
        for f in files_list:
            clean_name = remove_separator_chars(f, remove_suffixes)
            if clean_name in attn_clean_name:
                return f
        # 调试：打印所有候选文件的清理后名字
        if file_type and len(files_list) > 0:
            print(f"[调试] {file_type} 候选文件清理后的名字: {[remove_separator_chars(f, remove_suffixes) for f in files_list[:3]]}...")
            print(f"[调试] attention清理后的名字: {attn_clean_name}")
        return None

    print("[调试] 开始匹配文件...")
    match_count = 0
    fail_count = 0

    for attn_filename in attn_files:
        print(f"[调试] 处理attention文件: {attn_filename}")

        attn_clean_name = remove_separator_chars(attn_filename, ['_attention'])
        print(f"[调试] 清理后的attention文件名: {attn_clean_name}")

        tif_match = find_matching_file(tif_files, attn_clean_name, "TIF")
        xml_match = find_matching_file(xml_files, attn_clean_name, "XML")
        h5_match = find_matching_file(h5_files, attn_clean_name, "H5")
        feat_match = find_matching_file(feat_files, attn_clean_name, "Feature", ['_feat', '_features']) if feat_files else None
        cls_match = find_matching_file(cls_files, attn_clean_name, "Classifier", ['_cls', '_classifier']) if cls_files else None

        print(f"[调试] 匹配结果 - TIF: {tif_match if tif_match else '未找到'}, "
              f"XML: {xml_match if xml_match else '未找到'}, "
              f"H5: {h5_match if h5_match else '未找到'}, "
              f"Feature: {feat_match if feat_match else '未找到'}, "
              f"Classifier: {cls_match if cls_match else '未找到'}")

        required_optional_match = True
        if feat_dir and not feat_match:
            required_optional_match = False
        if cls_dir and not cls_match:
            required_optional_match = False

        if tif_match and xml_match and h5_match and required_optional_match:
            file_groups.append({
                'tif': os.path.join(tif_dir, tif_match),
                'xml': os.path.join(xml_dir, xml_match),
                'h5': os.path.join(h5_dir, h5_match),
                'attn': os.path.join(attn_dir, attn_filename),
                'feat': os.path.join(feat_dir, feat_match) if feat_match and feat_dir else None,
                'cls': os.path.join(cls_dir, cls_match) if cls_match and cls_dir else None,
                'base_name': os.path.splitext(attn_filename)[0]
            })
            match_count += 1
            print(f"[成功] 匹配成功: {attn_filename}")
        else:
            fail_count += 1
            fail_reason = []
            if not tif_match:
                fail_reason.append("TIF缺失")
            if not xml_match:
                fail_reason.append("XML缺失")
            if not h5_match:
                fail_reason.append("H5缺失")
            if feat_dir and not feat_match:
                fail_reason.append("Feature缺失")
            if cls_dir and not cls_match:
                fail_reason.append("Classifier缺失")
            print(f"[失败] 匹配失败: {attn_filename} - {', '.join(fail_reason)}")

    print(f"[总结] 匹配完成 - 成功: {match_count} 组, 失败: {fail_count} 组")
    print(f"[总结] 总共找到 {len(file_groups)} 组匹配的文件")

    return file_groups


class HeatmapTab(QWidget):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        param_group = QGroupBox("可视化参数")
        param_group.setStyleSheet("""
            QGroupBox {
                color: #e8eaf;
                font-weight: bold;
                border: 2px solid #3a3a42;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #2a2a3a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        param_layout = QVBoxLayout()

        param_row1 = QHBoxLayout()

        param_row1.addWidget(QLabel("可视化层级:"))
        self.vis_level = QSpinBox()
        self.vis_level.setRange(0, 10)
        self.vis_level.setValue(int(self.config.get('heatmap_vis_level', 3)))
        self.vis_level.valueChanged.connect(lambda: self.save_current_config())
        param_row1.addWidget(self.vis_level)

        param_row1.addWidget(QLabel("透明度(alpha):"))
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(float(self.config.get('heatmap_alpha', 0.3)))
        self.alpha.valueChanged.connect(lambda: self.save_current_config())
        param_row1.addWidget(self.alpha)

        param_layout.addLayout(param_row1)

        param_row2 = QHBoxLayout()

        param_row2.addWidget(QLabel("边距百分比:"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.5, 5.0)
        self.margin.setSingleStep(0.5)
        self.margin.setValue(float(self.config.get('heatmap_margin', 1.5)))
        self.margin.valueChanged.connect(lambda: self.save_current_config())
        param_row2.addWidget(self.margin)

        param_row2.addWidget(QLabel("过滤阈值:"))
        self.filter_thr = QDoubleSpinBox()
        self.filter_thr.setRange(0.0, 1.0)
        self.filter_thr.setSingleStep(0.1)
        self.filter_thr.setValue(float(self.config.get('heatmap_filter', 0.3)))
        self.filter_thr.valueChanged.connect(lambda: self.save_current_config())
        param_row2.addWidget(self.filter_thr)

        param_layout.addLayout(param_row2)

        param_row3 = QHBoxLayout()
        param_row3.addWidget(QLabel("ROI索引 (留空自动选择):"))
        self.roi_idx = QSpinBox()
        self.roi_idx.setRange(-1, 100)
        roi_val = self.config.get('heatmap_roi_idx', -1)
        self.roi_idx.setValue(int(roi_val) if roi_val != 'auto' else -1)
        self.roi_idx.setSpecialValueText("自动")
        self.roi_idx.valueChanged.connect(lambda: self.save_current_config())
        param_row3.addWidget(self.roi_idx)
        param_layout.addLayout(param_row3)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        batch_group = QGroupBox("批量处理")
        batch_group.setStyleSheet("""
            QGroupBox {
                color: #e8eaf;
                font-weight: bold;
                border: 2px solid #3a3a42;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #2a2a3a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        batch_layout = QVBoxLayout()

        self.tif_folder = self.create_folder_input("TIF文件:", batch_layout,
                                                self.config.get('heatmap_tif_dir', ''))

        self.xml_folder = self.create_folder_input("XML文件:", batch_layout,
                                                self.config.get('heatmap_xml_dir', ''))

        self.h5_folder = self.create_folder_input("H5文件:", batch_layout,
                                               self.config.get('heatmap_h5_dir', ''))

        self.attn_folder = self.create_folder_input("Attention文件:", batch_layout,
                                                  self.config.get('heatmap_attn_dir', ''))

        self.output_folder = self.create_folder_input("输出文件夹:", batch_layout,
                                                 self.config.get('heatmap_output_dir', ''))

        batch_btn_layout = QHBoxLayout()
        self.batch_btn = QPushButton("生成注意力热图")
        self.batch_btn.setMinimumHeight(40)
        self.batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #4da6ff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5c75ff;
            }
            QPushButton:pressed {
                background-color: #3b86e6;
            }
        """)
        self.batch_btn.clicked.connect(self.run_batch_processing)
        batch_btn_layout.addWidget(self.batch_btn)
        batch_layout.addLayout(batch_btn_layout)

        self.batch_progress_label = QLabel()
        self.batch_progress_label.setVisible(False)
        batch_layout.addWidget(self.batch_progress_label)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        self.setLayout(layout)

    def create_folder_input(self, label_text, layout, default_value=''):
        row = QHBoxLayout()
        input_field = QLineEdit()
        input_field.setPlaceholderText(f"选择{label_text}")
        input_field.setText(default_value)
        input_field.setStyleSheet("""
            QLineEdit {
                background-color: #353545;
                color: #e8eaf;
                border: 2px solid #3a3a42;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 2px solid #4da6ff;
            }
        """)
        input_field.textChanged.connect(lambda: self.save_current_config())
        btn = QPushButton("浏览...")
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4da6ff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5c75ff;
            }
            QPushButton:pressed {
                background-color: #3b86e6;
            }
        """)
        btn.clicked.connect(lambda: self.select_folder(input_field, label_text))
        row.addWidget(QLabel(label_text))
        row.addWidget(input_field)
        row.addWidget(btn)
        layout.addLayout(row)
        return input_field

    def select_folder(self, input_field, label_text):
        folder_path = QFileDialog.getExistingDirectory(self, f"选择{label_text}")
        if folder_path:
            input_field.setText(folder_path)

    def save_current_config(self):
        """保存当前配置"""
        config = {
            'heatmap_tif_dir': self.tif_folder.text(),
            'heatmap_xml_dir': self.xml_folder.text(),
            'heatmap_h5_dir': self.h5_folder.text(),
            'heatmap_attn_dir': self.attn_folder.text(),
            'heatmap_output_dir': self.output_folder.text(),
            'heatmap_vis_level': str(self.vis_level.value()),
            'heatmap_alpha': str(self.alpha.value()),
            'heatmap_margin': str(self.margin.value()),
            'heatmap_filter': str(self.filter_thr.value()),
            'heatmap_roi_idx': str(self.roi_idx.value()) if self.roi_idx.value() != -1 else 'auto'
        }
        save_config(config)

    def check_multihead_attention(self, first_attn_path):
        try:
            data = torch.load(first_attn_path)
            data = data.squeeze().cpu().numpy()

            if len(data.shape) > 1:
                # 检测到多头注意力，弹出对话框让用户选择
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("多头注意力检测")
                msg_box.setText(f"检测到多头注意力！\n\n注意力数据维度: {data.shape}\n\n请选择聚合方式：")
                msg_box.setInformativeText("Mean: 对所有头取平均值\nMax: 取所有头的最大值")
                msg_box.setStandardButtons(QMessageBox.NoButton)
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #2a2a3a;
                        color: #e8eaf;
                    }
                    QMessageBox QLabel {
                        color: #e8eaf;
                        background-color: #2a2a3a;
                    }
                """)

                mean_btn = msg_box.addButton("Mean", QMessageBox.ActionRole)
                max_btn = msg_box.addButton("Max", QMessageBox.ActionRole)
                mean_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4da6ff;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 10px 20px;
                        font-size: 14px;
                        font-weight: bold;
                        min-width: 100px;
                    }
                """)
                max_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #5c75ff;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 10px 20px;
                        font-size: 14px;
                        font-weight: bold;
                        min-width: 100px;
                    }
                """)
                msg_box.exec_()
                if msg_box.clickedButton() == mean_btn:
                    return 'mean'
                elif msg_box.clickedButton() == max_btn:
                    return 'max'
                else:
                    return 'mean'  # 默认
            else:
                return None  # 单头注意力，无需聚合
        except Exception as e:
            print(f"[错误] 检测多头注意力失败: {str(e)}")
            return 'mean'  # 出错时默认使用 mean

    def run_batch_processing(self):
        tif_dir = self.tif_folder.text()
        xml_dir = self.xml_folder.text()
        h5_dir = self.h5_folder.text()
        attn_dir = self.attn_folder.text()
        output_dir = self.output_folder.text()

        if not all([tif_dir, xml_dir, h5_dir, attn_dir, output_dir]):
            QMessageBox.warning(self, "警告", "请填写所有必填的文件夹路径！")
            return

        file_groups = find_matching_files_separate_paths(tif_dir, xml_dir, h5_dir, attn_dir)

        if not file_groups:
            QMessageBox.warning(self, "警告", "没有找到匹配的文件组！请检查文件命名是否一致。")
            return

        if file_groups:
            multihead_agg = self.check_multihead_attention(file_groups[0]['attn'])
        else:
            multihead_agg = None

        self.batch_btn.setEnabled(False)
        self.batch_progress_label.setVisible(True)
        self.batch_progress_label.setText("正在处理...")

        roi_idx = self.roi_idx.value() if self.roi_idx.value() != -1 else None

        params = {
            'vis_level': self.vis_level.value(),
            'alpha': self.alpha.value(),
            'margin_percentage': self.margin.value(),
            'filter_thr': self.filter_thr.value(),
            'rel_roi_idx': roi_idx,
            'multihead_agg': multihead_agg
        }

        self.batch_worker = BatchProcessingThread('heatmap', file_groups, output_dir, params)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.error.connect(self.on_batch_error)
        self.batch_worker.start()

    def on_batch_progress(self, current, total, filename):
        self.batch_progress_label.setText(f"正在处理: {filename} ({current}/{total})")

    def on_batch_finished(self, output_dir, success_count, failed_count):
        self.batch_btn.setEnabled(True)
        self.batch_progress_label.setVisible(False)
        QMessageBox.information(
            self, "批量处理完成",
            f"处理完成！\n成功: {success_count}\n失败: {failed_count}\n输出目录: {output_dir}"
        )

    def on_batch_error(self, error_msg):
        self.batch_btn.setEnabled(True)
        self.batch_progress_label.setVisible(False)
        QMessageBox.critical(self, "批量处理错误", error_msg)


class CamTab(QWidget):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        param_group = QGroupBox("可视化参数")
        param_group.setStyleSheet("""
            QGroupBox {
                color: #e8eaf;
                font-weight: bold;
                border: 2px solid #3a3a42;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #2a2a3a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        param_layout = QVBoxLayout()

        param_row1 = QHBoxLayout()

        param_row1.addWidget(QLabel("可视化层级:"))
        self.vis_level = QSpinBox()
        self.vis_level.setRange(0, 10)
        self.vis_level.setValue(int(self.config.get('cam_vis_level', 3)))
        self.vis_level.valueChanged.connect(lambda: self.save_current_config())
        param_row1.addWidget(self.vis_level)

        param_row1.addWidget(QLabel("透明度(alpha):"))
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(float(self.config.get('cam_alpha', 0.3)))
        self.alpha.valueChanged.connect(lambda: self.save_current_config())
        param_row1.addWidget(self.alpha)

        param_layout.addLayout(param_row1)

        param_row2 = QHBoxLayout()

        param_row2.addWidget(QLabel("边距百分比:"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.5, 5.0)
        self.margin.setSingleStep(0.5)
        self.margin.setValue(float(self.config.get('cam_margin', 1.5)))
        self.margin.valueChanged.connect(lambda: self.save_current_config())
        param_row2.addWidget(self.margin)

        param_row2.addWidget(QLabel("注意力阈值:"))
        self.filter_thr = QDoubleSpinBox()
        self.filter_thr.setRange(0.0, 1.0)
        self.filter_thr.setSingleStep(0.1)
        self.filter_thr.setValue(float(self.config.get('cam_filter', 0.2)))
        self.filter_thr.valueChanged.connect(lambda: self.save_current_config())
        param_row2.addWidget(self.filter_thr)

        param_layout.addLayout(param_row2)

        param_row3 = QHBoxLayout()

        param_row3.addWidget(QLabel("CAM阈值:"))
        self.filter_thr_cam = QDoubleSpinBox()
        self.filter_thr_cam.setRange(0.0, 1.0)
        self.filter_thr_cam.setSingleStep(0.1)
        self.filter_thr_cam.setValue(float(self.config.get('cam_filter_cam', 0.3)))
        self.filter_thr_cam.valueChanged.connect(lambda: self.save_current_config())
        param_row3.addWidget(self.filter_thr_cam)

        param_row3.addWidget(QLabel("ROI索引 (留空自动选择):"))
        self.roi_idx = QSpinBox()
        self.roi_idx.setRange(-1, 100)
        roi_val = self.config.get('cam_roi_idx', -1)
        self.roi_idx.setValue(int(roi_val) if roi_val != 'auto' else -1)
        self.roi_idx.setSpecialValueText("自动")
        self.roi_idx.valueChanged.connect(lambda: self.save_current_config())
        param_row3.addWidget(self.roi_idx)

        param_layout.addLayout(param_row3)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        batch_group = QGroupBox("批量处理")
        batch_group.setStyleSheet("""
            QGroupBox {
                color: #e8eaf;
                font-weight: bold;
                border: 2px solid #3a3a42;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #2a2a3a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        batch_layout = QVBoxLayout()

        self.tif_folder = self.create_folder_input("TIF文件:", batch_layout,
                                                self.config.get('cam_tif_dir', ''))

        self.xml_folder = self.create_folder_input("XML文件:", batch_layout,
                                                self.config.get('cam_xml_dir', ''))

        self.h5_folder = self.create_folder_input("H5文件:", batch_layout,
                                               self.config.get('cam_h5_dir', ''))

        self.attn_folder = self.create_folder_input("Attention文件:", batch_layout,
                                                  self.config.get('cam_attn_dir', ''))


        self.feat_folder = self.create_folder_input("特征文件:", batch_layout,
                                                 self.config.get('cam_feat_dir', ''))

        self.cls_folder = self.create_folder_input("分类器权重文件:", batch_layout,
                                                self.config.get('cam_cls_dir', ''))

        self.output_folder = self.create_folder_input("输出文件夹:", batch_layout,
                                                 self.config.get('cam_output_dir', ''))

        batch_btn_layout = QHBoxLayout()
        self.batch_btn = QPushButton("生成CAM图")
        self.batch_btn.setMinimumHeight(40)
        self.batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #5c75ff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6b85ff;
            }
            QPushButton:pressed {
                background-color: #4c65df;
            }
        """)
        self.batch_btn.clicked.connect(self.run_batch_processing)
        batch_btn_layout.addWidget(self.batch_btn)
        batch_layout.addLayout(batch_btn_layout)

        self.batch_progress_label = QLabel()
        self.batch_progress_label.setVisible(False)
        batch_layout.addWidget(self.batch_progress_label)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        self.setLayout(layout)

    def create_folder_input(self, label_text, layout, default_value=''):
        row = QHBoxLayout()
        input_field = QLineEdit()
        input_field.setPlaceholderText(f"选择{label_text}")
        input_field.setText(default_value)
        input_field.setStyleSheet("""
            QLineEdit {
                background-color: #353545;
                color: #e8eaf;
                border: 2px solid #3a3a42;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 2px solid #4da6ff;
            }
        """)

        input_field.textChanged.connect(lambda: self.save_current_config())
        btn = QPushButton("浏览...")
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4da6ff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5c75ff;
            }
            QPushButton:pressed {
                background-color: #3b86e6;
            }
        """)
        btn.clicked.connect(lambda: self.select_folder(input_field, label_text))
        row.addWidget(QLabel(label_text))
        row.addWidget(input_field)
        row.addWidget(btn)
        layout.addLayout(row)
        return input_field

    def select_folder(self, input_field, label_text):
        folder_path = QFileDialog.getExistingDirectory(self, f"选择{label_text}")
        if folder_path:
            input_field.setText(folder_path)

    def save_current_config(self):

        config = {
            'cam_tif_dir': self.tif_folder.text(),
            'cam_xml_dir': self.xml_folder.text(),
            'cam_h5_dir': self.h5_folder.text(),
            'cam_attn_dir': self.attn_folder.text(),
            'cam_feat_dir': self.feat_folder.text(),
            'cam_cls_dir': self.cls_folder.text(),
            'cam_output_dir': self.output_folder.text(),
            'cam_vis_level': str(self.vis_level.value()),
            'cam_alpha': str(self.alpha.value()),
            'cam_margin': str(self.margin.value()),
            'cam_filter': str(self.filter_thr.value()),
            'cam_filter_cam': str(self.filter_thr_cam.value()),
            'cam_roi_idx': str(self.roi_idx.value()) if self.roi_idx.value() != -1 else 'auto'
        }
        save_config(config)

    def check_cam_norm_method(self):

        try:
            print("[调试] 开始创建归一化方式选择对话框...")
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("选择归一化方式")
            msg_box.setText("请选择 CAM 归一化方式：")
            msg_box.setInformativeText("Softmax: 使用 Softmax 归一化\nMin-Max: 使用 Min-Max 归一化")
            msg_box.setStandardButtons(QMessageBox.NoButton)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2a2a3a;
                    color: #e8eaf;
                }
                QMessageBox QLabel {
                    color: #e8eaf;
                    background-color: #2a2a3a;
                }
                QPushButton {
                    background-color: #3a3a42;
                    color: #e8eaf;
                    border: 2px solid #4da6ff;
                    border-radius: 6px;
                    padding: 10px 20px;
                    min-width: 100px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #4a4a52;
                    border-color: #5ab6ff;
                }
            """)
            print("[调试] 添加按钮...")
            softmax_btn = msg_box.addButton("Softmax", QMessageBox.ActionRole)
            minmax_btn = msg_box.addButton("Min-Max", QMessageBox.ActionRole)
            print("[调试] 显示对话框...")
            msg_box.exec_()
            print("[调试] 获取用户选择...")
            if msg_box.clickedButton() == softmax_btn:
                print("[调试] 用户选择了 Softmax")
                return 'Softmax'
            elif msg_box.clickedButton() == minmax_btn:
                print("[调试] 用户选择了 Min-Max")
                return 'Min-Max'
            else:
                print("[调试] 用户未选择，使用默认 Softmax")
                return 'Softmax'
        except Exception as e:
            print(f"[错误] check_cam_norm_method 出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return 'Softmax'

    def check_cam_class_idx(self):
        try:
            print("[调试] 开始创建类别选择对话框...")
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("选择CAM类别")
            msg_box.setText("请选择 CAM 取值列：")
            msg_box.setInformativeText("类别0: 第一列\n类别1: 第二列")
            msg_box.setStandardButtons(QMessageBox.NoButton)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2a2a3a;
                    color: #e8eaf;
                }
                QMessageBox QLabel {
                    color: #e8eaf;
                    background-color: #2a2a3a;
                }
                QPushButton {
                    background-color: #3a3a42;
                    color: #e8eaf;
                    border: 2px solid #4da6ff;
                    border-radius: 6px;
                    padding: 10px 20px;
                    min-width: 100px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #4a4a52;
                    border-color: #5ab6ff;
                }
            """)
            print("[调试] 添加按钮...")
            class0_btn = msg_box.addButton("类别0", QMessageBox.ActionRole)
            class1_btn = msg_box.addButton("类别1", QMessageBox.ActionRole)
            print("[调试] 显示对话框...")
            msg_box.exec_()
            print("[调试] 获取用户选择...")
            if msg_box.clickedButton() == class0_btn:
                print("[调试] 用户选择了 类别0")
                return 0
            elif msg_box.clickedButton() == class1_btn:
                print("[调试] 用户选择了 类别1")
                return 1
            else:
                print("[调试] 用户未选择，使用默认 类别0")
                return 0
        except Exception as e:
            print(f"[错误] check_cam_class_idx 出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0

    def run_batch_processing(self):
        try:
            print("[调试] run_batch_processing 开始...")
            tif_dir = self.tif_folder.text()
            xml_dir = self.xml_folder.text()
            h5_dir = self.h5_folder.text()
            attn_dir = self.attn_folder.text()
            feat_dir = self.feat_folder.text()
            cls_dir = self.cls_folder.text()
            output_dir = self.output_folder.text()

            if not all([tif_dir, xml_dir, h5_dir, attn_dir, feat_dir, cls_dir, output_dir]):
                QMessageBox.warning(self, "警告", "请填写所有必填的文件夹路径！")
                return

            print("[调试] 查找匹配的文件...")
            file_groups = find_matching_files_separate_paths(tif_dir, xml_dir, h5_dir, attn_dir, feat_dir, cls_dir)
            print(f"[调试] 找到 {len(file_groups)} 个文件组")

            if not file_groups:
                QMessageBox.warning(self, "警告", "没有找到匹配的文件组！请检查文件命名是否一致。")
                return

            print("[调试] 调用 check_cam_norm_method...")
            norm_method = self.check_cam_norm_method()
            print(f"[调试] 归一化方式: {norm_method}")

            print("[调试] 调用 check_cam_class_idx...")
            class_idx = self.check_cam_class_idx()
            print(f"[调试] 类别索引: {class_idx}")

            self.batch_btn.setEnabled(False)
            self.batch_progress_label.setVisible(True)
            self.batch_progress_label.setText("正在处理...")

            roi_idx = self.roi_idx.value() if self.roi_idx.value() != -1 else None

            params = {
                'vis_level': self.vis_level.value(),
                'alpha': self.alpha.value(),
                'margin_percentage': self.margin.value(),
                'filter_thr': self.filter_thr.value(),
                'filter_thr_cam': self.filter_thr_cam.value(),
                'rel_roi_idx': roi_idx,
                'cam_norm_method': norm_method,
                'cam_class_idx': class_idx
            }
            print(f"[调试] 参数: {params}")

            print("[调试] 创建批量处理线程...")
            self.batch_worker = BatchProcessingThread('cam', file_groups, output_dir, params)
            self.batch_worker.progress.connect(self.on_batch_progress)
            self.batch_worker.finished.connect(self.on_batch_finished)
            self.batch_worker.error.connect(self.on_batch_error)
            print("[调试] 启动线程...")
            self.batch_worker.start()
            print("[调试] 线程已启动")
        except Exception as e:
            print(f"[错误] run_batch_processing 出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"批量处理失败: {str(e)}")
            self.batch_btn.setEnabled(True)
            self.batch_progress_label.setVisible(False)

    def on_batch_progress(self, current, total, filename):
        self.batch_progress_label.setText(f"正在处理: {filename} ({current}/{total})")

    def on_batch_finished(self, output_dir, success_count, failed_count):
        self.batch_btn.setEnabled(True)
        self.batch_progress_label.setVisible(False)
        QMessageBox.information(
            self, "批量处理完成",
            f"处理完成！\n成功: {success_count}\n失败: {failed_count}\n输出目录: {output_dir}"
        )

    def on_batch_error(self, error_msg):
        self.batch_btn.setEnabled(True)
        self.batch_progress_label.setVisible(False)
        QMessageBox.critical(self, "批量处理错误", error_msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 加载配置
        self.config = load_config()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("可视化软件主程序")
        self.setGeometry(100, 100, 900, 700)

        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        title_row = QHBoxLayout()
        title_label = QLabel("Attention-based MIL Visualization")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(26)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #4da6ff; padding: 10px; background-color: rgba(42, 42, 58, 100); border-radius: 8px;")
        title_row.addWidget(title_label)

        help_btn = QPushButton("?")
        help_btn.setFixedSize(40, 40)
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #4da6ff;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5c75ff;
            }
            QPushButton:pressed {
                background-color: #3b86e6;
            }
        """)
        help_btn.clicked.connect(self.show_help)
        title_row.addWidget(help_btn)
        main_layout.addLayout(title_row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3a3a42;
                background: #2a2a3a;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #353545;
                color: #e8eaf;
                padding: 12px 24px;
                margin-right: 4px;
                margin-bottom: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #4da6ff;
                color: white;
            }
            QTabBar::tab:hover {
                background: #5c75ff;
                color: white;
            }
        """)

        self.heatmap_tab = HeatmapTab(self.config)
        self.cam_tab = CamTab(self.config)

        self.tabs.addTab(self.heatmap_tab, "热力图可视化")
        self.tabs.addTab(self.cam_tab, "CAM可视化")

        main_layout.addWidget(self.tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def show_help(self):
        help_text = """
        <h3>可视化软件使用说明</h3>

        <h4>软件简介</h4>
        <p>本软件用于病理图像的可视化，支持生成注意力热力图和CAM（Class Activation Map）图。</p>

        <h4>文件匹配规则</h4>
        <p>软件会自动匹配不同文件夹中对应的文件。匹配时会忽略文件名中的下划线(_)和连字符(-)。</p>
        <p>例如：<br>
        &nbsp;&nbsp;• attention文件：<b>tumor_001_attention.pt</b> → <b>tumor001attention</b><br>
        &nbsp;&nbsp;• TIF文件：<b>tumor-001.tiff</b> → <b>tumor001</b><br>
        &nbsp;&nbsp;• 匹配成功，因为 "tumor001" 在 "tumor001attention" 中</p>

        <h4>参数说明</h4>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
        <tr><td><b>可视化层级</b></td><td>WSI的下采样层级 (0-10)，值越大图像越粗糙</td></tr>
        <tr><td><b>透明度(alpha)</b></td><td>热力图叠加在原图上的透明度 (0.0-1.0)</td></tr>
        <tr><td><b>边距百分比</b></td><td>ROI区域扩展的边距百分比</td></tr>
        <tr><td><b>注意力阈值</b></td><td>过滤低注意力patch的阈值 (0.0-1.0)</td></tr>
        <tr><td><b>CAM阈值</b></td><td>过滤低CAM值的阈值，仅CAM模式可用 (0.0-1.0)</td></tr>
        <tr><td><b>ROI索引</b></td><td>指定要可视化的ROI索引，留空则自动选择最大ROI</td></tr>
        </table>

        <h4>热力图可视化</h4>
        <p>需要提供的文件：<br>
        &nbsp;&nbsp;• TIF文件：病理图像文件 (.tif/.tiff)<br>
        &nbsp;&nbsp;• XML文件：ROI标注文件 (.xml)<br>
        &nbsp;&nbsp;• H5文件：patch坐标文件 (.h5)<br>
        &nbsp;&nbsp;• Attention文件：注意力权重文件 (.pt)</p>

        <h4>CAM可视化</h4>
        <p>需要提供的文件（比热力图多两个）：<br>
        &nbsp;&nbsp;• TIF文件：病理图像文件<br>
        &nbsp;&nbsp;• XML文件：ROI标注文件<br>
        &nbsp;&nbsp;• H5文件：patch坐标文件<br>
        &nbsp;&nbsp;• Attention文件：注意力权重文件<br>
        &nbsp;&nbsp;• 特征文件：patch特征文件 (.pt)<br>
        &nbsp;&nbsp;• 分类器权重文件：分类器权重文件 (.pt)</p>

        <h4>使用步骤</h4>
        <ol>
        <li>选择"热力图可视化"或"CAM可视化"标签页</li>
        <li>配置可视化参数</li>
        <li>分别选择各类型文件所在的文件夹</li>
        <li>选择输出文件夹</li>
        <li>点击"生成热力图"或"生成CAM图"按钮开始批量处理</li>
        </ol>
        """

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("帮助")
        msg_box.setText(help_text)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2a2a3a;
                color: #e8eaf;
            }
            QMessageBox QLabel {
                color: #e8eaf;
                background-color: #2a2a3a;
            }
        """)
        msg_box.exec_()


def main():
    app = QApplication(sys.argv)
    set_dark_theme(app)
    window = MainWindow()

    screen = app.desktop().screenGeometry()
    size = window.geometry()
    x = (screen.width() - size.width()) // 2
    y = (screen.height() - size.height()) // 2
    window.move(x, y)

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
