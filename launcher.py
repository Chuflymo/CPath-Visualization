# -*- coding: utf-8 -*-
"""
Pathology Visualization Software Launcher
"""
import sys
import os
import subprocess
import configparser
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


def check_dependencies(python_exe):
    required = ['numpy', 'matplotlib', 'cv2', 'openslide', 'h5py', 'torch', 'shapely', 'PIL', 'PyQt5']
    missing = []
    for pkg in required:
        try:
            subprocess.run([python_exe, '-c', f'import {pkg}'], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception:
            missing.append(pkg)
    return missing


def get_config_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


def get_saved_config():
    config_dir = get_config_dir()
    config_file = os.path.join(config_dir, 'launcher_config.ini')
    if os.path.exists(config_file):
        try:
            config = configparser.ConfigParser()
            config.read(config_file, encoding='utf-8')
            if 'Python' in config:
                path = config.get('Python', 'path', fallback='')
                if path and os.path.exists(path):
                    return path
        except Exception as e:
            print(f"Error reading config: {e}")
    return None


def save_config_file(python_path):
    config_dir = get_config_dir()
    config_file = os.path.join(config_dir, 'launcher_config.ini')
    config = configparser.ConfigParser()
    config['Python'] = {'path': python_path}
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            config.write(f)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_launch_command(python_exe):
    config_dir = get_config_dir()
    main_py = os.path.join(config_dir, 'main.py')
    return python_exe, main_py, config_dir


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.config_dir = get_config_dir()
        self.config_file = os.path.join(self.config_dir, 'launcher_config.ini')
        self.setup_ui()

    def setup_ui(self):
        self.colors = {
            'bg': '#1a1a2d',
            'card': '#2d2d35',
            'primary': '#4a9eff',
            'primary_dark': '#3a7fec',
            'secondary': '#7b68ee',
            'text_main': '#e2e8f0',
            'text_light': '#b5c5d5',
            'text_lighter': '#d4e4d9',
            'border': '#3d3d45',
            'success': '#10b981',
        }

        self.root.title("启动器")
        self.root.geometry("850x570")
        self.root.configure(bg=self.colors['bg'])

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 850) // 2
        y = (screen_height - 570) // 2
        self.root.geometry(f"+{x}+{y}")

        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=30, pady=20)

        title_frame = tk.Frame(main_container, bg=self.colors['card'], relief='flat', bd=0)
        title_frame.pack(fill='x', pady=(0, 15))

        title_label = tk.Label(
            title_frame,
            text="Attention-based MIL Visualization",
            font=('Segoe UI', 22, 'bold'),
            fg=self.colors['text_main'],
            bg=self.colors['card']
        )
        title_label.pack(pady=(12, 5))

        subtitle_label = tk.Label(
            title_frame,
            text="基于注意力的MIL模型可视化软件",
            font=('Segoe UI', 10),
            fg=self.colors['text_light'],
            bg=self.colors['card']
        )
        subtitle_label.pack(pady=(0, 12))

        config_frame = tk.Frame(main_container, bg=self.colors['card'], relief='flat', bd=0)
        config_frame.pack(fill='x', pady=(0, 12))

        card_title = tk.Label(
            config_frame,
            text="Python Environment",
            font=('Segoe UI', 12, 'bold'),
            fg=self.colors['text_main'],
            bg=self.colors['card']
        )
        card_title.pack(anchor='w', padx=20, pady=(12, 8))

        input_frame = tk.Frame(config_frame, bg=self.colors['card'])
        input_frame.pack(fill='x', padx=20, pady=(0, 6))

        tk.Label(
            input_frame,
            text="Python Path:",
            font=('Segoe UI', 9),
            fg=self.colors['text_light'],
            bg=self.colors['card']
        ).pack(side='left', padx=(0, 10))

        saved_python = get_saved_config()
        self.python_path_var = tk.StringVar(value=saved_python if saved_python else '')

        python_entry = tk.Entry(
            input_frame,
            textvariable=self.python_path_var,
            font=('Segoe UI', 10),
            width=50,
            relief='flat',
            bg='#3a3a45',
            fg=self.colors['text_main'],
            insertbackground=self.colors['bg'],
            selectbackground=self.colors['primary_dark']
        )
        python_entry.pack(side='left', padx=0)

        btn_frame = tk.Frame(config_frame, bg=self.colors['card'])
        btn_frame.pack(fill='x', padx=20, pady=(6, 8))

        browse_btn = tk.Button(
            btn_frame,
            text="浏览",
            font=('Segoe UI', 9),
            bg=self.colors['secondary'],
            fg='white',
            relief='flat',
            cursor='hand2',
            padx=18,
            pady=6,
            command=self.browse_python
        )
        browse_btn.pack(side='left', padx=(0, 8))

        help_btn = tk.Button(
            btn_frame,
            text="帮助",
            font=('Segoe UI', 9),
            bg=self.colors['primary'],
            fg='white',
            relief='flat',
            cursor='hand2',
            padx=18,
            pady=6,
            command=self.show_usage_help
        )
        help_btn.pack(side='left', padx=(0, 8))

        deps_frame = tk.Frame(main_container, bg=self.colors['card'], relief='flat', bd=0)
        deps_frame.pack(fill='x', pady=(0, 12))

        card_title = tk.Label(
            deps_frame,
            text="Required Dependencies",
            font=('Segoe UI', 12, 'bold'),
            fg=self.colors['text_main'],
            bg=self.colors['card']
        )
        card_title.pack(anchor='w', padx=20, pady=(12, 8))

        deps_list = """必需依赖库:
- numpy matplotlib
- opencv-python
- openslide-python
- h5py
- torch
- shapely
- Pillow
- PyQt5

安装命令:
pip install -r requirements.txt

或使用Conda:
For openslide: conda install -c conda-forge openslide openslide-python
For others: numpy matplotlib opencv-python openslide-python h5py torch shapely Pillow PyQt5"""

        deps_text = scrolledtext.ScrolledText(
            deps_frame,
            font=('Consolas', 10),
            fg=self.colors['text_light'],
            bg=self.colors['card'],
            height=7,
            width=0,
            wrap='none'
        )
        deps_text.pack(fill='both', expand=True, padx=20, pady=(0, 10))
        deps_text.insert(1.0, deps_list)
        deps_text.config(state='disabled')

        action_frame = tk.Frame(main_container, bg=self.colors['card'], relief='flat', bd=0)
        action_frame.pack(fill='x', pady=(0, 0))

        btn_frame = tk.Frame(action_frame, bg=self.colors['card'])
        btn_frame.pack(fill='x', padx=20, pady=(12, 12))

        save_btn = tk.Button(
            btn_frame,
            text="保存路径",
            font=('Segoe UI', 9),
            bg=self.colors['success'],
            fg='white',
            relief='flat',
            cursor='hand2',
            width=5,
            padx=24,
            pady=8,
            command=self.save_config
        )

        save_btn.pack(side='left', padx=6)

        self.launch_btn = tk.Button(
            btn_frame,
            text="启动主程序",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            relief='flat',
            cursor='hand2',
            padx=28,
            pady=10,
            command=self.launch_main
        )

        self.launch_btn.pack(side='left', padx=(200, 0))

        status_frame = tk.Frame(self.root, bg=self.colors['border'], height=32)
        status_frame.pack(side='bottom', fill='x')

        self.status_var = tk.StringVar(value="就绪")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=('Segoe UI', 9),
            fg=self.colors['text_lighter'],
            bg=self.colors['border']
        )
        status_label.pack(expand=True)

    def browse_python(self):
        file_path = filedialog.askopenfilename(
            title="Select Python Executable",
            filetypes=[("Python Executable", "*.exe"), ("All Files", "*.*")]
        )
        if file_path:
            self.python_path_var.set(file_path)

    def show_usage_help(self):
        help_text = """如何配置Python路径

方法1：使用"浏览"按钮
1. 点击"浏览..."按钮
2. 找到您的conda环境目录，例如：
   C:\\Users\\用户名\\.conda\\envs\\环境名\\
3. 选择 pythonw.exe 文件（注意：选择pythonw.exe，不是python.exe）

方法2：手动输入路径
1. 在"Python路径"输入框中直接输入完整路径
2. 路径示例：
   C:\\Users\\用户名\\.conda\\envs\\torch2.5.1\\pythonw.exe


配置完成后，点击"保存配置"按钮保存设置。
"""

        help_window = tk.Toplevel(self.root)
        help_window.title("帮助")
        help_window.geometry("640x500")
        help_window.configure(bg=self.colors['card'])
        help_window.resizable(False, False)

        screen_width = help_window.winfo_screenwidth()
        screen_height = help_window.winfo_screenheight()
        x = (screen_width - 640) // 2
        y = (screen_height - 500) // 2
        help_window.geometry(f"+{x}+{y}")

        content_frame = tk.Frame(help_window, bg=self.colors['card'])
        content_frame.pack(fill='both', expand=True, padx=30, pady=30)

        help_label = tk.Label(
            content_frame,
            text=help_text,
            justify='left',
            font=('Consolas', 11),
            fg=self.colors['text_main'],
            bg=self.colors['card']
        )
        help_label.pack(fill='both', expand=True)

        btn_frame = tk.Frame(help_window, bg=self.colors['card'])
        btn_frame.pack(fill='x', padx=30, pady=20)

        close_btn = tk.Button(
            btn_frame,
            text="关闭",
            font=('Segoe UI', 10),
            bg=self.colors['primary'],
            fg='white',
            relief='flat',
            cursor='hand2',
            padx=32,
            pady=10,
            command=help_window.destroy
        )
        close_btn.pack()

    def save_config(self):
        python_path = self.python_path_var.get().strip()
        if not python_path:
            messagebox.showwarning("警告", "请先配置Python路径!")
            return
        if save_config_file(python_path):
            messagebox.showinfo("成功", "配置已保存!")
        else:
            messagebox.showerror("错误", "保存配置失败，请检查写入权限。")

    def launch_main(self):
        python_path = self.python_path_var.get().strip()
        if not python_path:
            messagebox.showwarning("警告", "请先配置Python路径!")
            return

        if not os.path.exists(python_path):
            messagebox.showerror("错误", f"找不到Python可执行文件:\n{python_path}")
            return

        if python_path.lower().endswith('python.exe'):
            python_path = python_path[:-4] + 'pythonw.exe'

        python_exe, main_py, work_dir = get_launch_command(python_path)

        # 检查 main.py 是否存在
        if not os.path.exists(main_py):
            messagebox.showerror("错误", f"找不到main.py:\n{main_py}")
            self.status_var.set("启动失败")
            self.launch_btn.config(state='normal', cursor='hand2')
            self.root.update()
            return

        # 检查 visualization_core.py 是否存在
        core_py = os.path.join(work_dir, 'visualization_core.py')
        if not os.path.exists(core_py):
            messagebox.showwarning("警告", f"找不到visualization_core.py:\n{core_py}\n\n请确保该文件在同一目录下。")

        self.status_var.set("正在检查Python环境...")
        self.launch_btn.config(state='disabled', cursor='watch')
        self.root.update()

        try:
            missing = check_dependencies(python_path)
        except Exception as e:
            self.status_var.set("依赖检查失败")
            messagebox.showwarning("警告", f"依赖检查失败: {str(e)}")
            self.launch_btn.config(state='normal', cursor='hand2')
            self.root.update()
            return

        if missing:
            self.status_var.set("缺少依赖")
            missing_str = '\n'.join(f"  - {pkg}" for pkg in missing)
            reply = messagebox.askyesno(
                "依赖检查",
                f"缺少以下依赖库:\n\n{missing_str}\n\n是否继续?",
                icon='warning'
            )
            if not reply:
                self.status_var.set("就绪")
                self.launch_btn.config(state='normal', cursor='hand2')
                self.root.update()
                return

        self.status_var.set("正在启动主程序...")
        self.root.update()

        # 尝试直接启动，不隐藏窗口以便查看错误
        try:
            print(f"[Launcher] Python路径: {python_exe}")
            print(f"[Launcher] main.py路径: {main_py}")
            print(f"[Launcher] 工作目录: {work_dir}")

            # 使用 CREATE_NEW_PROCESS_GROUP 避免子进程被影响
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008

            # 先尝试读取 main.py 确保没有语法错误
            try:
                with open(main_py, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, main_py, 'exec')
                print(f"[Launcher] main.py 语法检查通过")
            except SyntaxError as e:
                error_msg = f"main.py 语法错误:\n文件: {e.filename}\n行号: {e.lineno}\n错误: {e.msg}"
                messagebox.showerror("错误", error_msg)
                self.status_var.set("启动失败")
                self.launch_btn.config(state='normal', cursor='hand2')
                self.root.update()
                return

            # 启动进程
            process = subprocess.Popen(
                [python_exe, main_py],
                cwd=work_dir,
                creationflags=CREATE_NEW_PROCESS_GROUP
            )

            # 等待一下检查进程状态
            import time
            time.sleep(2.0)

            if process.poll() is None:
                # 进程还在运行
                self.status_var.set("主程序已启动!")
                self.launch_btn.config(state='normal', cursor='hand2')
                print(f"[Launcher] 主程序启动成功 (PID: {process.pid})")
            else:
                # 进程已退出，获取错误输出
                stdout, stderr = process.communicate(timeout=1)
                error_info = ""
                if stdout:
                    error_info += f"标准输出:\n{stdout.decode('utf-8', errors='ignore')}\n\n"
                if stderr:
                    error_info += f"标准错误:\n{stderr.decode('utf-8', errors='ignore')}\n\n"

                error_msg = f"启动失败 (退出代码: {process.returncode})\n\n{error_info}"
                messagebox.showerror("错误", error_msg)
                self.status_var.set("启动失败")
                self.launch_btn.config(state='normal', cursor='hand2')

        except subprocess.TimeoutExpired:
            messagebox.showerror("错误", "启动超时")
            self.status_var.set("启动失败")
            self.launch_btn.config(state='normal', cursor='hand2')
        except FileNotFoundError as e:
            messagebox.showerror("错误", f"找不到文件:\n{str(e)}")
            self.status_var.set("启动失败")
            self.launch_btn.config(state='normal', cursor='hand2')
        except Exception as e:
            import traceback
            error_msg = f"启动失败:\n{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
            messagebox.showcritical("错误", error_msg)
            self.status_var.set("启动失败")
            self.launch_btn.config(state='normal', cursor='hand2')

        self.root.update()


def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
