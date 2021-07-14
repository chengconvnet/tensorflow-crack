from pathlib import Path


def mkdir(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

def load_current_env():
    '''获取当前目录的决对路径，且添加 Python 环境'''
    import os
    # 获取根目录
    try:  # colab 目录
        from google.colab import drive

        root = '/content/drive'  # colab 训练
        drive.mount(root)  # 挂载磁盘
        root = f'{root}/MyDrive'
    except:
        root = '.'  # 本地目录
    # 添加当前路径为 Python 包所在环境
    # 保证 colab 可以获取自定义的 .py 文件
    os.chdir(root)
