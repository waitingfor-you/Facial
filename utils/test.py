import os
import shutil


def clear_directory(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    # num = cnt
    for filename in os.listdir(folder_path):
        # if num == 0:
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子文件夹
            # num -= 1
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    file_count = len(os.listdir(folder_path))
    print(f"现在文件夹内容数量为:{file_count}")

clear_directory('../cache')