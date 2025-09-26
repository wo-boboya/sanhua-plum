import os
import shutil
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def get_valid_files(image_dir: str, label_dir: str, image_extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Tuple[
    List[str], Dict[str, str]]:
    """获取所有有效（存在对应标签）的图片文件名及扩展名映射"""
    valid_files = []
    image_extensions = [ext.lower() for ext in image_extensions]

    # 构建图片文件字典，键为基础名，值为扩展名
    image_files_dict = {}
    for f in os.listdir(image_dir):
        if any(f.lower().endswith(ext) for ext in image_extensions):
            base_name = os.path.splitext(f)[0]
            image_files_dict[base_name] = os.path.splitext(f)[1]

    # 构建标签文件集合
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    # 找出有效的文件
    valid_files = []
    valid_extensions = {}
    for base_name, ext in image_files_dict.items():
        if base_name in label_files:
            valid_files.append(base_name)
            valid_extensions[base_name] = ext

    print(f"\n警告：共跳过 {len(image_files_dict) - len(valid_files)} 个无标签文件")
    return valid_files, valid_extensions


def split_dataset(valid_files: List[str], ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> Tuple[
    List[str], List[str], List[str]]:
    """按比例划分数据集（随机打乱）"""
    total = len(valid_files)
    ratios = [round(r * total) for r in ratio]
    if sum(ratios) != total:
        ratios[-1] = total - sum(ratios[:-1])

    random.shuffle(valid_files)
    train_files = valid_files[:ratios[0]]
    val_files = valid_files[ratios[0]:ratios[0] + ratios[1]]
    test_files = valid_files[ratios[0] + ratios[1]:]

    return train_files, val_files, test_files


def copy_file(base_name: str, src_image_dir: str, src_label_dir: str, dest_image_dir: str, dest_label_dir: str,
              extensions_map: Dict[str, str]):
    """复制单个文件"""
    try:
        # 获取图片扩展名
        img_ext = extensions_map[base_name]

        # 复制图片
        img_src = os.path.join(src_image_dir, f"{base_name}{img_ext}")
        img_dest = os.path.join(dest_image_dir, f"{base_name}{img_ext}")
        shutil.copy2(img_src, img_dest)

        # 复制标签
        label_src = os.path.join(src_label_dir, f"{base_name}.txt")
        label_dest = os.path.join(dest_label_dir, f"{base_name}.txt")
        shutil.copy2(label_src, label_dest)

        return True
    except Exception as e:
        print(f"警告：复制文件 {base_name} 失败: {e}")
        return False


def copy_files(file_list: List[str], src_image_dir: str, src_label_dir: str, dest_dir: str, subdir: str,
               extensions_map: Dict[str, str], max_workers: int = 8):
    """使用多线程复制文件到目标目录"""
    image_dest = os.path.join(dest_dir, subdir, "images")
    label_dest = os.path.join(dest_dir, subdir, "labels")
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)

    print(f"\n复制 {subdir} 数据集...")
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有复制任务
        futures = {executor.submit(copy_file, base_name, src_image_dir, src_label_dir, image_dest, label_dest,
                                   extensions_map): base_name
                   for base_name in file_list}

        # 使用tqdm显示进度
        for future in tqdm(futures, desc=f"{subdir} 文件"):
            success = future.result()
            if success:
                success_count += 1

    print(f"{subdir} 数据集复制完成：成功 {success_count}/{len(file_list)}")


def main():
    # 配置参数（可根据实际情况修改）
    CONFIG = {
        "src_image_dir": r"Path to the file",
        "src_label_dir": r"Path to the file",
        "dest_root": r"Path to the file",
        "split_ratio": (0.7, 0.2, 0.1),
        "image_extensions": ['.jpg', '.jpeg', '.png'],
        "max_workers": 8  # 多线程复制的最大工作线程数
    }

    # 安装检查（如果未安装tqdm，提示用户安装）
    try:
        from tqdm import tqdm
    except ImportError:
        print("错误：缺少tqdm库，请先安装：pip install tqdm")
        return

    # 获取有效文件列表和扩展名映射
    valid_files, extensions_map = get_valid_files(
        CONFIG["src_image_dir"],
        CONFIG["src_label_dir"],
        CONFIG["image_extensions"]
    )
    print(f"找到 {len(valid_files)} 个有效匹配文件")

    # 划分数据集
    train_files, val_files, test_files = split_dataset(valid_files, CONFIG["split_ratio"])
    print(f"划分结果：训练集{len(train_files)}，验证集{len(val_files)}，测试集{len(test_files)}")

    # 调整复制顺序：先test，再val，最后train
    copy_files(test_files, CONFIG["src_image_dir"], CONFIG["src_label_dir"], CONFIG["dest_root"], "test",
               extensions_map, CONFIG["max_workers"])
    copy_files(val_files, CONFIG["src_image_dir"], CONFIG["src_label_dir"], CONFIG["dest_root"], "val", extensions_map,
               CONFIG["max_workers"])
    copy_files(train_files, CONFIG["src_image_dir"], CONFIG["src_label_dir"], CONFIG["dest_root"], "train",
               extensions_map, CONFIG["max_workers"])

    print("\n数据集划分完成！")


if __name__ == "__main__":
    main()