import os
import xml.etree.ElementTree as ET

def update_paths_and_filenames(annotations_dir, old_prefix, new_prefix, encoding="utf-8"):
    """
    更新标注文件中的路径和文件名。

    参数:
        annotations_dir (str): 包含标注文件（XML格式）的目录路径。
        old_prefix (str): 需要替换的旧路径前缀。
        new_prefix (str): 新路径前缀。
        encoding (str): XML文件的编码格式，默认为 'utf-8'。
    """
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 更新 <path> 标签
        path_element = root.find("path")
        if path_element is not None:
            old_path = path_element.text
            if old_path.startswith(old_prefix):
                # 替换路径
                new_path = old_path.replace(old_prefix, new_prefix)
                path_element.text = new_path
                print(f"Updated path for {xml_file}: {old_path} -> {new_path}")

        # 更新 <filename> 标签
        filename_element = root.find("filename")
        if filename_element is not None:
            old_filename = filename_element.text
            # 替换文件名中的中文字符
            new_filename = old_filename.replace("&#19981;&#30446;&#35270;&#21069;&#26041;", "non_object_foreground")
            filename_element.text = new_filename
            print(f"Updated filename for {xml_file}: {old_filename} -> {new_filename}")

        # 更新 <folder> 标签
        folder_element = root.find("folder")
        if folder_element is not None:
            old_folder = folder_element.text
            new_folder = old_folder.replace("&#19981;&#30446;&#35270;&#21069;&#26041;", "non_object_foreground")
            folder_element.text = new_folder
            print(f"Updated folder for {xml_file}: {old_folder} -> {new_folder}")

        # 保存更新后的XML文件
        tree.write(xml_path, encoding=encoding, xml_declaration=True)

    print("Path and filename update completed.")

# 示例调用
annotations_dir = r"Path to the file"
old_prefix="Path to the file"
new_prefix = r"Path to the file"
update_paths_and_filenames(annotations_dir, old_prefix, new_prefix)