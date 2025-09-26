import os
import xml.etree.ElementTree as ET

def update_folder(annotations_dir, new_folder_name, encoding="utf-8"):
    """
    更新标注文件中的 <folder> 标签。

    参数:
        annotations_dir (str): 包含标注文件（XML格式）的目录路径。
        new_folder_name (str): 新的 <folder> 标签值。
        encoding (str): XML文件的编码格式，默认为 'utf-8'。
    """
    # 确保标注文件目录存在
    if not os.path.exists(annotations_dir):
        print(f"Error: Directory '{annotations_dir}' does not exist.")
        return

    # 遍历标注文件目录
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)

        try:
            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 查找 <folder> 标签并更新其值
            folder_element = root.find("folder")
            if folder_element is not None:
                old_folder_name = folder_element.text
                folder_element.text = new_folder_name
                print(f"Updated folder for {xml_file}: {old_folder_name} -> {new_folder_name}")
            else:
                print(f"Warning: No <folder> tag found in {xml_file}. Skipping.")

            # 保存更新后的XML文件
            tree.write(xml_path, encoding=encoding, xml_declaration=True)

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    print("Folder update completed.")

# 示例调用
if __name__ == "__main__":
    annotations_dir = r"Path to the file" # 标注文件所在的目录
    new_folder_name = "New folder"  # 新的 <folder> 标签值
    update_folder(annotations_dir, new_folder_name)