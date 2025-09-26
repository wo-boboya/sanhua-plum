import os
import xml.etree.ElementTree as ET

def update_object_names(annotations_dir, name_mapping, encoding="utf-8"):
    """
    更新标注文件中的 <name> 标签。

    参数:
        annotations_dir (str): 包含标注文件（XML格式）的目录路径。
        name_mapping (dict): 旧类别名称到新类别名称的映射，例如 {"old_name": "new_name"}。
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

            # 遍历所有 <object> 标签并更新 <name> 标签
            objects = root.findall("object")
            updated = False
            for obj in objects:
                name_element = obj.find("name")
                if name_element is not None:
                    old_name = name_element.text
                    if old_name in name_mapping:
                        new_name = name_mapping[old_name]
                        name_element.text = new_name
                        print(f"Updated name for {xml_file}: {old_name} -> {new_name}")
                        updated = True

            # 如果有更新，保存修改后的XML文件
            if updated:
                tree.write(xml_path, encoding=encoding, xml_declaration=True)

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    print("Name update completed.")

# 示例调用
if __name__ == "__main__":
    annotations_dir = r"Path to the file"  # 标注文件所在的目录
    name_mapping = {
        "old class": "new class"
    }  # 类别名称映射
    update_object_names(annotations_dir, name_mapping)