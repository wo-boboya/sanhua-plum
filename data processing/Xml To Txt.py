import os
import xml.etree.ElementTree as ET


def convert_xml_to_yolo(xml_path, output_dir, class_mapping):
    """
    转换单个XML文件到YOLO格式TXT
    :param xml_path: XML文件路径
    :param output_dir: TXT输出目录
    :param class_mapping: 类别名称到索引的映射（如{'bing': 0, 'cong': 1}）
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 提取图像尺寸
        size_elem = root.find('size')
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)

        txt_lines = []
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                raise ValueError(f"未知类别: {class_name}，请检查class_mapping是否包含该类别")

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # 计算YOLO格式坐标（中心x, 中心y, 宽度, 高度，均归一化到0-1）
            dw = 1.0 / width
            dh = 1.0 / height
            x_center = (xmin + xmax) / 2.0 * dw
            y_center = (ymin + ymax) / 2.0 * dh
            box_width = (xmax - xmin) * dw
            box_height = (ymax - ymin) * dh

            txt_lines.append(
                f"{class_mapping[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        # 生成TXT文件名（与XML同名，扩展名改为txt）
        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        # 写入文件（保留6位小数，符合YOLO格式要求）
        with open(txt_path, 'w') as f:
            f.write('\n'.join(txt_lines))

    except Exception as e:
        print(f"转换 {xml_path} 出错: {str(e)}")


def process_batch(xml_dir, output_dir, class_list):
    """
    批量处理XML目录，生成YOLO格式TXT文件，并生成类别映射文件
    :param xml_dir: XML文件所在目录
    :param output_dir: TXT输出目录
    :param class_list: 类别列表（顺序决定索引，如['bing', 'cong']对应0和1）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    class_mapping = {cls: idx for idx, cls in enumerate(class_list)}  # 生成类别到索引的映射

    # 生成类别定义文件（YOLO训练所需）
    class_file = os.path.join(output_dir, 'classes.txt')
    with open(class_file, 'w') as f:
        f.write('\n'.join(class_list))  # 每行一个类别，顺序必须与索引一致

    # 遍历所有XML文件
    for file in os.listdir(xml_dir):
        if not file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, file)
        convert_xml_to_yolo(xml_path, output_dir, class_mapping)

    print(f"转换完成！共处理 {len(class_list)} 个类别，输出文件到 {output_dir}")


if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    XML_DIR = r"H:\九月三华李\增强\增强后\labels" # XML文件目录
    OUTPUT_DIR = r"H:\九月三华李\增强\增强后\txt"  # YOLO标签输出目录
    CLASS_LIST = ['Diseased fruit','Insect-damaged fruit','Bird-pecked fruit','Cracked fruit']  # 类别列表（顺序决定索引：0对应bing，1对应cong）

    # 执行批量转换
    process_batch(XML_DIR, OUTPUT_DIR, CLASS_LIST)
    print("所有文件转换完成！")