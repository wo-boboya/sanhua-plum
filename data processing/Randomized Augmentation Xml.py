import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from xml.dom import minidom
import albumentations as A
import random
import shutil
from tqdm import tqdm
import re


class DataAugmentor:
    def __init__(self, img_dir, xml_dir, output_dir, target_classes, target_count):
        """
        初始化数据增强器

        参数:
        img_dir: 原始图像目录
        xml_dir: 原始XML标注目录
        output_dir: 增强数据输出目录
        target_classes: 需要增强的目标类别列表
        target_count: 每个类别需要达到的图像数量
        """
        # 使用原始字符串处理路径，避免转义问题
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.output_dir = output_dir
        self.target_classes = target_classes
        self.target_count = target_count

        # 创建输出目录
        self.aug_img_dir = os.path.join(output_dir, "augmented_images")
        self.aug_xml_dir = os.path.join(output_dir, "augmented_annotations")
        os.makedirs(self.aug_img_dir, exist_ok=True)
        os.makedirs(self.aug_xml_dir, exist_ok=True)

        # 定义增强方法 - 使用正确的bbox_params
        self.transforms = {
            'flip': A.Compose([
                A.HorizontalFlip(p=1.0)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels'])),

            'rotate': A.Compose([
                A.RandomRotate90(p=1.0)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels'])),

            'brightness': A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    p=1.0
                )
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels'])),

            'noise': A.Compose([
                A.GaussNoise(p=1.0)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))
        }

    def parse_xml(self, xml_path):
        """解析XML文件，提取图像信息和边界框"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 提取图像信息
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)

            # 提取边界框
            boxes = []
            class_labels = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])
                class_labels.append(class_name)

            return filename, (width, height, depth), boxes, class_labels
        except Exception as e:
            print(f"解析XML文件 {xml_path} 时出错: {e}")
            return None, None, None, None

    def create_xml(self, filename, size, boxes, class_labels, output_path):
        """创建新的XML文件"""
        # 创建XML结构
        annotation = ET.Element("annotation")

        folder = ET.SubElement(annotation, "folder")
        folder.text = "augmented_dataset"

        filename_elem = ET.SubElement(annotation, "filename")
        filename_elem.text = filename

        path = ET.SubElement(annotation, "path")
        path.text = os.path.join(self.aug_img_dir, filename)

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        size_elem = ET.SubElement(annotation, "size")
        width = ET.SubElement(size_elem, "width")
        width.text = str(size[0])
        height = ET.SubElement(size_elem, "height")
        height.text = str(size[1])
        depth = ET.SubElement(size_elem, "depth")
        depth.text = str(size[2])

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        # 添加所有边界框
        for i, (box, class_label) in enumerate(zip(boxes, class_labels)):
            obj = ET.SubElement(annotation, "object")

            name = ET.SubElement(obj, "name")
            name.text = class_label

            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin_elem = ET.SubElement(bndbox, 'xmin')
            xmin_elem.text = str(int(box[0]))
            ymin_elem = ET.SubElement(bndbox, 'ymin')
            ymin_elem.text = str(int(box[1]))
            xmax_elem = ET.SubElement(bndbox, 'xmax')
            xmax_elem.text = str(int(box[2]))
            ymax_elem = ET.SubElement(bndbox, 'ymax')
            ymax_elem.text = str(int(box[3]))

        # 格式化并写入XML文件
        rough_string = ET.tostring(annotation, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  "))

    def count_images_per_class(self):
        """统计每个类别对应的图像数量"""
        class_image_counts = {cls: 0 for cls in self.target_classes}
        all_images = set()

        for xml_file in os.listdir(self.xml_dir):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(self.xml_dir, xml_file)
            filename, _, _, class_labels = self.parse_xml(xml_path)

            if filename is None or class_labels is None:
                continue

            # 记录所有图像
            all_images.add(filename)

            # 统计每个类别对应的图像数量
            for cls in class_labels:
                if cls in class_image_counts:
                    class_image_counts[cls] += 1

        return class_image_counts, len(all_images)

    def get_images_by_class(self):
        """获取每个类别对应的图像列表"""
        class_images = {cls: [] for cls in self.target_classes}

        for xml_file in os.listdir(self.xml_dir):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(self.xml_dir, xml_file)
            filename, _, _, class_labels = self.parse_xml(xml_path)

            if filename is None or class_labels is None:
                continue

            # 为每个类别添加图像
            for cls in class_labels:
                if cls in class_images:
                    class_images[cls].append({
                        'xml_path': xml_path,
                        'img_name': filename
                    })

        return class_images

    def apply_single_transform(self, image, boxes, class_labels, transform_name):
        """应用单个变换并返回结果"""
        transform = self.transforms[transform_name]

        # 确保boxes是numpy数组格式，这是albumentations所期望的
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)

        try:
            # 应用变换
            transformed = transform(
                image=image,
                bboxes=boxes_np,
                class_labels=class_labels
            )

            # 将bboxes转换回列表格式
            if len(transformed['bboxes']) > 0:
                transformed['bboxes'] = [list(bbox) for bbox in transformed['bboxes']]
            else:
                transformed['bboxes'] = []

            return transformed
        except Exception as e:
            print(f"应用变换 {transform_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def augment_data(self):
        """执行数据增强"""
        # 统计当前每个类别的图像数量
        class_image_counts, total_images = self.count_images_per_class()
        print("当前各类别对应的图像数量:")
        for cls, count in class_image_counts.items():
            print(f"{cls}: {count}")
        print(f"总图像数量: {total_images}")

        # 计算需要增强的数量
        augment_needs = {}
        for cls in self.target_classes:
            augment_needs[cls] = max(0, self.target_count - class_image_counts[cls])

        print("\n需要增强的图像数量:")
        for cls, need in augment_needs.items():
            print(f"{cls}: {need}")

        # 获取每个类别对应的图像
        class_images = self.get_images_by_class()
        print(f"\n每个类别对应的图像数量:")
        for cls, images in class_images.items():
            print(f"{cls}: {len(images)}")

        # 开始增强
        augmented_counts = {cls: 0 for cls in self.target_classes}
        augmentation_idx = 0

        # 持续增强直到满足需求
        total_augment_needs = sum(augment_needs.values())
        if total_augment_needs == 0:
            print("无需增强，各类别图像数量已满足要求")
            return

        pbar = tqdm(total=total_augment_needs, desc="增强进度")

        try:
            # 为每个类别单独增强
            for cls in self.target_classes:
                while augmented_counts[cls] < augment_needs[cls]:
                    # 随机选择一张包含当前类别的图像
                    if not class_images[cls]:
                        print(f"警告: 没有找到包含类别 {cls} 的图像")
                        break

                    img_info = random.choice(class_images[cls])
                    xml_path = img_info['xml_path']
                    img_name = img_info['img_name']

                    # 解析XML
                    filename, size, boxes, class_labels = self.parse_xml(xml_path)

                    if filename is None or boxes is None or class_labels is None:
                        continue

                    # 读取图像
                    img_path = os.path.join(self.img_dir, img_name)
                    if not os.path.exists(img_path):
                        print(f"警告: 图像文件 {img_path} 不存在，跳过")
                        continue

                    # 使用imdecode读取图像，避免中文路径问题
                    with open(img_path, 'rb') as f:
                        img_data = np.frombuffer(f.read(), np.uint8)
                    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                    if image is None:
                        print(f"警告: 无法读取图像 {img_path}，跳过")
                        continue

                    # 随机选择一种增强方法
                    transform_names = list(self.transforms.keys())
                    selected_transform = random.choice(transform_names)

                    # 应用选定的增强方法
                    transformed = self.apply_single_transform(image, boxes, class_labels, selected_transform)

                    if transformed is None:
                        continue

                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    transformed_class_labels = transformed['class_labels']

                    # 检查增强后的图像是否包含当前类别
                    if cls not in transformed_class_labels:
                        continue

                    # 过滤掉无效的边界框
                    valid_bboxes = []
                    valid_labels = []
                    for bbox, label in zip(transformed_bboxes, transformed_class_labels):
                        # 确保bbox是列表格式
                        if isinstance(bbox, tuple):
                            bbox = list(bbox)

                        xmin, ymin, xmax, ymax = bbox
                        # 检查边界框是否有效
                        if (xmax > xmin and ymax > ymin and
                                xmin >= 0 and ymin >= 0 and
                                xmax <= transformed_image.shape[1] and ymax <= transformed_image.shape[0]):
                            valid_bboxes.append(bbox)
                            valid_labels.append(label)

                    # 如果没有有效的边界框，跳过此次增强
                    if not valid_bboxes or cls not in valid_labels:
                        continue

                    # 生成新的文件名，包含增强类型信息
                    base_name = os.path.splitext(img_name)[0]
                    # 移除可能已有的增强标记
                    base_name = re.sub(r'_(flip|rotate|brightness|noise|combined)_\d+$', '', base_name)

                    new_img_name = f"{base_name}_{selected_transform}_{augmentation_idx}.jpg"
                    new_xml_name = f"{base_name}_{selected_transform}_{augmentation_idx}.xml"

                    # 保存增强后的图像
                    new_img_path = os.path.join(self.aug_img_dir, new_img_name)
                    success, encoded_image = cv2.imencode('.jpg', transformed_image)
                    if success:
                        with open(new_img_path, 'wb') as f:
                            f.write(encoded_image)
                    else:
                        print(f"警告: 无法保存图像 {new_img_path}，跳过")
                        continue

                    # 创建并保存新的XML文件
                    new_xml_path = os.path.join(self.aug_xml_dir, new_xml_name)
                    self.create_xml(new_img_name,
                                    (transformed_image.shape[1], transformed_image.shape[0], 3),
                                    valid_bboxes, valid_labels, new_xml_path)

                    augmentation_idx += 1
                    augmented_counts[cls] += 1
                    pbar.update(1)

        except KeyboardInterrupt:
            print("\n用户中断了增强过程")
        except Exception as e:
            print(f"\n增强过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pbar.close()

        print("\n增强完成!")
        print("增强后的各类别图像数量:")
        for cls in self.target_classes:
            print(f"{cls}: {class_image_counts[cls] + augmented_counts[cls]}")

        # 复制原始数据到输出目录
        print("\n正在复制原始数据到输出目录...")
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(self.img_dir, img_file)
                dst = os.path.join(self.aug_img_dir, img_file)
                if not os.path.exists(dst):
                    # 使用shutil.copy2复制文件
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"复制图像文件 {src} 到 {dst} 时出错: {e}")

        for xml_file in os.listdir(self.xml_dir):
            if xml_file.endswith('.xml'):
                src = os.path.join(self.xml_dir, xml_file)
                dst = os.path.join(self.aug_xml_dir, xml_file)
                if not os.path.exists(dst):
                    # 使用shutil.copy2复制文件
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"复制XML文件 {src} 到 {dst} 时出错: {e}")

        print("所有操作完成!")


# 使用示例
if __name__ == "__main__":

    IMG_DIR = r"Path to the file"
    XML_DIR = r"Path to the file"
    OUTPUT_DIR = r"Path to the file"

    # 需要增强的目标类别和目标数量
    TARGET_CLASSES = ["class1", "class2"]
    TARGET_COUNT = 1500

    # 创建增强器并执行增强
    augmentor = DataAugmentor(IMG_DIR, XML_DIR, OUTPUT_DIR, TARGET_CLASSES, TARGET_COUNT)
    augmentor.augment_data()