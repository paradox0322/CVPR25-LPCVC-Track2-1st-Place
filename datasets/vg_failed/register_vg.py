import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

# 定义VG数据集的JSON文件路径和图像文件夹路径
VG_JSON_FILE = "/seu_nvme/home/zhangmeng/220246538/zzt/datasets/visual_genome/refer_miniv.json"
VG_IMAGE_ROOT = "/seu_nvme/home/zhangmeng/220246538/zzt/datasets/visual_genome/VG_100K"

# 获取VG数据集的元信息
def get_vg_metadata():
    """
    返回VG数据集的元信息。
    """
    meta = {
        "thing_classes": ["object"],  # 示例类别名（需根据实际数据修改）
        "thing_colors": [(255, 0, 0)],  # 示例颜色（需根据实际数据修改）
    }
    return meta

# 加载VG数据集
def load_vg_json(json_file, image_dir):
    """
    Args:
        json_file (str): VG JSON标注文件路径。
        image_dir (str): 图像文件所在目录。
    Returns:
        list[dict]: 符合Detectron2标准格式的数据列表。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_dicts = []
    for item in data:
        # 提取字段
        image_id = item.get("image_id")
        polygons = item.get("Polygons")  # 注意：字段名为 "Polygons"（大写P）
        phrase = item.get("phrase")

        # 构造图像文件路径
        file_name = f"{image_id}.jpg"  # 假设图像文件扩展名为.jpg
        image_file = os.path.join(image_dir, file_name)

        # 将提取的信息存储到字典中
        record = {
            "file_name": image_file,
            "image_id": image_id,
            "phrase": phrase,
            "polygons": polygons,
        }

        dataset_dicts.append(record)

    return dataset_dicts

# 注册VG数据集
def register_vg_dataset(name, metadata, json_file, image_root):
    """
    注册VG数据集到Detectron2。
    Args:
        name (str): 数据集名称。
        metadata (dict): 数据集元信息。
        json_file (str): VG JSON标注文件路径。
        image_root (str): 图像文件根目录。
    """
    DatasetCatalog.register(
        name,
        lambda: load_vg_json(json_file, image_root)
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=json_file,
        evaluator_type="coco",  # 使用COCO评估器
        **metadata
    )

def register_vg_all(name):
    vg_metadata = get_vg_metadata()
    register_vg_dataset(
        name=name,
        metadata=vg_metadata,
        json_file= VG_JSON_FILE,
        image_root=VG_IMAGE_ROOT,
    )
    # dataset_dicts = DatasetCatalog.get(name)
    # print(f"Registered dataset '{name}' with {len(dataset_dicts)} images.")
    # print(dataset_dicts)
# 主函数：注册VG数据集
name = "vg"
register_vg_all(name)