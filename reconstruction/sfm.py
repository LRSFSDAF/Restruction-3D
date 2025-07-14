'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:50:16
LastEditTime: 2025-07-13 17:18:14
LastEditors: Damocles_lin
'''
import pycolmap
import numpy as np
import logging
import os
import sqlite3
from pathlib import Path
from utils import stats_utils

def extract_features(image_dir, database_path, device=pycolmap.Device.cuda):
    """特征提取，返回图像信息和特征点统计"""
    # 特征提取选项
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.num_threads = -1
    sift_options.use_gpu = True
    sift_options.gpu_index = "0"
    
    if not sift_options.check():
        logging.error("特征提取选项无效！")
        return None
    
    # 获取图像列表
    image_dir = Path(image_dir)
    image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    image_names = [f.name for f in image_files]
    total_images = len(image_names)
    
    # 记录图像分辨率
    image_resolutions = {}
    try:
        # 使用Pillow获取图像分辨率
        from PIL import Image
        pil_available = True
    except ImportError:
        logging.warning("Pillow未安装，无法获取图像分辨率")
        pil_available = False
    
    if pil_available:
        # 临时禁用PIL的详细日志
        original_pil_level = logging.getLogger('PIL').level
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
                    image_resolutions[img_file.name] = (width, height)
            except Exception as e:
                logging.warning(f"无法读取图像分辨率: {img_file} - {str(e)}")
                image_resolutions[img_file.name] = (0, 0)
        
        # 恢复原始日志级别
        logging.getLogger('PIL').setLevel(original_pil_level)
    else:
        # 即使没有Pillow，也填充默认值
        for img_file in image_files:
            image_resolutions[img_file.name] = (0, 0)
    
    logging.info(f"找到 {total_images} 张图像进行处理")    
    
    # 调用特征提取函数
    pycolmap.extract_features(
        database_path=database_path,
        image_path=str(image_dir),
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.AUTO,
        sift_options=sift_options,
        device=device
    )
    
    # 获取特征点统计
    total_keypoints = get_total_keypoints(database_path)
    
    return {
        "total_images": total_images,
        "image_resolutions": image_resolutions,
        "total_keypoints": total_keypoints
    }

def get_total_keypoints(database_path):
    """从数据库获取总特征点数量，并将每张图片的特征点数量输出到文本文件"""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # 1. 查询所有图像的rows字段（每张图像的特征点数量）
        cursor.execute("SELECT image_id, rows FROM keypoints")
        rows = cursor.fetchall()
        
        # 2. 计算所有图像的特征点总和
        total_keypoints = sum(row[1] for row in rows) if rows else 0
        
        # 3. 生成输出文本文件路径（与数据库同目录）
        db_dir = os.path.dirname(database_path)
        output_file = os.path.join(db_dir, "keypoints_per_image.txt")
        
        # 4. 将每张图片的特征点数量写入文本文件
        with open(output_file, 'w') as f:
            f.write("image_id, keypoints_count\n")
            for image_id, count in rows:
                f.write(f"{image_id}, {count}\n")
        
        conn.close()
        
        logging.info(f"总特征点数量: {total_keypoints}")
        logging.info(f"每张图像特征点数量已保存至: {output_file}")
        return total_keypoints
        
    except Exception as e:
        logging.error(f"获取特征点统计失败: {str(e)}")
        return 0

def match_features(database_path, device=pycolmap.Device.cuda):
    """特征匹配，返回匹配统计信息"""
    sift_matcher_options = pycolmap.SiftMatchingOptions()
    sift_matcher_options.num_threads = -1
    sift_matcher_options.use_gpu = True
    sift_matcher_options.gpu_index = "0"
    
    exhaustive_options = pycolmap.ExhaustiveMatchingOptions()
    verification_options = pycolmap.TwoViewGeometryOptions()
    
    if not sift_matcher_options.check():
        logging.error("特征匹配选项无效！")
        return None
    
    pycolmap.match_exhaustive(
        database_path=database_path,
        sift_options=sift_matcher_options,
        matching_options=exhaustive_options,
        verification_options=verification_options,
        device=device
    )
    
    # 获取匹配统计
    return get_matching_stats(database_path)

def get_matching_stats(database_path):
    """从数据库获取匹配统计信息（使用正确的COLMAP数据库模式）"""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # 1. 查询匹配对数量
        cursor.execute("SELECT COUNT(*) FROM matches")
        total_matches = cursor.fetchone()[0]
        
        # 2. 查询成功匹配的图像对数量
        cursor.execute("SELECT COUNT(DISTINCT pair_id) FROM matches")
        matched_image_pairs = cursor.fetchone()[0]
        
        # 3. 查询具有匹配的图像数量
        # 从图像表中获取所有图像ID
        cursor.execute("SELECT image_id FROM images")
        all_image_ids = [row[0] for row in cursor.fetchall()]
        
        # 从匹配表中获取有匹配的图像ID
        matched_image_ids = set()
        
        # 获取所有pair_id
        cursor.execute("SELECT DISTINCT pair_id FROM matches")
        pair_ids = [row[0] for row in cursor.fetchall()]
        
        # 解码pair_id获取图像ID（COLMAP使用pair_id = image_id1 * 2^32 + image_id2）
        for pair_id in pair_ids:
            image_id1 = pair_id >> 32
            image_id2 = pair_id & 0xFFFFFFFF
            matched_image_ids.add(image_id1)
            matched_image_ids.add(image_id2)
        
        matched_images_count = len(matched_image_ids)
        
        # 4. 查询每张图像的匹配数量
        per_image_matches = {}
        for image_id in all_image_ids:
            # 计算该图像在匹配中出现的次数
            cursor.execute("""
                SELECT COUNT(*) FROM matches
                WHERE pair_id IN (
                    SELECT pair_id FROM matches
                    WHERE (pair_id >> 32) = ? OR (pair_id & 0xFFFFFFFF) = ?
                )
            """, (image_id, image_id))
            count = cursor.fetchone()[0]
            per_image_matches[image_id] = count
        
        conn.close()
        
        logging.info(f"总匹配对数量: {total_matches}")
        logging.info(f"成功匹配的图像对数量: {matched_image_pairs}")
        logging.info(f"具有匹配的图像数量: {matched_images_count}")
        
        return {
            "total_matches": total_matches,
            "matched_image_pairs": matched_image_pairs,
            "matched_images_count": matched_images_count,
            "per_image_matches": per_image_matches
        }
    except Exception as e:
        logging.error(f"获取匹配统计失败: {str(e)}")
        return None

def incremental_reconstruction(database_path, image_path, output_path, image_stats, match_stats):
    """增量重建"""
    mapper_options = pycolmap.IncrementalPipelineOptions()
    
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=mapper_options
    )
    
    if not reconstructions:
        logging.error("重建失败！")
        return None
    
    # 取第一个重建模型
    reconstruction = list(reconstructions.values())[0]
    reconstruction.write(output_path)
    
    # 计算SfM统计信息
    sfm_stats = calculate_sfm_stats(reconstruction)
    
    # 添加图像和匹配统计
    sfm_stats.update({
        "total_images": image_stats["total_images"],
        "image_resolutions": image_stats["image_resolutions"],
        "total_keypoints": image_stats["total_keypoints"],
        "total_matches": match_stats["total_matches"],
        "matched_image_pairs": match_stats["matched_image_pairs"],
        "matched_images_count": match_stats["matched_images_count"]
    })
    
    logging.info(f"重建成功！包含 {sfm_stats['registered_images']} 张图像和 {sfm_stats['sparse_points']} 个点")
    logging.info(f"平均重投影误差: {sfm_stats['mean_reprojection_error']:.4f} 像素")
    
    # 保存统计信息
    stats_utils.save_sfm_stats(os.path.dirname(output_path), sfm_stats)
    
    return reconstruction, sfm_stats

def calculate_sfm_stats(reconstruction):
    """计算SfM统计信息"""
    registered_images = len(reconstruction.images)
    sparse_points = len(reconstruction.points3D)
    mean_reprojection_error = reconstruction.compute_mean_reprojection_error()
    
    return {
        "registered_images": registered_images,
        "sparse_points": sparse_points,
        "mean_reprojection_error": mean_reprojection_error
    }