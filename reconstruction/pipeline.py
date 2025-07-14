'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:50:16
LastEditTime: 2025-07-10 16:46:43
LastEditors: Damocles_lin
'''
import os
import logging
import numpy as np
from pathlib import Path
import pycolmap
from utils import logging_utils, timer, stats_utils, camera_utils
from .sfm import extract_features, match_features, incremental_reconstruction
from .mvs import dense_reconstruction

def run_colmap_pipeline(image_dir, output_dir):
    # 初始化计时器
    timer_obj = timer.Timer()
    
    # 配置日志
    log_file = logging_utils.configure_logging(output_dir)
    logging.info(f"开始COLMAP处理流程，图像目录: {image_dir}, 输出目录: {output_dir}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 确保目录存在
    image_dir = Path(image_dir)
    if not image_dir.exists():
        logging.error(f"图像目录不存在: {image_dir}")
        return
    
    # 1. 特征提取
    database_path = str(output_path / "database.db")
    image_path = str(image_dir)
    
    timer_obj.start("特征提取")
    image_stats = extract_features(image_dir, database_path)
    if not image_stats:
        return
    timer_obj.end()
    
    # 2. 特征匹配
    timer_obj.start("特征匹配")
    match_stats = match_features(database_path)
    if not match_stats:
        return
    timer_obj.end()

    # 3. 增量重建
    sparse_path = str(output_path / "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    timer_obj.start("增量重建")
    result = incremental_reconstruction(
        database_path, 
        image_path, 
        sparse_path,
        image_stats,
        match_stats
    )
    if not result:
        return
    reconstruction, sfm_stats = result
    timer_obj.end()

    # 4. 稠密重建
    timer_obj.start("稠密重建")
    mvs_stats = dense_reconstruction(output_dir, os.path.join(sparse_path, "0"), image_path)
    timer_obj.end()

    # 5. 保存重建结果
    timer_obj.start("保存重建结果")
    
    # 保存稀疏重建结果
    results_dir = os.path.join(output_dir, "dense", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存稀疏点云
    sparse_points = []
    for point3D_id, point3D in reconstruction.points3D.items():
        sparse_points.append([point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]])
    sparse_points = np.array(sparse_points)
    np.save(os.path.join(results_dir, "sparse_points.npy"), sparse_points)
    logging.info(f"保存稀疏点云: {sparse_points.shape[0]}个点")
    
    # 保存相机参数
    cameras = {}
    for camera_id, camera in reconstruction.cameras.items():
        cameras[camera_id] = {
            "model": int(camera.model),
            "params": camera.params,
            "width": camera.width,
            "height": camera.height
        }
    np.save(os.path.join(results_dir, "cameras.npy"), cameras)
    logging.info(f"保存{len(cameras)}个相机参数")
    
    # 保存图像位姿
    poses = {}
    for image_id, image in reconstruction.images.items():
        cam_from_world = image.cam_from_world() # 4x4矩阵
        rotation_matrix = cam_from_world.rotation.matrix()
        translation = cam_from_world.translation
        poses[image.name] = {
            "rotation": rotation_matrix,
            "translation": translation,
            "camera_id": image.camera_id,
            "cam_from_world": cam_from_world.matrix()
        }
    np.save(os.path.join(results_dir, "poses.npy"), poses)
    logging.info(f"保存{len(poses)}个相机位姿")
    
    # 打印相机示例
    camera_utils.print_camera_example(results_dir)

    timer_obj.end()
    
    # 记录并保存计时摘要
    summary = timer_obj.log_summary()
    stats_utils.save_timing_summary(output_dir, summary)
    
    # 保存整体统计信息
    stats_utils.save_overall_stats(output_dir, sfm_stats, mvs_stats)
    
    logging.info(f"处理流程完成！日志已保存到: {log_file}")
    
    # 返回统计信息
    return {
        "sfm_stats": sfm_stats,
        "mvs_stats": mvs_stats
    }