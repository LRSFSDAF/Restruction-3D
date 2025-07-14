'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:50:59
LastEditTime: 2025-07-08 14:18:11
LastEditors: Damocles_lin
'''
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

def save_sfm_stats(output_dir, stats):
    """保存SfM统计信息"""
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = stats_dir / "sfm_stats.txt"
    
    with open(stats_file, "w") as f:
        f.write("===== SfM重建统计信息 =====\n")
        f.write(f"总图像数量: {stats['total_images']}\n")
        
        # 记录图像分辨率
        if stats['image_resolutions']:
            # 获取所有分辨率
            resolutions = list(stats['image_resolutions'].values())
            unique_resolutions = set(resolutions)
            
            if len(unique_resolutions) == 1:
                w, h = next(iter(unique_resolutions))
                f.write(f"图像分辨率: {w}x{h} (所有图像相同)\n")
            else:
                f.write("图像分辨率:\n")
                for res, count in count_resolutions(resolutions):
                    f.write(f"  - {res[0]}x{res[1]}: {count} 张图像\n")
        
        f.write(f"总特征点数量: {stats['total_keypoints']}\n")
        f.write(f"总匹配对数量: {stats['total_matches']}\n")
        f.write(f"成功匹配的图像对数量: {stats['matched_image_pairs']}\n")
        f.write(f"具有匹配的图像数量: {stats['matched_images_count']}\n")
        f.write(f"注册图像数量: {stats['registered_images']}\n")
        f.write(f"稀疏点云数量: {stats['sparse_points']}\n")
        f.write(f"平均重投影误差: {stats['mean_reprojection_error']:.6f} 像素\n")
    
    logging.info(f"SfM统计信息已保存到: {stats_file}")

def count_resolutions(resolutions):
    """统计不同分辨率的数量"""
    from collections import Counter
    counter = Counter(resolutions)
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)

def save_mvs_stats(output_dir, stats):
    """保存MVS统计信息"""
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = stats_dir / "mvs_stats.txt"
    
    with open(stats_file, "w") as f:
        f.write("===== MVS重建统计信息 =====\n")
        f.write(f"稠密点云数量: {stats['dense_points']}\n")
        f.write(f"网格顶点数量: {stats['mesh_vertices']}\n")
        f.write(f"网格面片数量: {stats['mesh_triangles']}\n")
    
    logging.info(f"MVS统计信息已保存到: {stats_file}")

def save_overall_stats(output_dir, sfm_stats, mvs_stats):
    """保存整体统计信息"""
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = stats_dir / "overall_stats.txt"
    
    with open(stats_file, "w") as f:
        f.write("===== 三维重建统计摘要 =====\n")
        f.write("--- 输入图像 ---\n")
        f.write(f"总图像数量: {sfm_stats['total_images']}\n")
        
        # 分辨率信息
        if sfm_stats['image_resolutions']:
            resolutions = list(sfm_stats['image_resolutions'].values())
            unique_resolutions = set(resolutions)
            
            if len(unique_resolutions) == 1:
                w, h = next(iter(unique_resolutions))
                f.write(f"图像分辨率: {w}x{h}\n")
            else:
                f.write("图像分辨率分布:\n")
                for res, count in count_resolutions(resolutions):
                    f.write(f"  - {res[0]}x{res[1]}: {count} 张\n")
        
        f.write("\n--- 特征提取与匹配 ---\n")
        f.write(f"总特征点数量: {sfm_stats['total_keypoints']}\n")
        f.write(f"总匹配对数量: {sfm_stats['total_matches']}\n")
        f.write(f"成功匹配的图像对数量: {sfm_stats['matched_image_pairs']}\n")
        f.write(f"具有匹配的图像数量: {sfm_stats['matched_images_count']}\n")
        
        f.write("\n--- SfM重建 ---\n")
        f.write(f"注册图像数量: {sfm_stats['registered_images']}\n")
        f.write(f"稀疏点云数量: {sfm_stats['sparse_points']}\n")
        f.write(f"平均重投影误差: {sfm_stats['mean_reprojection_error']:.6f} 像素\n")
        
        f.write("\n--- MVS重建 ---\n")
        f.write(f"稠密点云数量: {mvs_stats['dense_points']}\n")
        f.write(f"网格顶点数量: {mvs_stats['mesh_vertices']}\n")
        f.write(f"网格面片数量: {mvs_stats['mesh_triangles']}\n")
    
    logging.info(f"整体统计信息已保存到: {stats_file}")

def save_timing_summary(output_dir, summary):
    """保存计时摘要到文件"""
    timing_dir = Path(output_dir) / "logs"
    timing_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_file = timing_dir / f"timing_summary_{timestamp}.txt"
    
    with open(timing_file, "w") as f:
        f.write(summary)
    
    logging.info(f"计时摘要已保存到: {timing_file}")