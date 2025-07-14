'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:50:16
LastEditTime: 2025-07-10 16:44:54
LastEditors: Damocles_lin
'''
import pycolmap
import logging
import os
import open3d as o3d
import numpy as np
from utils import stats_utils, camera_utils

def dense_reconstruction(output_dir, sparse_path, image_path):
    """执行稠密重建"""
    dense_path = os.path.join(output_dir, "dense")
    os.makedirs(dense_path, exist_ok=True)
    
    # 去畸变图像
    undistort_images(dense_path, sparse_path, image_path)
    
    # 立体匹配
    stereo_matching(dense_path)
    
    # 融合深度图生成稠密点云
    fused_path = os.path.join(dense_path, "fused.ply")
    fuse_depth_maps(dense_path, fused_path)
    
    # 保存重建结果
    results_dir = os.path.join(dense_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成网格
    mesh_path = os.path.join(dense_path, "meshed.ply")
    generate_mesh(fused_path, mesh_path)
    
    # 保存重建结果并获取MVS统计信息
    mvs_stats = save_reconstruction_results(results_dir, fused_path, mesh_path)
    
    return mvs_stats

def undistort_images(output_path, input_path, image_path):
    """去畸变图像"""
    pycolmap.undistort_images(
        output_path=output_path,
        input_path=input_path,
        image_path=image_path
    )

def stereo_matching(workspace_path):
    """立体匹配"""
    stereo_options = pycolmap.PatchMatchOptions()
    stereo_options.gpu_index = "0"

    pycolmap.patch_match_stereo(
        workspace_path=workspace_path,
        workspace_format="COLMAP",
        options=stereo_options
    )

def fuse_depth_maps(workspace_path, output_path):
    """融合深度图生成稠密点云"""
    fusion_options = pycolmap.StereoFusionOptions()
    
    pycolmap.stereo_fusion(
        output_path=output_path,
        workspace_path=workspace_path,
        workspace_format="COLMAP",
        options=fusion_options
    )

def generate_mesh(input_path, output_path):
    """生成网格"""
    if os.path.exists(input_path):
        poisson_options = pycolmap.PoissonMeshingOptions()       
        pycolmap.poisson_meshing(
            input_path=input_path,
            output_path=output_path,
            options=poisson_options
        )
    else:
        logging.error("无法生成网格，缺少输入点云")

def save_reconstruction_results(results_dir, fused_ply_path, mesh_path):
    """保存重建结果并返回MVS统计信息"""
    # 初始化统计信息
    dense_points_count = 0
    mesh_vertices_count = 0
    mesh_triangles_count = 0
    
    # 保存稠密点云
    if os.path.exists(fused_ply_path):
        pcd = o3d.io.read_point_cloud(fused_ply_path)
        if pcd and len(pcd.points) > 0:
            dense_points = np.asarray(pcd.points)
            dense_points_count = dense_points.shape[0]
            np.save(os.path.join(results_dir, "dense_points.npy"), dense_points)
            logging.info(f"保存稠密点云: {dense_points_count}个点")
        else:
            logging.error("读取稠密点云失败或点云为空")
    else:
        logging.error(f"稠密点云文件不存在: {fused_ply_path}")
    
    # 保存网格
    if os.path.exists(mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh and len(mesh.vertices) > 0:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            mesh_vertices_count = vertices.shape[0]
            mesh_triangles_count = triangles.shape[0]
            np.save(os.path.join(results_dir, "mesh_vertices.npy"), vertices)
            np.save(os.path.join(results_dir, "mesh_triangles.npy"), triangles)
            logging.info(f"保存网格: {mesh_vertices_count}个顶点, {mesh_triangles_count}个面")
        else:
            logging.error("读取网格失败或网格为空")
    else:
        logging.error(f"网格文件不存在: {mesh_path}")
    
    # 返回MVS统计信息
    return {
        "dense_points": dense_points_count,
        "mesh_vertices": mesh_vertices_count,
        "mesh_triangles": mesh_triangles_count
    }