'''
Description: 
Author: Damocles_lin
Date: 2025-06-30 13:43:19
LastEditTime: 2025-07-04 23:09:38
LastEditors: Damocles_lin
'''

import os
import sys
import numpy as np
import pycolmap
from pathlib import Path
import open3d as o3d
import logging
from datetime import datetime

# 配置日志系统
def configure_logging(output_dir):
    """配置pycolmap日志系统和文件日志"""
    # 创建日志目录
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"colmap_log_{timestamp}.txt"
    
    # 配置pycolmap日志
    pycolmap.logging.logtostderr = True     # 设置日志输出到控制台
    pycolmap.logging.verbose_level = 4      # 设置日志详细级别，最高详细级别
    pycolmap.logging.alsologtostderr = True
    pycolmap.logging.minloglevel = pycolmap.logging.Level.INFO.value    # 设置所有级别日志都输出到控制台
    
    # 创建文件日志处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 创建日志格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 获取根日志记录器并添加文件处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"日志文件已创建: {log_file}")
    return log_file

def run_colmap_pipeline(image_dir, output_dir):
    # 配置日志
    log_file = configure_logging(output_dir)
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
    
    logging.info("开始特征提取...")
    
    # 特征提取选项
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.num_threads = -1
    sift_options.use_gpu = True
    sift_options.gpu_index = "0"
    
    if not sift_options.check():
        logging.error("特征提取选项无效！")
        return
    
    # 获取图像列表
    image_names = [f.name for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    logging.info(f"找到 {len(image_names)} 张图像进行处理")
    
    # 调用特征提取函数
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_path,
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.AUTO,
        sift_options=sift_options,
        device=pycolmap.Device.cuda
    )
    
    # 2. 特征匹配
    logging.info("开始特征匹配...")
    
    sift_matcher_options = pycolmap.SiftMatchingOptions()
    sift_matcher_options.num_threads = -1
    sift_matcher_options.use_gpu = True
    sift_matcher_options.gpu_index = "0"
    
    exhaustive_options = pycolmap.ExhaustiveMatchingOptions()
    verification_options = pycolmap.TwoViewGeometryOptions()
    
    if not sift_matcher_options.check():
        logging.error("特征匹配选项无效！")
        return
    
    pycolmap.match_exhaustive(
        database_path=database_path,
        sift_options=sift_matcher_options,
        matching_options=exhaustive_options,
        verification_options=verification_options,
        device=pycolmap.Device.cuda
    )

    # 3. 增量重建
    sparse_path = str(output_path / "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    logging.info("开始增量重建...")
    mapper_options = pycolmap.IncrementalPipelineOptions()
    
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=sparse_path,
        options=mapper_options
    )
    
    if not reconstructions:
        logging.error("重建失败！")
        return
    
    # 取第一个重建模型
    reconstruction = list(reconstructions.values())[0]
    reconstruction.write(sparse_path)
    logging.info(f"重建成功！包含 {len(reconstruction.images)} 张图像和 {len(reconstruction.points3D)} 个点")
    
    # 4. 稠密重建
    dense_path = str(output_path / "dense")
    os.makedirs(dense_path, exist_ok=True)
    
    logging.info("开始稠密重建...")
    # 去畸变图像
    pycolmap.undistort_images(
        output_path=dense_path,
        input_path=os.path.join(sparse_path, "0"),
        image_path=image_path
    )
    
    # 立体匹配
    logging.info("开始立体匹配...")
    stereo_options = pycolmap.PatchMatchOptions()
    stereo_options.gpu_index = "0"

    pycolmap.patch_match_stereo(
        workspace_path=dense_path,
        workspace_format="COLMAP",
        options=stereo_options
    )
    
    # 融合深度图生成稠密点云
    logging.info("融合深度图生成稠密点云...")
    fused_path = os.path.join(dense_path, "fused.ply")
    fusion_options = pycolmap.StereoFusionOptions()
    
    pycolmap.stereo_fusion(
        output_path=fused_path,
        workspace_path=dense_path,
        workspace_format="COLMAP",
        options=fusion_options
    )
    
    # 5. 保存重建结果
    save_reconstruction_results(reconstruction, dense_path, fused_path)
    logging.info(f"处理流程完成！日志已保存到: {log_file}")

def save_reconstruction_results(reconstruction, dense_path, fused_ply_path):
    """保存重建结果到numpy格式"""
    # 创建保存目录
    results_dir = os.path.join(dense_path, "results")
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
    
    # 保存稠密点云
    if os.path.exists(fused_ply_path):
        pcd = o3d.io.read_point_cloud(fused_ply_path)
        if pcd and len(pcd.points) > 0:
            dense_points = np.asarray(pcd.points)
            np.save(os.path.join(results_dir, "dense_points.npy"), dense_points)
            logging.info(f"保存稠密点云: {dense_points.shape[0]}个点")
        else:
            logging.error("读取稠密点云失败或点云为空")
    else:
        logging.error(f"稠密点云文件不存在: {fused_ply_path}")
    
    # 生成并保存网格
    mesh_path = os.path.join(dense_path, "meshed.ply")
    if os.path.exists(fused_ply_path):
        poisson_options = pycolmap.PoissonMeshingOptions()       
        pycolmap.poisson_meshing(
            input_path=fused_ply_path,
            output_path=mesh_path,
            options=poisson_options
        )
        
        if os.path.exists(mesh_path):
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if mesh and len(mesh.vertices) > 0:
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                np.save(os.path.join(results_dir, "mesh_vertices.npy"), vertices)
                np.save(os.path.join(results_dir, "mesh_triangles.npy"), triangles)
                logging.info(f"保存网格: {vertices.shape[0]}个顶点, {triangles.shape[0]}个面")
            else:
                logging.error("读取网格失败或网格为空")
        else:
            logging.error(f"网格文件不存在: {mesh_path}")
    else:
        logging.error("无法生成网格，缺少稠密点云")
    
    logging.info(f"所有结果已保存到 {results_dir}")

if __name__ == "__main__":
    # 配置路径 - 使用实际路径替换这些占位符
    IMAGE_DIR = "./images"  # 替换为实际图像路径
    OUTPUT_DIR = "./output"  # 替换为输出目录
    
    # 运行完整流程
    run_colmap_pipeline(IMAGE_DIR, OUTPUT_DIR)