'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:51:27
LastEditTime: 2025-07-14 14:02:17
LastEditors: Damocles_lin
'''
def print_camera_example(results_dir):
    """加载并打印所有相机和位姿信息，生成总结文件"""
    try:
        import os
        import numpy as np
        import logging
        
        # 加载相机参数
        cameras_path = os.path.join(results_dir, "cameras.npy")
        poses_path = os.path.join(results_dir, "poses.npy")
        
        cameras = np.load(cameras_path, allow_pickle=True).item()
        poses = np.load(poses_path, allow_pickle=True).item()
        
        if not cameras or not poses:
            logging.warning("无法加载相机数据，文件可能为空")
            return
        
        # 创建输出文件
        summary_path = os.path.join(results_dir, "cameras_poses_summary.txt")
        with open(summary_path, 'w') as summary_file:  # 使用不同的变量名避免冲突
            # 相机模型ID映射
            model_names = {
                0: "SIMPLE_PINHOLE",
                1: "PINHOLE",
                2: "SIMPLE_RADIAL",
                3: "RADIAL",
                4: "OPENCV",
                5: "OPENCV_FISHEYE"
            }
            
            # 1. 输出所有相机信息
            summary_file.write("="*50 + "\n")
            summary_file.write(f"共找到 {len(cameras)} 个相机\n")
            summary_file.write("="*50 + "\n\n")
            
            for camera_id, camera_data in cameras.items():
                summary_file.write(f"===== 相机ID: {camera_id} =====\n")
                model_id = camera_data['model']
                model_name = model_names.get(model_id, f"未知模型({model_id})")
                params = camera_data['params']
                
                summary_file.write(f"相机模型: {model_name}\n")
                summary_file.write(f"相机参数: {params}\n")
                
                try:
                    if model_id == 0:  # SIMPLE_PINHOLE
                        focal, cx, cy = params
                        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
                    elif model_id == 1:  # PINHOLE
                        fx, fy, cx, cy = params
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    elif model_id in [2, 3]:  # RADIAL类型
                        focal, cx, cy, *_ = params
                        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
                    elif model_id == 4:  # OPENCV
                        fx, fy, cx, cy, *_ = params
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    else:
                        if len(params) >= 4:
                            # 取前四个参数，假设为fx, fy, cx, cy
                            fx, fy, cx, cy = params[:4]
                            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        else:
                            K = None
                    
                    if K is not None:
                        summary_file.write(f"内参矩阵K:\n{np.array2string(K, precision=6, suppress_small=True)}\n\n")
                    else:
                        summary_file.write("无法构建内参矩阵: 参数不足或模型不支持\n\n")
                except Exception as e:
                    summary_file.write(f"构建内参矩阵时出错: {str(e)}\n\n")
            
            # 2. 输出所有位姿信息
            summary_file.write("\n" + "="*50 + "\n")
            summary_file.write(f"共找到 {len(poses)} 个位姿\n")
            summary_file.write("="*50 + "\n\n")
            
            for image_name, pose_data in poses.items():
                summary_file.write(f"===== 图像: {image_name} =====\n")
                summary_file.write(f"相机ID: {pose_data['camera_id']}\n")
                
                R = pose_data['rotation']
                t = pose_data['translation']
                summary_file.write(f"旋转矩阵R:\n{np.array2string(R, precision=6, suppress_small=True)}\n")
                summary_file.write(f"平移向量t:\n{np.array2string(t, precision=6, suppress_small=True)}\n")
                
                # 构建外参矩阵[R|t]
                try:
                    Rt = np.hstack((R, t.reshape(3, 1)))
                    Rt_homog = np.vstack((Rt, [0, 0, 0, 1]))
                    summary_file.write(f"外参矩阵[R|t]:\n{np.array2string(Rt_homog, precision=6, suppress_small=True)}\n")
                except Exception as e:
                    summary_file.write(f"构建外参矩阵时出错: {str(e)}\n")
                
                # 输出相机到世界变换（如果存在）
                if 'cam_from_world' in pose_data:
                    c2w = pose_data['cam_from_world']
                    summary_file.write(f"相机到世界变换:\n{np.array2string(c2w, precision=6, suppress_small=True)}\n")
                
                summary_file.write("\n")  # 空行分隔不同位姿
        
        logging.info(f"成功生成相机和位姿总结文件: {summary_path}")
        logging.info(f"相机数量: {len(cameras)}, 位姿数量: {len(poses)}")
        
    except Exception as e:
        logging.error(f"处理相机数据时出错: {str(e)}")