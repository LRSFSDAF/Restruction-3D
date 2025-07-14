'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:53:45
LastEditTime: 2025-07-13 17:31:33
LastEditors: Damocles_lin
'''
from reconstruction.pipeline import run_colmap_pipeline
import argparse
import os
import time

DEFAULT_IMAGE_DIR = "./images"  # 默认图像目录
DEFAULT_OUTPUT_DIR = "./output"  # 默认输出目录

def main():
    # 记录总开始时间
    total_start = time.time()

    # 创建解析器对象
    parser = argparse.ArgumentParser(description="运行COLMAP三维重建流程")
    # 添加可选参数
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR, 
                        help=f"输入图像目录路径 (默认: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help=f"输出结果目录路径 (默认: {DEFAULT_OUTPUT_DIR})")
    # 解析参数
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.image_dir):
        print(f"警告: 图像目录 '{args.image_dir}' 不存在，使用默认路径 '{DEFAULT_IMAGE_DIR}'")
        args.image_dir = DEFAULT_IMAGE_DIR
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始处理: 图像目录={args.image_dir}, 输出目录={args.output_dir}")
    
    # 运行COLMAP流程
    run_colmap_pipeline(args.image_dir, args.output_dir)
    
    # 计算总耗时
    total_elapsed = time.time() - total_start
    print(f"处理完成！总耗时: {total_elapsed:.2f}秒")

if __name__ == "__main__":
    main()