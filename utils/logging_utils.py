'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:49:53
LastEditTime: 2025-07-08 21:08:58
LastEditors: Damocles_lin
'''
import logging
import pycolmap
import sys
from pathlib import Path
from datetime import datetime

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
    pycolmap.logging.minloglevel = pycolmap.logging.Level.INFO.value

    # 添加Pillow日志过滤器
    class PillowFilter(logging.Filter):
        def filter(self, record):
            # 过滤掉Pillow的调试日志
            if record.name.startswith('PIL.'):
                return record.levelno >= logging.WARNING
            return True    
    
    # 创建文件日志处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 创建日志格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(PillowFilter())
    
    # 获取根日志记录器并添加文件处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(PillowFilter())
    root_logger.addHandler(console_handler)

    # 同时设置Pillow的日志级别
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info(f"日志文件已创建: {log_file}")
    return log_file