'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:50:16
LastEditTime: 2025-07-06 16:50:25
LastEditors: Damocles_lin
'''
import time
import logging

class Timer:
    """计时器类，用于记录各步骤耗时"""
    def __init__(self):
        self.start_time = None
        self.step_times = {}
        self.current_step = None
        self.step_start_times = {}
    
    def start(self, step_name):
        """开始一个新步骤的计时"""
        if self.current_step:
            self.end()  # 结束当前步骤
            
        self.current_step = step_name
        self.start_time = time.time()
        self.step_start_times[step_name] = self.start_time
        logging.info(f"开始步骤: {step_name}")
    
    def end(self):
        """结束当前步骤的计时"""
        if self.current_step and self.start_time:
            elapsed = time.time() - self.start_time
            self.step_times[self.current_step] = elapsed
            logging.info(f"完成步骤: {self.current_step} | 耗时: {elapsed:.2f}秒")
            self.current_step = None
            self.start_time = None
    
    def get_step_time(self, step_name):
        """获取特定步骤的耗时"""
        return self.step_times.get(step_name, 0.0)
    
    def total_time(self):
        """计算总耗时"""
        return sum(self.step_times.values())
    
    def log_summary(self):
        """记录所有步骤的耗时摘要"""
        logging.info("\n===== 处理步骤耗时摘要 =====")
        for step, elapsed in self.step_times.items():
            logging.info(f"{step}: {elapsed:.2f}秒")
        logging.info(f"总耗时: {self.total_time():.2f}秒")
        logging.info("===========================\n")
        
        # 返回摘要字符串，可用于保存到文件
        summary_lines = ["===== 处理步骤耗时摘要 ====="]
        for step, elapsed in self.step_times.items():
            summary_lines.append(f"{step}: {elapsed:.2f}秒")
        summary_lines.append(f"总耗时: {self.total_time():.2f}秒")
        summary_lines.append("===========================\n")
        return "\n".join(summary_lines)