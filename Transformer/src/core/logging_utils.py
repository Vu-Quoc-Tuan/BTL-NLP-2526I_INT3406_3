import logging
import sys
import os
from pathlib import Path

def setup_logging(log_dir: str = "logs", log_name: str = "train.log", level: int = logging.INFO):
    """
    Thiết lập cấu hình ghi nhật ký (logging).
    - Ghi log ra file tại đường dẫn log_dir/log_name
    - Ghi log ra màn hình console
    """
    # Tạo thư mục log nếu chưa tồn tại
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Định dạng log: [Thời gian] [Level] [Tên module] - Nội dung
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Cấu hình logger gốc
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Xóa các handler cũ để tránh bị log duplicate khi hàm này được gọi nhiều lần
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler: Ghi log vào file
    file_path = os.path.join(log_dir, log_name)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # Stream Handler: Ghi log ra màn hình console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    
    logging.info(f"Logging initialized. Log file: {file_path}")
    return logger

def get_logger(name: str) -> logging.Logger:
    """Hàm tiện ích để lấy logger theo tên cụ thể."""
    return logging.getLogger(name)
