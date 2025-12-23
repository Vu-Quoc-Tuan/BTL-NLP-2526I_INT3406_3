from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os

@dataclass
class ModelConfig:
    d_model: int = 512       # Kích thước vector ẩn.
    n_layers: int = 6        # Số lớp Encoder và Decoder chồng lên nhau.
    heads: int = 8           # Số lượng Attention Heads.
    dropout: float = 0.1     # Tỉ lệ tắt neuron ngẫu nhiên để tránh overfitting.
    vocab_size: int = 32768  # Kích thước bộ từ điển.
    max_len: int = 120       # Độ dài tối đa của câu (tính theo token).

@dataclass
class TrainingConfig:
    batch_size: int = 64     # Số lượng câu học cùng một lúc.
    lr: float = 0.0001       # Tốc độ học (Learning Rate).
    epochs: int = 20         # Số vòng lặp qua toàn bộ dữ liệu.
    warmup_ratio: float = 0.07 # Tỉ lệ bước khởi động (7% tổng số bước).
    save_dir: str = "checkpoints" # Thư mục lưu model.
    log_dir: str = "logs"         # Thư mục lưu nhật ký.
    seed: int = 42                # Random Seed để tái lập kết quả.
    num_workers: int = 4          # Số luồng load dữ liệu. Tăng lên 4-8 nếu RAM nhiều.

@dataclass
class InferenceConfig:
    beam_size: int = 1            # Kích thước beam. 1 = Greedy.
    decoding_method: str = "greedy" # "greedy" hoặc "beam"
    eval_batch_size: int = 32     # Batch size cho model evaluation
    length_penalty: float = 0.6   # Phạt câu ngắn/dài (alpha). Thường dùng 0.6 - 1.0.

@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "transformer-from-scratch"
    api_key: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None

@dataclass
class DataConfig:
    min_len: int = 3
    max_ratio: float = 2.0
    bpe_dropout: float = 0.1

@dataclass
class AppConfig:
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    @classmethod
    def load(cls, path: str) -> 'AppConfig':
        """Load cấu hình từ file YAML."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")
            
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Parse thủ công từ dict sang dataclass (để demo kiểm soát chặt chẽ kiểu dữ liệu)
        # Trong thực tế có thể dùng thư viện dacite hoặc pydantic
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))
        inference_cfg = InferenceConfig(**config_dict.get('inference', {}))
        wandb_cfg = WandbConfig(**config_dict.get('wandb', {}))
        data_cfg = DataConfig(**config_dict.get('data', {}))
        
        return cls(
            model=model_cfg, 
            training=training_cfg, 
            inference=inference_cfg, 
            wandb=wandb_cfg,
            data=data_cfg
        )
