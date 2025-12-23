import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import os
import logging
from tqdm import tqdm
from ..core.config import TrainingConfig
import wandb
import sacrebleu
import re

logger = logging.getLogger(__name__)

class Trainer:
    """
    Class quản lý quy trình huấn luyện (Training Loop).
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 scheduler = None,
                 tokenizer = None,
                 max_len: int = 120,
                 device: str = "cpu"):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion # Hàm loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Xử lý Config (Chấp nhận AppConfig hoặc TrainingConfig để tương thích ngược)
        if hasattr(config, 'training') and hasattr(config, 'wandb'):
            self.app_config = config
            self.config = config.training
            self.wandb_config = config.wandb
        else:
            self.config = config
            self.wandb_config = None

        self.scheduler = scheduler
        
        # Init WandB
        if self.wandb_config and self.wandb_config.enabled:
            # Kiểm tra xem wandb đã được init chưa (ví dụ trong notebook hoặc lần chạy trước)
            if wandb.run is None:
                logger.info("Initializing WandB...")
                wandb.init(
                    project=self.wandb_config.project,
                    entity=self.wandb_config.entity,
                    name=self.wandb_config.name,
                    config=vars(self.config) 
                )
            else:
                 logger.info("WandB already initialized.")

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        
        self.global_step = 0
        
        # Tạo thư mục lưu checkpoint
        os.makedirs(self.config.save_dir, exist_ok=True)

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_idx+1}/{self.config.epochs}")
        
        for batch in pbar:
            # batch là output của Dataset (src, tgt, src_text, tgt_text)
            encoder_input = batch['encoder_input'].to(self.device) # (B, Seq_Len)
            decoder_input = batch['decoder_input'].to(self.device) # (B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(self.device) # (B, 1, 1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(self.device) # (B, 1, Seq_Len, Seq_Len)
            label = batch['label'].to(self.device) # (B, Seq_Len)
            
            # 1. Lan truyền xuôi
            # Encoder output: (B, Seq_Len, d_model)
            encoder_output = self.model.encode(encoder_input, encoder_mask)
            # Decoder output: (B, Seq_Len, d_model)
            decoder_output = self.model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            # Logits: (B, Seq_Len, Vocab_Size)
            proj_output = self.model.project(decoder_output)
            
            # 2. Tính Loss
            # Cần flatten về (B * Seq_Len, Vocab_Size) để tính CrossEntropy
            # CrossEntropyLoss của PyTorch yêu cầu input (N, C) và target (N)
            loss = self.criterion(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))
            
            # 3. Lan truyền ngược
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping: Cực kỳ quan trọng với Transformer để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # WandB Log (Step-wise)
            if self.wandb_config and self.wandb_config.enabled:
                current_lr = self.optimizer.param_groups[0]['lr']
                train_ppl = torch.exp(loss).item()
                wandb.log({
                    "train_loss": loss.item(),
                    "train_ppl": train_ppl,
                    "lr": current_lr,
                    "epoch": self.global_step / len(self.train_loader) # Epoch (float)
                })
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Cập nhật progress bar
            # Lấy current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
            
        avg_loss = total_loss / len(self.train_loader)
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch_idx+1} | Train Loss: {avg_loss:.4f} | Train PPL: {avg_ppl:.2f} | LR: {current_lr:.2e}")
        return avg_loss

    def evaluate(self, epoch_idx: int):
        """Đánh giá model trên tập Validation (Loss)."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating Loss"):
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                encoder_mask = batch['encoder_mask'].to(self.device)
                decoder_mask = batch['decoder_mask'].to(self.device)
                label = batch['label'].to(self.device)
                
                # Forward (giống train nhưng không backward)
                encoder_output = self.model.encode(encoder_input, encoder_mask)
                decoder_output = self.model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                proj_output = self.model.project(decoder_output)
                
                # Tính Loss
                loss = self.criterion(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Epoch {epoch_idx+1} | Val Loss: {avg_loss:.4f} | Val PPL: {avg_ppl:.2f}")
        return avg_loss

    def evaluate_bleu(self, epoch_idx: int):
        """Đánh giá BLEU Score trên tập Validation."""
        if not self.tokenizer:
            logger.warning("No tokenizer provided. Skipping BLEU evaluation.")
            return 0.0
            
        logger.info("Computing BLEU on Validation Set...")
        self.model.eval()
        
        hypotheses = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating BLEU"):
                encoder_input = batch['encoder_input'].to(self.device)
                encoder_mask = batch['encoder_mask'].to(self.device)
                tgt_text = batch['tgt_text'] # Ground truth text
                
                # Greedy Decode Batch (đã hỗ trợ trong greedy_decode custom hoặc loop)
                # Dùng loop cho chắc hoặc dùng batch greedy decode nếu đã implement
                # Hiện tại greedy_decode hỗ trợ batch
                
                # Lấy config decoding
                beam_size = 3 # Beam size mặc định cho validation
                if hasattr(self, 'app_config') and hasattr(self.app_config, 'inference'):
                     beam_size = self.app_config.inference.beam_size if self.app_config.inference.beam_size else 1
                
                if beam_size > 1:
                    # Dùng Optimized Batched Beam Search
                    model_out_ids = self.batched_beam_search_decode(
                        encoder_input, 
                        encoder_mask, 
                        beam_size=beam_size, 
                        max_len=self.max_len + 5,
                        start_symbol=self.tokenizer.sos_token_id
                    )
                    # Lưu ý: batched_beam_search_decode trả về (Batch, Seq_Len) của beam tốt nhất
                else:
                    # Dùng Optimized Greedy Decode
                    model_out_ids = self.greedy_decode(
                        encoder_input, 
                        encoder_mask, 
                        max_len=self.max_len + 5, 
                        start_symbol=self.tokenizer.sos_token_id
                    )
                
                # Detokenize
                model_out_ids_list = model_out_ids.tolist()
                for i, out_ids in enumerate(model_out_ids_list):
                    # Decode
                    pred_text = self.tokenizer.decode(out_ids)
                    
                    # Hậu xử lý (sửa lỗi khoảng trắng dấu câu) qua detokenize
                    if hasattr(self.tokenizer, 'detokenize'):
                         pred_text = self.tokenizer.detokenize(pred_text)
                         # Ref text cũng cần xử lý tương tự để tính BLEU công bằng
                         ref_text = self.tokenizer.detokenize(tgt_text[i])
                    else:
                         # Fallback nếu chưa update tokenizer (hoặc dùng tokenizer khác)
                         pred_text = re.sub(r'\s+([.,!?:;])', r'\1', pred_text).strip()
                         ref_text = re.sub(r'\s+([.,!?:;])', r'\1', tgt_text[i]).strip()
                    
                    hypotheses.append(pred_text)
                    references.append(ref_text)
        
        # Tính toán BLEU
        # references cho corpus_bleu yêu cầu dạng List[List[str]] (hỗ trợ nhiều reference cho 1 câu)
        # Nên cần bọc references trong một list
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        
        logger.info(f"Epoch {epoch_idx+1} | Val BLEU: {bleu.score:.2f}")
        return bleu.score

    def save_checkpoint(self, epoch_idx: int, val_loss: float, val_bleu: float):
        path = os.path.join(self.config.save_dir, f"checkpoint_epoch_{epoch_idx+1:02d}.pt")
        # Ưu tiên lưu toàn bộ AppConfig (chứa cả model, data, inference config)
        # Nếu không có lưu TrainingConfig
        config_to_save = getattr(self, 'app_config', self.config)
        
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config_to_save,
            'val_loss': val_loss,
            'val_bleu': val_bleu
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def fit(self):
        logger.info(f"Starting training on device: {self.device}")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)
            val_bleu = self.evaluate_bleu(epoch)
            
            # WandB Log (Epoch-wise)
            if self.wandb_config and self.wandb_config.enabled:
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                wandb.log({
                    "val_loss": val_loss, 
                    "val_ppl": val_ppl,
                    "val_bleu": val_bleu,
                    "epoch": epoch
                })
            
            # Lưu checkpoint sau mỗi epoch
            self.save_checkpoint(epoch, val_loss, val_bleu)

        # Kết thúc training: Dịch thử vài câu
        if self.tokenizer:
            self.sample_translation(num_samples=3)

    def beam_search_decode(self, source, source_mask, beam_size=3, max_len=None, length_penalty_alpha=0.6, start_symbol=None, end_symbol=None):
        sos_idx = self.tokenizer.sos_token_id
        eos_idx = self.tokenizer.eos_token_id
        if start_symbol is None: start_symbol = sos_idx
        if end_symbol is None: end_symbol = eos_idx
        if max_len is None: max_len = 60

        dev = self.device
        
        # 1. Encode (Chỉ làm 1 lần)
        # encoder_output: (1, Seq_Len, d_model)
        encoder_output = self.model.encode(source, source_mask)
        
        # 2. Khởi tạo Beam
        # Mỗi phần tử trong beam là 1 tuple: (tensor_sequence, score, finished)
        # score là log_prob tích lũy
        current_beams = [(torch.empty(1, 1).fill_(start_symbol).type_as(source).to(dev), 0.0, False)]
        
        for i in range(max_len):
            all_candidates = []
            
            # Mở rộng từng beam
            for seq, score, finished in current_beams:
                if finished:
                    all_candidates.append((seq, score, True))
                    continue
                
                # Tạo mask
                decoder_mask = torch.tril(torch.ones((1, seq.size(1), seq.size(1)))).type_as(source).to(dev)
                
                # Decode
                out = self.model.decode(seq, encoder_output, source_mask, decoder_mask)
                
                # Lấy token cuối: (1, d_model)
                prob = self.model.project(out[:, -1])
                
                # Lấy log_softmax để cộng dồn score
                log_probs = torch.log_softmax(prob, dim=1) # (1, Vocab_Size)
                
                # Lấy Top-K token tốt nhất cho nhánh này
                # Để tối ưu, nếu beam_size nhỏ thì lấy luôn k cái
                topk_probs, topk_ids = torch.topk(log_probs, beam_size, dim=1)
                
                for k in range(beam_size):
                    next_token_id = topk_ids[0, k].item()
                    next_token_prob = topk_probs[0, k].item()
                    
                    new_score = score + next_token_prob
                    new_seq = torch.cat([seq, torch.empty(1, 1).type_as(source).fill_(next_token_id).to(dev)], dim=1)
                    
                    is_finished = (next_token_id == end_symbol)
                    all_candidates.append((new_seq, new_score, is_finished))
            
            # Sắp xếp tất cả candidates theo score (giảm dần)
            # Áp dụng Length Penalty khi so sánh: Score = log_prob / (len ** alpha)
            ordered = sorted(all_candidates, key=lambda x: x[1] / (x[0].size(1) ** length_penalty_alpha), reverse=True)
            
            # Pruning: Chỉ giữ lại top Beam_Size
            current_beams = ordered[:beam_size]
            
            # Kiểm tra nếu tất cả top beams đều đã finish
            if all([x[2] for x in current_beams]):
                break
                
        # Trả về sequence của beam tốt nhất
        best_seq = current_beams[0][0]
        return best_seq.squeeze(0)

    def batched_beam_search_decode(self, source, source_mask, beam_size=3, max_len=None, length_penalty_alpha=0.6, start_symbol=None, end_symbol=None):
        """
        Phiên bản vectorized của Beam Search, hỗ trợ xử lý cả batch cùng lúc.
        Tăng tốc độ inference đáng kể so với loop.
        """
        sos_idx = self.tokenizer.sos_token_id
        eos_idx = self.tokenizer.eos_token_id
        pad_idx = self.tokenizer.pad_token_id
        
        if start_symbol is None: start_symbol = sos_idx
        if end_symbol is None: end_symbol = eos_idx
        if max_len is None: max_len = 60

        dev = self.device
        batch_size = source.size(0)

        # 1. Encode (Batch, Seq_Len, d_model)
        encoder_output = self.model.encode(source, source_mask)
        
        # 2. Chuẩn bị cho Beam Search
        # Chúng ta sẽ mở rộng (expand) encoder output và mask để phù hợp với (Batch * Beam_Size)
        # (Batch, Seq, Dim) -> (Batch, Beam, Seq, Dim) -> (Batch * Beam, Seq, Dim)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, beam_size, 1, 1).view(batch_size * beam_size, -1, encoder_output.size(-1))
        # (Batch, 1, 1, Seq) -> (Batch, Beam, 1, 1, Seq) -> (Batch * Beam, 1, 1, Seq)
        source_mask = source_mask.unsqueeze(1).repeat(1, beam_size, 1, 1, 1).view(batch_size * beam_size, 1, 1, -1)

        # Decoder input ban đầu: (Batch * Beam, 1) filled with SOS
        decoder_input = torch.empty(batch_size * beam_size, 1).fill_(start_symbol).type_as(source).to(dev)

        # Beam Scores: (Batch, Beam)
        # Khởi tạo điểm: Beam đầu tiên = 0, các beam còn lại = -inf để ép model chọn beam đầu tiên ở bước 1
        scores = torch.full((batch_size, beam_size), -float('inf'), device=dev)
        scores[:, 0] = 0.0
        
        # Flatten scores để cộng dồn dễ dàng: (Batch * Beam)
        scores = scores.view(-1) 

        # Để lưu lại các token đã chọn để reconstruct đường đi
        # Mỗi bước sẽ lưu (indices của beam trước, token mới)
        step_history = []
        
        # Trạng thái đã hoàn thành của từng beam (Batch * Beam)
        # Nếu True nghĩa là beam này đã gặp EOS
        is_finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=dev)

        # Trạng thái đã hoàn thành của từng beam (Batch * Beam)
        # Nếu True nghĩa là beam này đã gặp EOS
        is_finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=dev)

        # KV Cache Storage
        past_key_values = None

        for i in range(max_len):
            # 1. Chuẩn bị Input cho Decoder
            if i == 0:
                # Step 0: Input là SOS token
                # decoder_input đang là (Batch * Beam, 1) chứa SOS
                current_input = decoder_input
                
                # Mask: Triangular mask cho step 0 (1x1)
                decoder_mask = torch.tril(torch.ones((1, 1, 1), device=dev)).type_as(source)
            else:
                # Step > 0: Input là token vừa sinh ra ở bước trước
                # decoder_input đã được cập nhật full sequence, ta chỉ lấy token cuối
                current_input = decoder_input[:, -1:] # (Batch * Beam, 1)
                
                # Mask: Khi dùng Cache, ta chỉ decode cho 1 token (query len = 1).
                # Trong Inference không padding, token hiện tại luôn được attend vào toàn bộ quá khứ.
                # Do đó, decoder_mask có thể là None (Attend All).
                decoder_mask = None 
            
            # 2. Decode & Update Cache
            # out: (Batch * Beam, 1, d_model)
            # new_cache: List[Tuple(Keys, Values)]
            out, new_cache = self.model.decode(current_input, encoder_output, source_mask, decoder_mask, past_key_values=past_key_values, use_cache=True)
            
            # 3. Project & Select Top-K
            # Lấy token cuối (cũng là token duy nhất của out)
            logits = self.model.project(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Cập nhật điểm
            next_scores = scores.unsqueeze(1) + log_probs
            
            # Xử lý Finished Beams
            if is_finished.any():
                mask_finished = is_finished.unsqueeze(1).expand_as(next_scores)
                next_scores[mask_finished] = -float('inf')
                finished_indices = torch.nonzero(is_finished).squeeze()
                if finished_indices.numel() > 0:
                    next_scores[finished_indices, pad_idx] = scores[finished_indices]

            # Top-K Selection
            next_scores = next_scores.view(batch_size, -1)
            topk_scores, topk_indices = torch.topk(next_scores, beam_size, dim=1)
            
            prev_beam_indices = torch.div(topk_indices, self.tokenizer.vocab_size, rounding_mode='floor')
            next_tokens = topk_indices % self.tokenizer.vocab_size
            
            step_history.append((prev_beam_indices, next_tokens))
            
            scores = topk_scores.view(-1)
            
            # 4. Sắp xếp lại & Cập nhật Cấu trúc Dữ liệu
            batch_offset = (torch.arange(batch_size, device=dev) * beam_size).unsqueeze(1)
            gather_indices = (batch_offset + prev_beam_indices).view(-1) # (Batch * Beam)
            
            # Cập nhật decoder_input (Full Sequence để return sau này)
            selected_sequences = decoder_input[gather_indices]
            next_tokens_flat = next_tokens.view(-1, 1)
            decoder_input = torch.cat([selected_sequences, next_tokens_flat], dim=1)
            
            # Cập nhật Cache (QUAN TRỌNG: Sắp xếp lại theo Beam)
            # Cấu trúc new_cache: List of (self_k, self_v). (Cross cache là None)
            # self_k shape: (Batch * Beam, h, Seq_Len_Past, d_k)
            reordered_cache = []
            for layer_cache in new_cache:
                self_kv, cross_kv = layer_cache
                # Sắp xếp lại Self Attention Cache
                # self_kv là (key, value)
                k, v = self_kv
                k_new = k[gather_indices]
                v_new = v[gather_indices]
                
                reordered_cache.append(((k_new, v_new), None))
                
            past_key_values = reordered_cache
            
            # Cập nhật is_finished
            newly_finished = (next_tokens_flat.squeeze(1) == end_symbol)
            is_finished = is_finished[gather_indices]
            is_finished = is_finished | newly_finished
            
            if is_finished.all():
                break
                
        # --- Tái tạo kết quả tốt nhất ---
        # Chúng ta lấy beam có điểm cao nhất (cột 0 của topk cuối cùng) cho mỗi batch
        # Sắp xếp lại lần cuối dựa trên Length Penalty
        # decoder_input: (Batch * Beam, Final_Len)
        # scores: (Batch * Beam)
        
        # Reshape về (Batch, Beam)
        final_scores = scores.view(batch_size, beam_size)
        final_seqs = decoder_input.view(batch_size, beam_size, -1)
        
        # Tính Length Penalty
        # mask != pad & mask != eos (tùy, thường chỉ ignore pad)
        seq_lens = (final_seqs != pad_idx).sum(dim=2).float()
        final_scores_penalized = final_scores / (seq_lens ** length_penalty_alpha)
        
        # Lấy kết quả tốt nhất
        _, best_indices = torch.max(final_scores_penalized, dim=1) # (Batch,)
        
        # Gom các chuỗi tốt nhất
        best_sequences = []
        for b in range(batch_size):
            best_idx = best_indices[b].item()
            seq = final_seqs[b, best_idx]
            best_sequences.append(seq)
            
        return torch.stack(best_sequences) # (Batch, Seq_Len)

    def greedy_decode(self, source, source_mask, max_len=None, start_symbol=None):
        """Hàm dịch (Greedy Decode) có sử dụng KV Cache."""
        sos_idx = self.tokenizer.sos_token_id
        eos_idx = self.tokenizer.eos_token_id
        if start_symbol is None:
            start_symbol = sos_idx
        if max_len is None:
            max_len = 60

        dev = self.device
        batch_size = source.size(0)
        
        # 1. Encode
        encoder_output = self.model.encode(source, source_mask)
        
        # 2. Decode từng bước
        # Khởi tạo decoder input với token SOS: (Batch, 1)
        decoder_input = torch.empty(batch_size, 1).fill_(start_symbol).long().to(dev)
        
        # Trạng thái giữ xem câu nào đã xong (gặp EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(dev)
        
        # KV Cache
        past_key_values = None
        
        # Để lưu toàn bộ câu (cho việc return)
        full_indices = decoder_input
        
        for i in range(max_len):
            # Với Cache: Input luôn chỉ là token cuối cùng vừa sinh ra (trừ bước 0)
            if i == 0:
                current_input = decoder_input
                # Mask trivial cho bước đầu
                decoder_mask = torch.tril(torch.ones((1, 1, 1))).type_as(source).to(dev)
            else:
                current_input = decoder_input # decoder_input đã được gán bằng next_word (Last token) ở cuối loop
                # Mask không cần thiết (hoặc full 1) vì ta luôn attend hết quá khứ
                decoder_mask = None
            
            # Decode
            # out: (Batch, 1, d_model)
            out, past_key_values = self.model.decode(
                current_input, 
                encoder_output, 
                source_mask, 
                decoder_mask, 
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Project sang vocab
            prob = self.model.project(out[:, -1]) # (Batch, Vocab)
            
            # Lấy token có xác suất cao nhất
            _, next_word = torch.max(prob, dim=1)
            
            # Cập nhật input cho vòng lặp sau (chỉ lấy token mới nhất)
            decoder_input = next_word.unsqueeze(1) # (Batch, 1)
            
            # Lưu vào kết quả
            full_indices = torch.cat([full_indices, decoder_input], dim=1)
            
            # Kiểm tra EOS
            is_eos = (next_word == eos_idx)
            finished = finished | is_eos
            
            # Nếu tất cả batch đều xong thì dừng sớm
            if finished.all():
                break
            
        return full_indices # (Batch, Seq_Len)

    def sample_translation(self, loader: DataLoader = None, num_samples: int = 3):
        """Lấy ngẫu nhiên vài câu từ loader và dịch thử."""
        target_loader = loader if loader else self.val_loader
        logger.info(f"--- Translating {num_samples} sample sentences ---")
        self.model.eval()
        
        count = 0
        with torch.no_grad():
            for batch in target_loader:
                # Lấy 1 batch
                encoder_input = batch['encoder_input'].to(self.device)
                encoder_mask = batch['encoder_mask'].to(self.device)
                src_text = batch['src_text']
                tgt_text = batch['tgt_text']
                
                # Duyệt qua từng câu trong batch
                for i in range(encoder_input.size(0)):
                    count += 1
                    src = encoder_input[i].unsqueeze(0) # (1, Seq_Len)
                    mask = encoder_mask[i].unsqueeze(0) # (1, 1, 1, Seq_Len)
                    
                    # Gọi hàm Greedy Decode
                    model_out_ids = self.greedy_decode(src, mask, max_len=self.max_len + 5, start_symbol=self.tokenizer.sos_token_id)
                    
                    # Chuyển IDs -> Text (model_out_ids là (1, Seq), cần flatten)
                    model_out_text = self.tokenizer.decode(model_out_ids[0].tolist())
                    
                    # Detokenize để hiển thị đẹp
                    if hasattr(self.tokenizer, 'detokenize'):
                        model_out_text = self.tokenizer.detokenize(model_out_text)
                        src_display = self.tokenizer.detokenize(src_text[i]) # Optional: làm đẹp cả SRC nếu muốn
                        tgt_display = self.tokenizer.detokenize(tgt_text[i]) # Optional: làm đẹp cả TGT nếu muốn
                    else:
                        src_display = src_text[i]
                        tgt_display = tgt_text[i]

                    print("-" * 50)
                    print(f"SRC: {src_display}")
                    print(f"TGT: {tgt_display}")
                    print(f"PRED: {model_out_text}")
                    
                    if count >= num_samples:
                        print("-" * 50)
                        return
