pip install -r requirements.txt


python scripts/dedup_minhash.py --src data/raw/train.en --tgt data/raw/train.vi --out_dir data/dedup --threshold 0.8

python scripts/preprocess_vlsp.py --src_in data/dedup/train.en --tgt_in data/dedup/train.vi --out_dir data/clean --max_tokens 256


python scripts/train_qwen_lora.py \
 --direction en2vi --run_id Qwen_en2vi_A0_mini \
 --model_name qwen/qwen-2.5b-instruct \
 --src data/clean/train.en --tgt data/clean/train.vi --subset 2000 \
 --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 --lr 2e-4 --batch_size 2 --epochs 1


python scripts/generate.py --model_name qwen/qwen-2.5b-instruct --adapter_path runs/Qwen_en2vi_A1/lora_en2vi --direction en2vi --input data/clean/dev.en --output outputs/dev.hyp.en2vi.A1.vi


python scripts/rl_train_grpo.py \
 --model_name qwen/qwen-2.5b-instruct \
 --sft_adapter runs/Qwen_en2vi_A1/lora_en2vi \
 --init_adapter runs/Qwen_en2vi_A1/lora_en2vi \
 --rl_src data/rl_subset/en.txt --rl_tgt data/rl_subset/vi.txt \
 --run_id Qwen_en2vi_A1_rl --direction en2vi --epochs 1 --batch_size 8 --lr 3e-6 --kl_coef 0.01





python scripts/dedup_minhash.py --src data/raw/train.en --tgt data/raw/train.vi --out_dir data/dedup --threshold 0.8

python scripts/preprocess_vlsp.py --src_in data/dedup/train.en --tgt_in data/dedup/train.vi --out_dir data/clean --max_tokens 256

head -n 50000 data/clean/train.en > data/rl_subset/en.txt
head -n 50000 data/clean/train.vi > data/rl_subset/vi.txt

python scripts/train_qwen_lora.py --direction en2vi --run_id Qwen_en2vi_A1 --src data/clean/train.en --tgt data/clean/train.vi --lora_r 16 --lr 2e-4 --batch_size 4 --epochs 3 --max_len 256


python scripts/run_eval_all.py --run_id Qwen_en2vi_A1 --direction en2vi

python scripts/rl_train_grpo.py --model_name qwen/qwen-2.5b-instruct --sft_adapter runs/Qwen_en2vi_A1/lora_en2vi --init_adapter runs/Qwen_en2vi_A1/lora_en2vi --rl_src data/rl_subset/en.txt --rl_tgt data/rl_subset/vi.txt --run_id Qwen_en2vi_A1_rl --direction en2vi --epochs 1 --batch_size 8 --lr 3e-6


# point generate to runs/Qwen_en2vi_A1_rl/lora_rl
python scripts/generate.py --model_name qwen/qwen-2.5b-instruct --adapter_path runs/Qwen_en2vi_A1_rl/lora_rl --direction en2vi --input data/clean/dev.en --output outputs/dev.hyp.en2vi.A1_rl.vi
python scripts/eval_bleu.py --hyp outputs/dev.hyp.en2vi.A1_rl.vi --ref data/clean/dev.vi


