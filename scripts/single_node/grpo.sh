export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

# 1 GPU
accelerate launch --num_processes=1 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
# 4 GPU
# accelerate launch --num_processes=4 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
