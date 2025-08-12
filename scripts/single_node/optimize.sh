export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

# 1 GPU
accelerate launch --num_processes=1 scripts/optimize_sd3.py --config config/optimize.py:ocr

# 4 GPU
# accelerate launch --num_processes=4 scripts/optimize_sd3.py --config config/optimize.py:ocr --config.sample.train_batch_size=8