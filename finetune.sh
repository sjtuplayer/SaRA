cd examples
python3 finetune.py \
   --config=configs/Barbie.json \
   --output_dir=results \
   --threshold=2e-3 \
   --lr_scheduler=cosine \
   --progressive_iter=2500 \
   --lambda_rank=0.0005 \
