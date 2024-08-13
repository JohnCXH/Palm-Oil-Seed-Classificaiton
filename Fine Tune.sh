 torchrun -m training.main \
    --batch-size 50 \
    --precision fp32 \
    --workers 4 \
    --save-frequency 10 \
    --name "Final Report/Generation 7 Prompt" \
    --seed 420 \
    --logs="logs" \
    --dataset-type csv \
    --report-to tensorboard \
    --csv-separator=";" \
    --train-data "CSV/Train Seed [Default,Masked,Augmented,Crappified,Interpolated].csv" \
    --val-data "CSV/Valid Seed [Validation] [NT].csv" \
    --csv-img-key filename \
    --csv-caption-key captions \
    --warmup 10000 \
    --lr=5e-5 \
    --beta1=0.9 \
    --beta2=0.98 \
    --wd=0.1 \
    --epochs=30 \
    --model "ViT-B-32" \
    --pretrained "datacomp_xl_s13b_b90k"

   #--model "ViT-H-14-378-quickgelu" \
   #--pretrained dfn5b
   #--pretrained datacomp_xl_s13b_b90k 
