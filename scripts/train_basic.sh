#!/bin/bash

# Basic SD3.5 LoRA Training Script
# This script provides a simple starting point for training LoRA adapters

echo "🚀 Starting SD3.5 LoRA Training (Basic Configuration)"
echo "=================================================="

# Check if accelerate is configured
if ! accelerate env >/dev/null 2>&1; then
    echo "⚠️ Accelerate not configured. Running accelerate config..."
    accelerate config
fi

# Set default values (can be overridden by environment variables)
MODEL_NAME=${MODEL_NAME:-"stabilityai/stable-diffusion-3.5-medium"}
DATASET_DIR=${DATASET_DIR:-"./examples/dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/sd35-lora-basic"}
RESOLUTION=${RESOLUTION:-1024}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-10}
RANK=${RANK:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-4}

echo "📋 Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Resolution: $RESOLUTION"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  LoRA Rank: $RANK"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
accelerate launch train_text_to_image_lora_sd35.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --train_data_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --resolution $RESOLUTION \
  --train_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --rank $RANK \
  --learning_rate $LEARNING_RATE \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --validation_prompt "a beautiful landscape with mountains and a lake" \
  --validation_epochs 2 \
  --num_validation_images 4 \
  --checkpointing_steps 500 \
  --seed 42 \
  --report_to tensorboard

echo ""
echo "✅ Training completed! Check your results in: $OUTPUT_DIR"
echo "📊 View training logs with: tensorboard --logdir $OUTPUT_DIR/logs" 

#eample
accelerate launch t2i-lora-sd35-3.py \
  --pretrained_model_name_or_path="/scratch/chans/sd-models/stable-diffusion-3.5-medium" \
  --train_data_dir="/home/chans/worldccub/lora_dataset" \
  --output_dir="/scratch/chans/lora-output-t2i-sd35-3" \
  --resolution 1024 \
  --train_batch_size 1 \
  --num_train_epochs 10 \
  --rank 128 \
  --learning_rate 4e-4 \
  --text_encoder_lr 2e-5 \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 250 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --train_text_encoder \
  --report_to wandb \
  --seed 42