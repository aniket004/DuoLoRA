export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# for subjectplop test set
# for subject
export LORA_PATH="PATH/content_loras/lora-sdxl-slime"
export INSTANCE_DIR="PATH/subjectplop_testset/objects/slime"
export PROMPT="a sbu1 slime in szn1 style"

# for style
export LORA_PATH2="PATH/style_loras/lora-sdxl-sty_8"
export INSTANCE_DIR2="PATH/subjectplop_testset/styles/sty_8"
export PROMPT2="a sbu2 temple of in szn2 style"

export PROMPT3="a sbu1 slime in szn2 style"

# general 
export OUTPUT_DIR="PATH/output/test/cycle_rank_layer_prior-sdxl-slime-sty8_sim_0_cycle_0.01_rank_0.l"
export OUTPUT_INFERENCE_DIR="PATH/output/test"
export VALID_PROMPT="a sbu1 slime in szn2 style"


########################################################
# clear all the .db files in the subdirectory

# Find and remove all .db files
find "$INSTANCE_DIR" -type f -name '*.db' -exec rm -rf {} +

echo "All *.db files have been removed from ${INSTANCE_DIR} and its subdirectories."

# Find and remove all .db files
find "$INSTANCE_DIR2" -type f -name '*.db' -exec rm -rf {} +

echo "All *.db files have been removed from ${INSTANCE_DIR2} and its subdirectories."

#################################
### Train the LoRA ##############
#################################

python train_dreambooth_ziplora_rank_layer_prior_cycle_rev_init_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --output_dir=$OUTPUT_DIR \
    --lora_name_or_path=$LORA_PATH \
    --instance_prompt="${PROMPT}" \
    --instance_data_dir=$INSTANCE_DIR \
    --lora_name_or_path_2=$LORA_PATH2 \
    --instance_prompt_2="${PROMPT2}" \
    --instance_prompt_3="${PROMPT3}" \
    --instance_data_dir_2=$INSTANCE_DIR2 \
    --resolution=1024 \
    --train_batch_size=1 \
    --learning_rate=0.01 \
    --similarity_lambda=0.0 \
    --cycle_lambda=0.1 \
    --rev_cycle_lambda=0.1 \
    --rank_lambda=0.01 \
    --nuclear_norm_coeff=1.0 \
    --sparsity_coeff=1.0 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=100 \
    --content_init_thresh=0.1\
    --style_init_thresh=0.0\
    --seed="0" \
    --mixed_precision="fp16" \
    --merger_init_method="boolean" \
    --gradient_checkpointing \
    --use_8bit_adam \

##################
### test #########
##################

# inference from the generated model
python inference_composite.py --ziplora_name_or_path $OUTPUT_DIR --out_dir $OUTPUT_INFERENCE_DIR --prompt 'a sbu1 slime in szn2 style in Ancient temple hall with carved pillars in dramatic lighting style'

