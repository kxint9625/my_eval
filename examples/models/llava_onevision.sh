export HF_HOME="~/.cache/huggingface"

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=/mnt/data/kexin/spatialvlm/LLaVA-NeXT/outputs/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_0822/checkpoint-13500,conv_template=qwen_1_5,device_map=None,model_name=llava_qwen \
    --tasks=nuscenes_quan \
    --batch_size=1