# validate_staged_model.py
import sys
import torch
from pathlib import Path
import re

# 确保项目根目录在 sys.path 中
try:
    import diffsynth
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))

# 确保导入的是修改后的 WanVideoPipeline
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.data.video import save_video
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor

def find_latest_checkpoint_step(checkpoint_dir: Path) -> Optional[str]:
    """在目录中查找最新步骤的 checkpoint 文件"""
    steps = [int(f.stem.split('_')[-1]) for f in checkpoint_dir.glob("*.pth") if f.stem.split('_')[-1].isdigit()]
    if not steps:
        return None
    return str(max(steps))

# --- 1. 配置路径和参数 ---
# LoRA 路径（您训练时使用的原始 LoRA）
LORA_PATH = "/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/lora_outputs/inkwash_style_v1/epoch-18.safetensors"

# **重要**: 指向您第一步训练的输出根目录
# 例如：'./runs/staged_training_final/staged_lr0.0001_warmup2000_17557XXXXX'
TRAINING_OUTPUT_DIR = Path("./runs/staged_training_final/staged_lr0.0001_warmup2000_17557XXXXX") # << 替换为您的实际路径

CHECKPOINT_DIR = TRAINING_OUTPUT_DIR / "checkpoints"

# 自动查找最新步骤
LATEST_STEP = find_latest_checkpoint_step(CHECKPOINT_DIR)
if LATEST_STEP is None:
    print(f"Error: No checkpoints found in {CHECKPOINT_DIR}")
    sys.exit(1)
print(f"Found latest checkpoint step: {LATEST_STEP}")

TEMPORAL_MODULE_PATH = CHECKPOINT_DIR / f"temporal_module_step_{LATEST_STEP}.pth"
FLOW_PREDICTOR_PATH = CHECKPOINT_DIR / f"flow_predictor_step_{LATEST_STEP}.pth"
TRAINED_LORA_PATH = CHECKPOINT_DIR / f"lora_step_{LATEST_STEP}.pth"

# 与训练时一致的参数
LATENT_CHANNELS = 16
STYLE_DIM = 128

device = torch.device("cuda")
dtype = torch.float16 # 使用 float16 以匹配训练

# --- 2. 加载模型和所有训练好的模块 ---
print("Step 1: Loading base WanVideoPipeline...")
pipe = WanVideoPipeline.from_pretrained(torch_dtype=dtype)
pipe.to(device)

print(f"Step 2: Loading originally trained LoRA style from {LORA_PATH}...")
pipe.load_lora(pipe.dit, LORA_PATH, alpha=1.0)
# 如果分阶段训练也更新了 LoRA，可以选择加载训练后的 LoRA
if TRAINED_LORA_PATH.exists():
    print(f"Step 2.1: Overwriting with fine-tuned LoRA from {TRAINED_LORA_PATH}...")
    pipe.load_lora(pipe.dit, TRAINED_LORA_PATH, alpha=1.0)

print("Step 3: Loading and Attaching Temporal Modules...")
temporal_module = TemporalModule(latent_channels=LATENT_CHANNELS, style_dim=STYLE_DIM).to(device, dtype=dtype).eval()
temporal_module.load_state_dict(torch.load(TEMPORAL_MODULE_PATH, map_location=device))

if FLOW_PREDICTOR_PATH.exists():
    flow_predictor = LatentFlowPredictor(in_channels=LATENT_CHANNELS).to(device, dtype=dtype).eval()
    flow_predictor.load_state_dict(torch.load(FLOW_PREDICTOR_PATH, map_location=device))
    print("Flow Predictor loaded.")
else:
    flow_predictor = None
    print("Flow Predictor checkpoint not found, will use None.")

# 挂载到 Pipeline 实例
pipe.temporal_module = temporal_module
pipe.flow_predictor = flow_predictor
print("Temporal modules attached to pipeline.")

# --- 3. 执行推理 (带时序平滑) ---
prompt = "a majestic dragon flying through the clouds, fast and graceful movement, inkwash style, fluid lines."
output_path_smoothed = "video_smoothed_dragon_4sec.mp4"

print(f"\nStep 4: Generating SMOOTHED video (4 seconds)...")
video_smoothed = pipe(
    prompt, 
    seed=42, 
    num_frames=64, # 增加帧数
    height=512, 
    width=512,
    num_inference_steps=50,
    cfg_scale=7.0
)
save_video(video_smoothed, output_path_smoothed, fps=16) # 提高 fps
print(f"Success! Smoothed video saved as: {output_path_smoothed}")

# --- 4. 生成基线对比视频 ---
print("\n--- Generating BASELINE video for comparison ---")
# 卸载模块以生成基线
pipe.temporal_module = None 
pipe.flow_predictor = None 

output_path_baseline = "video_baseline_dragon_4sec.mp4"

video_baseline = pipe(
    prompt, 
    seed=42, 
    num_frames=64,
    height=512, 
    width=512,
    num_inference_steps=50,
    cfg_scale=7.0
)
save_video(video_baseline, output_path_baseline, fps=16)
print(f"Success! Baseline video saved as: {output_path_baseline}")