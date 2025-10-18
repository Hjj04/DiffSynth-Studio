#!/usr/bin/env python3
"""
generate_all_levels.py

批量生成所有4个级别的评估视频
Level 0-3 × 11 Prompts × 3 Seeds = 132 videos

修复版本：兼容PyTorch 2.6+
"""

import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict

# 添加项目路径
repo_root = Path(__file__).resolve().parents[4]
sys.path.append(str(repo_root))

from diffsynth.pipelines.wan_video_new import WanVideoPipeline
from diffsynth.utils import ModelConfig
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.utils import load_state_dict as load_state_dict_from_file
from diffsynth.modules.temporal_module import TemporalModule
from diffsynth.modules.latent_flow_predictor import LatentFlowPredictor

# 导入配置
config_dir = Path(__file__).resolve().parent.parent / "config"
sys.path.append(str(config_dir))
from prompts import EVALUATION_PROMPTS, RANDOM_SEEDS, LEVEL_CONFIGS

utils_dir = Path(__file__).resolve().parent.parent / "utils"
sys.path.append(str(utils_dir))
from video_utils import save_video_frames

class MultiLevelGenerator:
    """四级对比实验的视频生成器"""
    
    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    ):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        
        # 初始化基础Pipeline
        print("正在初始化基础Pipeline...")
        model_configs = [
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ]
        self.base_pipe = WanVideoPipeline.from_pretrained(
            device=device,
            torch_dtype=dtype,
            model_configs=model_configs
        )
        
        # 保存基础DiT权重的路径（从已加载的模型中获取）
        self.base_dit_state = None
        
        # 存储各级别的组件
        self.lora_loader = GeneralLoRALoader(device=device, torch_dtype=dtype)
        self.temporal_module = None
        self.flow_predictor = None
        
        print("✓ 基础Pipeline初始化完成")
    
    def _save_base_dit_state(self):
        """保存基础DiT的state_dict到内存"""
        if self.base_dit_state is None:
            print("  保存基础DiT状态到内存...")
            self.base_dit_state = {
                k: v.clone().cpu() for k, v in self.base_pipe.dit.state_dict().items()
            }
    
    def _restore_base_dit_state(self):
        """从内存恢复基础DiT的state_dict"""
        if self.base_dit_state is not None:
            print("  从内存恢复基础DiT状态...")
            # 将state_dict移回GPU
            state_dict_gpu = {
                k: v.to(self.device) for k, v in self.base_dit_state.items()
            }
            self.base_pipe.dit.load_state_dict(state_dict_gpu)
    
    def _apply_lora(self, lora_path: str, alpha: float = 0.8):
        """应用LoRA权重"""
        print(f"  加载LoRA: {Path(lora_path).name}")
        lora_state = load_state_dict_from_file(
            lora_path,
            torch_dtype=self.dtype,
            device=self.device
        )
        self.lora_loader.load(self.base_pipe.dit, lora_state, alpha=alpha)
    
    def _load_dit_finetuned(self, dit_path: str):
        """加载微调后的DiT权重"""
        print(f"  加载微调DiT: {Path(dit_path).name}")
        # 🔧 修复：添加 weights_only=False
        state_dict = torch.load(dit_path, map_location=self.device, weights_only=False)
        self.base_pipe.dit.load_state_dict(state_dict)
    
    def _load_temporal_module(self, checkpoint_path: str):
        """加载Temporal Module"""
        print(f"  加载Temporal Module: {Path(checkpoint_path).name}")
        
        # 初始化模块
        self.temporal_module = TemporalModule(
            latent_channels=16,
            style_dim=None
        ).to(self.device)
        
        self.flow_predictor = LatentFlowPredictor(
            in_channels=16
        ).to(self.device)
        
        # 🔧 修复：添加 weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载EMA权重
        if "ema" in checkpoint:
            for name, param in self.temporal_module.named_parameters():
                if name in checkpoint["ema"]:
                    param.data = checkpoint["ema"][name]
        else:
            self.temporal_module.load_state_dict(checkpoint["temporal_module"])
        
        # 加载Flow Predictor
        if "flow_predictor" in checkpoint:
            self.flow_predictor.load_state_dict(checkpoint["flow_predictor"])
        
        self.temporal_module.eval()
        self.flow_predictor.eval()
    
    def generate_single_video(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        seed: int,
        height: int = 256,
        width: int = 256,
        use_temporal: bool = False
    ) -> torch.Tensor:
        """
        生成单个视频
        
        Returns:
            torch.Tensor: (T, C, H, W) 格式的视频帧
        """
        with torch.no_grad():
            if not use_temporal:
                # 标准生成
                output = self.base_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    seed=seed
                )
                
                # PIL Image列表 -> Tensor
                from PIL import Image
                import numpy as np
                
                if isinstance(output, list) and len(output) > 0 and isinstance(output[0], Image.Image):
                    frames_list = []
                    for img in output:
                        img_np = np.array(img)
                        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                        frames_list.append(img_tensor)
                    frames = torch.stack(frames_list, dim=0)
                else:
                    raise ValueError(f"未知的输出格式: {type(output)}")
                
                return frames
            
            else:
                # 使用Temporal Module生成
                # 先标准生成
                output = self.base_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    seed=seed
                )
                
                # 转换为Tensor
                from PIL import Image
                import numpy as np
                
                frames_list = []
                for img in output:
                    img_np = np.array(img)
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                    frames_list.append(img_tensor)
                frames = torch.stack(frames_list, dim=0).unsqueeze(0)  # (1, T, C, H, W)
                frames = frames.to(self.device, dtype=self.dtype)
                
                # VAE编码
                video_for_vae = [frames[0].permute(1, 0, 2, 3)]  # [(C, T, H, W)]
                latents = self.base_pipe.vae.encode(video_for_vae, device=self.device, tiled=False)
                latents = latents.permute(0, 2, 1, 3, 4).to(torch.float32)  # (1, T, C, H, W)
                
                # 应用Temporal Module
                smoothed_list = [latents[:, 0]]
                for t in range(1, latents.shape[1]):
                    z_prev = smoothed_list[-1]
                    z_cur = latents[:, t]
                    
                    flow = self.flow_predictor(z_prev, z_cur)
                    z_fused, _, _ = self.temporal_module(
                        z_prev, z_cur,
                        s_prev=None, s_cur=None,
                        flow=flow
                    )
                    smoothed_list.append(z_fused)
                
                smoothed_latents = torch.stack(smoothed_list, dim=1)
                
                # VAE解码
                smoothed_for_decode = smoothed_latents.permute(0, 2, 1, 3, 4).to(self.dtype)
                frames_smoothed = self.base_pipe.vae.decode(
                    smoothed_for_decode,
                    device=self.device,
                    tiled=False
                )
                frames_smoothed = frames_smoothed[:, :3].permute(0, 2, 1, 3, 4)
                
                return frames_smoothed[0].cpu()  # (T, C, H, W)
    
    def generate_level(
        self,
        level_id: str,
        output_dir: Path,
        overwrite: bool = False
    ):
        """
        生成特定级别的所有视频
        
        Args:
            level_id: 级别ID（level_0, level_1, level_2, level_3）
            output_dir: 输出目录
            overwrite: 是否覆盖已存在的视频
        """
        level_config = LEVEL_CONFIGS[level_id]
        level_num = level_id.split('_')[1]  # "level_0" -> "0"
        
        print(f"\n{'='*80}")
        print(f"开始生成: {level_config['name']}")
        print(f"描述: {level_config['description']}")
        print(f"{'='*80}\n")
        
        # 保存基础DiT状态（仅第一次）
        self._save_base_dit_state()
        
        # 重置Pipeline到基础状态
        print("重置Pipeline...")
        self._restore_base_dit_state()
        
        # 应用级别配置
        if level_config.get("use_lora") and level_config.get("lora_path"):
            self._apply_lora(level_config["lora_path"])
        
        if level_config.get("dit_finetuned"):
            self._load_dit_finetuned(level_config["dit_finetuned"])
        
        if level_config.get("use_temporal") and level_config.get("temporal_checkpoint"):
            self._load_temporal_module(level_config["temporal_checkpoint"])
        
        # 生成所有视频
        total_videos = len(EVALUATION_PROMPTS) * len(RANDOM_SEEDS)
        pbar = tqdm(total=total_videos, desc=f"Level {level_num}")
        
        for prompt_config in EVALUATION_PROMPTS:
            for seed in RANDOM_SEEDS:
                # 构建输出文件名
                video_filename = f"video_{level_num}_{prompt_config['id']}_seed{seed}.mp4"
                video_path = output_dir / video_filename
                
                # 检查是否已存在
                if video_path.exists() and not overwrite:
                    pbar.update(1)
                    pbar.set_postfix_str(f"跳过: {video_filename}")
                    continue
                
                try:
                    # 生成视频
                    frames = self.generate_single_video(
                        prompt=prompt_config["english"],
                        negative_prompt=prompt_config["negative"],
                        num_frames=prompt_config["num_frames"],
                        seed=seed,
                        use_temporal=level_config.get("use_temporal", False)
                    )
                    
                    # 保存
                    save_video_frames(frames, video_path, fps=8)
                    pbar.set_postfix_str(f"完成: {video_filename}")
                
                except Exception as e:
                    pbar.set_postfix_str(f"错误: {video_filename}")
                    print(f"\n生成失败: {video_filename}")
                    print(f"错误: {e}")
                    import traceback
                    traceback.print_exc()
                
                pbar.update(1)
        
        pbar.close()
        print(f"\n✓ {level_config['name']} 完成!\n")

def main():
    parser = argparse.ArgumentParser(description="批量生成所有级别的评估视频")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_outputs/videos",
        help="视频输出目录"
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs='+',
        default=["level_0", "level_1", "level_2", "level_3"],
        choices=["level_0", "level_1", "level_2", "level_3"],
        help="要生成的级别"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="精度类型"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的视频"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # dtype映射
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    
    # 初始化生成器
    generator = MultiLevelGenerator(
        device=args.device,
        dtype=dtype_map[args.dtype]
    )
    
    # 生成所有级别
    for level_id in args.levels:
        generator.generate_level(
            level_id=level_id,
            output_dir=output_dir,
            overwrite=args.overwrite
        )
    
    print("\n" + "="*80)
    print("✅ 所有视频生成完成!")
    print(f"保存位置: {output_dir}")
    print(f"总计视频数: {len(list(output_dir.glob('*.mp4')))}")
    print("="*80)

if __name__ == "__main__":
    main()