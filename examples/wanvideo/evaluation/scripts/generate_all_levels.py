#!/usr/bin/env python3
"""
generate_all_levels.py - 批量生成所有级别的评估视频（修正版）

核心修复：
1. Temporal Module直接挂载到Pipeline实例
2. 在去噪循环的每一步中应用时序平滑
3. 不再使用后处理方式

修复前的错误逻辑：
  生成完整视频 -> VAE编码 -> 应用Temporal Module -> VAE解码

修复后的正确逻辑：
  在每个去噪步骤中 -> 应用Temporal Module平滑潜在向量 -> 继续去噪
  
作者: AI Assistant
日期: 2025-01-16
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import traceback

# ============================================================================
# 添加项目路径
# ============================================================================

# 获取项目根目录
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

# 添加配置目录
config_dir = Path(__file__).resolve().parent.parent / "config"
sys.path.insert(0, str(config_dir))

# ============================================================================
# 导入依赖
# ============================================================================

from diffsynth import ModelManager
from diffsynth.pipelines.wan_video_new import WanVideoPipeline
from diffsynth.models.wan_video_temporal_module import WanVideoTemporalModule
from diffsynth.models.wan_video_latent_flow_predictor import WanVideoLatentFlowPredictor
from diffsynth.data.video import save_video

# 导入评估配置
from prompts import EVALUATION_PROMPTS, RANDOM_SEEDS, LEVEL_CONFIGS

# ============================================================================
# 多级别视频生成器（修正版）
# ============================================================================

class MultiLevelVideoGenerator:
    """
    多级别视频生成器
    
    核心修复：确保Temporal Module在去噪过程中被调用，而非后处理
    
    工作流程：
    1. 加载基础模型（DiT + VAE + T5）
    2. 根据级别配置加载LoRA/微调权重
    3. 将Temporal Module直接挂载到Pipeline实例
    4. 调用Pipeline时，自动在去噪循环中应用时序平滑
    """
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        base_model_id: str = "models--Wan2.1",
        enable_vram_management: bool = True
    ):
        """
        初始化生成器
        
        Args:
            device: 计算设备
            dtype: 数据类型
            base_model_id: 基础模型ID
            enable_vram_management: 是否启用显存管理
        """
        self.device = device
        self.dtype = dtype
        self.base_model_id = base_model_id
        self.enable_vram_management = enable_vram_management
        
        print("=" * 80)
        print("初始化多级别视频生成器（修正版）")
        print("=" * 80)
        print(f"设备: {device}")
        print(f"数据类型: {dtype}")
        print(f"基础模型: {base_model_id}")
        print(f"显存管理: {enable_vram_management}")
        print("=" * 80)
        print()
        
        # 初始化基础Pipeline
        self._initialize_base_pipeline()
        
        # 保存原始DiT状态（用于级别切换）
        print("[2/2] 保存基础DiT状态...")
        self.original_dit_state = {
            k: v.clone().cpu() 
            for k, v in self.pipe.dit.state_dict().items()
        }
        print("✓ 基础DiT状态已保存")
        print()
        
        print("=" * 80)
        print("✓ 初始化完成")
        print("=" * 80)
        print()
    
    def _initialize_base_pipeline(self):
        """
        初始化基础Pipeline
        
        加载：
        1. T5文本编码器
        2. DiT（扩散变换器）
        3. VAE（变分自编码器）
        """
        print("[1/2] 加载基础模型...")
        
        # 创建模型管理器
        model_manager = ModelManager(
            torch_dtype=self.dtype,
            device=self.device
        )
        
        # 定义要加载的模型文件
        model_files = [
            ("T5文本编码器", "models_t5_umt5-xxl-enc-bf16.pth"),
            ("DiT扩散模型", "diffusion_pytorch_model.safetensors"),
            ("VAE编码器", "Wan2.1_VAE.pth")
        ]
        
        # 加载模型
        for model_name, file_pattern in model_files:
            print(f"  - 加载 {model_name}...")
            model_manager.load_model(
                model_id=self.base_model_id,
                origin_file_pattern=file_pattern
            )
        
        print()
        print("  ✓ 所有基础模型已加载")
        print()
        
        # 从模型管理器创建Pipeline
        print("  - 创建Pipeline实例...")
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        
        # 启用显存管理
        if self.enable_vram_management:
            print("  - 启用显存管理...")
            self.pipe.enable_vram_management()
        
        print("  ✓ Pipeline创建完成")
        print()
    
    def _restore_dit_base_state(self):
        """
        恢复DiT到基础状态
        
        在切换级别前调用，确保从干净状态开始
        """
        print("  [1/4] 恢复DiT到基础状态...")
        
        # 将保存的状态加载回DiT
        state_dict_on_device = {
            k: v.to(self.device) 
            for k, v in self.original_dit_state.items()
        }
        
        self.pipe.dit.load_state_dict(state_dict_on_device, strict=True)
        
        print("    ✓ DiT已恢复到基础状态")
    
    def _load_level_specific_weights(self, level_config: dict):
        """
        加载级别特定的权重
        
        Args:
            level_config: 级别配置字典
            
        支持：
        1. 微调的DiT权重
        2. LoRA权重
        """
        print("  [2/4] 加载级别特定权重...")
        
        # 检查是否有微调的DiT
        if level_config.get("dit_finetuned"):
            dit_path = level_config["dit_finetuned"]
            
            if not Path(dit_path).exists():
                print(f"    ⚠️  警告: DiT权重文件不存在: {dit_path}")
                print("    继续使用基础DiT")
                return
            
            print(f"    - 加载微调DiT: {Path(dit_path).name}")
            
            try:
                # 加载检查点
                checkpoint = torch.load(
                    dit_path,
                    map_location=self.device,
                    weights_only=False
                )
                
                # 根据检查点结构提取state_dict
                if isinstance(checkpoint, dict):
                    if "dit" in checkpoint:
                        state_dict = checkpoint["dit"]
                    elif "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # 加载到DiT
                self.pipe.dit.load_state_dict(state_dict, strict=False)
                
                print("    ✓ 微调DiT加载成功")
                
            except Exception as e:
                print(f"    ✗ 加载微调DiT失败: {e}")
                print("    继续使用基础DiT")
        
        # 检查是否有LoRA
        elif level_config.get("use_lora") and level_config.get("lora_path"):
            lora_path = level_config["lora_path"]
            
            if not Path(lora_path).exists():
                print(f"    ⚠️  警告: LoRA权重文件不存在: {lora_path}")
                print("    继续使用基础DiT")
                return
            
            print(f"    - 加载LoRA: {Path(lora_path).name}")
            
            try:
                # 使用Pipeline的内置LoRA加载方法
                lora_alpha = level_config.get("lora_alpha", 1.0)
                
                self.pipe.load_lora(
                    self.pipe.dit,
                    lora_path,
                    alpha=lora_alpha
                )
                
                print(f"    ✓ LoRA加载成功 (alpha={lora_alpha})")
                
            except Exception as e:
                print(f"    ✗ 加载LoRA失败: {e}")
                print("    继续使用基础DiT")
        
        else:
            print("    - 使用基础DiT（无LoRA/微调）")
    
    def _attach_temporal_modules(self, level_config: dict):
        """
        挂载Temporal Module到Pipeline
        
        这是核心修复点！
        
        修复前：在生成后作为后处理应用
        修复后：直接挂载到Pipeline，在去噪循环中自动调用
        
        Args:
            level_config: 级别配置字典
        """
        print("  [3/4] 配置Temporal Module...")
        
        if not level_config.get("use_temporal"):
            print("    - 此级别不使用Temporal Module")
            
            # 确保卸载任何已有的Temporal Module
            self.pipe.temporal_module = None
            self.pipe.flow_predictor = None
            
            return
        
        # 检查必需的路径
        temporal_path = level_config.get("temporal_module_path")
        flow_path = level_config.get("flow_predictor_path")
        
        if not temporal_path or not flow_path:
            print("    ✗ 错误: 缺少Temporal Module或Flow Predictor路径")
            self.pipe.temporal_module = None
            self.pipe.flow_predictor = None
            return
        
        if not Path(temporal_path).exists():
            print(f"    ✗ 错误: Temporal Module文件不存在: {temporal_path}")
            self.pipe.temporal_module = None
            self.pipe.flow_predictor = None
            return
        
        if not Path(flow_path).exists():
            print(f"    ✗ 错误: Flow Predictor文件不存在: {flow_path}")
            self.pipe.temporal_module = None
            self.pipe.flow_predictor = None
            return
        
        print(f"    - 加载Temporal Module: {Path(temporal_path).name}")
        print(f"    - 加载Flow Predictor: {Path(flow_path).name}")
        
        try:
            # 创建Temporal Module实例
            temporal_module = WanVideoTemporalModule(
                latent_channels=16,
                style_dim=None  # 根据实际模型配置调整
            )
            
            # 加载权重
            temporal_checkpoint = torch.load(
                temporal_path,
                map_location=self.device,
                weights_only=False
            )
            
            # 根据检查点结构提取state_dict
            if isinstance(temporal_checkpoint, dict):
                if "temporal_module" in temporal_checkpoint:
                    temporal_state = temporal_checkpoint["temporal_module"]
                elif "model_state_dict" in temporal_checkpoint:
                    temporal_state = temporal_checkpoint["model_state_dict"]
                elif "state_dict" in temporal_checkpoint:
                    temporal_state = temporal_checkpoint["state_dict"]
                else:
                    temporal_state = temporal_checkpoint
            else:
                temporal_state = temporal_checkpoint
            
            temporal_module.load_state_dict(temporal_state)
            temporal_module = temporal_module.to(self.device, dtype=self.dtype)
            temporal_module.eval()
            
            # 创建Flow Predictor实例
            flow_predictor = WanVideoLatentFlowPredictor(
                in_channels=16
            )
            
            # 加载权重
            flow_checkpoint = torch.load(
                flow_path,
                map_location=self.device,
                weights_only=False
            )
            
            # 根据检查点结构提取state_dict
            if isinstance(flow_checkpoint, dict):
                if "flow_predictor" in flow_checkpoint:
                    flow_state = flow_checkpoint["flow_predictor"]
                elif "model_state_dict" in flow_checkpoint:
                    flow_state = flow_checkpoint["model_state_dict"]
                elif "state_dict" in flow_checkpoint:
                    flow_state = flow_checkpoint["state_dict"]
                else:
                    flow_state = flow_checkpoint
            else:
                flow_state = flow_checkpoint
            
            flow_predictor.load_state_dict(flow_state)
            flow_predictor = flow_predictor.to(self.device, dtype=self.dtype)
            flow_predictor.eval()
            
            # 【核心修复】直接挂载到Pipeline实例
            # 这样在Pipeline的__call__方法中会自动在去噪循环的每一步使用
            self.pipe.temporal_module = temporal_module
            self.pipe.flow_predictor = flow_predictor
            
            print("    ✓ Temporal Module已挂载到Pipeline")
            print("    ✓ 将在去噪循环的每一步中应用时序平滑")
            
        except Exception as e:
            print(f"    ✗ 挂载Temporal Module失败: {e}")
            print(f"    错误详情: {traceback.format_exc()}")
            
            self.pipe.temporal_module = None
            self.pipe.flow_predictor = None
    
    def _finalize_configuration(self, level_config: dict):
        """
        完成配置
        
        Args:
            level_config: 级别配置字典
        """
        print("  [4/4] 完成配置...")
        
        # 设置模型为评估模式
        self.pipe.dit.eval()
        if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
            self.pipe.vae.eval()
        if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
            self.pipe.text_encoder.eval()
        
        # 禁用梯度计算
        torch.set_grad_enabled(False)
        
        print("    ✓ 所有模型已设为评估模式")
        print("    ✓ 梯度计算已禁用")
    
    def configure_for_level(self, level_config: dict):
        """
        为特定级别配置模型
        
        Args:
            level_config: 级别配置字典
            
        配置步骤：
        1. 恢复DiT到基础状态
        2. 加载级别特定权重（LoRA或微调DiT）
        3. 挂载Temporal Module（如果需要）
        4. 完成配置
        """
        level_name = level_config.get("name", "Unknown")
        
        print()
        print("-" * 80)
        print(f"配置模型用于: {level_name}")
        print("-" * 80)
        
        # 执行配置步骤
        self._restore_dit_base_state()
        self._load_level_specific_weights(level_config)
        self._attach_temporal_modules(level_config)
        self._finalize_configuration(level_config)
        
        print("-" * 80)
        print(f"✓ {level_name} 配置完成")
        print("-" * 80)
        print()
    
    def generate_single_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        cfg_scale: float = 7.0,
        seed: int = 42
    ):
        """
        生成单个视频
        
        核心修复：直接调用Pipeline，内部自动在去噪循环中使用Temporal Module
        
        Args:
            prompt: 提示词
            negative_prompt: 负面提示词
            num_frames: 帧数
            height: 高度
            width: 宽度
            num_inference_steps: 去噪步数
            cfg_scale: 分类器自由引导强度
            seed: 随机种子
            
        Returns:
            视频帧列表（PIL Image）
        """
        # 设置随机种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # 直接调用Pipeline
        # 如果temporal_module和flow_predictor已挂载，
        # Pipeline的__call__方法会在去噪循环中自动使用它们
        video_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed
        )
        
        return video_frames
    
    def generate_level(
        self,
        level_id: str,
        output_dir: Path,
        overwrite: bool = False,
        save_metadata: bool = True
    ):
        """
        生成特定级别的所有评估视频
        
        Args:
            level_id: 级别ID（例如 "level_0", "level_3"）
            output_dir: 输出目录
            overwrite: 是否覆盖已存在的视频
            save_metadata: 是否保存元数据
            
        Returns:
            生成统计信息字典
        """
        # 验证级别ID
        if level_id not in LEVEL_CONFIGS:
            raise ValueError(f"未知的级别ID: {level_id}")
        
        level_config = LEVEL_CONFIGS[level_id]
        level_num = level_id.split('_')[1]
        
        print()
        print("=" * 80)
        print(f"开始生成: {level_config['name']} (Level {level_num})")
        print("=" * 80)
        
        # 配置模型
        self.configure_for_level(level_config)
        
        # 准备输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计
        total_videos = len(EVALUATION_PROMPTS) * len(RANDOM_SEEDS)
        generated_count = 0
        skipped_count = 0
        failed_count = 0
        
        # 元数据
        metadata = {
            "level": level_id,
            "level_name": level_config["name"],
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(EVALUATION_PROMPTS),
            "total_seeds": len(RANDOM_SEEDS),
            "total_videos": total_videos,
            "videos": []
        }
        
        # 生成视频
        print()
        print(f"生成 {total_videos} 个视频...")
        print("-" * 80)
        
        pbar = tqdm(
            total=total_videos,
            desc=f"Level {level_num}",
            unit="video",
            ncols=100
        )
        
        for prompt_config in EVALUATION_PROMPTS:
            for seed in RANDOM_SEEDS:
                # 构建文件名
                video_filename = f"video_{level_num}_{prompt_config['id']}_seed{seed}.mp4"
                video_path = output_dir / video_filename
                
                # 检查是否已存在
                if video_path.exists() and not overwrite:
                    skipped_count += 1
                    pbar.set_postfix({"status": "skipped", "file": video_filename})
                    pbar.update(1)
                    
                    if save_metadata:
                        metadata["videos"].append({
                            "filename": video_filename,
                            "prompt_id": prompt_config["id"],
                            "seed": seed,
                            "status": "skipped"
                        })
                    
                    continue
                
                # 生成视频
                try:
                    pbar.set_postfix({"status": "generating", "file": video_filename})
                    
                    video_frames = self.generate_single_video(
                        prompt=prompt_config["english"],
                        negative_prompt=prompt_config.get("negative", ""),
                        num_frames=prompt_config["num_frames"],
                        height=256,
                        width=256,
                        num_inference_steps=50,
                        cfg_scale=7.0,
                        seed=seed
                    )
                    
                    # 保存视频
                    save_video(video_frames, str(video_path), fps=8)
                    
                    generated_count += 1
                    pbar.set_postfix({"status": "success", "file": video_filename})
                    
                    # 记录元数据
                    if save_metadata:
                        metadata["videos"].append({
                            "filename": video_filename,
                            "prompt_id": prompt_config["id"],
                            "prompt": prompt_config["english"],
                            "seed": seed,
                            "num_frames": prompt_config["num_frames"],
                            "status": "success"
                        })
                
                except Exception as e:
                    failed_count += 1
                    pbar.set_postfix({"status": "failed", "file": video_filename})
                    
                    print()
                    print(f"✗ 生成 {video_filename} 失败:")
                    print(f"  错误: {str(e)}")
                    print(f"  详情: {traceback.format_exc()}")
                    print()
                    
                    # 记录元数据
                    if save_metadata:
                        metadata["videos"].append({
                            "filename": video_filename,
                            "prompt_id": prompt_config["id"],
                            "seed": seed,
                            "status": "failed",
                            "error": str(e)
                        })
                
                pbar.update(1)
        
        pbar.close()
        
        # 保存元数据
        if save_metadata:
            metadata["generated"] = generated_count
            metadata["skipped"] = skipped_count
            metadata["failed"] = failed_count
            
            metadata_path = output_dir / f"metadata_{level_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print()
        print("=" * 80)
        print(f"Level {level_num} 生成完成")
        print("=" * 80)
        print(f"成功生成: {generated_count}/{total_videos}")
        print(f"跳过已存在: {skipped_count}")
        print(f"生成失败: {failed_count}")
        print("=" * 80)
        print()
        
        return {
            "total": total_videos,
            "generated": generated_count,
            "skipped": skipped_count,
            "failed": failed_count
        }
    
    def generate_all_levels(
        self,
        output_dir: Path,
        levels: list = None,
        overwrite: bool = False
    ):
        """
        生成所有级别的视频
        
        Args:
            output_dir: 输出目录
            levels: 要生成的级别列表（默认：所有）
            overwrite: 是否覆盖已存在的视频
            
        Returns:
            所有级别的生成统计
        """
        if levels is None:
            levels = ["level_0", "level_1", "level_2", "level_3"]
        
        results = {}
        
        for level_id in levels:
            try:
                result = self.generate_level(
                    level_id=level_id,
                    output_dir=output_dir,
                    overwrite=overwrite
                )
                results[level_id] = result
                
            except Exception as e:
                print()
                print("=" * 80)
                print(f"✗ Level {level_id} 生成失败")
                print("=" * 80)
                print(f"错误: {str(e)}")
                print(f"详情: {traceback.format_exc()}")
                print("=" * 80)
                print()
                
                results[level_id] = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        return results

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="生成所有级别的评估视频（修正版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成所有级别
  python generate_all_levels.py --output_dir ../../evaluation_outputs/videos
  
  # 仅生成Level 3
  python generate_all_levels.py --output_dir ../../evaluation_outputs/videos --levels level_3
  
  # 覆盖已存在的视频
  python generate_all_levels.py --output_dir ../../evaluation_outputs/videos --overwrite

关键修复说明:
  此版本确保Temporal Module在去噪循环的每一步中被调用，
  而不是作为后处理步骤。这是核心创新的正确实现。
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="视频输出目录"
    )
    
    parser.add_argument(
        "--levels",
        type=str,
        nargs='+',
        default=None,
        choices=["level_0", "level_1", "level_2", "level_3"],
        help="要生成的级别（默认：全部）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备（默认：cuda）"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="数据类型（默认：float16）"
    )
    
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="models--Wan2.1",
        help="基础模型ID"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的视频"
    )
    
    parser.add_argument(
        "--no_vram_management",
        action="store_true",
        help="禁用显存管理"
    )
    
    args = parser.parse_args()
    
    # 转换dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("=" * 80)
    print("多级别视频生成器（修正版）")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"设备: {args.device}")
    print(f"数据类型: {args.dtype}")
    print(f"覆盖模式: {args.overwrite}")
    print("=" * 80)
    print()
    
    # 记录开始时间
    start_time = datetime.now()
    
    try:
        # 初始化生成器
        generator = MultiLevelVideoGenerator(
            device=args.device,
            dtype=dtype,
            base_model_id=args.base_model_id,
            enable_vram_management=not args.no_vram_management
        )
        
        # 生成视频
        results = generator.generate_all_levels(
            output_dir=output_dir,
            levels=args.levels,
            overwrite=args.overwrite
        )
        
        # 保存总结
        summary = {
            "timestamp": datetime.now().isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "configuration": {
                "device": args.device,
                "dtype": args.dtype,
                "base_model_id": args.base_model_id,
                "vram_management": not args.no_vram_management,
                "overwrite": args.overwrite
            },
            "results": results
        }
        
        summary_path = output_dir / "generation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印最终总结
        print()
        print("=" * 80)
        print("所有级别生成完成")
        print("=" * 80)
        print(f"总耗时: {summary['duration_seconds']:.2f} 秒")
        print(f"总结已保存到: {summary_path}")
        print("=" * 80)
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print("✗ 生成过程发生严重错误")
        print("=" * 80)
        print(f"错误: {str(e)}")
        print(f"详情: {traceback.format_exc()}")
        print("=" * 80)
        print()
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
