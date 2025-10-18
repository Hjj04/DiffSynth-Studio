#!/usr/bin/env python3
"""
prompts.py

水墨画视频生成评估的Prompt配置
包含11个精心设计的测试场景，用于评估模型的时序连贯性和风格保持能力
"""

from typing import List, Dict

# 评估Prompt列表（中英双语，生成时使用英文）
EVALUATION_PROMPTS: List[Dict[str, str]] = [
    {
        "id": "01_slow_calligraphy",
        "chinese": "黑白水墨风，一位穿汉服的女子在古桥上慢速挥毫，长笔触、墨迹随笔尖流动并自然渗化，镜头缓慢横移，8帧，要求帧间连贯、笔触连续、无跳帧、留白意境明显。",
        "english": "Black-and-white ink-wash style, a woman in hanfu slowly painting on an ancient bridge, long continuous brush strokes with ink flow and natural bleeding, slow lateral camera pan, 8 frames. Require frame-to-frame continuity, continuous strokes, no flicker or broken strokes, strong negative space.",
        "negative": "no flicker, no jitter, no broken strokes, no sudden color shifts",
        "num_frames": 17,  # 会被向上取整到17
        "focus": "连贯细节展示"
    },
    {
        "id": "02_detailed_stroke",
        "chinese": "一位穿汉服的女子在古桥上挥毫，慢速横向平移，持续挥笔，笔触为连续长stroke，墨迹随动作自然流动，要求帧间连贯、无跳帧。",
        "english": "Ink wash painting style, a woman in traditional hanfu calligraphy on an ancient stone bridge, slow horizontal camera pan, continuous long brush strokes with natural ink flow following the motion, 8 frames. Emphasis on frame-to-frame coherence and stroke continuity.",
        "negative": "no flicker, no jitter, no broken strokes, no sudden color shifts",
        "num_frames": 17,
        "focus": "笔触连贯性测试"
    },
    {
        "id": "03_fast_action",
        "chinese": "水墨风，少年猛然挥剑，笔触与墨滴被快速甩出，强调高速动作下墨迹轨迹，8帧，强制无断裂、无抖动。",
        "english": "Ink wash style, a young warrior swiftly swinging a sword, brush strokes and ink droplets rapidly flung out, emphasizing ink trajectory during high-speed motion, 8 frames. Strictly no broken strokes or jitter.",
        "negative": "no motion blur artifacts, no temporal discontinuity, no stroke breaks",
        "num_frames": 17,
        "focus": "快速动作压力测试"
    },
    {
        "id": "04_negative_space",
        "chinese": "一片留白的宣纸上，笔尖缓慢拖过，留白区域随笔触轻微变化，展示空白面积随时间平滑变化，8帧，要求低抖动。",
        "english": "On a blank rice paper with strong negative space, a brush tip slowly drags across, the blank area subtly changes with the stroke, showing smooth temporal variation of white space, 8 frames. Require minimal jitter.",
        "negative": "no flickering in white areas, no abrupt changes in negative space",
        "num_frames": 17,
        "focus": "留白稳定性测试"
    },
    {
        "id": "05_wet_to_dry",
        "chinese": "单笔长划，湿墨开始向外渗透，随后逐渐变干，强调渗化连续性与边缘形变，8帧，帧间连贯。",
        "english": "A single long brush stroke, wet ink begins to bleed outward and gradually dries, emphasizing continuous bleeding and edge deformation, 8 frames. Frame-to-frame coherence required.",
        "negative": "no sudden drying, no discontinuous bleeding patterns",
        "num_frames": 17,
        "focus": "渐进湿度变化"
    },
    {
        "id": "06_occlusion_recovery",
        "chinese": "画面中手掌短暂遮挡笔尖，然后移开，墨迹在遮挡处连续延展，检查遮挡引起的断裂与恢复质量，8帧。",
        "english": "In the frame, a hand briefly occludes the brush tip, then moves away, ink continuously extends at the occlusion point, testing occlusion-induced breaks and recovery quality, 8 frames.",
        "negative": "no broken strokes after occlusion, no temporal artifacts during reveal",
        "num_frames": 17,
        "focus": "遮挡恢复测试"
    },
    {
        "id": "07_edge_consistency",
        "chinese": "复杂的树枝线稿，用钢笔式水墨描边，强调边缘轮廓保留与连贯性，8帧，要求高边缘一致性（no broken edges）。",
        "english": "Complex tree branch line art, outlined with pen-style ink wash, emphasizing edge contour preservation and coherence, 8 frames. Require high edge consistency (no broken edges).",
        "negative": "no broken edges, no edge flickering, no line discontinuity",
        "num_frames": 17,
        "focus": "边缘一致性压力测试"
    },
    {
        "id": "08_multi_brush",
        "chinese": "两只毛笔交替在纸上画出交织的笔触，墨迹相互叠加并流动，检验笔触重叠处的连贯性，8帧。",
        "english": "Two brushes alternately drawing interwoven strokes on paper, ink overlapping and flowing together, testing coherence at stroke intersections, 8 frames.",
        "negative": "no broken strokes at intersections, no temporal aliasing in overlapping regions",
        "num_frames": 17,
        "focus": "多笔触交织测试"
    },
    {
        "id": "09_long_zoom",
        "chinese": "慢平移长镜头，画面从近景的笔触细节逐渐拉远到宣纸全貌，观察缩放与笔触连贯性，16帧，low speed。",
        "english": "Slow panning long shot, gradually zooming from close-up brush stroke details to full view of rice paper, observing zoom and stroke coherence, 16 frames, low speed.",
        "negative": "no scale-induced flickering, no detail loss during zoom",
        "num_frames": 17,
        "focus": "长镜头缩放测试"
    },
    {
        "id": "10_particle_merge",
        "chinese": "墨滴在纸上溅开并形成花瓣形态，随后风吹使墨瓣缓慢移动，8帧，检查粒子/滴状墨迹合并与连贯。",
        "english": "Ink droplets splash on paper forming petal shapes, then wind slowly moves the ink petals, 8 frames, testing particle/droplet ink merging and coherence.",
        "negative": "no particle disappearance, no abrupt shape changes",
        "num_frames": 17,
        "focus": "粒子合并测试"
    },
    {
        "id": "11_calligraphy_stroke",
        "chinese": "一剂行草书法笔画，笔势连贯、收笔自然，强调笔锋延续，8帧。",
        "english": "A running-cursive calligraphy stroke, with continuous brush momentum and natural ending, emphasizing brush tip continuity, 8 frames.",
        "negative": "no broken brush tips, no momentum discontinuity",
        "num_frames": 17,
        "focus": "书法笔触测试"
    }
]

# 随机种子配置
RANDOM_SEEDS = [42, 100, 400]

# 四个实验级别配置
LEVEL_CONFIGS = {
    "level_0": {
        "name": "Level 0 (Absolute Baseline)",
        "description": "无LoRA，无Temporal Module",
        "use_lora": False,
        "use_temporal": False,
    },
    "level_1": {
        "name": "Level 1 (Style Baseline)",
        "description": "有LoRA，无Temporal Module",
        "use_lora": True,
        "use_temporal": False,
        "lora_path": "/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/lora_outputs/inkwash_style_v1/epoch-18.safetensors",
    },
    "level_2": {
        "name": "Level 2 (Joint Style Training)",
        "description": "仅使用微调后的DiT（风格已内化），无Temporal Module",
        "use_lora": False,
        "use_temporal": False,
        "dit_finetuned": "/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio/runs/staged_training_final_oom_fix/checkpoints/dit_finetuned_step_final.pth",
    },
    "level_3": {
        "name": "Level 3 (Fully Enhanced)",
        "description": "微调后的DiT + Temporal Module",
        "use_lora": False,
        "use_temporal": True,
        "dit_finetuned": "/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio/runs/staged_training_final_oom_fix/checkpoints/dit_finetuned_step_final.pth",
        # 🔧 分别指定两个文件
        "temporal_module_path": "/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio/runs/staged_training_final_oom_fix/checkpoints/temporal_module_step_final.pth",
        "flow_predictor_path": "/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio/runs/staged_training_final_oom_fix/checkpoints/flow_predictor_step_final.pth"
    }
}

def get_total_videos():
    """计算总视频数量"""
    return len(EVALUATION_PROMPTS) * len(LEVEL_CONFIGS) * len(RANDOM_SEEDS)

if __name__ == "__main__":
    print("="*80)
    print("评估配置总览")
    print("="*80)
    print(f"Prompt数量: {len(EVALUATION_PROMPTS)}")
    print(f"实验级别数: {len(LEVEL_CONFIGS)}")
    print(f"随机种子数: {len(RANDOM_SEEDS)}")
    print(f"总视频数量: {get_total_videos()}")
    print("\n实验级别:")
    for level_id, config in LEVEL_CONFIGS.items():
        print(f"  {level_id}: {config['name']}")
        print(f"    描述: {config['description']}")
    print("="*80)
