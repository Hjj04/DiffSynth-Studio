#!/bin/bash
set -e

################################################################################
# 完整评估流程脚本
# 功能：从视频生成到所有指标评估的端到端流程
#
# 使用方法：
#   bash run_complete_evaluation_pipeline.sh [OPTIONS]
#
# 选项：
#   --skip-generation     跳过视频生成步骤
#   --skip-tlpips        跳过tLPIPS评估
#   --skip-warping       跳过Warping Error评估
#   --skip-fid           跳过FID评估
#   --skip-fvd           跳过FVD评估
#   --skip-edge          跳过Edge IoU评估
#   --levels LEVEL_LIST  仅评估指定级别（例如：level_0 level_3）
#   --overwrite          覆盖已存在的文件
################################################################################

# ============================================================================
# 配置部分
# ============================================================================

# 项目根目录
PROJECT_ROOT="/share/project/chengweiwu/code/Chinese_ink/hanzhe/code/DiffSynth-Studio"

# 输出目录
OUTPUT_BASE="${PROJECT_ROOT}/evaluation_outputs"
VIDEOS_DIR="${OUTPUT_BASE}/videos"
METRICS_DIR="${OUTPUT_BASE}/metrics"
LOGS_DIR="${OUTPUT_BASE}/logs"

# 模型和检查点
CHECKPOINT_DIR="${PROJECT_ROOT}/runs/staged_training_final_oom_fix/checkpoints"
TEMPORAL_CKPT="${CHECKPOINT_DIR}/checkpoint_step_final.pth"

# 真实数据路径（需要根据实际情况修改）
REAL_DATA_FRAMES="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/final_inkwash_dataset/frames"
REAL_VIDEOS_DIR="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/final_inkwash_dataset/videos"

# 脚本目录
SCRIPTS_DIR="${PROJECT_ROOT}/examples/wanvideo/evaluation/scripts"

# GPU设置
export CUDA_VISIBLE_DEVICES=0

# 评估级别
ALL_LEVELS=("level_0" "level_1" "level_2" "level_3")
EVAL_LEVELS=("${ALL_LEVELS[@]}")

# 标志
SKIP_GENERATION=false
SKIP_TLPIPS=false
SKIP_WARPING=false
SKIP_FID=false
SKIP_FVD=false
SKIP_EDGE=false
OVERWRITE=false

# ============================================================================
# 解析命令行参数
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --skip-tlpips)
            SKIP_TLPIPS=true
            shift
            ;;
        --skip-warping)
            SKIP_WARPING=true
            shift
            ;;
        --skip-fid)
            SKIP_FID=true
            shift
            ;;
        --skip-fvd)
            SKIP_FVD=true
            shift
            ;;
        --skip-edge)
            SKIP_EDGE=true
            shift
            ;;
        --levels)
            shift
            EVAL_LEVELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EVAL_LEVELS+=("$1")
                shift
            done
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --help)
            echo "使用方法: $0 [OPTIONS]"
            echo ""
            echo "选项:"
            echo "  --skip-generation     跳过视频生成步骤"
            echo "  --skip-tlpips        跳过tLPIPS评估"
            echo "  --skip-warping       跳过Warping Error评估"
            echo "  --skip-fid           跳过FID评估"
            echo "  --skip-fvd           跳过FVD评估"
            echo "  --skip-edge          跳过Edge IoU评估"
            echo "  --levels LEVEL_LIST  仅评估指定级别"
            echo "  --overwrite          覆盖已存在的文件"
            echo "  --help               显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 辅助函数
# ============================================================================

# 打印分隔线
print_separator() {
    echo "================================================================================"
}

# 打印带时间戳的消息
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 打印步骤标题
print_step() {
    echo ""
    print_separator
    log_message "步骤 $1: $2"
    print_separator
    echo ""
}

# 检查命令是否成功
check_status() {
    if [ $? -eq 0 ]; then
        log_message "✓ $1 完成"
    else
        log_message "✗ $1 失败"
        exit 1
    fi
}

# 检查文件是否存在
check_file_exists() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

# 检查目录是否存在且非空
check_dir_not_empty() {
    if [ -d "$1" ] && [ "$(ls -A $1)" ]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# 环境检查
# ============================================================================

print_separator
log_message "环境检查"
print_separator

echo ""
log_message "检查目录结构..."

# 创建必要的目录
mkdir -p "${VIDEOS_DIR}"
mkdir -p "${METRICS_DIR}"
mkdir -p "${LOGS_DIR}"

log_message "✓ 输出目录已创建"

# 检查脚本是否存在
REQUIRED_SCRIPTS=(
    "${SCRIPTS_DIR}/generate_all_levels.py"
    "${SCRIPTS_DIR}/eval_tlpips.py"
    "${SCRIPTS_DIR}/eval_warping_error_all_levels.py"
    "${SCRIPTS_DIR}/eval_fid.py"
    "${SCRIPTS_DIR}/eval_fvd.py"
    "${SCRIPTS_DIR}/eval_edge_iou.py"
    "${SCRIPTS_DIR}/summarize_all_levels.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        log_message "✗ 缺少脚本: $script"
        exit 1
    fi
done

log_message "✓ 所有必需脚本已就绪"

# 检查真实数据（FID/FVD需要）
if [ "$SKIP_FID" = false ] || [ "$SKIP_FVD" = false ]; then
    if [ ! -d "$REAL_DATA_FRAMES" ] && [ "$SKIP_FID" = false ]; then
        log_message "⚠️  警告: 真实数据帧目录不存在: $REAL_DATA_FRAMES"
        log_message "   FID评估将被跳过"
        SKIP_FID=true
    fi
    
    if [ ! -d "$REAL_VIDEOS_DIR" ] && [ "$SKIP_FVD" = false ]; then
        log_message "⚠️  警告: 真实视频目录不存在: $REAL_VIDEOS_DIR"
        log_message "   FVD评估将被跳过"
        SKIP_FVD=true
    fi
fi

# 打印配置信息
echo ""
print_separator
log_message "配置信息"
print_separator
echo "  项目根目录: ${PROJECT_ROOT}"
echo "  视频输出目录: ${VIDEOS_DIR}"
echo "  指标输出目录: ${METRICS_DIR}"
echo "  日志目录: ${LOGS_DIR}"
echo "  评估级别: ${EVAL_LEVELS[@]}"
echo ""
echo "  跳过视频生成: ${SKIP_GENERATION}"
echo "  跳过tLPIPS: ${SKIP_TLPIPS}"
echo "  跳过Warping Error: ${SKIP_WARPING}"
echo "  跳过FID: ${SKIP_FID}"
echo "  跳过FVD: ${SKIP_FVD}"
echo "  跳过Edge IoU: ${SKIP_EDGE}"
echo "  覆盖已存在文件: ${OVERWRITE}"
print_separator

# 记录开始时间
START_TIME=$(date +%s)
LOG_FILE="${LOGS_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"

log_message "日志文件: ${LOG_FILE}"
echo ""

# 将输出重定向到日志文件（同时保留终端输出）
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

# ============================================================================
# 步骤1: 生成评估视频
# ============================================================================

STEP_NUM=1

if [ "$SKIP_GENERATION" = false ]; then
    print_step $STEP_NUM "生成评估视频 (所有级别)"
    
    OVERWRITE_FLAG=""
    if [ "$OVERWRITE" = true ]; then
        OVERWRITE_FLAG="--overwrite"
    fi
    
    LEVELS_FLAG=""
    if [ ${#EVAL_LEVELS[@]} -lt ${#ALL_LEVELS[@]} ]; then
        LEVELS_FLAG="--levels ${EVAL_LEVELS[@]}"
    fi
    
    cd "${SCRIPTS_DIR}/.."
    
    python3 scripts/generate_all_levels.py \
        --output_dir "${VIDEOS_DIR}" \
        --device cuda \
        --dtype fp16 \
        ${OVERWRITE_FLAG} \
        ${LEVELS_FLAG}
    
    check_status "视频生成"
    
    STEP_NUM=$((STEP_NUM + 1))
else
    log_message "⏩ 跳过视频生成步骤"
fi

# ============================================================================
# 步骤2: 评估tLPIPS (Temporal LPIPS)
# ============================================================================

if [ "$SKIP_TLPIPS" = false ]; then
    print_step $STEP_NUM "评估tLPIPS (时序感知LPIPS)"
    
    cd "${SCRIPTS_DIR}/.."
    
    python3 scripts/eval_tlpips.py \
        --video_dir "${VIDEOS_DIR}" \
        --output_file "${METRICS_DIR}/tlpips_results.json" \
        --device cuda
    
    check_status "tLPIPS评估"
    
    STEP_NUM=$((STEP_NUM + 1))
else
    log_message "⏩ 跳过tLPIPS评估"
fi

# ============================================================================
# 步骤3: 评估Warping Error (所有级别)
# ============================================================================

if [ "$SKIP_WARPING" = false ]; then
    print_step $STEP_NUM "评估Warping Error (所有级别)"
    
    cd "${SCRIPTS_DIR}/.."
    
    for level in "${EVAL_LEVELS[@]}"; do
        log_message "评估 ${level}..."
        
        if [ "$level" == "level_3" ] && [ -f "$TEMPORAL_CKPT" ]; then
            # Level 3使用Temporal Module
            python3 scripts/eval_warping_error_all_levels.py \
                --videos_dir "${VIDEOS_DIR}" \
                --level "${level}" \
                --output_file "${METRICS_DIR}/warping_error_${level}.json" \
                --temporal_checkpoint "${TEMPORAL_CKPT}" \
                --device cuda
        else
            # Level 0/1/2使用光流方法
            python3 scripts/eval_warping_error_all_levels.py \
                --videos_dir "${VIDEOS_DIR}" \
                --level "${level}" \
                --output_file "${METRICS_DIR}/warping_error_${level}.json" \
                --device cuda
        fi
        
        check_status "Warping Error评估 (${level})"
        echo ""
    done
    
    STEP_NUM=$((STEP_NUM + 1))
else
    log_message "⏩ 跳过Warping Error评估"
fi

# ============================================================================
# 步骤4: 评估FID (所有级别)
# ============================================================================

if [ "$SKIP_FID" = false ]; then
    print_step $STEP_NUM "评估FID (Fréchet Inception Distance)"
    
    cd "${SCRIPTS_DIR}/.."
    
    for level in "${EVAL_LEVELS[@]}"; do
        log_message "评估 ${level}..."
        
        python3 scripts/eval_fid.py \
            --videos_dir "${VIDEOS_DIR}" \
            --real_data_dir "${REAL_DATA_FRAMES}" \
            --level "${level}" \
            --output_file "${METRICS_DIR}/fid_${level}.json" \
            --device cuda \
            --batch_size 50
        
        check_status "FID评估 (${level})"
        echo ""
    done
    
    STEP_NUM=$((STEP_NUM + 1))
else
    log_message "⏩ 跳过FID评估"
fi

# ============================================================================
# 步骤5: 评估FVD (所有级别)
# ============================================================================

if [ "$SKIP_FVD" = false ]; then
    print_step $STEP_NUM "评估FVD (Fréchet Video Distance)"
    
    cd "${SCRIPTS_DIR}/.."
    
    for level in "${EVAL_LEVELS[@]}"; do
        log_message "评估 ${level}..."
        
        python3 scripts/eval_fvd.py \
            --videos_dir "${VIDEOS_DIR}" \
            --real_videos_dir "${REAL_VIDEOS_DIR}" \
            --level "${level}" \
            --output_file "${METRICS_DIR}/fvd_${level}.json" \
            --device cuda
        
        check_status "FVD评估 (${level})"
        echo ""
    done
    
    STEP_NUM=$((STEP_NUM + 1))
else
    log_message "⏩ 跳过FVD评估"
fi

# ============================================================================
# 步骤6: 评估Edge IoU (所有级别)
# ============================================================================

if [ "$SKIP_EDGE" = false ]; then
    print_step $STEP_NUM "评估Edge IoU (边缘一致性)"
    
    cd "${SCRIPTS_DIR}/.."
    
    for level in "${EVAL_LEVELS[@]}"; do
        log_message "评估 ${level}..."
        
        python3 scripts/eval_edge_iou.py \
            --videos_dir "${VIDEOS_DIR}" \
            --level "${level}" \
            --output_file "${METRICS_DIR}/edge_iou_${level}.json" \
            --device cuda \
            --canny_threshold1 50 \
            --canny_threshold2 150
        
        check_status "Edge IoU评估 (${level})"
        echo ""
    done
    
    STEP_NUM=$((STEP_NUM + 1))
else
    log_message "⏩ 跳过Edge IoU评估"
fi

# ============================================================================
# 步骤7: 汇总所有评估结果
# ============================================================================

print_step $STEP_NUM "汇总所有评估结果"

cd "${SCRIPTS_DIR}/.."

python3 scripts/summarize_all_levels.py \
    --metrics_dir "${METRICS_DIR}" \
    --output_file "${METRICS_DIR}/all_levels_summary.json"

check_status "结果汇总"

# ============================================================================
# 完成
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
print_separator
log_message "✅ 完整评估流程完成！"
print_separator
echo ""
echo "  总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "  结果位置:"
echo "    - 生成视频: ${VIDEOS_DIR}/"
echo "    - 评估指标: ${METRICS_DIR}/"
echo "    - 汇总报告: ${METRICS_DIR}/all_levels_summary.json"
echo "    - 日志文件: ${LOG_FILE}"
echo ""
print_separator

# 显示汇总结果
if [ -f "${METRICS_DIR}/all_levels_summary.json" ]; then
    echo ""
    log_message "评估结果预览:"
    echo ""
    python3 -c "
import json
import sys

try:
    with open('${METRICS_DIR}/all_levels_summary.json', 'r') as f:
        data = json.load(f)
    
    if 'comparison_table' in data:
        print('  指标对比表:')
        print('  ' + '-'*100)
        
        header = f\"  {'指标':<20} {'趋势':<15}\"
        for level in ['level_0', 'level_1', 'level_2', 'level_3']:
            header += f\" {level:<18}\"
        print(header)
        print('  ' + '-'*100)
        
        for metric_name, metric_data in data['comparison_table'].items():
            trend = '越低越好 ↓' if metric_data.get('lower_is_better', True) else '越高越好 ↑'
            row = f\"  {metric_name:<20} {trend:<15}\"
            
            for level in ['level_0', 'level_1', 'level_2', 'level_3']:
                value = metric_data.get('values', {}).get(level, 'N/A')
                if isinstance(value, (int, float)):
                    row += f\" {value:<18.6f}\"
                else:
                    row += f\" {str(value):<18}\"
            print(row)
        
        print('  ' + '-'*100)
    
except Exception as e:
    print(f'  无法加载汇总结果: {e}')
"
fi

echo ""
log_message "评估流程脚本执行完毕"
print_separator
