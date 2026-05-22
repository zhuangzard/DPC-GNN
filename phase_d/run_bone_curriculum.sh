#!/bin/bash
# 骨头课程学习训练脚本
# 流程: Brain → Kidney → Myocardium → Cartilage → Bone

set -e
cd ~/workspace/DPC-GNN/phase_d

CKPT_DIR=~/workspace/DPC-GNN/multi_tissue
LOG_DIR=~/workspace/DPC-GNN/multi_tissue
RESULTS_FILE=~/workspace/DPC-GNN/multi_tissue/curriculum_results.json

echo "========================================"
echo "骨头课程学习训练"
echo "========================================"

# Stage 1: Brain (已有 checkpoint，跳过)
echo "[1/5] Brain (E=1kPa) - 使用已有checkpoint"

# Stage 2: Kidney (已有 checkpoint，跳过)  
echo "[2/5] Kidney (E=10kPa) - 使用已有checkpoint"

# Stage 3: Myocardium (已有 checkpoint，跳过)
echo "[3/5] Myocardium (E=30kPa) - 使用已有checkpoint"

# Stage 4: Cartilage (已有 checkpoint，跳过)
echo "[4/5] Cartilage (E=500kPa) - 使用已有checkpoint"

# Stage 5: Bone - 需要训练
echo "[5/5] Bone (E=10GPa) - 开始训练"

# Bone 参数
BONE_E=10000000000
BONE_NU=0.30
BONE_RHO=1900
BONE_LR=5e-5
BONE_LAMBDA=100.0
BONE_J_THRESH=0.05
BONE_EPOCHS=1000

echo ""
echo "Bone 配置:"
echo "  E = $BONE_E Pa (10 GPa)"
echo "  lr = $BONE_LR"
echo "  lambda_barrier = $BONE_LAMBDA"
echo "  j_threshold = $BONE_J_THRESH"
echo "  epochs = $BONE_EPOCHS"
echo "  warm_start = Cartilage checkpoint"
echo ""

# 启动 Bone 训练
nohup python3 -u train_phase_d.py \
  --stage d2 \
  --no-detach \
  --lr $BONE_LR \
  --lambda-barrier $BONE_LAMBDA \
  --j-threshold $BONE_J_THRESH \
  --warm-start $CKPT_DIR/cartilage/ckpt/phase_d/final_model_d2.pt \
  --E $BONE_E --nu $BONE_NU --rho $BONE_RHO \
  --checkpoint-dir $CKPT_DIR/bone/ckpt/phase_d \
  > $LOG_DIR/bone/results/phase_d_bone.log 2>&1 &

BONE_PID=$!
echo "Bone PID: $BONE_PID"

# 等待并监控
echo ""
echo "开始监控 Bone 训练..."
while kill -0 $BONE_PID 2>/dev/null; do
    sleep 60
    if [ -f $LOG_DIR/bone/results/phase_d_bone.log ]; then
        EPOCH=$(grep -oE "Ep [0-9]+" $LOG_DIR/bone/results/phase_d_bone.log | tail -1 | grep -oE "[0-9]+")
        J_MIN=$(grep "J_min" $LOG_DIR/bone/results/phase_d_bone.log | tail -1 | grep -oE "J_min=-?[0-9.]+" | head -1)
        echo "[$(date +%H:%M)] Bone: Epoch=$EPOCH, $J_MIN"
    fi
done

echo ""
echo "Bone 训练完成！"
grep "Best 50-step" $LOG_DIR/bone/results/phase_d_bone.log
