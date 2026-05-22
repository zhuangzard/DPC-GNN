"""位移缩放模块测试脚本

测试内容:
1. 不同E值的缩放因子计算
2. 缩放后位移量级对比
3. 反缩放恢复原始值验证
4. Bone场景专项测试
"""
import numpy as np
import torch
import sys
from material_scaling import MaterialAwareScaling, MATERIALS


def test_scale_factors():
    """测试1: 不同材料的缩放因子"""
    print("\n" + "=" * 70)
    print("测试 1: 各材料缩放因子计算")
    print("=" * 70)
    
    scaler = MaterialAwareScaling(E_ref=10000.0, exponent=0.5)
    
    print(f"\n参考杨氏模量 E_ref = {scaler.E_ref:.2e} Pa")
    print(f"缩放公式: α = (E_ref/E)^{scaler.exponent}")
    print("\n" + "-" * 50)
    print(f"{'材料':<15} {'E (Pa)':<15} {'α':<12} {'效果'}")
    print("-" * 50)
    
    for name, props in MATERIALS.items():
        E = props["E"]
        alpha = scaler.scale_factor(E)
        effect = "放大" if alpha > 1 else "缩小"
        print(f"{name:<15} {E:<15.0e} {alpha:<12.6f} {effect} {alpha:.2e}x")
    
    print("-" * 50)


def test_displacement_scaling():
    """测试2: 位移缩放与反缩放"""
    print("\n" + "=" * 70)
    print("测试 2: 位移缩放与反缩放验证")
    print("=" * 70)
    
    scaler = MaterialAwareScaling(E_ref=10000.0, exponent=0.5)
    
    # 模拟受力产生的原始位移 u = F/E (假设F=1N)
    F = 1.0  # 假设施加1N力
    
    print(f"\n假设施加力 F = {F} N")
    print(f"位移 u ∝ F/E")
    print("\n" + "-" * 70)
    print(f"{'材料':<12} {'E (Pa)':<12} {'u_raw (mm)':<14} {'u_scaled':<14} {'反缩放误差'}")
    print("-" * 70)
    
    for name, props in MATERIALS.items():
        E = props["E"]
        
        # 原始位移 (模拟值)
        u_raw = F / E  # m
        u_raw_mm = u_raw * 1000  # mm
        
        # 缩放
        u_scaled = scaler.scale_displacement(u_raw, E)
        
        # 反缩放
        u_recovered = scaler.unscale_displacement(u_scaled, E)
        
        # 误差
        error = abs(u_recovered - u_raw) / abs(u_raw) if u_raw != 0 else 0
        
        print(f"{name:<12} {E:<12.0e} {u_raw_mm:<14.6e} {u_scaled:<14.6e} {error:.2e}")
    
    print("-" * 70)


def test_magnitude_unification():
    """测试3: 验证缩放后位移量级统一"""
    print("\n" + "=" * 70)
    print("测试 3: 缩放后位移量级对比 (关键测试)")
    print("=" * 70)
    
    scaler = MaterialAwareScaling(E_ref=10000.0, exponent=0.5)
    
    # 统一力下的位移
    F = 1.0  # N
    
    print(f"\n假设各材料受力 F = {F} N")
    print("位移缩放前后对比:\n")
    print("-" * 70)
    print(f"{'材料':<12} {'原始位移(m)':<18} {'缩放后位移':<18} {'提升倍数'}")
    print("-" * 70)
    
    scaled_displacements = []
    
    for name, props in MATERIALS.items():
        E = props["E"]
        u_raw = F / E  # m
        u_scaled = scaler.scale_displacement(u_raw, E)
        scaled_displacements.append((name, u_scaled))
        
        improvement = u_scaled / u_raw if u_raw != 0 else 0
        print(f"{name:<12} {u_raw:<18.6e} {u_scaled:<18.6e} {improvement:<10.0f}x")
    
    print("-" * 70)
    
    # 计算量级差异
    values = [v for _, v in scaled_displacements]
    max_val = max(values)
    min_val = min(values)
    ratio = max_val / min_val if min_val != 0 else float('inf')
    
    print(f"\n缩放后位移范围: [{min_val:.2e}, {max_val:.2e}] m")
    print(f"最大值/最小值 = {ratio:.2f} (理想接近1)")
    
    if ratio < 10:
        print("✓ 缩放后位移量级基本统一!")
    else:
        print("⚠ 缩放后位移量级仍有差异")


def test_bone_scenario():
    """测试4: Bone专项测试 (E=10GPa)"""
    print("\n" + "=" * 70)
    print("测试 4: Bone场景专项测试 (E = 10 GPa)")
    print("=" * 70)
    
    scaler = MaterialAwareScaling(E_ref=10000.0, exponent=0.5)
    
    E_bone = 10_000_000_000  # 10 GPa
    alpha_bone = scaler.scale_factor(E_bone)
    
    print(f"\n骨材料参数:")
    print(f"  杨氏模量 E = {E_bone:.2e} Pa = {E_bone/1e9:.0f} GPa")
    print(f"  参考模量 E_ref = {scaler.E_ref:.2e} Pa")
    print(f"  缩放因子 α = (E_ref/E)^0.5 = {alpha_bone:.6f}")
    print(f"  → Bone位移将被放大 {1/alpha_bone:.0f} 倍")
    
    # 模拟实际位移
    print(f"\n模拟Bone在不同受力下的位移:")
    print("-" * 60)
    print(f"{'力 (N)':<12} {'原始位移(mm)':<18} {'缩放后(m)':<18} {'适合float32?'}")
    print("-" * 60)
    
    forces = [0.001, 0.01, 0.1, 1.0, 10.0]
    float32_min = 1e-7  # float32有效精度下限
    
    for F in forces:
        u_raw = F / E_bone  # m
        u_raw_mm = u_raw * 1000
        u_scaled = scaler.scale_displacement(u_raw, E_bone)
        
        suitable = "✓ 是" if u_scaled > float32_min else "✗ 否"
        print(f"{F:<12.3f} {u_raw_mm:<18.6f} {u_scaled:<18.6e} {suitable}")
    
    print("-" * 60)
    print(f"\nfloat32 有效精度下限: ~{float32_min:.0e}")
    print("Bone原始位移 ~1e-6 mm，缩放后 ~1e-3 m，进入float32安全区!")


def test_numpy_torch_compatibility():
    """测试5: NumPy和PyTorch兼容性"""
    print("\n" + "=" * 70)
    print("测试 5: NumPy和PyTorch兼容性")
    print("=" * 70)
    
    scaler = MaterialAwareScaling(E_ref=10000.0, exponent=0.5)
    E_kidney = 10000.0
    
    # NumPy测试
    u_np = np.array([0.001, 0.002, 0.003])
    u_scaled_np = scaler.scale_displacement(u_np, E_kidney)
    u_recovered_np = scaler.unscale_displacement(u_scaled_np, E_kidney)
    
    print(f"\nNumPy数组测试:")
    print(f"  原始位移: {u_np}")
    print(f"  缩放后:   {u_scaled_np}")
    print(f"  反缩放:   {u_recovered_np}")
    print(f"  误差:     {np.max(np.abs(u_recovered_np - u_np)):.2e}")
    
    # PyTorch测试
    u_torch = torch.tensor([0.001, 0.002, 0.003])
    u_scaled_torch = scaler.scale_displacement(u_torch, E_kidney)
    u_recovered_torch = scaler.unscale_displacement(u_scaled_torch, E_kidney)
    
    print(f"\nPyTorch张量测试:")
    print(f"  原始位移: {u_torch}")
    print(f"  缩放后:   {u_scaled_torch}")
    print(f"  反缩放:   {u_recovered_torch}")
    print(f"  误差:     {torch.max(torch.abs(u_recovered_torch - u_torch)).item():.2e}")


def test_batch_scaling():
    """测试6: 批量数据测试"""
    print("\n" + "=" * 70)
    print("测试 6: 批量数据缩放 (模拟GNN batch)")
    print("=" * 70)
    
    scaler = MaterialAwareScaling(E_ref=10000.0, exponent=0.5)
    
    # 模拟batch: 不同材料的混合
    batch_size = 5
    u_batch = np.array([
        1e-6,   # brain位移
        1e-7,   # kidney位移  
        3e-8,   # myocardium位移
        2e-9,   # cartilage位移
        1e-10,  # bone位移
    ])
    E_batch = np.array([
        MATERIALS["brain"]["E"],
        MATERIALS["kidney"]["E"],
        MATERIALS["myocardium"]["E"],
        MATERIALS["cartilage"]["E"],
        MATERIALS["bone"]["E"],
    ])
    
    print(f"\nBatch数据 (位移单位: m):")
    print("-" * 50)
    materials_list = ["brain", "kidney", "myocardium", "cartilage", "bone"]
    for i, mat in enumerate(materials_list):
        print(f"  {mat}: u={u_batch[i]:.2e}, E={E_batch[i]:.2e}")
    
    # 批量缩放
    u_scaled_batch = scaler.scale_displacement(u_batch, E_batch)
    u_recovered_batch = scaler.unscale_displacement(u_scaled_batch, E_batch)
    
    print(f"\n缩放后位移: {u_scaled_batch}")
    print(f"反缩放位移: {u_recovered_batch}")
    print(f"最大误差: {np.max(np.abs(u_recovered_batch - u_batch)):.2e}")
    
    # 检查量级统一性
    print(f"\n缩放前位移范围: [{u_batch.min():.2e}, {u_batch.max():.2e}]")
    print(f"缩放后位移范围: [{u_scaled_batch.min():.2e}, {u_scaled_batch.max():.2e}]")
    print(f"范围缩小比: {u_batch.max()/u_batch.min():.0f}x → {u_scaled_batch.max()/u_scaled_batch.min():.1f}x")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "█" * 70)
    print("  材料自适应位移缩放模块 - 单元测试")
    print("█" * 70)
    
    test_scale_factors()
    test_displacement_scaling()
    test_magnitude_unification()
    test_bone_scenario()
    test_numpy_torch_compatibility()
    test_batch_scaling()
    
    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
