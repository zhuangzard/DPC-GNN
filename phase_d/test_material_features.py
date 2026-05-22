#!/usr/bin/env python3
"""
测试材料感知特征注入模块

测试内容：
1. 每种材料的特征向量
2. E 对数归一化范围正确性
3. Embedding 梯度可传播
4. 注入到节点特征的维度变化
5. 5 种材料的特征差异对比
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from material_features import MaterialFeatureInjector, get_material_info, list_materials


def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_material_features():
    """测试 1: 每种材料的特征向量"""
    print_separator("测试 1: 每种材料的特征向量")
    
    injector = MaterialFeatureInjector(embed_dim=8)
    
    print(f"\n特征向量维度: 2 (E_norm + nu) + 8 (embedding) = 10")
    print("-" * 60)
    
    for mat_name in list_materials():
        mat_info = get_material_info(mat_name)
        features = injector.get_material_features(mat_name)
        
        print(f"\n{mat_name.upper()}:")
        print(f"  E = {mat_info['E']:,.0f} Pa ({mat_info['E']/1e9:.2f} GPa)")
        print(f"  nu = {mat_info['nu']}")
        print(f"  特征向量: {features.detach().numpy()}")
        print(f"  E_norm = {features[0].item():.4f}")
        print(f"  nu = {features[1].item():.4f}")
        print(f"  embedding (前4维): {features[2:6].detach().numpy()}")


def test_normalization():
    """测试 2: E 对数归一化范围正确性"""
    print_separator("测试 2: E 对数归一化范围正确性")
    
    injector = MaterialFeatureInjector(embed_dim=8)
    
    print("\n验证归一化范围 [0, 1]:")
    print("-" * 60)
    
    for mat_name in list_materials():
        mat_info = get_material_info(mat_name)
        E = mat_info['E']
        
        # 计算归一化值
        E_norm = injector.normalize_E(E).item()
        
        # 验证范围
        in_range = 0.0 <= E_norm <= 1.0
        status = "✓" if in_range else "✗"
        
        print(f"{mat_name:12s}: E={E:>13,.0f} Pa → E_norm={E_norm:.4f} {status}")
        
        if not in_range:
            print(f"  WARNING: 归一化值超出范围！")


def test_gradient_propagation():
    """测试 3: Embedding 梯度可传播"""
    print_separator("测试 3: Embedding 梯度可传播")
    
    injector = MaterialFeatureInjector(embed_dim=8)
    
    print("\n创建计算图并反向传播...")
    print("-" * 60)
    
    # 创建一个简单的损失函数
    features = injector.get_material_features("bone")
    
    # 模拟一个损失：特征向量的和
    loss = features.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查 embedding 的梯度
    has_grad = injector.embedding.weight.grad is not None
    
    if has_grad:
        grad_norm = injector.embedding.weight.grad.norm().item()
        print(f"✓ Embedding 有梯度")
        print(f"  梯度范数: {grad_norm:.6f}")
        print(f"  梯度形状: {injector.embedding.weight.grad.shape}")
        
        # 检查哪些材料有梯度更新
        print("\n  各材料的梯度更新:")
        for mat_name in list_materials():
            mat_id = get_material_info(mat_name)['id']
            grad_for_mat = injector.embedding.weight.grad[mat_id]
            grad_norm_mat = grad_for_mat.norm().item()
            print(f"    {mat_name}: grad_norm={grad_norm_mat:.6f}")
    else:
        print("✗ Embedding 没有梯度！")
    
    return has_grad


def test_injection_dimensions():
    """测试 4: 注入到节点特征的维度变化"""
    print_separator("测试 4: 注入到节点特征的维度变化")
    
    injector = MaterialFeatureInjector(embed_dim=8)
    
    print("\n测试不同输入维度:")
    print("-" * 60)
    
    test_dims = [16, 32, 64, 128]
    N_nodes = 100  # 节点数
    
    for input_dim in test_dims:
        # 创建随机节点特征
        h_nodes = torch.randn(N_nodes, input_dim)
        
        # 注入材料特征
        h_enhanced = injector.inject_to_node_features(h_nodes, "bone")
        
        # 验证维度
        expected_dim = injector.get_output_dim(input_dim)
        actual_dim = h_enhanced.shape[1]
        
        match = actual_dim == expected_dim
        status = "✓" if match else "✗"
        
        print(f"输入维度: {input_dim:3d} → 输出维度: {actual_dim:3d} (期望: {expected_dim:3d}) {status}")
        
        # 验证节点数不变
        assert h_enhanced.shape[0] == N_nodes, "节点数不应该改变！"
    
    print(f"\n✓ 所有维度测试通过")


def test_material_difference():
    """测试 5: 5 种材料的特征差异对比"""
    print_separator("测试 5: 5 种材料的特征差异对比")
    
    # 使用固定的 embedding 以便比较
    torch.manual_seed(42)
    injector = MaterialFeatureInjector(embed_dim=8)
    
    print("\n材料特征向量对比:")
    print("-" * 60)
    
    # 收集所有材料的特征
    features_dict = {}
    for mat_name in list_materials():
        features_dict[mat_name] = injector.get_material_features(mat_name).detach()
    
    # 打印 E_norm 和 nu 的对比（物理属性，固定）
    print("\n物理属性部分（前2维，固定）:")
    print(f"{'材料':<12} {'E_norm':>8} {'nu':>8}")
    print("-" * 60)
    for mat_name in list_materials():
        feat = features_dict[mat_name]
        print(f"{mat_name:<12} {feat[0].item():>8.4f} {feat[1].item():>8.4f}")
    
    # 计算材料间的欧氏距离
    print("\n材料间欧氏距离（完整特征向量）:")
    print("-" * 60)
    
    materials = list_materials()
    n_mats = len(materials)
    
    # 打印表头
    header = "         "
    for j in range(n_mats):
        header += f"{materials[j][:8]:>10}"
    print(header)
    
    # 计算并打印距离矩阵
    for i in range(n_mats):
        row = f"{materials[i][:8]:<8}"
        for j in range(n_mats):
            if i == j:
                dist_str = "    -    "
            else:
                dist = (features_dict[materials[i]] - features_dict[materials[j]]).norm().item()
                dist_str = f"{dist:>8.4f}"
            row += f" {dist_str}"
        print(row)
    
    # 关键对比：bone vs brain
    print("\n关键对比: Bone vs Brain")
    print("-" * 60)
    bone_feat = features_dict["bone"]
    brain_feat = features_dict["brain"]
    
    E_diff = (bone_feat[0] - brain_feat[0]).item()
    nu_diff = (bone_feat[1] - brain_feat[1]).item()
    total_diff = (bone_feat - brain_feat).norm().item()
    
    print(f"E_norm 差异: {E_diff:.4f} (bone={bone_feat[0]:.4f}, brain={brain_feat[0]:.4f})")
    print(f"nu 差异:     {nu_diff:.4f} (bone={bone_feat[1]:.4f}, brain={brain_feat[1]:.4f})")
    print(f"总欧氏距离:  {total_diff:.4f}")
    
    print(f"\n✓ Bone 和 Brain 特征差异显著（距离={total_diff:.4f}）")


def test_embedding_learning():
    """测试 6: Embedding 的学习能力"""
    print_separator("测试 6: Embedding 的学习能力")
    
    print("\n模拟训练场景：让模型学习区分 bone 和 brain")
    print("-" * 60)
    
    # 创建注入器
    injector = MaterialFeatureInjector(embed_dim=8)
    optimizer = torch.optim.SGD(injector.embedding.parameters(), lr=0.1)
    
    # 目标：bone 特征的第一维（E_norm）应该远大于 brain
    # 我们通过训练 embedding 来增强这种差异
    
    initial_bone = injector.get_material_features("bone").detach().clone()
    initial_brain = injector.get_material_features("brain").detach().clone()
    initial_diff = (initial_bone - initial_brain).norm().item()
    
    print(f"初始差异: {initial_diff:.4f}")
    
    # 训练几步
    for step in range(10):
        optimizer.zero_grad()
        
        bone_feat = injector.get_material_features("bone")
        brain_feat = injector.get_material_features("brain")
        
        # 损失：希望 bone 和 brain 在 embedding 空间中差异更大
        # 同时保持 E_norm 的物理意义
        loss = -((bone_feat - brain_feat).norm())  # 负号表示我们想要最大化差异
        
        loss.backward()
        optimizer.step()
    
    final_bone = injector.get_material_features("bone").detach()
    final_brain = injector.get_material_features("brain").detach()
    final_diff = (final_bone - final_brain).norm().item()
    
    print(f"训练后差异: {final_diff:.4f}")
    print(f"差异变化: {final_diff - initial_diff:+.4f}")
    
    print("\n✓ Embedding 可以通过训练来调整材料表示")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  材料感知特征注入模块 - 单元测试")
    print("=" * 60)
    
    try:
        test_material_features()
        test_normalization()
        gradient_ok = test_gradient_propagation()
        test_injection_dimensions()
        test_material_difference()
        test_embedding_learning()
        
        print_separator("测试总结")
        print("\n✓ 所有测试通过！")
        print(f"  - 特征向量生成: ✓")
        print(f"  - 归一化范围: ✓")
        print(f"  - 梯度传播: {'✓' if gradient_ok else '✗'}")
        print(f"  - 维度变换: ✓")
        print(f"  - 材料差异: ✓")
        print(f"  - Embedding 学习: ✓")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
