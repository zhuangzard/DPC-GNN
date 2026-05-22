"""
材料感知特征注入模块
将材料属性(E, ν, embedding)注入节点特征
"""

import torch
import torch.nn as nn

MATERIALS = {
    "brain":      {"E": 1000.0,        "nu": 0.49, "id": 0},
    "kidney":     {"E": 10000.0,       "nu": 0.45, "id": 1},
    "myocardium": {"E": 30000.0,       "nu": 0.40, "id": 2},
    "cartilage":  {"E": 500000.0,      "nu": 0.30, "id": 3},
    "bone":       {"E": 10000000000.0, "nu": 0.30, "id": 4},
}

class MaterialFeatureInjector(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(5, embed_dim)
        self.E_min, self.E_max = 3.0, 10.0  # log10范围
    
    def normalize_E(self, E):
        log_E = torch.log10(torch.clamp(E, min=1e-10))
        return (log_E - self.E_min) / (self.E_max - self.E_min)
    
    def get_features(self, material_name, E, nu):
        mat_id = MATERIALS[material_name]["id"]
        embed = self.embedding(torch.tensor(mat_id))
        E_norm = self.normalize_E(E)
        return torch.cat([E_norm.unsqueeze(0), torch.tensor([nu]), embed])
    
    def inject(self, h_nodes, material_name, E, nu):
        """h_nodes: (N, feat_dim) -> (N, feat_dim + 2 + embed_dim)"""
        mat_feat = self.get_features(material_name, E, nu)
        N = h_nodes.shape[0]
        return torch.cat([h_nodes, mat_feat.unsqueeze(0).expand(N, -1)], dim=-1)

# 测试
if __name__ == "__main__":
    injector = MaterialFeatureInjector(embed_dim=8)
    
    print("=== 材料特征向量 ===")
    for name, mat in MATERIALS.items():
        feat = injector.get_features(name, torch.tensor(mat["E"]), mat["nu"])
        print(f"{name:15s}: shape={feat.shape}, E_norm={feat[0]:.4f}, nu={feat[1]:.2f}")
    
    # 测试注入
    h = torch.randn(100, 32)  # 100节点, 32维特征
    h_bone = injector.inject(h, "bone", torch.tensor(1e10), 0.3)
    print(f"\n注入前: {h.shape}, 注入后: {h_bone.shape}")
    assert h_bone.shape == (100, 32 + 2 + 8), "维度错误"
    print("✅ 特征注入测试通过")
    
    # 测试梯度
    h.requires_grad_(True)
    h_enhanced = injector.inject(h, "bone", torch.tensor(1e10), 0.3)
    loss = h_enhanced.sum()
    loss.backward()
    assert h.grad is not None, "梯度未传播"
    print("✅ 梯度传播测试通过")
