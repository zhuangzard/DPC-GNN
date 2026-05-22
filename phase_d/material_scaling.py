"""
材料自适应位移缩放模块
解决硬材料位移信号微弱导致的梯度消失问题
"""

import torch
import torch.nn as nn

MATERIALS = {
    "brain":      {"E": 1000.0,        "nu": 0.49, "rho": 1040},
    "kidney":     {"E": 10000.0,       "nu": 0.45, "rho": 1050},
    "myocardium": {"E": 30000.0,       "nu": 0.40, "rho": 1060},
    "cartilage":  {"E": 500000.0,      "nu": 0.30, "rho": 1100},
    "bone":       {"E": 10000000000.0, "nu": 0.30, "rho": 1900},
}

class MaterialAwareScaling:
    def __init__(self, E_ref=10000.0, exponent=0.5):
        self.E_ref = E_ref
        self.exponent = exponent
    
    def scale_factor(self, E):
        """alpha = (E_ref/E)^exponent"""
        return (self.E_ref / E) ** self.exponent
    
    def scale_displacement(self, u, E):
        """GNN输入前缩放"""
        return u * self.scale_factor(E)
    
    def unscale_displacement(self, u_scaled, E):
        """GNN输出后恢复"""
        return u_scaled / self.scale_factor(E)

# 单元测试
if __name__ == "__main__":
    scaler = MaterialAwareScaling()
    
    print("=== 材料缩放因子 ===")
    for name, mat in MATERIALS.items():
        alpha = scaler.scale_factor(mat["E"])
        u_typical = 0.01  # 10mm
        u_scaled = scaler.scale_displacement(torch.tensor([u_typical]), mat["E"])
        print(f"{name:15s}: alpha={alpha:.6f}, 10mm→{u_scaled.item():.3f}mm (scaled)")
    
    # 验证反缩放
    for name, mat in MATERIALS.items():
        u = torch.tensor([5.0])
        u_scaled = scaler.scale_displacement(u, mat["E"])
        u_recovered = scaler.unscale_displacement(u_scaled, mat["E"])
        assert torch.allclose(u, u_recovered), f"Failed for {name}"
    print("\n✅ 所有反缩放测试通过")
