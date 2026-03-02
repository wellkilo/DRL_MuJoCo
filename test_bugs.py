#!/usr/bin/env python3
"""简单的测试脚本来检查项目中的潜在bug"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).parent))

def test_models_forward() -> bool:
    """测试 models.py 的 forward 方法是否正确"""
    print("测试 models.py 的 forward 方法...")
    try:
        import torch
        from drl.models import ActorCritic
        
        # 创建一个简单的模型
        model = ActorCritic(obs_dim=11, action_dim=3, hidden_sizes=(64, 64))
        
        # 创建一个测试输入
        obs = torch.randn(2, 11)  # batch_size=2, obs_dim=11
        
        # 前向传播
        mean, value = model.forward(obs)
        
        print(f"  ✓ 策略输出 shape: {mean.shape}")
        print(f"  ✓ 价值输出 shape: {value.shape}")
        
        # 检查输出维度是否正确
        assert mean.shape == (2, 3), f"期望 (2, 3), 得到 {mean.shape}"
        assert value.shape == (2,), f"期望 (2,), 得到 {value.shape}"
        
        print("  ✓ models.py forward 方法正常！")
        return True
    except AssertionError as e:
        print(f"  ✗ models.py 测试失败: {e}")
        return False
    except Exception as e:
        print(f"  ✗ models.py 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loader() -> bool:
    """测试配置加载器"""
    print("\n测试配置加载器...")
    try:
        from drl.config_loader import load_config
        
        cfg = load_config("config/config.yaml")
        
        print(f"  ✓ 环境名称: {cfg.env_name}")
        print(f"  ✓ Actor 数量: {cfg.num_actors}")
        print(f"  ✓ 学习率: {cfg.lr}")
        print(f"  ✓ 隐藏层大小: {cfg.hidden_sizes}")
        
        print("  ✓ 配置加载器正常！")
        return True
    except Exception as e:
        print(f"  ✗ 配置加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_loss() -> bool:
    """测试 PPO 损失函数计算"""
    print("\n测试 PPO 损失函数...")
    try:
        import torch
        import numpy as np
        from drl.models import ActorCritic
        from drl.ray_components import _ppo_loss, _to_tensor
        
        # 创建模型
        model = ActorCritic(obs_dim=11, action_dim=3, hidden_sizes=(64, 64))
        
        # 创建一些假的批量数据
        batch = []
        for _ in range(10):
            batch.append({
                "obs": np.random.randn(11).astype(np.float32),
                "act": np.random.randn(3).astype(np.float32),
                "logp": -1.0,
                "adv": 0.1,
                "ret": 10.0
            })
        
        # 转换为张量
        batch_t = _to_tensor(batch, torch.device("cpu"))
        
        # 计算损失
        loss, metrics = _ppo_loss(
            model, 
            batch_t,
            clip_ratio=0.2,
            vf_coef=0.5,
            ent_coef=0.01
        )
        
        print(f"  ✓ 总损失: {loss.item():.4f}")
        print(f"  ✓ 策略损失: {metrics['policy_loss']:.4f}")
        print(f"  ✓ 价值损失: {metrics['value_loss']:.4f}")
        print(f"  ✓ 熵: {metrics['entropy']:.4f}")
        
        print("  ✓ PPO 损失函数正常！")
        return True
    except Exception as e:
        print(f"  ✗ PPO 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    print("=" * 60)
    print("检查 DRL_MuJoCo 项目中的 Bug")
    print("=" * 60)
    
    results = []
    results.append(("models.py forward 方法", test_models_forward()))
    results.append(("配置加载器", test_config_loader()))
    results.append(("PPO 损失函数", test_ppo_loss()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！除了已发现的 models.py bug 外，没有发现其他明显的 bug。")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
    print("=" * 60)


if __name__ == "__main__":
    main()

