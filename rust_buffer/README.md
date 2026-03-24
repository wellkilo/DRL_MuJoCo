# Rust Buffer - 高性能回放缓冲区

## 构建

```bash
cd rust_buffer
pip install maturin
maturin develop --release
```

## 特性

- SoA（Structure of Arrays）连续内存布局
- Rayon 并行 GAE 计算
- PyO3 Python 绑定
- 内置 Advantage 标准化

## 性能预期

| 操作 | Python 现有 | Rust 改造后 | 加速比 |
|------|------------|------------|--------|
| 存储 16384 条经验 | ~12ms | ~0.8ms | 15x |
| GAE 计算（8×2048 步） | ~45ms | ~2ms | 20x |
| 数据序列化（Ray 传输） | ~30ms | ~3ms | 10x |
| Advantage 标准化 | ~5ms | ~0.1ms | 50x |
