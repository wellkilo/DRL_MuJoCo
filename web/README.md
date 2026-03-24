# DRL MuJoCo TypeScript 前端

## 开发

```bash
cd web
npm install
npm run dev
```

开发时 Vite 会通过代理转发 `/api` 和 `/ws` 请求到 FastAPI 后端（http://127.0.0.1:8000）

## 构建

```bash
npm run build
```

构建产物在 `dist/` 目录下，FastAPI 后端会自动检测并使用该目录。

## 特性

- 完整的 TypeScript 类型安全
- React 18 + Zustand 状态管理
- 通用图表组件（一个组件替代 12 个手动创建的图表）
- 16 个训练指标的完整展示
- WebSocket 实时数据推送
- 响应式设计
