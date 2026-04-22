import type { NextConfig } from 'next';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// [FIX] 通过环境变量区分 dev 和静态导出两种场景
//   - dev 模式 (`npm run dev`)：不加 output:'export'，rewrites 才会生效
//   - 生产构建 (`NEXT_EXPORT=1 npm run build`)：加 output:'export'，由 FastAPI 统一托管
// 原代码同时写了 output:'export' + rewrites，rewrites 在静态导出模式下会被 Next.js
// 完全忽略，导致 `next dev` (:3000) 下点击训练按钮时 POST /api/... 会直接打到 3000
// 而不是代理到后端 8000，从而"没反应"。
const isExport = process.env.NEXT_EXPORT === '1';

const nextConfig: NextConfig = {
  ...(isExport ? { output: 'export' as const } : {}),
  images: {
    unoptimized: true,
  },
  // Disable devtools to avoid Next.js 15.3+ bug:
  // "Could not find the module segment-explorer-node.js#SegmentViewNode in the React Client Manifest"
  devtools: false,
  outputFileTracingRoot: path.join(__dirname, '..'),
  // 仅在非静态导出模式下生效：把 /api/* 与 /ws/* 代理到 FastAPI 8000
  async rewrites() {
    if (isExport) return [];
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
      {
        source: '/ws/:path*',
        destination: 'http://127.0.0.1:8000/ws/:path*',
      },
      {
        source: '/output/:path*',
        destination: 'http://127.0.0.1:8000/output/:path*',
      },
    ];
  },
};

export default nextConfig;
