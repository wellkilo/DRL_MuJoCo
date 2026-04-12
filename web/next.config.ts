import type { NextConfig } from 'next';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Disable devtools to avoid Next.js 15.3+ bug:
  // "Could not find the module segment-explorer-node.js#SegmentViewNode in the React Client Manifest"
  devtools: false,
  outputFileTracingRoot: path.join(__dirname, '..'),
  // These rewrites work with `next dev` but are ignored in static export.
  // In production, FastAPI serves both the static files and the API.
  async rewrites() {
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