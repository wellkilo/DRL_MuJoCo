import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DRL MuJoCo Web UI',
  description: '实时监控分布式强化学习训练过程',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="min-h-screen bg-gradient-to-br from-[#667eea] to-[#764ba2] p-4 md:p-8" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}