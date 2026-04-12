import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DRL MuJoCo — 分布式训练监控',
  description: '实时监控分布式强化学习训练过程',
  icons: {
    icon: '/icon.svg',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <head suppressHydrationWarning>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var theme = localStorage.getItem('drl-mujoco-theme');
                  if (theme === 'light') {
                    document.documentElement.classList.remove('dark');
                  } else {
                    document.documentElement.classList.add('dark');
                  }
                } catch(e) {
                  document.documentElement.classList.add('dark');
                }
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-bg-base p-4 md:p-6 lg:p-8" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}