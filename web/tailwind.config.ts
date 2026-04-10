import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Surface colors (CSS variables for theme switching)
        'surface-900': 'var(--surface-900)',
        'surface-800': 'var(--surface-800)',
        'surface-700': 'var(--surface-700)',
        'surface-600': 'var(--surface-600)',
        'surface-500': 'var(--surface-500)',
        'surface-400': 'var(--surface-400)',
        // Accent colors
        primary: '#6366f1',
        'primary-light': '#818cf8',
        'primary-dark': '#4f46e5',
        accent: '#06b6d4',
        'accent-light': '#22d3ee',
        success: '#10b981',
        'success-light': '#34d399',
        danger: '#ef4444',
        'danger-light': '#f87171',
        warning: '#f59e0b',
        'warning-light': '#fbbf24',
        // Chart colors
        'chart-blue': '#6366f1',
        'chart-cyan': '#06b6d4',
        'chart-amber': '#f59e0b',
        'chart-emerald': '#10b981',
        'chart-rose': '#f43f5e',
        'chart-violet': '#8b5cf6',
        // Text (CSS variables)
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-muted': 'var(--text-muted)',
        // Border (CSS variables)
        'border-dark': 'var(--border-dark)',
        'border-light': 'var(--border-light)',
        // Background
        'bg-base': 'var(--bg-base)',
        'bg-card': 'var(--bg-card)',
        'bg-hover': 'var(--bg-hover)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(99, 102, 241, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
};

export default config;