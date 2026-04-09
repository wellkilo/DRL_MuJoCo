import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#667eea',
        'primary-dark': '#764ba2',
        success: '#28a745',
        'success-light': '#20c997',
        danger: '#dc3545',
        warning: '#ffc107',
        'bg-light': '#f8f9fa',
        border: '#e9ecef',
        'text-primary': '#495057',
        'text-secondary': '#6c757d',
      },
    },
  },
  plugins: [],
};

export default config;