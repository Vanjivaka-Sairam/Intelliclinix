module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // CVAT-inspired color scheme
        'cvat-primary': '#1890ff',
        'cvat-primary-hover': '#40a9ff',
        'cvat-primary-active': '#096dd9',
        'cvat-success': '#52c41a',
        'cvat-warning': '#faad14',
        'cvat-error': '#ff4d4f',
        'cvat-info': '#1890ff',
        
        // Background colors
        'cvat-bg-primary': '#f5f5f5',
        'cvat-bg-secondary': '#ffffff',
        'cvat-bg-tertiary': '#fafafa',
        'cvat-bg-header': '#001529',
        'cvat-bg-header-light': '#002140',
        
        // Text colors
        'cvat-text-primary': '#262626',
        'cvat-text-secondary': '#595959',
        'cvat-text-tertiary': '#8c8c8c',
        'cvat-text-white': '#ffffff',
        
        // Border colors
        'cvat-border': '#d9d9d9',
        'cvat-border-light': '#f0f0f0',
        'cvat-border-dark': '#434343',
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'cvat': '0 2px 8px rgba(0, 0, 0, 0.15)',
        'cvat-light': '0 1px 3px rgba(0, 0, 0, 0.12)',
      },
    },
  },
  plugins: [],
}

