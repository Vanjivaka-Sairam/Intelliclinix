/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5328/api/:path*',
      },
      {
        source: '/auth/:path*',
        destination: 'http://localhost:5328/auth/:path*',
      },
      {
        source: '/cvat/:path*',
        destination: 'http://localhost:5328/cvat/:path*',
      },
      {
        source: '/static/:path*',
        destination: 'http://localhost:5328/static/:path*',
      },
    ]
  },
}

module.exports = nextConfig

