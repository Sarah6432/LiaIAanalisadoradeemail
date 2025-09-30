import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/classify-email",
        destination: "http://127.0.0.1:8000/classify-email",
      },
      {
        source: "/api/classify-batch",
        destination: "http://127.0.0.1:8000/classify-batch/",
      },
    ];
  },
};

export default nextConfig;
