module.exports = {
  '/ocr_segments': {
    target: 'http://localhost:8080/',
    changeOrigin: true,
    secure: false,
  },
  '/health': {
    target: 'http://localhost:8080/',
    changeOrigin: true,
    secure: false,
  },
  '/train/start': {
    target: 'http://localhost:8080/',
    changeOrigin: true,
    secure: false,
  },
};
