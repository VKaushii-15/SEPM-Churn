"""
Simple HTTP server for the frontend
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def log_message(self, format, *args):
        logger.info("%s - %s" % (self.client_address[0], format%args))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    PORT = 3000
    server = HTTPServer(('0.0.0.0', PORT), MyHTTPRequestHandler)
    
    logger.info("="*60)
    logger.info("🌐 FRONTEND SERVER STARTED")
    logger.info("="*60)
    logger.info(f"📂 Serving from: {os.getcwd()}")
    logger.info(f"🌍 Access at: http://localhost:{PORT}")
    logger.info(f"📄 Main page: http://localhost:{PORT}/index.html")
    logger.info("="*60)
    logger.info("Press Ctrl+C to stop the server")
    logger.info("="*60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n✓ Server stopped")
