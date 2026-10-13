"""python -m studio --host 127.0.0.1 --port 8765"""
import argparse
import ipaddress
import os


def main():
    parser=argparse.ArgumentParser(description='SDV Studio local research dashboard')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    try:loopback=ipaddress.ip_address(args.host).is_loopback
    except ValueError:loopback=args.host=='localhost'
    if not loopback and len(os.environ.get('SDV_AUTH_TOKEN',''))<24:parser.error('외부 공개는 24자 이상의 SDV_AUTH_TOKEN을 명시적으로 설정해야 합니다.')
    if not 1<=args.port<=65535:parser.error('포트는 1~65535여야 합니다.')
    import uvicorn
    from .api import create_app
    from .http_protocol import SDVH11Protocol
    uvicorn.run(create_app(),host=args.host,port=args.port,log_level='info',timeout_graceful_shutdown=28,
                http=SDVH11Protocol)

if __name__=='__main__':main()
