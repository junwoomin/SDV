"""Existing random RPC and traffic-manager port selection."""
import random

def get_random_ports():
    PORT = random.randint(1024, 30000)

    # TM_PORT는 PORT와 다르게 설정되도록 보장
    while True:
        TM_PORT = random.randint(1024, 30000)
        if TM_PORT != PORT:
            break

    return PORT, TM_PORT
