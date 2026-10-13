"""Small durable receipt writer. Research code runs in its configured interpreter."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:raise SystemExit('Missing internal research command')
    # Group signals also reach the research child. Keep the receipt writer alive
    # until the child has released sensors/checkpoints and actually exited.
    signal.signal(signal.SIGINT,lambda *_:None);signal.signal(signal.SIGTERM,lambda *_:None)
    rc=-1
    try:
        child=subprocess.Popen(command)
        rc=child.wait()
    except BaseException as exc:
        print('SDV worker failed:',exc,flush=True)
    finally:
        path=Path(args.receipt);temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps({'return_code':rc,'finished_unix':time.time()}),encoding='utf-8');os.replace(temp,path)
    return rc if rc>=0 else 128-rc

if __name__=='__main__':sys.exit(main())
