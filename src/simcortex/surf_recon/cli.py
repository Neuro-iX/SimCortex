# src/simcortex/surf_recon/cli.py

import sys
from simcortex.surf_recon.train    import train_app
from simcortex.surf_recon.predict  import predict_app
from simcortex.surf_recon.evaluate import eval_app

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    # remove command so Hydra only sees --config-name, etc.
    sys.argv.pop(1)
    if cmd == "train":
        train_app()
    elif cmd == "predict":
        predict_app()
    elif cmd in ("eval", "evaluate"):
        eval_app()
    else:
        print("Usage: python -m simcortex.surf_recon.cli [train|predict|eval] --config-name <name>")

if __name__ == "__main__":
    main()
