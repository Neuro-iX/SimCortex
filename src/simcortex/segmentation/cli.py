# segmentation/cli.py
import sys
from simcortex.segmentation.train    import train_app
from simcortex.segmentation.predict  import predict_app
from simcortex.segmentation.evaluate import evaluate_app

def usage():
    print("Usage: python -m simcortex.segmentation.cli [train|predict|eval] --config-name <yaml>")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    cmd = sys.argv[1]
    # Shift sys.argv so Hydra sees the right args
    sys.argv.pop(1)
    if cmd == "train":
        train_app()     # Hydra main for training
    elif cmd == "predict":
        predict_app()   # Hydra main for prediction
    elif cmd == "eval":
        evaluate_app()  # Hydra main for evaluation
    else:
        usage()
        sys.exit(1)

