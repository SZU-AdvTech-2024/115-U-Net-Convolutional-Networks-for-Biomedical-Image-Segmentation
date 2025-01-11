from argparse import ArgumentParser


def add_predict_args(parser: ArgumentParser) -> None:
    parser.add_argument("--image_path", "-i", default=None, type=str)
    parser.add_argument("--image_dir", default=None, type=str)
    parser.add_argument("--weights_path", "-w", default=None, type=str)
    
