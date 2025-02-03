import argparse
import torch
from foundation_models import OpenCLIP, CLIP


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['openclip', 'clip'], help='model to finetune')
    args = parser.parse_args()

    model_dict = {'openclip': OpenCLIP, 'clip': CLIP}
    model = model_dict[args.model](device)
    model.load_model()



