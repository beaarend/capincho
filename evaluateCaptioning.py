import argparse
import random
import torch
from textLoader import TextLoader
from decoder import model_from_json
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--embeddings', type=str, default='embeddings/textEmbeddings.pkl', required=True,
                        help='path to embeddings file')
    parser.add_argument('-m', '--model', type=str, default='experiments/checkpoint.pt', required=True,
                        help='path to the saved model')
    args = parser.parse_args()
    data = TextLoader(args.embeddings, has_embeddings=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_json(args.model, device)
    model.eval()

    for i in tqdm([random.randint(0, len(data)) for i in range(10)]):
        generated = model.caption(data[i]['embeddings'], max_tokens=20, )
        print(data[i]['captions'], generated)


