import json
import os.path
import random
from embeddingsLoader import COCODataset
import torchvision.datasets as dset
from decoder import OPT
import torch
import argparse
from PIL import Image, ImageFont, ImageDraw
import evaluate
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='experiments/coco_opt-350m_openclip_text-only.json',
                        help='experiment path')
    parser.add_argument('--embeddings', type=str, default='coco_openCLIP_val')
    parser.add_argument('--qualitative', action='store_true', help='run qualitative evaluation')
    args = parser.parse_args()
    with open(args.experiment, 'r') as f:
        config = json.load(f)

    precision = torch.float16 if config['fp'] == 'fp16' else torch.float32

    decoder = OPT(config['model_name'], device, prefix_length=config['prefix_len'], precision=precision,
                  add_noise=config['text_only'])

    if not config['full_finetune']:
        decoder.lora_model(config['rank'], config['alpha'], config['dropout'])

    checkpoint = torch.load(config['checkpoint_path'])
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()

    embeddings = COCODataset(path='datasets_torchvision/embeddings/{}.pkl'.format(args.embeddings), n_captions=1)
    coco = dset.CocoCaptions(root=f'datasets_torchvision/coco_2017/val2017',
                             annFile=f'datasets_torchvision/coco_2017/annotations/captions_val2017.json', )

    print('\n Evaluating captioning \n')
    if args.qualitative:
        for i in [random.randint(0, len(coco)) for i in range(10)]:
            generated = decoder.caption(embeddings[i][0].to(device, dtype=precision), max_tokens=20, )
            text = 'GT: {}\n generated: {}'.format(coco[i][1][0], generated[0])

            image = coco[i][0]
            w, h = image.size[:2]
            text_board = Image.new('RGB', (w, 50), (255, 255, 255))
            font = ImageFont.truetype("fonts/ARIAL.TTF", 16)
            ImageDraw.Draw(text_board).text((1, 1), text, (0, 0, 0), font=font)

            dst = Image.new('RGB', (w, h + 60), (255, 255, 255))
            dst.paste(image, (0, 0))
            dst.paste(text_board, (0, h))
            dst.save('plots/caption/captions_{}.png'.format(i))
    else:
        ann = COCO('datasets_torchvision/coco_2017/annotations/captions_val2017.json')
        name = os.path.basename(args.experiment)
        if not os.path.exists(f'results/{name}'):
            print(f'results/{name}')
            img_embeds, text_embeds = embeddings[:]
            captions = [decoder.caption(torch.tensor(e, dtype=precision, device=device).squeeze(0))[0] for e in
                        tqdm(img_embeds, total=len(img_embeds))]
            ids = ann.getImgIds()
            results = []
            for i, id in enumerate(ids):
                results.append({'image_id': id, 'caption': captions[i]})

            with open(f'results/{name}', 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f'loading results from {name}')
        res = ann.loadRes(f'results/{name}')
        coco_eval = COCOEvalCap(ann, res)
        coco_eval.evaluate()

        experiment_result = {}
        for metric, score in coco_eval.eval.items():
            experiment_result[metric] = score

        with open('results/experiment_result.json', 'r') as f:
            json_dict = json.load(f)
            json_dict[f'{name}'] = experiment_result

        with open(f'results/experiment_result.json', 'w') as f:
            json.dump(json_dict, f, indent=2)



