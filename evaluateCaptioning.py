import json
import os.path
import random
from embeddingsDataset import COCODataset
from decoder import OPT
import torch
import argparse
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='experiments/opt-350m-coco.json',
                        help='experiment path')
    parser.add_argument('--embeddings', type=str, default='embeddings/coco_openCLIP_val.pkl')
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

    embeddings = COCODataset(path=args.embeddings, n_captions=1)

    print('\n Evaluating captioning \n')
    coco = COCO('datasets_torchvision/coco_2017/annotations/captions_val2017.json')
    if args.qualitative:
        for i in [random.randint(0, len(embeddings)) for i in range(10)]:
            input_emb = embeddings[i]['image_embeddings'][0].to(device, dtype=precision)
            generated = decoder.caption(input_emb, max_tokens=20, )
            ann_id = coco.getAnnIds(embeddings[i]['image_id'])
            ann = coco.loadAnns(ann_id)
            text = 'GT: {}\n generated: {}'.format(ann[0]['caption'], generated[0])

            image = Image.open('datasets_torchvision/coco_2017/val2017/{}'.format(embeddings[i]['image_name']))
            w, h = image.size[:2]
            text_board = Image.new('RGB', (w, 50), (255, 255, 255))
            font = ImageFont.truetype("fonts/ARIAL.TTF", 16)
            ImageDraw.Draw(text_board).text((1, 1), text, (0, 0, 0), font=font)

            dst = Image.new('RGB', (w, h + 60), (255, 255, 255))
            dst.paste(image, (0, 0))
            dst.paste(text_board, (0, h))
            dst.save('plots/caption/captions_{}'.format(embeddings[i]['image_name']))

    else:
        name = os.path.basename(args.experiment)
        if not os.path.exists(f'results/{name}'):
            print(f'generating captions, results/{name}')
            data = embeddings[:]
            captions = [decoder.caption(torch.tensor(e, dtype=precision, device=device).squeeze(0))[0] for e in tqdm(data['image_embeddings'], total=len(data['image_embeddings']))]
            results = []
            print(captions)
            for i in range(len(captions)):
                results.append({'image_id': embeddings[i]['image_id'], 'caption': captions[i]})

            with open(f'results/{name}', 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f'loading results from {name}')

        res = coco.loadRes(f'results/{name}')
        coco_eval = COCOEvalCap(coco, res)
        coco_eval.evaluate()

        experiment_result = {}
        for metric, score in coco_eval.eval.items():
            experiment_result[metric] = score

        with open('results/experiment_result.json', 'r') as f:
            json_dict = json.load(f)
            json_dict[f'{name}'] = experiment_result

        with open(f'results/experiment_result.json', 'w') as f:
            json.dump(json_dict, f, indent=2)



