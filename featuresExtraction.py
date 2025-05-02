import argparse
import torch
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import pickle
import foundation_models
from util import dataset_path
from trainLoRA import run_lora_training

from dataLoader import DatasetHandler

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from lora import LoRAWrapper
from util import apply_lora

from peft import get_peft_model, LoraConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='split to use')
    parser.add_argument('--model', choices=['openclip', 'clip', 'coca'], default='openclip', help='model to use')
    parser.add_argument('--backbone', default='ViT-B/16', type=str)

    parser.add_argument('--lora', action='store_true', help='use lora', default=True)

    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--dataset',type=str, default='rsicd', choices=['coco', 'rsicd'])

    parser.add_argument('--save_path', type=str, default='embeddings/rsicd_lora_train.pkl')

    args = parser.parse_args()

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP,
                  'capivara': foundation_models.Capivara}
    
    model = model_dict[args.model](device)

    model.load_model()

    if args.lora:
        model = LoRAWrapper(model, encoder='both')
        model.backbone.to(device)

    run_lora_training(model, args)
    exit()

    model.backbone.eval()

    

    if(args.dataset == 'coco'):
        new_dataset_path = dataset_path + 'COCO/'
    elif(args.dataset == 'rsicd'):
        new_dataset_path = dataset_path + 'RSICD/'

    #coco = COCO(f'datasets_torchvision/coco_2017/annotations/captions_{args.split}2017.json')
    #loaded_dataset = COCO(os.path.join(dataset_path, 'COCO', 'annotations', f'captions_{args.split}2017.json'))
    loaded_dataset = DatasetHandler(os.path.join(new_dataset_path, 'annotations', f'{args.split}_split.json'))

    ids = loaded_dataset.get_image_ids()
    imgs = loaded_dataset.load_images(ids)

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'texts_embeddings': [], }
    for i, image in enumerate(tqdm(imgs)):
        data['image_name'].append(image['filename'])

        data['image_id'].append(ids[i])
        # img_embeds = model.visual_embedding('datasets_torchvision/coco_2017/{}2017/{}'.format(args.split,
        #                                                                                       image['file_name']))
        #img_embeds = model.visual_embedding(os.path.join(dataset_path, 'COCO', f'{args.split}2017', image['file_name']))
        
        img_embeds = model.visual_embedding(os.path.join(new_dataset_path, f'rsicd_{args.split}', image['filename']))

        data['image_embeddings'].append(img_embeds.detach().cpu())

        ann = loaded_dataset.load_annotations(loaded_dataset.get_annotation_ids(ids[i]))

        texts = [e['raw'] for e in ann]
        text_embeds = model.language_embedding(texts[:5])
        data['texts_embeddings'].append(text_embeds.detach().cpu())

    with open(args.save_path, 'wb') as f:
        pickle.dump(data, f)


