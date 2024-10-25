import torch
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import pandas
device = torch.device("cuda" if torch.cuda.is_available() else "")


def plot_curves(training, validation, output_name, type):
    plt.plot(training, label=f'training {type}')
    plt.plot(validation, label=f'validation {type}')

    plt.text(len(training), training[-1], f'{training[-1]:.3}')
    plt.text(len(validation), validation[-1], f'{validation[-1]:.3}')

    plt.title(f'{type} curves {output_name}')
    plt.legend()
    plt.savefig(f'plots/experiment training/{output_name}')
    plt.clf()


def coco_texts():
    data = dset.CocoCaptions(root=f'datasets_torchvision/coco_2017/train2017',
                             annFile=f'datasets_torchvision/coco_2017/annotations/captions_train2017.json', )
    texts = []
    for img, caption in data:
        texts += caption[:5]
    data = {'texts': texts}
    df = pandas.DataFrame(data)
    df.to_csv(f'datasets_torchvision/coco_2017/texts.csv', index=False)


if __name__ == '__main__':
    coco_texts()

