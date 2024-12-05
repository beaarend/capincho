
def plot_curves(training, validation, output_name, type):
    import matplotlib.pyplot as plt
    plt.plot(training, label=f'training {type}')
    plt.plot(validation, label=f'validation {type}')

    plt.text(len(training), training[-1], f'{training[-1]:.3}')
    plt.text(len(validation), validation[-1], f'{validation[-1]:.3}')

    plt.title(f'{type} curves {output_name}')
    plt.legend()
    plt.savefig(f'plots/experiment training/{output_name}')
    plt.clf()


def coco_texts():
    import pandas
    import torchvision.datasets as dset
    data = dset.CocoCaptions(root=f'datasets_torchvision/coco_2017/train2017',
                             annFile=f'datasets_torchvision/coco_2017/annotations/captions_train2017.json', )
    texts = []
    for img, caption in data:
        texts += caption[:5]
    data = {'texts': texts}
    df = pandas.DataFrame(data)
    df.to_csv(f'datasets_torchvision/coco_2017/texts.csv', index=False)


def model_size(model):
    import torch
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    return size_model / 8e6


def learnable_parameters(model):
    learnable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            learnable += param.numel()

    print(f'total params: {total / 1e6}M,  learnable params: {learnable / 1e6}M')


if __name__ == '__main__':
    coco_texts()

