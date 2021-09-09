import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from proxy_anchor.dataset import CUBirds
from proxy_anchor.utils import evaluate_cos


if __name__ == "__main__":
    eval_tr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds_eval = CUBirds("/home/i", "eval", eval_tr)
    dl_eval = DataLoader(dataset=ds_eval, batch_size=100, shuffle=False, num_workers=10,
                         pin_memory=False, drop_last=False)

    EMB_SIZE = 128
    # model = resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, EMB_SIZE)

    body = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    head = nn.Sequential(nn.Linear(384, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Linear(1024, EMB_SIZE))
    # head = nn.Linear(384, EMB_SIZE)
    model = nn.Sequential(body, head)
    model = nn.DataParallel(model)

    model.cuda().eval()
    model.load_state_dict(torch.load("output/model_t.pt"))

    evaluate_cos(model, dl_eval)
    evaluate_cos(body, dl_eval)

    # model.fc = nn.Identity()
    # evaluate_cos(model, dl_eval)
