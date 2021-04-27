"""MobileNetV1 for Semantic Segmentation"""
import torch
import torch.nn.functional as F

from light.model.base import BaseModel
from light.nn import _FCNHead


class MobileNetSeg(BaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenet', pretrained_base=False, **kwargs):
        super(MobileNetSeg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        self.head = _FCNHead(1024, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(512, nclass, **kwargs)

    def forward(self, x):
        size = x.size()[2:]

        _, _, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


def get_mobilenet_seg(dataset='citys', pretrained=False, root='~/.torch/models',
                      pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from light.data import datasets
    model = MobileNetSeg(datasets[dataset].NUM_CLASS, backbone='mobilenet',
                         pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from ..model import get_model_file
        model.load_state_dict(torch.load(get_model_file('mobilenet_%s_best_model' % (acronyms[dataset]), root=root)))
    return model


if __name__ == '__main__':
    model = get_mobilenet_seg()