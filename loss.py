import torch

class DSCLoss(object):
    def __call__(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))
def dscloss(input, target):
    smooth = 1.

    iflat = torch.round(input.view(-1))
    tflat = torch.round(target.view(-1))
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))

def DSC(input, target):
    smooth = 1.
    iflat = input.flatten() > 0.5
    tflat = target.flatten() > 0.5
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))