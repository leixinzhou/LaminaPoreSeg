from torch.utils.data import DataLoader
from dataset import *
from loss import dscloss, DSC
import loss
import unet
import yaml
import argparse
import time
from tensorboardX import SummaryWriter
from torch import optim
from torch import nn
import shutil
import os
import torch.nn.functional as F
from unet import FCNSigmoid
import PIL, torch
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage import morphology
from matplotlib import patches, lines
from matplotlib.patches import Polygon

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)

# train


def train(model, criterion, optimizer, input_img_gt):
    model.train()
    D = model(input_img_gt['img'])
    loss = criterion(D, input_img_gt['gt'])
    with torch.no_grad():
        dsc = dscloss(D, input_img_gt['gt'])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy(), dsc.detach().cpu().numpy()
# val


def val(model, criterion, input_img_gt):
    model.eval()
    D = model(input_img_gt['img'])
    loss = criterion(D, input_img_gt['gt'])
    with torch.no_grad():
        dsc = dscloss(D, input_img_gt['gt'])

    return loss.detach().cpu().numpy(), dsc.detach().cpu().numpy()
# learn


def learn(model, hps):
    since = time.time()
    writer = SummaryWriter(hps['learning']['checkpoint_path'])
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = hps['gpu'])
        model.cuda()
        model = nn.DataParallel(model)

    else:
        raise NotImplementedError("CPU version is not implemented!")

    aug_dict = {"Jitter": 
                    ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,hue=0.4), 
                "Affine": 
                    RandomAffine(
                        degrees=360, translate=(0.2, 0.2),scale=(0.8, 1.3),shear=(10, 20)),
                "Hflip": 
                    RandomHorizontalFlip(),
                "Vflip": 
                    RandomVerticalFlip(),
                "ResizeCrop":
                    RandomResizedCrop(size=(128,128),scale=(0.3,1.0)),
                "ToTensor":
                    ToTensor()}
    transform_tr = transforms.Compose([aug_dict[i] for i in hps['learning']['augmentation']])
    transform_val = transforms.Compose([ToTensor()])
    tr_dataset = LaminaDataset(prefix="../../pre", transform=transform_tr)
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=hps['learning']['batch_size'], 
                            shuffle=True)
    val_dataset = LaminaDataset(prefix="../../after", transform=transform_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hps['learning']['batch_size'], 
                            shuffle=False)

    optimizer = getattr(optim, hps['learning']['optimizer'])(
        [{'params': model.parameters(), 'lr': hps['learning']['lr']}
         ])
    # scheduler = getattr(optim.lr_scheduler,
    #                     hps.learning.scheduler)(optimizer, factor=hps.learning.scheduler_params.factor,
    #                                             patience=hps.learning.scheduler_params.patience,
    #                                             threshold=hps.learning.scheduler_params.threshold,
    #                                             threshold_mode=hps.learning.scheduler_params.threshold_mode)
    try:
        loss_func = getattr(nn, hps['learning']['loss'])()
    except AttributeError:
        try: 
            loss_func = getattr(loss, hps['learning']['loss'])()
        except AttributeError:
            raise AttributeError(hps['learning']['loss']+" is not implemented!")

    if os.path.isfile(hps['learning']['resume_path']):
        print('loading checkpoint: {}'.format(hps['learning']['resume_path']))
        checkpoint = torch.load(hps['learning']['resume_path'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps['learning']['resume_path']))

    epoch_start = 0
    best_loss = hps['learning']['best_loss']

    for epoch in range(epoch_start, hps['learning']['total_iterations']):
        # tr_loss_g = 0
        tr_loss_d = 0
        tr_loss_dsc = 0
        tr_mb = 0
        print("Epoch: " + str(epoch))
        for step, batch in enumerate(tr_loader):
            batch = {key: value.cuda() for (key, value) in batch.items() }
            m_batch_loss, m_batch_dsc = train(model, loss_func, optimizer, batch)
            tr_loss_dsc += m_batch_dsc
            tr_loss_d += m_batch_loss
            tr_mb += 1
            print("         mini batch train loss: "+ "%.5e" % m_batch_loss + 
                            " mini batch train dsc: "+ "%.5e" % m_batch_dsc)
        epoch_tr_loss_d = tr_loss_d / tr_mb
        epoch_tr_loss_dsc = tr_loss_dsc / tr_mb
        writer.add_scalar('data/train_dsc', epoch_tr_loss_dsc, epoch)
        writer.add_scalar('data/train_loss', epoch_tr_loss_d, epoch)
        
        # print("     tr_loss_g: " + "%.5e" % epoch_tr_loss_g)
        print("     tr_loss: " + "%.5e" % epoch_tr_loss_d+
                    "   tr_dice: " + "%.5e" % epoch_tr_loss_dsc)
        # scheduler.step(epoch_tr_loss)

        val_loss_dsc = 0
        val_loss_d = 0
        val_mb = 0
        for step, batch in enumerate(val_loader):
            batch = {key: value.cuda() for (key, value) in batch.items() }
            m_batch_loss, m_batch_dsc = val(model, loss_func, batch)
            val_loss_dsc += m_batch_dsc
            val_loss_d += m_batch_loss
            val_mb += 1
            print("         mini batch val loss: "+ "%.5e" % m_batch_loss+
                            " mini batch val dice: "+ "%.5e" % m_batch_dsc)
        epoch_val_loss_dsc = val_loss_dsc / val_mb
        epoch_val_loss_d = val_loss_d / val_mb
        writer.add_scalar('data/val_dsc', epoch_val_loss_dsc, epoch)
        writer.add_scalar('data/val_loss', epoch_val_loss_d, epoch)
        # print("     val_loss_g: " + "%.5e" % epoch_val_loss_g)
        print("     val_loss: " + "%.5e" % epoch_val_loss_d+
                    "   val_dsc: " + "%.5e" % epoch_val_loss_dsc)

        if epoch_val_loss_dsc > best_loss:
            best_loss = epoch_val_loss_dsc
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                },
                path=hps['learning']['checkpoint_path'],
            )

    writer.export_scalars_to_json(os.path.join(
        hps['learning']['checkpoint_path'], "all_scalars.json"))
    writer.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

def infer(model, hps):
    since = time.time()
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = hps['gpu'])
        model.cuda()
        model = nn.DataParallel(model)
    else:
        raise NotImplementedError("CPU version is not implemented!")
    transform_test = transforms.Compose([ToTensor()])
    test_dataset = LaminaDataset(prefix="../../after", transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, 
                            shuffle=False)
    if os.path.isfile(hps['test']['resume_path']):
        print('loading checkpoint: {}'.format(hps['test']['resume_path']))
        checkpoint = torch.load(hps['test']['resume_path'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps['test']['resume_path']))
    model.eval()

    if not os.path.isdir(hps['test']['pred_dir']):
        os.mkdir(hps['test']['pred_dir'])
    
    dsc_list = []
    for step, batch in enumerate(test_loader):
        batch = {key: value.cuda() for (key, value) in batch.items() }
        img = batch['img'].squeeze().detach().cpu().numpy() * 255
        gt = batch['gt'].squeeze().detach().cpu().numpy() * 255
        pred = model(batch['img']).squeeze().detach().cpu().numpy() > 0.5
        img = img.astype(np.uint8)
        gt = gt.astype(np.uint8)
        # remove small objects
        pred = morphology.remove_small_objects(pred, min_size=8)
        pred = (pred * 255).astype(np.uint8)
        dsc_list.append(DSC(pred, gt))
        # pred_img = PIL.Image.fromarray(pred, 'L')
        # # plt.imshow(pred, cmap='binary')
        # # plt.show()
        # # break
        # pred_img.save(os.path.join(hps['test']['pred_dir'], "%d.png" % step),
        #                 format='png')
        
        contours_gt = find_contours(gt, 122)
        contours_pred = find_contours(pred, 122)
        _, axes = plt.subplots(1,2, figsize=(10,10))
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(img, cmap='gray')
        for verts in contours_gt:
            p = Polygon(np.fliplr(verts), facecolor='none', edgecolor='b')
            axes[1].add_patch(p)
        for verts in contours_pred:
            p = Polygon(np.fliplr(verts), facecolor='none', edgecolor='r')
            axes[1].add_patch(p)
        plt.savefig(os.path.join(hps['test']['pred_dir'], "%d.png" % step),
                        format='png', dpi=100, bbox_inches='tight')
        plt.close()
    dsc_list = np.array(dsc_list)
    print(dsc_list.mean(), dsc_list.std())
    np.savetxt(os.path.join(hps['test']['pred_dir'], "dsc.txt"), np.array([dsc_list.mean(), dsc_list.std()]))
   

def main():
    # read configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyperparams', default='./para/hparas_unet.json',
                        type=str, metavar='FILE.PATH',
                        help='path to hyperparameters setting file (default: ./para/hparas_unet.json)')

    args = parser.parse_args()
    try:
        with open(args.hyperparams, "r") as config_file:
            hps = yaml.load(config_file)
    except IOError:
        print('Couldn\'t read hyperparameter setting file')
    if hps['learning']['loss']=="DSCLoss":
        net = FCNSigmoid(in_channels=1,
                 up_mode='transpose', depth=hps['network']['depth'],
                  start_filts=hps['network']['start_filts'])
    else:
        print("Network for " + hps['learning']['loss']+" not implemented.")
    if hps['test']['mode']:
        infer(net, hps)
    else:
        try:
            learn(net, hps)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), os.path.join(
                hps['learning']['checkpoint_path'], 'INTERRUPTED.pth'))
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)



if __name__ == '__main__':
    main()
