from model import model_init, BraggNN
import torch, argparse, os, time, sys, shutil, logging
from util import str2bool, str2tuple, s2ituple
from torch.utils.data import DataLoader
from dataset import BraggNNDataset
import numpy as np

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
parser.add_argument('-mbsz',   type=int, default=512, help='mini batch size')
parser.add_argument('-maxep',  type=int, default=500, help='max training epoches')
parser.add_argument('-fcsz',   type=s2ituple, default='16_8_4_2', help='size of dense layers')
parser.add_argument('-psz',    type=int, default=11, help='working patch size')
parser.add_argument('-aug',    type=int, default=1, help='augmentation size')

args, unparsed = parser.parse_known_args()

if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")

itr_out_dir = args.expName + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

logging.basicConfig(filename=os.path.join(itr_out_dir, 'BraggNN.log'), level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def main(args):
    logging.info("[%.3f] loading data into CPU memory, it will take a while ... ..." % (time.time(), ))
    ds_train = BraggNNDataset(psz=args.psz, rnd_shift=args.aug, use='train')
    dl_train = DataLoader(dataset=ds_train, batch_size=args.mbsz, shuffle=True,\
                          num_workers=8, prefetch_factor=args.mbsz, drop_last=True, pin_memory=True)

    ds_valid = BraggNNDataset(psz=args.psz, rnd_shift=0, use='validation')
    dl_valid = DataLoader(dataset=ds_valid, batch_size=args.mbsz, shuffle=False, \
                          num_workers=8, prefetch_factor=args.mbsz, drop_last=False, pin_memory=True)
 
    logging.info("[%.3f] loaded training set with %d samples, and validation set with %d samples " % (\
                 time.time(), len(ds_train), len(ds_valid)))

    model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)
    _ = model.apply(model_init) # init model weights and bias
    
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        if gpus > 1:
            logging.info("This implementation only makes use of one GPU although %d are visiable" % gpus)
        model = model.to(torch_devs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    time_on_training = 0
    for epoch in range(args.maxep):
        ep_tick = time.time()
        time_comp = 0
        for X_mb, y_mb in dl_train:
            it_comp_tick = time.time()

            optimizer.zero_grad()
            pred = model.forward(X_mb.to(torch_devs))
            loss = criterion(pred, y_mb.to(torch_devs))
            loss.backward()
            optimizer.step() 

            time_comp += 1000 * (time.time() - it_comp_tick)

        time_e2e = 1000 * (time.time() - ep_tick)
        time_on_training += time_e2e

        _prints = '[%.3f] Epoch: %05d, loss: %.4f, elapse: %.2fms/epoch (computation=%.1fms/epoch, %.2f%%)' % (\
                   time.time(), epoch, args.psz * loss.cpu().detach().numpy(), time_e2e, time_comp, 100*time_comp/time_e2e)
        logging.info(_prints)

        pred_val, gt_val = [], []
        for X_mb_val, y_mb_val in dl_valid:
            with torch.no_grad():
                _pred = model.forward(X_mb_val.to(torch_devs))
                pred_val.append(_pred.cpu().numpy())
                gt_val.append(y_mb_val.numpy())
        pred_val = np.concatenate(pred_val, axis=0)
        gt_val   = np.concatenate(gt_val,   axis=0)

        pred_train = pred.cpu().detach().numpy()  
        true_train = y_mb.cpu().numpy()  
        l2norm_train = np.sqrt((true_train[:,0] - pred_train[:,0])**2   + (true_train[:,1] - pred_train[:,1])**2) * args.psz
        l2norm_val   = np.sqrt((gt_val[:,0]     - pred_val[:,0])**2     + (gt_val[:,1]     - pred_val[:,1])**2)   * args.psz

        logging.info('[Train] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels).' % (\
                     (epoch, l2norm_train.shape[0], l2norm_train.mean()) + tuple(np.percentile(l2norm_train, (50, 75, 95, 99.5))) ) )

        logging.info('[Valid] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels) \n' % (\
                     (epoch, l2norm_val.shape[0], l2norm_val.mean()) + tuple(np.percentile(l2norm_val, (50, 75, 95, 99.5))) ) )

        torch.save(model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))

    logging.info("Trained for %3d epoches, each with %d steps (BS=%d) took %.3f seconds" % (\
                 args.maxep, len(dl_train), args.mbsz, time_on_training*1e-3))

if __name__ == "__main__":
    main(args)
