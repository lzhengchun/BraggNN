from model import model_init, BraggNN
import torch, argparse, os, time, sys, shutil
from util import str2bool, str2tuple, s2ituple
from torch.utils.data import DataLoader
from dataset import BraggNNDataset, load_val_dataset
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
parser.add_argument('-print',  type=str2bool, default=True, help='1:print to terminal; 0: redirect to file')

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

# redirect print to a file
if args.print == 0:
    sys.stdout = open(os.path.join(itr_out_dir, 'iter-prints.log'), 'w') 

def main(args):
    print("[Info] loading data into CPU memory, it will take a while ... ...")
    mb_data_iter = DataLoader(dataset=BraggNNDataset(psz=11, rnd_shift=args.aug), batch_size=args.mbsz, shuffle=True,\
                              drop_last=True, pin_memory=True)

    X_mb_val, y_mb_val = load_val_dataset(psz=args.psz, mbsz=None, rnd_shift=0, dev=torch_devs)
 
    model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)
    _ = model.apply(model_init) # init model weights and bias
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(torch_devs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 

    for epoch in range(args.maxep+1):
        ep_tick = time.time()
        time_comp = 0
        for X_mb, y_mb in mb_data_iter:
            it_comp_tick = time.time()

            optimizer.zero_grad()
            pred = model.forward(X_mb.to(torch_devs))
            loss = criterion(pred, y_mb.to(torch_devs))
            loss.backward()
            optimizer.step() 

            time_comp += 1000 * (time.time() - it_comp_tick)

        time_e2e = 1000 * (time.time() - ep_tick)

        _prints = '[Info] @ %.1f Epoch: %05d, loss: %.4f, elapse: %.2fms/epoch (computation=%.1fms/epoch, %.2f%%)' % (\
                   time.time(), epoch, args.psz * loss.cpu().detach().numpy(), time_e2e, time_comp, 100*time_comp/time_e2e)
        print(_prints)
        with torch.no_grad():
            pred_val = model.forward(X_mb_val).cpu().numpy()            

        pred_train = pred.cpu().detach().numpy()  
        true_train = y_mb.cpu().numpy()  
        l2norm_train = np.sqrt((true_train[:,0] - pred_train[:,0])**2   + (true_train[:,1] - pred_train[:,1])**2) * args.psz
        l2norm_val   = np.sqrt((y_mb_val[:,0]   - pred_val[:,0])**2     + (y_mb_val[:,1]   - pred_val[:,1])**2)   * args.psz

        print('[Train] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels).' % (\
                (epoch, l2norm_train.shape[0], l2norm_train.mean()) + tuple(np.percentile(l2norm_train, (50, 75, 95, 99.5))) ) )

        print('[Valid] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels) \n' % (\
                (epoch, l2norm_val.shape[0], l2norm_val.mean()) + tuple(np.percentile(l2norm_val, (50, 75, 95, 99.5))) ) )

        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
        else:
            torch.save(model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
        sys.stdout.flush()
        
if __name__ == "__main__":
    main(args)
