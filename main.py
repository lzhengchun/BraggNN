from model import model_init, BraggNN
import torch, argparse, os, time, sys, shutil
from util import str2bool, str2tuple, s2ituple
from data import bkgdGen, gen_train_batch_bg, get1batch4test
import numpy as np

parser = argparse.ArgumentParser(description='HEDM peak finding model.')
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-lr',     type=float,default=5e-4, help='learning rate')
parser.add_argument('-mbsize', type=int, default=512, help='mini batch size')
parser.add_argument('-maxep',  type=int, default=100000, help='max training epoches')
parser.add_argument('-fcsz',  type=s2ituple, default='16_8_4_2', help='size of dense layers')
parser.add_argument('-psz', type=int, default=11, help='working patch size')
parser.add_argument('-aug', type=int, default=1, help='augmentation size')
parser.add_argument('-print',  type=str2bool, default=False, help='1:print to terminal; 0: redirect to file')

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
    mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(mb_size=args.mbsize, psz=args.psz, \
                                                             dev=torch_devs, rnd_shift=args.aug), \
                           max_prefetch=args.mbsize*4)

    X_mb, y_mb = mb_data_iter.next()
    model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)
    _ = model.apply(model_init) # init model weights and bias
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(torch_devs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0) # no decay needed when augment is on

    for epoch in range(args.maxep+1):
        time_it_st = time.time()
        X_mb, y_mb = mb_data_iter.next()
        time_data = 1000 * (time.time() - time_it_st)

        optimizer.zero_grad()
        pred = model.forward(X_mb)
        loss = criterion(pred, y_mb)
        loss.backward()
        optimizer.step() 
        time_e2e = 1000 * (time.time() - time_it_st)
        itr_prints = '[Info] @ %.1f Epoch: %05d, loss: %.4f, elapse: %.2fs/itr' % (\
                    time.time(), epoch, args.psz * loss.cpu().detach().numpy(), (time.time() - time_it_st), )

        if epoch % 2000 == 0:
            if epoch == 0: 
                X_mb_val, y_mb_val = get1batch4test(psz=args.psz, mb_size=None, idx=None, rnd_shift=0, dev=torch_devs)

            with torch.no_grad():
                pred_val = model.forward(X_mb_val).cpu().numpy()            

            pred_train = pred.cpu().detach().numpy()  
            true_train = y_mb.cpu().numpy()  
            l2norm_train = np.sqrt((true_train[:,0] - pred_train[:,0])**2   + (true_train[:,1] - pred_train[:,1])**2) * args.psz
            l2norm_val   = np.sqrt((y_mb_val[:,0] - pred_val[:,0])**2 + (y_mb_val[:,1] - pred_val[:,1])**2) * args.psz

            print('[Train] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels). time_data: %.2fms, time_e2e: %.2fms' % (\
                 (epoch, l2norm_train.shape[0], l2norm_train.mean()) + tuple(np.percentile(l2norm_train, (50, 75, 95, 99.5))) + (time_data, time_e2e) ) )

            print('[Valid] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels) \n' % (\
                 (epoch, l2norm_val.shape[0], l2norm_val.mean()) + tuple(np.percentile(l2norm_val, (50, 75, 95, 99.5))) ) )

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
            else:
                torch.save(model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
        sys.stdout.flush()
        
if __name__ == "__main__":
    main(args)
