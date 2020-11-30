import numpy as np 
import h5py, threading, random
import queue as Queue
import h5py, torch
from skimage.feature import peak_local_max

class bkgdGen(threading.Thread):
    def __init__(self, data_generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = data_generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            # block if necessary until a free slot is available
            self.queue.put(item, block=True, timeout=None)
        self.queue.put(None)

    def next(self):
        # block if necessary until an item is available
        next_item = self.queue.get(block=True, timeout=None)
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def gen_train_batch_bg(mb_size, psz, rnd_shift=0, dev='cuda'):
    with h5py.File('./dataset/frames-exp4train.hdf5', 'r') as h5fd: 
        frames  = h5fd['frames'][:]

    with h5py.File('./dataset/peaks-exp4train-psz%d.hdf5' % psz, "r") as h5fd: 
        train_N = int(0.8 * h5fd['peak_fidx'].shape[0])

        mask    = h5fd['npeaks'][:train_N] == 1 # use only single-peak patches
        mask    = mask & ((h5fd['deviations'][:train_N] >= 0) & (h5fd['deviations'][:train_N] < 1))

        peak_fidx= h5fd['peak_fidx'][:train_N][mask]
        peak_row = h5fd['peak_row'][:train_N][mask]
        peak_col = h5fd['peak_col'][:train_N][mask]

    features = np.zeros((mb_size, 1, psz, psz), dtype=np.float32)
    labels   = np.zeros((mb_size, 2), dtype=np.float32)
    while True:
        peak_idx = random.sample(range(0, peak_fidx.shape[0]), mb_size)
        for i, _idx in enumerate(peak_idx):
            _frame = frames[peak_fidx[_idx]]
            if rnd_shift > 0:
                row_shift = np.random.randint(-rnd_shift, rnd_shift+1)
                col_shift = np.random.randint(-rnd_shift, rnd_shift+1)
            else:
                row_shift, col_shift = 0, 0
            prow_rnd  = int(peak_row[_idx]) + row_shift
            pcol_rnd  = int(peak_col[_idx]) + col_shift

            row_base = max(0, prow_rnd-psz//2)
            col_base = max(0, pcol_rnd-psz//2 )

            crop_img = _frame[row_base:(prow_rnd + psz//2 + psz%2), \
                              col_base:(pcol_rnd + psz//2  + psz%2)]
            if((crop_img > 0).sum() == 1): continue # ignore single non-zero peak
            c_pad_l = (psz - crop_img.shape[1]) // 2
            c_pad_r = psz - c_pad_l - crop_img.shape[1]

            r_pad_t = (psz - crop_img.shape[0]) // 2
            r_pad_b = psz - r_pad_t - crop_img.shape[0]

            crop_img = np.pad(crop_img, ((r_pad_t, r_pad_b), (c_pad_l, c_pad_r)), mode='constant')

            if crop_img.max() != crop_img.min():
                features[i, 0] = (crop_img - crop_img.min()) / (crop_img.max() - crop_img.min())
            else:
                print('[WARN] encounter uniform values in the patch for training, discard')
                continue

            px = (peak_col[_idx] - col_base + c_pad_l) / psz
            py = (peak_row[_idx] - row_base + r_pad_t) / psz

            labels[i] = (px, py)

        yield torch.from_numpy(features).to(dev), torch.from_numpy(labels).to(dev)

def get1batch4test(psz, idx=None, mb_size=None, rnd_shift=0, dev='cuda'):
    h5fd_fr = h5py.File('./dataset/frames-exp4train.hdf5', 'r')

    with h5py.File('./dataset/peaks-exp4train-psz%d.hdf5' % psz, "r") as h5fd_pk: 
        train_N = int(0.8 * h5fd_pk['peak_fidx'].shape[0])

        mask    = h5fd_pk['npeaks'][train_N:] == 1
        mask    = mask & ((h5fd_pk['deviations'][train_N:] >= 0) & (h5fd_pk['deviations'][train_N:] < 1))

        peak_fidx= h5fd_pk['peak_fidx'][train_N:][mask]
        peak_row = h5fd_pk['peak_row'][train_N:][mask]
        peak_col = h5fd_pk['peak_col'][train_N:][mask]
        
    if mb_size is None: 
        if idx is None:
            mb_size = peak_fidx.shape[0]
        else:
            mb_size = len(idx)
    else:
        mb_size = min(mb_size, peak_fidx.shape[0])
    features = []
    labels   = []

    if idx is None: idx = random.sample(range(0, peak_fidx.shape[0]), mb_size)
    for i, _idx in enumerate(idx):
        _frame = h5fd_fr['frames'][peak_fidx[_idx]]
        if rnd_shift > 0:
            row_shift = np.random.randint(-rnd_shift, rnd_shift+1)
            col_shift = np.random.randint(-rnd_shift, rnd_shift+1)
        else:
            row_shift, col_shift = 0, 0
        prow_rnd  = int(peak_row[_idx]) + row_shift
        pcol_rnd  = int(peak_col[_idx]) + col_shift

        row_base = max(0, prow_rnd-psz//2)
        col_base = max(0, pcol_rnd-psz//2 )

        crop_img = _frame[row_base:(prow_rnd + psz//2 + psz%2), \
                          col_base:(pcol_rnd + psz//2  + psz%2)]

        c_pad_l = (psz - crop_img.shape[1]) // 2
        c_pad_r = psz - c_pad_l - crop_img.shape[1]

        r_pad_t = (psz - crop_img.shape[0]) // 2
        r_pad_b = psz - r_pad_t - crop_img.shape[0]

        crop_img = np.pad(crop_img, ((r_pad_t, r_pad_b), (c_pad_l, c_pad_r)), mode='constant')

        if crop_img.max() != crop_img.min():
            features.append((crop_img - crop_img.min()) / (crop_img.max() - crop_img.min()))
        else:
            print('[WARN] encounter uniform values in the patch for testing, discard')
            continue

        px = (peak_col[_idx] - col_base + c_pad_l) / psz
        py = (peak_row[_idx] - row_base + r_pad_t) / psz

        labels.append((px, py))
    h5fd_fr.close()
    return torch.from_numpy(np.expand_dims(features, 1)).to(dev), np.array(labels)
