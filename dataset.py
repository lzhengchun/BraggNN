from torch.utils.data import Dataset
import numpy as np
import h5py, torch, random

class BraggNNDataset(Dataset):
    def __init__(self, psz=11, rnd_shift=0):
        self.psz = psz 
        self.rnd_shift = rnd_shift
        with h5py.File('./dataset/frames-exp4train.hdf5', 'r') as h5fd: 
            self.frames  = h5fd['frames'][:]

        with h5py.File('./dataset/peaks-exp4train-psz%d.hdf5' % psz, "r") as h5fd: 
            train_N = int(0.8 * h5fd['peak_fidx'].shape[0])

            mask = h5fd['npeaks'][:train_N] == 1 # use only single-peak patches
            mask = mask & ((h5fd['deviations'][:train_N] >= 0) & (h5fd['deviations'][:train_N] < 1))

            self.peak_fidx= h5fd['peak_fidx'][:train_N][mask]
            self.peak_row = h5fd['peak_row'][:train_N][mask]
            self.peak_col = h5fd['peak_col'][:train_N][mask]
        self.len = self.peak_fidx.shape[0]

    def __getitem__(self, idx):
        _frame = self.frames[self.peak_fidx[idx]]
        if self.rnd_shift > 0:
            row_shift = np.random.randint(-self.rnd_shift, self.rnd_shift+1)
            col_shift = np.random.randint(-self.rnd_shift, self.rnd_shift+1)
        else:
            row_shift, col_shift = 0, 0
        prow_rnd  = int(self.peak_row[idx]) + row_shift
        pcol_rnd  = int(self.peak_col[idx]) + col_shift

        row_base = max(0, prow_rnd-self.psz//2)
        col_base = max(0, pcol_rnd-self.psz//2 )

        crop_img = _frame[row_base:(prow_rnd + self.psz//2 + self.psz%2), \
                            col_base:(pcol_rnd + self.psz//2  + self.psz%2)]
        # if((crop_img > 0).sum() == 1): continue # ignore single non-zero peak
        c_pad_l = (self.psz - crop_img.shape[1]) // 2
        c_pad_r = self.psz - c_pad_l - crop_img.shape[1]

        r_pad_t = (self.psz - crop_img.shape[0]) // 2
        r_pad_b = self.psz - r_pad_t - crop_img.shape[0]

        crop_img = np.pad(crop_img, ((r_pad_t, r_pad_b), (c_pad_l, c_pad_r)), mode='constant')

        if crop_img.max() != crop_img.min():
            _min, _max = crop_img.min().astype(np.float32), crop_img.max().astype(np.float32)
            feature = (crop_img - _min) / (_max - _min)
        else:
            feature = crop_img

        px = (self.peak_col[idx] - col_base + c_pad_l) / self.psz
        py = (self.peak_row[idx] - row_base + r_pad_t) / self.psz

        return feature[np.newaxis], np.array([px, py]).astype(np.float32)

    def __len__(self):
        return self.len


def load_val_dataset(psz, mbsz=None, rnd_shift=0, dev='cuda'):
    h5fd_fr = h5py.File('./dataset/frames-exp4train.hdf5', 'r')

    with h5py.File('./dataset/peaks-exp4train-psz%d.hdf5' % psz, "r") as h5fd_pk: 
        train_N = int(0.8 * h5fd_pk['peak_fidx'].shape[0])

        mask    = h5fd_pk['npeaks'][train_N:] == 1
        mask    = mask & ((h5fd_pk['deviations'][train_N:] >= 0) & (h5fd_pk['deviations'][train_N:] < 1))

        peak_fidx= h5fd_pk['peak_fidx'][train_N:][mask]
        peak_row = h5fd_pk['peak_row'][train_N:][mask]
        peak_col = h5fd_pk['peak_col'][train_N:][mask]

    if mbsz is None:
        mbsz = peak_fidx.shape[0]
    else:
        mbsz = min(mbsz, peak_fidx.shape[0])

    features, labels = [], []

    for i in range(mbsz):
        _frame = h5fd_fr['frames'][peak_fidx[i]]
        if rnd_shift > 0:
            row_shift = np.random.randint(-rnd_shift, rnd_shift+1)
            col_shift = np.random.randint(-rnd_shift, rnd_shift+1)
        else:
            row_shift, col_shift = 0, 0
        prow_rnd  = int(peak_row[i]) + row_shift
        pcol_rnd  = int(peak_col[i]) + col_shift

        row_base = max(0, prow_rnd-psz//2)
        col_base = max(0, pcol_rnd-psz//2 )

        crop_img = _frame[row_base:(prow_rnd + psz//2 + psz%2), \
                          col_base:(pcol_rnd + psz//2 + psz%2)]

        c_pad_l = (psz - crop_img.shape[1]) // 2
        c_pad_r = psz - c_pad_l - crop_img.shape[1]

        r_pad_t = (psz - crop_img.shape[0]) // 2
        r_pad_b = psz - r_pad_t - crop_img.shape[0]

        crop_img = np.pad(crop_img, ((r_pad_t, r_pad_b), (c_pad_l, c_pad_r)), mode='constant')
        features.append((crop_img - crop_img.min()) / (crop_img.max() - crop_img.min()))

        px = (peak_col[i] - col_base + c_pad_l) / psz
        py = (peak_row[i] - row_base + r_pad_t) / psz

        labels.append((px, py))
    h5fd_fr.close()
    return torch.from_numpy(np.expand_dims(features, 1)).to(dev), np.array(labels)