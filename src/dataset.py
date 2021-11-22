import torch
from torch.utils.data.dataset import Dataset

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class IEMOCAPDataset(Dataset):
    def __init__(self, audio_data, audio_label, l_only=False, a_only=False, v_only=False):
        super(IEMOCAPDataset, self).__init__()

        if a_only:
            self.audio = torch.Tensor(audio_data)
            self.labels = torch.Tensor(audio_label)

        # Note: this is STILL an numpy array
        self.meta = {"meta_audio": audio_data,
                     "meta_label": audio_label
                     }

        self.l_only, self.a_only, self.v_only = l_only, a_only, v_only
        self.n_modalities = v_only + l_only + a_only

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):

        if self.n_modalities == 1 and self.a_only == 1:
            return 1, self.audio.shape[1], 1
        else:
            return 1, 1, self.audio.shape[1]

    def get_dim(self):
        if self.n_modalities == 1 and self.a_only == 1:
            return 1, self.audio.shape[2], 1
        else:
            return 1, 1, self.audio.shape[2]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        audio_data = (index, self.audio[index])
        audio_data_label = self.labels[index]

        return audio_data, audio_data_label
