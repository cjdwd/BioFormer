import torch

class kiba_conf():
    def __init__(self):
        self.dataset = 'kiba'

        ###train settings
        self.batch_size = 192
        self.num_gpus = 4
        self.device_ids = [1,0,2,3]
        self.test_mode = False
        self.epochs = 60
        self.test_epo = 1
        self.print_step = 64
        self.criterion = torch.nn.BCELoss()
        self.model_type = "fpn_bilstm"

        ###data settings
        self.split_names = ['train','valid','test']
        self.root_dir = './data/my_data/'
        self.contactmap_path = self.root_dir + 'my_data/kiba/kiba_contactmap/'
        self.save_prefix = './model_pkl/' + self.dataset + '_' + self.model_type + '_'

class kd_conf():
    def __init__(self):
        self.dataset = 'kd'

        ###train settings
        self.batch_size = 192
        self.num_gpus = 4
        self.device_ids = [1,0,2,3]
        self.test_mode = False
        self.epochs = 60
        self.test_epo = 1
        self.print_step = 64
        self.criterion = torch.nn.BCELoss()
        self.model_type = "fpn_bilstm"

        ###data settings
        self.split_names = ['train', 'valid', 'test']
        self.root_dir = './data/my_data/'
        self.contactmap_path = self.root_dir + 'my_data/bindingdb/Kd/Kd_contactmap/'
        self.save_prefix = './model_pkl/' + self.dataset + '_' + self.model_type + '_'

class dude_conf():
    def __init__(self):
        self.dataset = 'dude'

        ###train settings
        self.batch_size = 198
        self.num_gpus = 4
        self.device_ids = [1,0,2,3]
        self.test_mode = False
        self.epochs = 60
        self.test_epo = 10
        self.print_step = 64
        self.model_type = "fpn_bilstm"
        self.criterion = torch.nn.BCELoss()

        ###data settings
        self.split_names = ['train', 'test', 'test']
        self.root_dir = './data/my_data/'
        self.contactmap_path = self.root_dir + 'my_data/dude/dude_contactmap/'
        self.save_prefix = './model_pkl/' + self.dataset + '_' + self.model_type + '_'

def drugVQA_conf():
    modelArgs = {}

    modelArgs['batch_size'] = 1
    modelArgs['lstm_hid_dim'] = 64
    modelArgs['d_a'] = 32
    modelArgs['r'] = 10
    modelArgs['n_chars_smi'] = 260
    modelArgs['n_chars_seq'] = 21
    modelArgs['dropout'] = 0.2
    modelArgs['in_channels'] = 8
    modelArgs['cnn_channels'] = 32
    modelArgs['cnn_layers'] = 4
    modelArgs['emb_dim'] = 30
    modelArgs['dense_hid'] = 64
    modelArgs['task_type'] = 0
    modelArgs['n_classes'] = 1
    modelArgs['gene_emb_dim'] = 4
    modelArgs['gene_hidden_dim'] = 2
    modelArgs['gene_kernel_size'] = 512
    modelArgs['gene_stride'] = 32
    modelArgs['gene_out_channels'] = 2
    modelArgs['gene_linear_in_dim'] = 6400*4
    modelArgs['gene_linear_out_dim'] = 128

    return modelArgs