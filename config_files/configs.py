class Config(object):
    def __init__(self):
        # Cross-domain Training
        self.num_epoch = 15
        self.batch_size = 128

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.5
        self.beta2 = 0.99
        self.lr = 1e-3
        self.lr_disc = 1e-3
        self.weight_decay = 3e-4

        # scheduler
        self.step_size = 10
        self.gamma = 0.1

        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']

        self.sequence_len = 3000

        self.adast_params = ADAST_params_configs()
        self.base_model = base_model_Configs()


class ADAST_params_configs(object):
    def __init__(self):
        self.disc_wt = 1
        self.src_clf_wt = 1

        self.trg_clf_wt = 1e-2
        self.similarity_wt = 1e-3

        self.self_training_iterations = 2


class base_model_Configs(object):
    def __init__(self):
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.1
        self.num_classes = 5

        # features characteristics
        self.final_out_channels = 128
        self.features_len = 29

        self.sequence_len = 3000
        # discriminator
        self.disc_hid_dim = 100
