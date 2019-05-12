#coding=utf-8
import warnings

class DefaultConfigs(object):
    env='default'
    model_name = "multimodal"
    
    train_data = "./data/train/" # where is your train images data
    test_data = "./data/test/"   # your test data
    train_vis="./data/npy/train_visit"  # where is your train visits data
    test_vis="./data/npy/test_visit"
    load_model_path = None
    
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    debug_file='./tmp/debug'
    submit = "./submit/"
    
    num_classes = 9
    img_weight = 100
    img_height = 100
  
    channels = 3
    vis_channels=7
    vis_weight=24
    vis_height=26

    lr = 0.001
    lr_decay = 0.5
    weight_decay =0e-5
    batch_size = 64
    epochs = 30
    
def parse(self, kwargs):
    """
    update config by kwargs
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfigs.parse = parse
config = DefaultConfigs()
