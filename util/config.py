import os
import yaml
import torch
import datetime
import pytz

class Config:
    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False) and name in self.__dict__:
            raise AttributeError(f"[Config] attribute '{name}' is frozen and read-only. Call `unfreeze()` first if you need to modify it.")
        super().__setattr__(name, value)
                
    def __init__(self, args):
        
        super().__setattr__('_frozen', False)
        for key, value in vars(args).items():
            setattr(self, key, value)
            
        self.classes = {"FPV1": 0, "Lightbridge1": 1, "Ocusync_mini1": 2, "Ocusync21": 3, "Ocusync31": 4, "Ocusync41": 5, "Skylink11": 6, "Skylink21": 7}
        self.mat_key = "summed_submatrices"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_dir = 'experiments'
        os.makedirs(self.base_dir, exist_ok=True)
        
        current_time = datetime.datetime.now(pytz.utc)
        current_time = current_time.astimezone(pytz.timezone("Asia/Shanghai"))
        
        log_name = 'UAV_' + str(current_time.strftime("%Y%m%d_%H%M%S"))
        
        self.cfg_dir = f'{self.base_dir}/{log_name}'
        self.model_dir = f'{self.cfg_dir}/models'
        self.log_dir = f'{self.cfg_dir}/log'
        self.result_dir = f'{self.cfg_dir}/result'
        
        # self.make_dir()
        # self.save_config()
        
    def save_config(self):
        file_path = os.path.join(self.cfg_dir, 'config.yaml')
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f)
            
    def make_dir(self):
        
        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
    def freeze(self):
        """Freeze the config  all existing attributes become read-only."""
        super().__setattr__('_frozen', True)

    def unfreeze(self):
        """(Optional) Un-freeze the config for debugging."""
        super().__setattr__('_frozen', False)