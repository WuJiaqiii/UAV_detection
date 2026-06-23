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
            
        self.classes = {"Background": 0, "Lightbridge": 1, "Ocusync2": 2, "Ocusync3": 3, "Ocusync4": 4, "Skylink1": 5, "FPV": 6}
        # self.classes = {"Background": 0, "DJI_Phantom3": 1, "DJI_Phantom4Pro": 2, "DJI_MATRICE200": 3, "DJI_MATRICE100": 4, "DJI_Air2S": 5, "DJI_Mini3Pro": 6,
        #                 "DJI_Inspire2": 7, "DJI_MavicPro": 8, "DJI_Mini2": 9, "DJI_Mavic3": 10, "DJI_MATRICE300": 11, "DJI_Phantom4ProRTK": 12,
        #                 "DJI_MATRICE30T": 13, "DJI_AVATA": 14, "DJI_CommunicationModuleDIY": 15, "DJI_MATRICE600Pro": 16, "VBar_Controller": 17,
        #                 "FrSkyX20_Controller": 18, "FutabaT6IZ_Controller": 19, "TaranisPlus_Controller": 20, "RadioLinkAT9S_Controller": 21, 
        #                 "FutabaT14SG_Controller": 22, "YunzhuoT12_Controller": 23, "YunzhuoT10_Controller": 24}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_dir = 'experiments'
        os.makedirs(self.base_dir, exist_ok=True)
        
        current_time = datetime.datetime.now(pytz.utc)
        current_time = current_time.astimezone(pytz.timezone("Asia/Shanghai"))
        
        log_name = 'UAV_' + str(current_time.strftime("%Y%m%d_%H%M%S"))
        
        self.cfg_dir = f'{self.base_dir}/{log_name}'
        self.model_dir = f'{self.cfg_dir}/models'
        self.log_dir = f'{self.cfg_dir}/log'
        self.cache_dir = f'{self.cfg_dir}/cache'
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
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
    def freeze(self):
        """Freeze the config  all existing attributes become read-only."""
        super().__setattr__('_frozen', True)

    def unfreeze(self):
        """(Optional) Un-freeze the config for debugging."""
        super().__setattr__('_frozen', False)