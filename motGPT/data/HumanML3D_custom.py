from .HumanML3D import HumanML3DDataModule
from .humanml.dataset_m2t_token_custom import Motion2TextDatasetTokenCustom

class HumanML3DDataModuleCustom(HumanML3DDataModule):
    def __init__(self, cfg, **kwargs):
        # Initialize parent class, this will set basic parameters (paths, mean/std, etc.)
        super().__init__(cfg, **kwargs)
        
        # Override Dataset selection logic
        # We define a new STAGE name: "token_custom"
        if cfg.TRAIN.STAGE == "token_custom":
            print("[DataModule] Using Custom Token Dataset for M2T Training")
            self.Dataset = Motion2TextDatasetTokenCustom
            self.DatasetEval = Motion2TextDatasetTokenCustom
            
            # Ensure some parameters are passed correctly
            self.hparams.code_path = cfg.DATASET.CODE_PATH
