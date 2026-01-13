from .HumanML3D import HumanML3DDataModule
from .humanml.dataset_m2t_token_custom import Motion2TextDatasetTokenCustom

class HumanML3DDataModuleCustom(HumanML3DDataModule):
    def __init__(self, cfg, **kwargs):
        # 初始化父类，这会设置基本参数 (paths, mean/std 等)
        super().__init__(cfg, **kwargs)
        
        # 覆盖 Dataset 类选择逻辑
        # 我们定义一个新的 STAGE 名称: "token_custom"
        if cfg.TRAIN.STAGE == "token_custom":
            print("[DataModule] Using Custom Token Dataset for M2T Training")
            self.Dataset = Motion2TextDatasetTokenCustom
            self.DatasetEval = Motion2TextDatasetTokenCustom
            
            # 确保一些参数正确传递
            self.hparams.code_path = cfg.DATASET.CODE_PATH
