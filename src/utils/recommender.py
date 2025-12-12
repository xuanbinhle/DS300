# coding: utf-8
# @email  : enoche.chow@gmail.com

import os
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    r"""Base class for all models."""

    def pre_epoch_processing(self):
        """Hook chạy trước mỗi epoch (nếu model cần làm gì đó)."""
        pass

    def post_epoch_processing(self):
        """Hook chạy sau mỗi epoch (nếu model cần làm gì đó)."""
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.
        Args:
            interaction: batch data (tùy bạn định nghĩa trong pipeline).

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.
        Args:
            interaction: batch data (chứa thông tin user & item cần predict).

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""Full-sort prediction function.
        Given users, calculate the scores between users and all candidate items.
        Args:
            interaction: batch data (thường chỉ chứa user_id / user_index).

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
                          shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def __str__(self):
        """Model prints with number of trainable parameters."""
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """Abstract general recommender.

    - Cung cấp thông tin dataset: số user/item, tên field cho user/item.
    - Cung cấp tiện ích load feature multimodal (vision/text) dạng npy.
    - Các model cụ thể (VD: VBPR) chỉ cần kế thừa và implement:
        * __init__
        * calculate_loss
        * predict
        * full_sort_predict
    """

    def __init__(self, config, dataloader):
        super().__init__()
        # ====== Lưu lại config / dataloader / dataset để subclass dùng ======
        self.config = config
        self.dataloader = dataloader
        self.dataset = dataloader.dataset

        # ====== 1. Thông tin ID field ======
        self.USER_ID = config['USER_ID_FIELD']           # ví dụ: 'customer_index'
        self.ITEM_ID = config['ITEM_ID_FIELD']           # ví dụ: 'product_index'
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID

        # ====== 2. Số lượng user / item ======
        # dataset phải implement get_user_num(), get_item_num()
        self.n_users = self.dataset.get_user_num()
        self.n_items = self.dataset.get_item_num()

        # ====== 3. Thông tin training cơ bản ======
        self.batch_size = config['train_batch_size']
        self.device = torch.device(config['device'])

        # ====== 4. Side features (multimodal) ======
        #   v_feat: vision feature [n_items, d_v]
        #   t_feat: text feature   [n_items, d_t]
        self.v_feat: torch.Tensor | None = None
        self.t_feat: torch.Tensor | None = None
        self.v_dim: int | None = None
        self.t_dim: int | None = None
        self._load_side_features()

    # ---------------------------------------------------------------------
    #   HELPER: LOAD FEATURES
    # ---------------------------------------------------------------------
    def _load_side_features(self):
        """Load vision/text features từ .npy nếu là model multimodal và không end2end."""

        dataset_dir = self.config['dataset_dir']
        v_feat_file = self.config['vision_feature_file']
        t_feat_file = self.config['text_feature_file']
        
        # Load vision features
        if v_feat_file is not None:
            v_feat_path = os.path.join(dataset_dir, v_feat_file)
            if os.path.isfile(v_feat_path):
                v_arr = np.load(v_feat_path, allow_pickle=True)
                v_tensor = torch.from_numpy(v_arr).float().to(self.device)
                self.v_feat = v_tensor
                self.v_dim = v_tensor.size(-1)

        # Load text features
        if t_feat_file is not None:
            t_feat_path = os.path.join(dataset_dir, t_feat_file)
            if os.path.isfile(t_feat_path):
                t_arr = np.load(t_feat_path, allow_pickle=True)
                t_tensor = torch.from_numpy(t_arr).float().to(self.device)
                self.t_feat = t_tensor
                self.t_dim = t_tensor.size(-1)

        # Nếu config multimodal == True mà không load => báo lỗi
        if self.config['is_multimodal_model'] and self.v_feat is None and self.t_feat is None:
            raise ValueError(
                f"[GeneralRecommender] Multimodal model nhưng cả vision_feature_file "
                f"và text_feature_file đều không load được. Kiểm tra đường dẫn hoặc config."
            )

    # ---------------------------------------------------------------------
    #   PROPERTIES: CHECK FEATURE EXISTENCE
    # ---------------------------------------------------------------------
    @property
    def has_visual(self) -> bool:
        return self.v_feat is not None

    @property
    def has_text(self) -> bool:
        return self.t_feat is not None

    # ---------------------------------------------------------------------
    #   HELPER: GET FEATURES THEO ITEM INDEX
    # ---------------------------------------------------------------------
    def get_visual_features(self, item_indices: torch.LongTensor) -> torch.Tensor:
        """Trả về vision feature cho các item_index (shape: [B, d_v])."""
        if self.v_feat is None:
            raise RuntimeError("Visual features (v_feat) chưa được load.")
        return self.v_feat[item_indices]

    def get_text_features(self, item_indices: torch.LongTensor) -> torch.Tensor:
        """Trả về text feature cho các item_index (shape: [B, d_t])."""
        if self.t_feat is None:
            raise RuntimeError("Text features (t_feat) chưa được load.")
        return self.t_feat[item_indices]