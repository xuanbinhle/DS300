from torch.utils.data import Dataset
import pandas as pd
import os

class RecDataset(Dataset):
    """Interaction dataset for recommender systems.

    - Chứa DataFrame `df` với ít nhất 2 cột:
        * user id  (config['USER_ID_FIELD'])
        * item id  (config['ITEM_ID_FIELD'])
        * (tùy chọn) cột split label (train/val/test)
    - Có thể load trực tiếp từ file CSV (nếu df=None) hoặc dùng df đã chuẩn hoá sẵn (df!=None).
    - Cung cấp:
        * split() -> [train_ds, val_ds, test_ds]
        * get_user_num(), get_item_num()
        * __len__(), __getitem__()
    """
    
    def __init__(self, config: dict, df: pd.DataFrame):
        super().__init__()
        self.config = config
        
        # fields
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        
        # Load dataframe
        if df is not None:
            self.df = df
        else:
            self.load_inter_graph(config['interation_filename'])
            
        self.item_num = self.df[self.iid_field].max() + 1
        self.user_num = self.df[self.uid_field].max() + 1

    def load_inter_graph(self, filename: str):
        inter_file = os.path.join(self.config['dataset_dir'], filename)
        self.df = pd.read_csv(inter_file)

    def copy(self, new_df):
        nxt = RecDataset(self.config, new_df)
        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt
        
    def split(self):
        train_parts, val_parts, test_parts = [], [], []
        for customer_id, group in self.df.groupby(self.uid_field):
            assert len(group) >= 3, f'Length of group is not suitable of {customer_id}: {len(group)}'
            train_parts.append(group.iloc[:-2])
            val_parts.append(group.iloc[[-2]])
            test_parts.append(group.iloc[[-1]])
        
        train_df = pd.concat(train_parts).reset_index(drop=True)
        val_df = pd.concat(val_parts).reset_index(drop=True) if val_parts else pd.DataFrame(columns=self.df.columns)
        test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=self.df.columns)
        
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets (Drop user in val/test set if user not exist in train set)
            train_users = set(train_df[self.uid_field].values)
            val_df = val_df[val_df[self.uid_field].isin(train_users)].reset_index(drop=True)
            test_df = test_df[test_df[self.uid_field].isin(train_users)].reset_index(drop=True)
        return self.copy(train_df), self.copy(val_df), self.copy(test_df)

    def shuffle(self) -> None:
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df.iloc[idx]
    
    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)