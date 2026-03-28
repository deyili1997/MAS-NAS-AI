import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pandas.arrays import ArrowStringArray

class PreTrainEHRDataset(Dataset):
    def __init__(self, ehr_pretrain_data, tokenizer, token_type, mask_rate, max_adm_num, skip_transform=False):
        self.tokenizer = tokenizer
        self.token_type = token_type
        assert mask_rate > 0, "Mask rate must be greater than 0"
        self.mask_rate = float(mask_rate)
        self.token_type_id_dict = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[MASK]": 2,
            "diag": 3,
            "med": 4,
            "lab": 5,
            "pro": 6,
        }
        
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0]
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(["[CLS]"], voc_type="all")[0]
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(["[MASK]"], voc_type="all")[0]
        # keep CLS admission index non-zero to satisfy existing collate sanity checks
        self.cls_adm_index = max_adm_num + 1

        self.type_id_to_name = {
            self.token_type_id_dict["diag"]: "diag",
            self.token_type_id_dict["med"]: "med",
            self.token_type_id_dict["lab"]: "lab",
            self.token_type_id_dict["pro"]: "pro",
        }

        def _safe_to_list(value):
            if isinstance(value, (list, tuple, set, np.ndarray, ArrowStringArray)):
                return [str(v) for v in value if v is not None and str(v) != "nan"]
            return []

        def _transform_pretrain_data(data):
            hadm_records = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient = []
                for _, row in item_df.iterrows():
                    admission = []
                    if "diag" in token_type:
                        admission.append(_safe_to_list(row.get('ICD9_CODE', [])))
                    if "med" in token_type:
                        admission.append(_safe_to_list(row.get('NDC', [])))
                    if "lab" in token_type:
                        admission.append(_safe_to_list(row.get('LAB_TEST', [])))
                    if "pro" in token_type:
                        admission.append(_safe_to_list(row.get('PRO_CODE', [])))
                    patient.append(admission)
                hadm_records[subject_id] = patient
            return hadm_records

        if not skip_transform:
            self.records = _transform_pretrain_data(ehr_pretrain_data)
            self.subject_ids = list(self.records.keys())

    def __len__(self):
        return len(self.records)

    def get_ids(self):
        return self.subject_ids

    def _random_token_id_of_type(self, token_type_id):
        voc_name = self.type_id_to_name.get(token_type_id)
        if voc_name is None:
            return None
        token = self.tokenizer.random_token(voc_name)
        return self.tokenizer.convert_tokens_to_ids([token], voc_type="all")[0]

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        admissions = self.records[subject_id]

        input_ids = [self.cls_token_id]
        token_types = [self.token_type_id_dict["[CLS]"]]
        adm_index = [self.cls_adm_index]
        mlm_labels = [-100]

        for adm_i, adm in enumerate(admissions, start=1): # 0 is for [PAD]
            for type_i, codes in enumerate(adm):
                code_type = self.token_type[type_i]
                type_id = self.token_type_id_dict[code_type]
                for code in codes:
                    try:
                        code_id = self.tokenizer.convert_tokens_to_ids([code], voc_type="all")[0]
                    except KeyError:
                        continue
                    input_ids.append(code_id)
                    token_types.append(type_id)
                    adm_index.append(adm_i)
                    mlm_labels.append(-100)

        candidate_positions = list(range(1, len(input_ids)))
        if candidate_positions:
            n_to_mask = max(1, int(round(len(candidate_positions) * self.mask_rate)))
            n_to_mask = min(n_to_mask, len(candidate_positions))
            mask_positions = np.random.choice(candidate_positions, size=n_to_mask, replace=False)

            for pos in mask_positions:
                original_id = input_ids[pos]
                mlm_labels[pos] = original_id
                p = np.random.rand()
                if p < 0.8:
                    if self.mask_token_id is not None:
                        input_ids[pos] = self.mask_token_id
                    else:
                        rand_id = self._random_token_id_of_type(token_types[pos])
                        if rand_id is not None:
                            input_ids[pos] = rand_id
                elif p < 0.9:
                    rand_id = self._random_token_id_of_type(token_types[pos])
                    if rand_id is not None:
                        input_ids[pos] = rand_id
                # p >= 0.9 keeps original token

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        token_types = torch.tensor([token_types], dtype=torch.long)
        adm_index = torch.tensor([adm_index], dtype=torch.long)
        mlm_labels = torch.tensor([mlm_labels], dtype=torch.long)

        return input_ids, token_types, adm_index, mlm_labels


class FineTuneEHRDataset(PreTrainEHRDataset):
    def __init__(self, ehr_finetune_data, tokenizer, token_type, max_adm_num, task):
        super().__init__(ehr_finetune_data, tokenizer, token_type, mask_rate=0.15, 
                         max_adm_num=max_adm_num, skip_transform=True)
        self.task = task
        
        def _transform_finetune_data(data):
            hadm_records = {}
            labels = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient = []
                for _, row in item_df.iterrows():
                    admission = []
                    hadm_id = row['HADM_ID']
                    if "diag" in token_type:
                        admission.append(list(row['ICD9_CODE']))
                    if "med" in token_type:
                        admission.append(list(row['NDC']))
                    if "lab" in token_type:
                        admission.append(list(row['LAB_TEST'])) 
                    if "pro" in token_type:
                        admission.append(list(row['PRO_CODE']))
                    patient.append(admission)
                    # admission of structure[[diag], [med], [pro], [lab]]
                    # note that one patient records are transformed into multiple samples
                    # one sample represents all previous admissions until the current admission
                    hadm_records[hadm_id] = list(patient)
                    labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"], row["READMISSION"]]
            return hadm_records, labels

        self.records, self.labels = _transform_finetune_data(ehr_finetune_data)
    
    def __getitem__(self, idx): # return tokenized input_ids, token_types, adm_index, age_sex, labels
        hadm_id = list(self.records.keys())[idx]
        admissions = self.records[hadm_id]
        input_ids = [self.cls_token_id]
        token_types = [self.token_type_id_dict['[CLS]']]  # token type for CLS token
        adm_index = [self.cls_adm_index]
        
        for adm_i, adm in enumerate(admissions, start=1): # 0 is for [PAD]
            for type_i, codes in enumerate(adm):
                code_type = self.token_type[type_i]
                type_id = self.token_type_id_dict[code_type]
                for code in codes:
                    try:
                        code_id = self.tokenizer.convert_tokens_to_ids([code], voc_type="all")[0]
                    except KeyError:
                        continue
                    input_ids.append(code_id)
                    token_types.append(type_id)
                    adm_index.append(adm_i)

        # build labels based on the task
        if self.task == "death":
            # predict if the patient will die in the hospital
            labels = torch.tensor([self.labels[hadm_id][0]]).float()
        elif self.task == "stay":
            # predict if the patient will stay in the hospital for more than 7 days
            labels = (torch.tensor([self.labels[hadm_id][1]]) > 7).float()
        elif self.task == "readmission":
            # predict if the patient will be readmitted within 1 month
            labels = torch.tensor([self.labels[hadm_id][2]]).float()
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # convert input_tokens to ids
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        # convert token_types to tensor
        token_types = torch.tensor([token_types], dtype=torch.long)
        # convert adm_index to tensor
        adm_index = torch.tensor([adm_index], dtype=torch.long)

        return input_ids, token_types, adm_index, labels
    
    
def batcher(tokenizer, mode = "pretrain"):
    def batcher_dev(batch):
        raw_input_ids, raw_token_types, raw_adm_index, raw_labels = \
            [feat[0] for feat in batch], [feat[1] for feat in batch], [feat[2] for feat in batch], [feat[3] for feat in batch]

        seq_len = torch.tensor([x.size(1) for x in raw_input_ids])
        max_n_tokens = seq_len.max().item()
        
        pad_token_id = tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0]
        token_type_pad_id = 0
        adm_index_pad_id = 0

        input_ids = torch.cat([F.pad(raw_input_id, (0, max_n_tokens - raw_input_id.size(1)), "constant", pad_token_id) for raw_input_id in raw_input_ids], dim=0)
        token_types = torch.cat([F.pad(raw_token_type, (0, max_n_tokens - raw_token_type.size(1)), "constant", token_type_pad_id) for raw_token_type in raw_token_types], dim=0)
        adm_index = torch.cat([F.pad(raw_adm_idx, (0, max_n_tokens - raw_adm_idx.size(1)), "constant", adm_index_pad_id) for raw_adm_idx in raw_adm_index], dim=0)
        if mode == "pretrain":
            labels = torch.cat([F.pad(raw_labels, (0, max_n_tokens - raw_labels.size(1)), "constant", -100) for raw_labels in raw_labels], dim=0)
        else:
            labels = torch.stack(raw_labels, dim=0).float()
            labels = labels.squeeze(-1)
        attn_mask = (input_ids != pad_token_id).long() # get attention mask
        
        if mode == "pretrain":
            assert input_ids.shape == token_types.shape == adm_index.shape == labels.shape == attn_mask.shape
        else:
            assert input_ids.shape == token_types.shape == adm_index.shape == attn_mask.shape
        # sanity check
        for row in range(input_ids.size(0)):
            count_ids = (input_ids[row] != pad_token_id).sum().item()
            count_types = (token_types[row] != token_type_pad_id).sum().item()
            count_adm = (adm_index[row] != adm_index_pad_id).sum().item()
            expected = seq_len[row].item()

            assert count_ids == count_types == count_adm == expected, \
                (f"Row {row}: counts mismatch. "
                f"ids={count_ids}, types={count_types}, adm={count_adm}, expected seq_len={expected}")

        return input_ids, token_types, adm_index, attn_mask, labels
    return batcher_dev