from transformers import AutoTokenizer
from dataset.utils import dsets
from dataset.utils.datasetbase import DatasetBase




class S2SDataset(DatasetBase):
    NAME = "oedataset"  # open-ended dataset

    def __init__(self, accelerator, args):
        super().__init__()

        self.args = args
        self.accelerator = accelerator

        accelerator.wait_for_everyone()

        if args.evaluate:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True, padding_side="left"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True, padding_side="right"
            )
        # Set pad_token if not already set (e.g., for Qwen models)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.bos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.bos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        dset_class: dsets.ClassificationDataset = getattr(dsets, args.dataset)
        print(self.args.multi_answer)
        self.dset = dset_class(
            self.tokenizer,
            add_space=args.add_space,
            max_seq_len=args.max_seq_len,
            multianswer=self.args.multi_answer,
        )
        
        # For generative tasks, we don't have a fixed number of labels
        # Set to 0 as a placeholder (not used for generation)
        self.num_labels = 0

    def get_loaders(self):
        """
        Returns the train and test data loaders.
        """

        if self.accelerator.is_local_main_process:
            print("=====================================")
            print(f"Loading {self.args.dataset} dataset.")
            print("=====================================")

        self.train_dataloader = self.dset.loader(
            batch_size=self.args.batch_size,  # training batch size
            split="train",  # training split name in dset
            subset_size=self.args.subset_size,
        )

        total_data_count = 0
        for batch in self.train_dataloader:
            total_data_count += batch[0]["input_ids"].size(0)
        self.num_samples = total_data_count

        if self.accelerator.is_local_main_process:
            print(
                f"Loaded {self.args.dataset} training dataset. Total samples: {self.num_samples}"
            )

        if self.args.testing_set == "val":
            self.test_dataloader = self.dset.loader(
                batch_size=self.args.batch_size,  # training batch size
                split="validation",  # training split name in dset
                subset_size=self.args.subset_size,  # Apply subset to test set too
            )
        else:
            self.test_dataloader = self.dset.loader(
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=self.args.subset_size,  # Apply subset to test set too
            )
            self.val_dataloader = self.dset.loader(
                batch_size=self.args.batch_size,  # training batch size
                split="validation",  # training split name in dset
                subset_size=self.args.subset_size,  # Apply subset to validation set too
            )

        if self.accelerator.is_local_main_process:
            print("=====================================")
            print(f"Loaded {self.args.dataset} testing dataset.")
            print("=====================================")
