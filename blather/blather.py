import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np

class Blather():
    
    def __init__(self):
        self.tokenizer = tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        self.configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.configuration)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def read(self, data, epochs=1, batch_size = 4):
        dataset = self.GPT2Dataset(data, tokenizer=self.tokenizer, max_length=103)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        
        train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )

        validation_dataloader = DataLoader(
                    val_dataset, 
                    sampler = SequentialSampler(val_dataset), 
                    batch_size = batch_size 
                )
        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

        

        
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        learning_rate = 5e-4
        warmup_steps = 1e2
        epsilon = 1e-8

       
        optimizer = AdamW(self.model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
        
        total_steps = len(train_dataloader) * epochs


        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = warmup_steps, 
                                                    num_training_steps = total_steps)
        
        training_stats = []


        for epoch_i in range(0, epochs):

            print('Training...')


            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                self.model.zero_grad()        

                outputs = self.model(  b_input_ids,
                                  labels=b_labels, 
                                  attention_mask = b_masks,
                                  token_type_ids=None
                                )

                loss = outputs[0]  

                batch_loss = loss.item()
                total_train_loss += batch_loss

                loss.backward()

                optimizer.step()

                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)       

            print("Running Validation...")

      

            self.model.eval()

            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:

                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                with torch.no_grad():        

                    outputs  = self.model(b_input_ids, 
        #                            token_type_ids=None, 
                                     attention_mask = b_masks,
                                    labels=b_labels)

                    loss = outputs[0]  

                batch_loss = loss.item()
                total_eval_loss += batch_loss        

            avg_val_loss = total_eval_loss / len(validation_dataloader)


            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss
                }
            )
            return training_stats
        
    def write(self, context):
        self.model.eval()


        generated = torch.tensor(self.tokenizer.encode(context)).unsqueeze(0)
        generated = generated.to(self.device)


        sample_outputs = self.model.generate(
                                        generated, 
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = 300,
                                        top_p=0.95, 
                                        num_return_sequences=3
                                        )

        return self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

    def save(self, file_location):
        if file_location[-2:]=="pt":
            torch.save(self.model, file_location)
            return None
        else :
            print("File must be a torch file in the format model.pt")
            return None

    def load(self, file_location):
        if file_location[-2:]=="pt":
            self.model = torch.load(file_location)
            self.model.eval()
            return None
        else :
            print("File must be a torch file in the format model.pt")
            return None
    
    class GPT2Dataset(Dataset):

    
        def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=103):

            self.input_ids = []
            self.attn_masks = []

            for txt in txt_list:

                encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx] 
