import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader,Dataset
import torch
from multiprocessing import freeze_support
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from torch.optim import lr_scheduler
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.core import LightningDataModule,LightningModule
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


class BorderDataset(Dataset):

    def __init__(self,data):
        self.data=data
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        source_encoding=self.tokenizer(self.data.text.iloc[index]+'<eos>', return_tensors='pt',padding='max_length',max_length=560)
        target=torch.tensor(self.data.label.iloc[index])
        

        return dict(

        input_ids = source_encoding["input_ids"].squeeze(),
        lm_labels = target,
        attention_mask = source_encoding['attention_mask'].squeeze(),

        )

class BorderDataModule(LightningDataModule):

    def __init__(
    self,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ): 
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.train_df = train_df
        self.val_df = val_df

    def setup(self,stage=None):
        self.train_dataset = BorderDataset(self.train_df)
        self.validation_dataset = BorderDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = 4, shuffle=True, num_workers = 8,persistent_workers=True)

    def val_dataloader(self): 
        return DataLoader(self.validation_dataset, batch_size = 4, num_workers = 8,persistent_workers=True)

class GemmaFineTuner(LightningModule):
    def __init__(self,learning_rate):
        super(GemmaFineTuner,self).__init__()
        model = AutoModelForSequenceClassification.from_pretrained( "google/gemma-2-2b-it",torch_dtype=torch.bfloat16,num_labels=2,attn_implementation='flash_attention_2')
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj']
        )
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self.training_step_outputs=[]
        self.validation_step_outputs=[]
        self.test_step_outputs=[]
        self.validation_step_answer=0
        self.test_step_answer=0
        self.learning_rate=learning_rate

    def forward(self,input_ids,attention_mask,lm_labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels
            )
    
    def training_step(self,batch,batch_idx):
        
        outputs = self.forward(
        
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            lm_labels=batch['lm_labels']
        )
        loss=outputs.loss
        
        self.log('train_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        self.training_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        avg_train_loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def validation_step(self,batch,batch_idx):
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            lm_labels=batch['lm_labels']
        )
        loss=outputs.loss
        self.validation_step_outputs.append(loss)
        self.validation_step_answer=self.validation_step_answer+torch.eq(outputs.logits.argmax(dim=1),batch['lm_labels']).float().sum()
        self.log('val_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def on_validation_epoch_end(self):
        num=len(val)
        avg_val_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        acc=self.validation_step_answer/num
        print("********\n")
        print(acc)
        print("********\n")
        self.log('val_acc',acc,on_epoch=True,prog_bar=True,logger=True)
        self.validation_step_outputs.clear()
        self.validation_step_answer=0
        tensorboard_logs = {"avg_val_loss": avg_val_loss,'avg_val_acc':acc}
        return {"avg_val_loss": avg_val_loss, "avg_val_acc":acc, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def test_step(self,batch,batch_idx):
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            lm_labels=batch['lm_labels']
        )
        loss=outputs.loss
        self.test_step_outputs.append(loss)
        self.test_step_answer=self.test_step_answer+torch.eq(outputs.logits.argmax(dim=1),batch['lm_labels']).float().sum()
        self.log('test_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        return loss
    
    def on_test_epoch_end(self):
        num=len(val)
        avg_test_loss = torch.stack([x for x in self.test_step_outputs]).mean()
        acc=self.test_step_answer/num
        print("********\n")
        print(acc)
        print("********\n")
        self.log('test_acc',acc,on_epoch=True,prog_bar=True,logger=True)
        self.test_step_outputs.clear()
        self.test_step_answer=0
        tensorboard_logs ={"avg_test_loss":avg_test_loss,"avg_test_acc":acc}
        return {"avg_test_loss": avg_test_loss,"log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def predict_step(self,batch):
        input_ids=batch['source_ids']
        attention_mask=batch['source_mask']
        outputs=self.model.generate(input_ids.cuda(),attention_mask=attention_mask.cuda(),max_length=2)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8,weight_decay=0.01)
        return {
            'optimizer': optimizer,
            'lr_scheduler':{"scheduler":lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-8),"interval":"epoch"}
        }

if __name__ == '__main__':

    freeze_support()
    

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3,mode='min')
    checkpoint_callback = ModelCheckpoint(
        dirpath='outputcheckpoint', #save at this folder
        filename="google/gemma-2-2b-it-{epoch:02d}-{val_loss:.2f}", #name for the checkpoint, before i was using "best-checkpoint"
        save_top_k=2, #save all epochs, before it was only the best (1)
        verbose=True, #output something when a model is saved
        monitor="val_loss", #monitor the validation loss
        mode="min" #save the model with minimum validation loss
        )

    logger = TensorBoardLogger(save_dir='tf_dir')
    
    train=pd.read_csv(r'data_trainf.csv',sep='|')
    val=pd.read_csv(r'data_valf.csv',sep='|')
    test_df=pd.read_csv(r'data_testf.csv',sep='|')
    data_module=BorderDataModule(train_df=train,val_df=val)
    data_module.setup()
    
    train_dls=DataLoader(BorderDataset(train),batch_size = 4, num_workers = 8,persistent_workers=True)
    val_dls=DataLoader(BorderDataset(val),batch_size = 4, num_workers = 8,persistent_workers=True)
    test_dls=DataLoader(BorderDataset(test_df),batch_size = 4, num_workers = 8,persistent_workers=True)
    
    model=GemmaFineTuner(learning_rate=1e-4)


    while 1:
        
        control=int(input('Enter for function: 1-train 2-test  :\n'))
        
        if control==1:
            torch.set_float32_matmul_precision("medium")
            trainer = Trainer(
                max_epochs = 20,
                logger = logger,
                callbacks=[checkpoint_callback,early_stopping_callback]
                )
            trainer.fit(model,datamodule=data_module)
            
        if control==2:
            
            model=GemmaFineTuner.load_from_checkpoint('',learning_rate=1e-4)
            
            num=len(test_df)
            
            temp=[]
            
            predictions=trainer.predict(model,dataloaders=test_dls)
            a=torch.cat(predictions,dim=0)
            pre=torch.split(a,1,dim=1)[1]
            for data in test_dls:
                temp.append(data['target_ids'])
            b=torch.cat(temp,dim=0)
            label=torch.split(b,1,dim=1)[0]
            res=torch.eq(pre,label)
            print("\n")
            print("********\n")
            print('acc:',res.float().sum()/num)
            print("********\n")

