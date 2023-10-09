if __name__ == '__main__':

  import os
  os.environ["CUDA_VISIBLE_DEVICES"]="0"

  from typing import Optional, Union
  import pandas as pd, numpy as np, torch
  from datasets import Dataset # 这个的使用方法 hugging face 上面有教程
  from dataclasses import dataclass
  from transformers import AutoTokenizer
  from transformers import EarlyStoppingCallback
  from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
  from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

  # TRAIN WITH SUBSET OF 60K
  # NUM_TRAIN_SAMPLES = 1_024
  NUM_TRAIN_SAMPLES = None

  # PARAMETER EFFICIENT FINE TUNING
  # PEFT REQUIRES 1XP100 GPU NOT 2XT4
  USE_PEFT = True
  # USE_PEFT = True # 这个的全称是 pretrained efficient finetuning, hugging face 上面有教程

  # NUMBER OF LAYERS TO FREEZE
  # DEBERTA LARGE HAS TOTAL OF 24 LAYERS
  FREEZE_LAYERS = 18
  # FREEZE_LAYERS = 24
  # FREEZE_LAYERS = 20


  # BOOLEAN TO FREEZE EMBEDDINGS
  FREEZE_EMBEDDINGS = True
  # LENGTH OF CONTEXT PLUS QUESTION ANSWER
  # 我需要搞懂这个长度到底指的是什么，尤其是 context 和 question 的长度是怎么分配的。256 不可能 cover 全部。
  # 因为如果模型没能在:
  # 足够长的 input 中训练，那么positional encoding 很差的模型就不好 extrapolate
  # MAX_INPUT = 256
  MAX_INPUT = 1024 # 调整这个的大小的时候，每次都需要重新跑一下dataset

  # HUGGING FACE MODEL
  MODEL = 'microsoft/deberta-v3-large'
  VER=f'{FREEZE_LAYERS}_{MAX_INPUT}_60kdata'

  checkpoint_folder = MODEL.split('/')[-1] + '_checkpoints'
  dataset_folder = MODEL.split('/')[-1] + '_datasets'
  df_valid = pd.read_csv('../input/60k-data-with-context-v2/train_with_context2.csv')
  print('Validation data size:', df_valid.shape )
  df_valid

  df_train = pd.read_csv('..//input/60k-data-with-context-v2/all_12_with_context2.csv')
# df_train = pd.read_csv('../input/99k-context/RACE_with_context_original.csv')

  print('size of dataset', len(df_train))
  # df_train = df_train.drop(columns="source")
  if 'source' in df_train.columns:
      df_train = df_train.drop(columns="source")
  df_train = df_train.fillna('')
  if NUM_TRAIN_SAMPLES:
      df_train = df_train.sample(NUM_TRAIN_SAMPLES) # taken NUM_TRAIN_SAMPLES of samples here
  print('Train data size:', df_train.shape )
  df_train

  option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
  index_to_option = {v: k for k,v in option_to_index.items()}

  # 等于说训练的时候模型是可以看到context的，因此与预测保持一致
  def preprocess(example, tokenizer):
      first_sentence = [ "[CLS] " + example['context'] ] * 5
      second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
      tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first',
                                    max_length=MAX_INPUT, add_special_tokens=False)
      tokenized_example['label'] = option_to_index[example['answer']]

      return tokenized_example

  def race_preprocess(example, tokenizer, max_length=MAX_INPUT):
      context = example["context"].replace("\n", " ")
      first_sentence = ["[CLS] " + context] * 4
      second_sentences = [
          " #### " + example["prompt"] + " [SEP] " + example[option] + " [SEP]"
          for option in "ABCD"
      ]
      tokenized_example = tokenizer(
          first_sentence,
          second_sentences,
          truncation="only_first",
          max_length=max_length,
          add_special_tokens=False,
      )
      tokenized_example["label"] = option_to_index[example["answer"]]

      return tokenized_example

  @dataclass
  class DataCollatorForMultipleChoice:
      tokenizer: PreTrainedTokenizerBase
      padding: Union[bool, str, PaddingStrategy] = True
      max_length: Optional[int] = None
      pad_to_multiple_of: Optional[int] = None
      
      def __call__(self, features):
          label_name = 'label' if 'label' in features[0].keys() else 'labels'
          labels = [feature.pop(label_name) for feature in features]
          batch_size = len(features)
          num_choices = len(features[0]['input_ids'])
          flattened_features = [
              [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
          ]
          flattened_features = sum(flattened_features, [])
          
          # Huggingface tokenizer padding
          batch = self.tokenizer.pad(
              flattened_features,
              padding=self.padding,
              max_length=self.max_length,
              pad_to_multiple_of=self.pad_to_multiple_of, # this is related to mixed precision training
              return_tensors='pt',
          )

          batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
          batch['labels'] = torch.tensor(labels, dtype=torch.int64)
          return batch
      
  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  dataset_valid = Dataset.from_pandas(df_valid)
  dataset = Dataset.from_pandas(df_train)
  if '__index_level_0__' in dataset._info.features: # 加了这行防爆
      print('removing __index_level_0__')
      dataset = dataset.remove_columns(["__index_level_0__"])
  dataset
  from functools import partial
  preprocess = partial(preprocess, tokenizer=tokenizer)
  race_preprocess = partial(race_preprocess, tokenizer=tokenizer)

  tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])

  num_choices = len([x for x in dataset.column_names if x in 'ABCDEFG'])
  dataset_path = f'./{dataset_folder}/tokenized_dataset_{MAX_INPUT}_{num_choices}'
  if os.path.exists(dataset_path):
      print(f'getting tokenized_dataset_{MAX_INPUT} from disk')
      tokenized_dataset = Dataset.load_from_disk(dataset_path)
  else:
      if num_choices == 5:
          tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
      elif num_choices == 4:
          tokenized_dataset = dataset.map(race_preprocess, remove_columns=dataset.column_names)
          
      tokenized_dataset.save_to_disk(dataset_path)

  # changed by cxzheng, fuck! stucked!

  tokenized_dataset # 他跑到 21100 附近的时候会卡住，有点奇怪

  def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

  def compute_metrics(p):
      predictions = p.predictions.tolist()
      labels = p.label_ids.tolist()
      return {"map@3": map_at_3(predictions, labels)}

  model = AutoModelForMultipleChoice.from_pretrained(MODEL)

  # if USE_PEFT:
  #     print('We are using PEFT.')
  #     from peft import LoraConfig, get_peft_model, TaskType
  #     peft_config = LoraConfig(
  #         r=8, lora_alpha=4, task_type=TaskType.SEQ_CLS, lora_dropout=0.1,
  #         bias="none", inference_mode=False,
  #         target_modules=["query_proj", "value_proj"],
  #         modules_to_save=['classifier','pooler'],
  #     )
  #     model = get_peft_model(model, peft_config)
  #     model.print_trainable_parameters()
  if USE_PEFT:
      print('We are using PEFT.')
      from peft import LoraConfig, get_peft_model, TaskType
      peft_config = LoraConfig(
          r=12, lora_alpha=6, task_type=TaskType.SEQ_CLS, lora_dropout=0.1,
          bias="none", inference_mode=False,
          target_modules=["query_proj", "value_proj"],
          modules_to_save=['classifier','pooler'],
      )
      model = get_peft_model(model, peft_config)
      model.print_trainable_parameters()

    
  if FREEZE_EMBEDDINGS:
      print('Freezing embeddings.')
      for param in model.deberta.embeddings.parameters():
          param.requires_grad = False

  if FREEZE_LAYERS>0:
      print(f'Freezing {FREEZE_LAYERS} layers.')
      for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
          for param in layer.parameters():
              param.requires_grad = False
      # Newly added for v3
      for layer in model.deberta.encoder.layer[FREEZE_LAYERS:]:
          for param in layer.parameters():
              param.requires_grad = True

      total_params = sum(p.numel() for p in model.parameters())
      model_parameters = filter(lambda p: p.requires_grad, model.parameters())
      trainable_params = sum([np.prod(p.size()) for p in model_parameters])
      print(trainable_params, total_params, trainable_params/total_params)

  from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
  from transformers import TrainerState, TrainerControl, TrainerCallback
  class SavePeftModelCallback(TrainerCallback):
      def on_save(
          self,
          args: TrainingArguments,
          state: TrainerState,
          control: TrainerControl,
          **kwargs,
      ):
          checkpoint_folder = os.path.join(
              args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
          )

          peft_model_path = os.path.join(checkpoint_folder, "torch_model")
          # peft_model_path = os.path.join(checkpoint_folder)
          kwargs["model"].base_model.save_pretrained(peft_model_path)

          # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
          # if os.path.exists(pytorch_model_path):
          #     os.remove(pytorch_model_path)
          return control

  # batch_size = 8 # Try this if possible for 18 512
  batch_size = 4 # for 16 512

  # effective_batch_size = 1024
  effective_batch_size = 512
  # effective_batch_size = 256

  # effective_batch_size = 128
  GRAD_ACCUM = effective_batch_size // batch_size
  SAVING_STEP = 4
  LOGGING_STEPS = SAVING_STEP
  print(SAVING_STEP)
  training_args = TrainingArguments(
      # warmup_ratio=0.1, 
      # warmup_ratio=0.0, 
      warmup_ratio = 0.03,
      # warmup_ratio = 0.0,
      # learning_rate = 1e-4,
      learning_rate = 2.28e-5,
      # learning_rate = 2.28e-5 * 1.6,

      # max_grad_norm = 2.0,
      max_grad_norm = 1.0,

      # max_grad_norm = 0.3,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      num_train_epochs=3,  # 2.5
      report_to='none',
      output_dir = f'./{checkpoint_folder}/{VER}',
      overwrite_output_dir=True,
      fp16=True,
      # gradient_accumulation_steps=8,
      gradient_accumulation_steps=GRAD_ACCUM,
      logging_steps=LOGGING_STEPS,
      evaluation_strategy='steps',
      eval_steps=SAVING_STEP,
      save_strategy="steps",
      save_steps=SAVING_STEP,
      load_best_model_at_end=False,
      # metric_for_best_model='map@3',
      metric_for_best_model='eval_loss',
      seed=666,
      # lr_scheduler_type='linear',
      lr_scheduler_type='cosine',
      # lr_scheduler_type='cosine_with_restarts',    
      # lr_scheduler_type='reduce_lr_on_plateau',
      # weight_decay=0.01,
      # weight_decay=1e-6, # set this slightly higher to reduce oscillation
      weight_decay=1e-3, # set this slightly higher to reduce oscillation
      # weight_decay=3e-4, # set this slightly higher to reduce oscillation
      save_total_limit=5,
      
  )
  # training_args = training_args.set_optimizer(name="adamw_torch", beta1=0.9, beta2=0.98, weight_decay=training_args.weight_decay)
  # training_args = training_args.set_lr_scheduler(name="reduce_lr_on_plateau", )
  trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
    compute_metrics = compute_metrics,
    callbacks=[SavePeftModelCallback] if USE_PEFT else None,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
  )

  # trainer.train(resume_from_checkpoint=True)
  trainer.train()


  if USE_PEFT:
      trainer.model.save_pretrained(f'model_v{VER}') # 我改了这个
  else:
      trainer.save_model(f'model_v{VER}')

  # I think I read from some parts of the discussion that some length of input during training could be changed.
  # Training longer during training will hopefully cover the length of even the longest sentence in testing.
  # This is some problem with model extrapolation.
  # Basically, if the model is not using very good positional encoding, it will perform badly in sequences longer than what it has been trained on.