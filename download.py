from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
MODEL = 'microsoft/deberta-v3-large'
model = AutoModelForMultipleChoice.from_pretrained(MODEL)