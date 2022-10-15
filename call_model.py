import tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import io


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
    
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

        
def generate_reply(inp, num_gen=1):
    device = torch.device('cpu')
    #'cuda' if torch.cuda.is_available() else 'cpu'
    input_text = "<s>" + str(inp) + "[SEP]"
    load_path = 'model.pth'
    
    if torch.cuda.is_available():
        with open(load_path, 'rb') as file:
            model = pickle.load(file, encoding='latin1')
    else:
        with open(load_path, 'rb') as file:
            model = CPU_Unpickler(file).load()
    
    
    
    input_ids = tokenizer.encode_plus(input_text, return_tensors='pt').to(device)
    out = model.generate(input_ids, do_sample=True, max_length=64, num_return_sequences=num_gen, 
                         top_p=0.95, top_k=20, bad_words_ids=[[1], [5]], no_repeat_ngram_size=3)
    
    
    print(">", "あなた")
    print(inp)
    print(">", "totu")
    for sent in tokenizer.batch_decode(out):
        sent = sent.split('[SEP]</s>')[1]
        sent = sent.replace('</s>', '')
        sent = sent.replace('<br>', '\n')
        print(sent)