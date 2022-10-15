import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

def generate_reply(inp, num_gen=1):
    output='output/output'
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained("output/output")
    tokenizer.do_lower_case = True
    
    model = AutoModelForCausalLM.from_pretrained(output)
    model.to(device)
    model.eval()
    
    input_text = "<s>" + str(inp) + "[SEP]"
    input_ids = tokenizer.encode(input_text,return_tensors='pt').to(device)
    out = model.generate(input_ids, do_sample=True, max_length=64, num_return_sequences=num_gen, 
                         top_p=0.95, top_k=20, bad_words_ids=[[1], [5]], no_repeat_ngram_size=3)

    for sent in tokenizer.batch_decode(out):
        sent = sent.split('[SEP]</s>')[1]
        sent = sent.replace('</s>', '')
        sent = sent.replace('<br>', '\n')
    return sent