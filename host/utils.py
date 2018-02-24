import pickle
import torch
from torch.autograd import Variable

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def char_tensor(string,char2index):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = char2index[string[c]]
    return Variable(tensor)

def evaluate(prime_str='A', predict_len=200, temperature=0.8, index2char=None, char2index=None,decoder=None):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str,char2index)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = index2char[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char,char2index)

    return predicted