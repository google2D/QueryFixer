from transformers import BertModel, AutoTokenizer
import torch

class Encoder:
    def __init__(self):
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = False)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Function to encode a sequence/sentence using the bert embedding model
    def bert_encode_sequence(self, sequence):

        # encode the sentence
        encoded = self.bert_tokenizer.encode_plus(
            text=sequence,
            add_special_tokens=True,  # add [CLS] and [SEP] tokens
            max_length = 30,  # set the maximum length of a sentence
            truncation = True, # truncate longer sentences to max_length
            padding='max_length',  # add [PAD] tokens to shorter sentences
            return_attention_mask = True,  # generate the attention mask (so that no attention is given to padding tokens)
            return_tensors = 'pt',  # return encoding results as PyTorch tensors
        )

        # Each token is assigned an id by the tokenizer
        token_ids = encoded['input_ids']

        # Attention mask is used to ensure that the model does not pay attention to padding tokens
        # Value of mask is 0 if token is a padding token, else 1
        attention_mask = encoded['attention_mask']

        # BERT accepts two sentences at once, so token_type_ids is used to indicate
        # whether each token belongs to sentence A (0) or sentence B (1),
        # but we only have one sentence per query, so all tokens are assigned 0
        token_type_ids = encoded['token_type_ids']

        # set the BERT model in evaluation mode
        self.bert_model.eval()

        # Forward pass through the bert model to get contextual embeddings of each token (without calculating gradients)
        with torch.no_grad():
            # output of shape <batch_size, max_length, embedding_size>
            last_hidden_states = self.bert_model(token_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["last_hidden_state"]

        # mean pooled embedding of shape <1, hidden_size>
        mean_pooled_embedding = last_hidden_states.mean(axis=1)

        return mean_pooled_embedding