import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import MarianTokenizer


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = math.sqrt(dim)

    def forward(self, queries, keys):
        energy = torch.matmul(queries, keys.transpose(1, 2))
        return F.softmax(energy / self.scale, dim=-1)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        return lstm_output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = ScaledDotProductAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = self.embedding(x)  # Removed unnecessary unsqueeze(1)
        lstm_output, (hidden, cell) = self.lstm(x.unsqueeze(1), (hidden, cell))
        attention_weights = self.attention(hidden[-1].unsqueeze(1), encoder_outputs)
        context_vector = torch.bmm(attention_weights, encoder_outputs)
        prediction = self.fc(torch.cat((lstm_output, context_vector), dim=2))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, input_seq, target_seq):
        batch_size, max_length = target_seq.size()
        outputs = torch.zeros(batch_size, max_length, self.decoder.fc.out_features)

        encoder_outputs, hidden, cell = self.encoder(input_seq)
        target_input_token = target_seq[:, 0]

        for t in range(1, max_length):
            decoder_output, hidden, cell = self.decoder(
                target_input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = decoder_output.squeeze(1)
            target_input_token = target_seq[:, t]

        return outputs


tokenizer = MarianTokenizer.from_pretrained("src/NMT/artifacts/tokenizer")

model = Seq2Seq(
    vocab_size=len(tokenizer),
    embed_dim=128,
    hidden_dim=512,
    num_layers=4,
    dropout=0.2,
)