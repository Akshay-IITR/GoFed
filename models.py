import torch
import torch.nn as nn


class LSTM_GRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers, f_dropout, s_dropout):
        super(LSTM_GRU, self).__init__()
        # LSTM layer (bidirectional)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(f_dropout)
        # GRU layer (bidirectional)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(s_dropout)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout1(x)
        x, _ = self.gru(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Get the last time step
        x = self.fc(x)
        return x

class BiGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers, f_dropout, s_dropout):
        super(BiGRU, self).__init__()
        # First GRU layer (bidirectional)
        self.gru1 = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(f_dropout)
        # Second GRU layer (bidirectional)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(s_dropout)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Get the last time step
        x = self.fc(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers, f_dropout, s_dropout, kernel_size=3, num_filters=64):
        super(CNN_LSTM, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # LSTM layers
        self.lstm1 = nn.LSTM(num_filters, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(f_dropout)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(s_dropout)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Get the last time step
        x = self.fc(x)
        return x

class CNN_GRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers, f_dropout, s_dropout, kernel_size=3, num_filters=64):
        super(CNN_GRU, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # GRU layers
        self.gru1 = nn.GRU(num_filters, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(f_dropout)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(s_dropout)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_filters)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Get the last time step
        x = self.fc(x)
        return x