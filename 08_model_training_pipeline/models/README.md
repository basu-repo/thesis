# Models

All new trajectory-prediction model definitions should live here.

The registry already reserves the planned experiment families:

- `lstm_goal`
- `cnn_lstm`
- `gnn_lstm`
- `cnn_gnn_lstm`
- `cnn_gnn_transformer`
- `cnn_gnn_lstm_transformer`

The intention is to keep every model definition in plain Python modules rather
than notebook-only cells, so training runs remain reproducible after kernel
restarts.

