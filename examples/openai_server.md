# OpenAI-compatible server

Start a Qwen or Muse server that accepts up to eight simultaneous requests:

```bash
cd /home/twidmer/Documents/git/warp-nn
.venv/bin/python examples/openai_server.py /path/to/model --host 0.0.0.0 --port 8000 --max-batch-size 8 --api-key CHOOSE_A_SECRET
```

`--max-batch-size` accepts `1`, `2`, `4`, or `8`. It controls the maximum
number of active requests, not a permanently fixed kernel width: the server
automatically uses B1, B2, B4, or B8 according to current demand. Weights are
shared, but larger limits reserve more per-request KV and recurrent state.

`--host 0.0.0.0` allows connections from other computers on the LAN. At
startup the server prints its local URL, detected LAN URL, model ID, and a
ready-to-copy client command. Keep the default `127.0.0.1` if remote access is
not needed. A firewall may need to allow the selected TCP port. Use an API key
on any network you do not fully trust; this small server does not provide TLS.

On the other computer, copy `openai_client.py` and run the command printed by
the server. The client has no dependencies beyond Python. Without arguments it
asks for the LAN URL and discovers the model automatically:

```bash
python openai_client.py
```

Alternatively:

```bash
python openai_client.py --url http://192.168.1.5:8000/v1 --api-key CHOOSE_A_SECRET
```

The API implements `GET /v1/models` and `POST /v1/chat/completions`, so an
ordinary OpenAI-compatible client can also use the printed URL as its base URL.
