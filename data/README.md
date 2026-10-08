# Dados

Os dados de tráfego não são versionados neste repositório. Isso evita publicar
arquivos de vários gigabytes, registros individuais de rede, endereços IP,
timestamps, shards de clientes e artefatos derivados.

O pipeline WiFi espera o arquivo abaixo:

```text
data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
```

Ele pertence ao conjunto público CSE-CIC-IDS2018. Para obtê-lo no caminho
esperado, execute a partir da raiz do projeto:

```bash
bash scripts/fetch_cicids2018_tuesday.sh
```

O pré-processamento criará arquivos locais em `data/wifi/processed/`. Esses
arquivos, o Dataset Arrow tokenizado e os shards federados permanecem ignorados
pelo Git.
