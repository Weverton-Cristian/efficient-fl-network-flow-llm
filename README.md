# Efficient FL Network-Flow LLM

Projeto dedicado a aprendizado federado eficiente para análise de fluxos de
rede WiFi com modelos de linguagem pequenos.

Este repositório parte do pipeline WiFi/CIC-IDS2018 do projeto
[`FL_NETWORK_FLOW_LLM`](https://github.com/Weverton-Cristian/FL_NETWORK_FLOW_LLM),
no commit `6b8c40457e6604ececb56046fd25feaa4e1f4b75`. O repositório de origem permanece
como registro do código utilizado no artigo. A evolução experimental passa a
acontecer aqui.

## Escopo

O snapshot inicial contém somente o núcleo WiFi necessário para:

- transformar fluxos de rede em texto;
- treinar `HuggingFaceTB/SmolLM-135M` com LoRA em cenário federado;
- agregar atualizações com FedAvg/FedProx;
- avaliar classificação binária e limiares operacionais;
- reproduzir a configuração WiFi usada como ponto de partida.

O roadmap deste novo projeto inclui:

- QLoRA e quantização INT8/INT4;
- compressão de atualizações federadas;
- redução do consumo de RAM e VRAM;
- telemetria de tempo, memória, utilização e energia;
- testes rápidos e perfis reprodutíveis;
- melhorias de arquitetura e qualidade do código.

QLoRA e a nova telemetria ainda não fazem parte deste primeiro snapshot. A
configuração atual é a baseline LoRA preservada para permitir comparações
controladas.

## Conteúdo deliberadamente excluído

Não são versionados:

- CSVs brutos ou processados;
- Dataset Arrow e shards de clientes;
- checkpoints, modelos ou adapters;
- logs e resultados completos;
- ambientes virtuais e caches;
- processadores de ransomware/HDFS;
- scripts e artefatos de comparação com outros projetos;
- relatórios HTML/PDF gerados pelo trabalho anterior.

## Estrutura

```text
configs/config_wifi.yaml        configuração baseline WiFi
data/README.md                  obtenção e posicionamento dos dados
scripts/fetch_cicids2018_tuesday.sh
scripts/evaluate_existing_experiment.py
src/data_processing/            processamento WiFi
src/federated_learning/         clientes, servidor e agregação
src/models/                     inicialização do modelo e LoRA
src/evaluation/                 avaliação operacional
src/utils/                      utilitários Hugging Face e PEFT
main.py                         entrada do pipeline
```

## Instalação

Requer Python 3.10+ e uma GPU CUDA para os experimentos principais.

```bash
git clone https://github.com/Weverton-Cristian/efficient-fl-network-flow-llm.git
cd efficient-fl-network-flow-llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados

Baixe o arquivo Tuesday do CSE-CIC-IDS2018:

```bash
bash scripts/fetch_cicids2018_tuesday.sh
```

O arquivo será colocado em:

```text
data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
```

## Execução da baseline

A configuração preservada assume que os dados processados e o modelo já estão
disponíveis localmente. Para uma primeira execução com download e processamento:

```bash
python main.py --config configs/config_wifi.yaml \
  --set force_reprocess_data=true \
  --set hf_offline=false \
  --set hf_local_files_only=false
```

Depois que os artefatos locais forem criados:

```bash
python main.py --config configs/config_wifi.yaml
```

## Princípio de reprodutibilidade

A baseline do artigo não deve ser silenciosamente alterada. Novas experiências
de QLoRA, quantização, compressão ou profiling deverão usar novos arquivos em
`configs/`, nomes de simulação próprios e registrar versões de software e
hardware.

## Licença e citação

O projeto mantém a licença MIT e os metadados de citação do código de origem.
Consulte [`LICENSE`](LICENSE) e [`CITATION.cff`](CITATION.cff).
