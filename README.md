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
- telemetria de tempo, memória, utilização e energia (implementada);
- testes rápidos e perfis reprodutíveis;
- melhorias de arquitetura e qualidade do código.

QLoRA ainda não faz parte desta etapa. A telemetria está habilitada no perfil
controlado usado para as comparações.

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
configs/config_wifi.yaml        perfil controlado WiFi com telemetria
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

## Execução do perfil controlado

A configuração atual assume que os dados processados e o modelo já estão
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

## Telemetria de treinamento

A configuração principal habilita a telemetria e usa 5 rodadas, 10 clientes,
seleção de 20% dos clientes por rodada e LoRA rank 4. A execução é direta:

```bash
python main.py --config configs/config_wifi.yaml
```

Os arquivos são atualizados de forma atômica durante o treinamento em
`results/<simulation_name>/`:

- `client_resource_metrics.csv`: tempos do cliente, CPU/RAM do processo e da
  árvore de subprocessos, VRAM, utilização, potência e energia quando
  disponíveis;
- `round_resource_metrics.csv`: duração da fase de clientes, agregação,
  checkpoint, status e totais de sucesso/falha por rodada;
- `training_resource_summary.json`: resumo acumulado da execução.

`psutil` fornece CPU/RAM e `nvidia-ml-py` fornece métricas NVML da GPU. No modo
não estrito padrão, coletores indisponíveis são registrados no campo
`telemetry_errors` sem interromper o treinamento. O monitor é criado dentro de
cada processo de cliente, enquanto somente o servidor grava os arquivos.

As métricas `cuda_memory_*` pertencem ao alocador CUDA do processo do cliente.
Já `gpu_memory_used_*`, utilização, potência e energia vêm do NVML e representam
a GPU inteira; em uma GPU compartilhada, também podem incluir outras cargas.
O monitor associa o índice CUDA lógico à GPU NVML por UUID ou pelo mapeamento de
`CUDA_VISIBLE_DEVICES`, registrando a origem dessa associação no CSV.

## Princípio de reprodutibilidade

`configs/config_wifi.yaml` é a única configuração autoritativa do experimento.
As comparações LoRA/QLoRA devem manter 5 rodadas, 10 clientes e a mesma fração
de seleção, alterar o `simulation_name` quando necessário e registrar cada etapa
em um commit.

## Licença e citação

O projeto mantém a licença MIT e os metadados de citação do código de origem.
Consulte [`LICENSE`](LICENSE) e [`CITATION.cff`](CITATION.cff).
