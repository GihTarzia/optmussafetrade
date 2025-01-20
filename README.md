# Optimus Safe Trade

Sistema automatizado de trading para opções binárias com machine learning e otimização automática.

## Características

- Análise técnica avançada
- Machine Learning adaptativo
- Auto-otimização de parâmetros
- Gestão de risco dinâmica
- Backtesting integrado
- Logging detalhado

## Requisitos

- Python 3.8+
- Dependências listadas em requirements.txt

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/trading-bot.git
cd trading-bot
```

## ChatGPT ou Claude

> Olá! Estou trabalhando em um projeto de trading bot para opções binárias em Python. O sistema:
>
> 1. Utiliza machine learning (MLPredictor) e análise técnica (AnalisePadroesComplexos) para gerar sinais
> 2. Tem um sistema de gestão de risco adaptativo
> 3. Usa banco de dados SQLite para armazenar histórico
> 4. Monitora principalmente pares forex
>
> Atualmente:
>
> - A acurácia mínima dos modelos está em 55%
> - Estamos usando indicadores como RSI, MACD, Bollinger Bands
> - O sistema analisa 13 ativos a cada 30 segundos
>
> Quero melhorar [especifique o componente/funcionalidade], porque [explique o problema ou objetivo].
>
> O código relevante atual é:
> [cole o trecho de código relacionado]

## Árvore

```


opitimussafetrade/
│
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── parametros.py
│
├── data/
│   ├── __init__.py
│   ├── trading_bot.db
│   └── trading_bot.log
│
├── models/
│   ├── __init__.py
│   ├── ml_predictor.py
│   ├── analise_padroes.py
│   ├── gestao_risco.py
│   └── auto_ajuste.py
│
├── utils/
│   ├── __init__.py
│   ├── database.py
│   └── logger.py
│
├── requirements.txt
├── main.py
└── README.md
```
