import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yfinance as yf
import ta
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Optional
from pathlib import Path
import traceback
import joblib  # Importação correta do joblib

class MLPredictor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.models_path = Path("models/saved")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Configurações
        ml_config = self.config.get('ml_config')
        self.cache_timeout = self.config.get('ml_parametros.cache_timeout')
        self.min_probabilidade = ml_config['min_probabilidade']
        self.min_accuracy = ml_config['min_accuracy']
        self.max_depth = ml_config['max_depth']
        self.learning_rate = ml_config['learning_rate']
        self.n_estimators = ml_config['n_estimators']
        self.min_confirmacoes = self.config.get('ml_config.min_confirmacoes')
        
        # Novos parâmetros otimizados por tipo de ativo
        self.parametros_modelo = {
            'forex': {
                'max_depth': 6,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'scale_pos_weight': 1.0,
                'objective': 'binary:logistic',
                'random_state': 42
            },
        }
        
        # Sugestão: reduzir temporariamente para testar
        self.min_probabilidade = 0.65  # Reduzir para 65%
        self.min_accuracy = 0.60       # Reduzir para 60%
        
    async def inicializar_modelos(self):
        """Inicializa ou carrega modelos para cada ativo"""
        try:
            ativos = self.config.get_ativos_ativos()
            for ativo in ativos:
                self.logger.info(f"Inicializando modelo para {ativo}")
                
                # Inicializa ou carrega o scaler
                self.scalers[ativo] = StandardScaler()
                
                # Obtém dados históricos para treinar o scaler
                dados_treino = await self._obter_dados_treino(ativo)
                if dados_treino is not None and not dados_treino.empty:
                    features = await self._preparar_features(dados_treino, ativo)
                    if features is not None:
                        # Treina o scaler
                        self.scalers[ativo].fit(features)
                        self.logger.info(f"Scaler treinado para {ativo}")
                
                # Tenta carregar modelo existente
                modelo_path = self.models_path / f"{ativo.replace('=', '_')}_model.pkl"
                if modelo_path.exists():
                    self.logger.info(f"Carregando modelo existente para {ativo}")
                    self.models[ativo] = joblib.load(modelo_path)
                else:
                    self.logger.info(f"Criando novo modelo para {ativo}")
                    self.models[ativo] = XGBClassifier(
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        n_estimators=self.n_estimators,
                        objective='binary:logistic',
                        random_state=42
                    )
                    # Treinar com dados históricos se disponíveis
                    if dados_treino is not None and not dados_treino.empty:
                        await self._treinar_modelo_inicial(ativo, dados_treino)
                    
            self.logger.info("Todos os modelos inicializados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar modelos: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    async def _obter_dados_treino(self, ativo: str) -> Optional[pd.DataFrame]:
        """Obtém dados históricos para treino"""
        try:
            # Obtém dados dos últimos 7 dias para treino
            dados = yf.download(
                ativo,
                period="7d",
                interval="1m",
                progress=False
            )
            
            if dados.empty:
                self.logger.warning(f"Sem dados históricos para {ativo}")
                return None
                
            # Padroniza nomes das colunas
            dados.columns = [col.lower() for col in dados.columns]
            return dados
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados de treino para {ativo}: {str(e)}")
            return None

    async def _treinar_modelo_inicial(self, ativo: str, dados: pd.DataFrame):
        """Treina modelo com dados históricos"""
        try:
            features = await self._preparar_features(dados, ativo)
            if features is None:
                return
                
            # Cria labels simples baseados na direção do preço
            y = (dados['close'].shift(-5) > dados['close']).astype(int)
            y = y[:-5]  # Remove últimos registros sem label
            
            # Remove registros correspondentes das features
            features = features[:-5]
            
            if len(features) > 0:
                # Treina o modelo
                self.models[ativo].fit(features, y)
                self.logger.info(f"Modelo treinado para {ativo}")
                
                # Salva o modelo
                modelo_path = self.models_path / f"{ativo.replace('=', '_')}_model.pkl"
                joblib.dump(self.models[ativo], modelo_path)
                
        except Exception as e:
            self.logger.error(f"Erro ao treinar modelo para {ativo}: {str(e)}")

    def _preparar_features_treino(self, dados: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepara features para treinamento"""
        try:
            # Padroniza nomes das colunas
            dados.columns = [col.lower() for col in dados.columns]
            
            # Cria DataFrame de features
            features = pd.DataFrame(index=dados.index)
            
            # RSI
            rsi = ta.momentum.RSIIndicator(dados['close'], window=14).rsi()
            features['rsi'] = rsi.fillna(method='ffill')
            
            # MACD
            macd = ta.trend.MACD(dados['close'])
            features['macd'] = macd.macd().fillna(method='ffill')
            features['macd_signal'] = macd.macd_signal().fillna(method='ffill')
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(dados['close'])
            features['bb_high'] = bb.bollinger_hband().fillna(method='ffill')
            features['bb_low'] = bb.bollinger_lband().fillna(method='ffill')
            
            # Momentum
            features['momentum'] = ta.momentum.ROCIndicator(dados['close']).roc().fillna(method='ffill')
            
            # Preenche NaN restantes
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar features para treino: {str(e)}")
            return None

    async def prever(self, dados: pd.DataFrame, ativo: str) -> Optional[Dict]:
        """Realiza previsão para um ativo"""
        try:
            self.logger.info(f"Iniciando previsão para {ativo}")
            
            # Verifica se tem dados suficientes
            if len(dados) < 30:
                self.logger.warning(f"Dados insuficientes para {ativo}")
                return None
            
            # Prepara features
            features = await self._preparar_features(dados, ativo)
            if features is None or features.empty:
                return None
            
            # Normaliza features
            if ativo not in self.scalers:
                self.scalers[ativo] = StandardScaler()
                features_scaled = self.scalers[ativo].fit_transform(features)
            else:
                features_scaled = self.scalers[ativo].transform(features)

            # Faz previsão
            if ativo not in self.models:
                self.logger.warning(f"Modelo não encontrado para {ativo}")
                return None
            
            # Obtém probabilidades para ambas as classes
            probabilidades = self.models[ativo].predict_proba(features_scaled)
            prob_put = float(probabilidades[-1][0])
            prob_call = float(probabilidades[-1][1])
            
            # 1. Exige probabilidade mínima de 72% (aumentado de 70%)
            if max(prob_put, prob_call) < 0.72:
                self.logger.info(f"Probabilidade insuficiente: PUT={prob_put:.4f}, CALL={prob_call:.4f}")
                return None
            
            # Define direção com base na maior probabilidade
            direcao = 'CALL' if prob_call > prob_put else 'PUT'
            prob = prob_call if direcao == 'CALL' else prob_put
            
            # 2. Análise de Tendência mais robusta
            sma_curta = dados['close'].rolling(5).mean()
            sma_media = dados['close'].rolling(10).mean()  # Adicionada média intermediária
            sma_longa = dados['close'].rolling(20).mean()
            
            tendencia_curta = sma_curta.iloc[-1] > sma_media.iloc[-1]
            tendencia_longa = sma_media.iloc[-1] > sma_longa.iloc[-1]
            
            # Exige concordância das tendências
            if direcao == 'CALL' and not (tendencia_curta and tendencia_longa):
                self.logger.info("CALL rejeitado: tendências não confirmam")
                return None
            elif direcao == 'PUT' and (tendencia_curta or tendencia_longa):
                self.logger.info("PUT rejeitado: tendências não confirmam")
                return None
            
            # 3. Momentum mais restritivo
            roc = ta.momentum.ROCIndicator(dados['close'], window=10)
            roc_atual = roc.roc().iloc[-1]
            
            # Calcula momentum em diferentes períodos
            mom_curto = ta.momentum.ROCIndicator(dados['close'], window=5).roc().iloc[-1]
            mom_medio = ta.momentum.ROCIndicator(dados['close'], window=10).roc().iloc[-1]
            mom_longo = ta.momentum.ROCIndicator(dados['close'], window=20).roc().iloc[-1]
            
            # Calcula aceleração do momentum
            mom_aceleracao = mom_curto - mom_medio
            
            # Define limites mínimos de momentum baseado na direção
            if direcao == 'CALL':
                if mom_curto < 0.02:  # Momentum mínimo de 0.02% no curto prazo
                    self.logger.info(f"Momentum curto muito fraco para CALL: {mom_curto:.4f}")
                    return None
                if mom_medio < 0:  # Momentum médio precisa ser positivo
                    self.logger.info(f"Momentum médio negativo para CALL: {mom_medio:.4f}")
                    return None
                if mom_aceleracao < 0:  # Momentum não pode estar desacelerando
                    self.logger.info(f"Momentum desacelerando para CALL: {mom_aceleracao:.4f}")
                    return None
                
            else:  # PUT
                if mom_curto > -0.02:  # Momentum mínimo de -0.02% no curto prazo
                    self.logger.info(f"Momentum curto muito fraco para PUT: {mom_curto:.4f}")
                    return None
                if mom_medio > 0:  # Momentum médio precisa ser negativo
                    self.logger.info(f"Momentum médio positivo para PUT: {mom_medio:.4f}")
                    return None
                if mom_aceleracao > 0:  # Momentum não pode estar acelerando
                    self.logger.info(f"Momentum acelerando para PUT: {mom_aceleracao:.4f}")
                    return None
                
            # Verifica consistência do momentum
            if abs(mom_curto) > abs(mom_medio) * 2:  # Movimento muito brusco
                self.logger.info(f"Movimento muito brusco: curto={mom_curto:.4f}, médio={mom_medio:.4f}")
                return None
            
            # Verifica alinhamento com tendência de longo prazo
            if (mom_longo > 0) != (direcao == 'CALL'):
                self.logger.info(f"Momentum não alinhado com tendência de longo prazo")
                return None
            
            # 4. Volume mínimo aumentado
            volume_atual = dados['volume'].iloc[-1]
            volume_medio = dados['volume'].rolling(20).mean()
            volume_ratio = volume_atual / volume_medio.iloc[-1]
            
            # Verifica tendência de volume (últimas 3 barras vs média)
            volume_tendencia = dados['volume'].iloc[-3:].mean() / volume_medio.iloc[-1]
            
            # Define limites de volume baseado no horário
            hora_atual = pd.Timestamp.now().hour
            
            # Ajusta expectativas de volume por período
            if 8 <= hora_atual <= 12:  # Período mais ativo
                vol_min_ratio = 1.2  # Volume 20% acima da média
            elif 13 <= hora_atual <= 17:  # Período intermediário
                vol_min_ratio = 1.1  # Volume 10% acima da média
            else:  # Períodos menos ativos
                vol_min_ratio = 1.3  # Exige mais volume para confirmar movimento
            
            # Validações de volume
            if volume_ratio < vol_min_ratio:
                self.logger.info(f"Volume muito baixo: {volume_ratio:.2f}x média")
                return None
            
            if volume_tendencia < 0.8:  # Volume caindo
                self.logger.info(f"Tendência de queda no volume: {volume_tendencia:.2f}")
                return None
            
            # Verifica consistência do volume
            volume_std = dados['volume'].iloc[-5:].std() / volume_medio.iloc[-1]
            if volume_std > 0.5:  # Volume muito irregular
                self.logger.info(f"Volume muito irregular: std={volume_std:.2f}")
                return None
            
            # Calcula volatilidade
            volatilidade = dados['close'].pct_change().std()
            
            # Define limites de volatilidade baseado no par
            vol_min = 0.015  # Aumentado de ~0.005
            vol_max = 0.25   # Aumentado de 0.12
            
            # Ajusta limites para pares com JPY (20% mais altos)
            if 'JPY' in ativo:
                vol_min *= 1.2
                vol_max *= 1.2
            
            if volatilidade < vol_min:
                self.logger.info(f"Volatilidade muito baixa: {volatilidade:.6f}")
                return None
            elif volatilidade > vol_max:
                self.logger.info(f"Volatilidade muito alta: {volatilidade:.6f}")
                return None
            
            # Adiciona verificação de tendência de volatilidade
            vol_media = dados['close'].pct_change().rolling(20).std().mean()
            vol_ratio = volatilidade / vol_media
            
            if vol_ratio < 0.7:  # Volatilidade caindo muito
                self.logger.info(f"Tendência de queda na volatilidade: {vol_ratio:.2f}")
                return None
            elif vol_ratio > 2.0:  # Volatilidade subindo muito rápido
                self.logger.info(f"Volatilidade aumentando muito rápido: {vol_ratio:.2f}")
                return None
            
            # 6. RSI mais restritivo
            rsi = ta.momentum.RSIIndicator(dados['close']).rsi()
            rsi_atual = rsi.iloc[-1]
            
            # Calcula tendência do RSI
            rsi_tendencia = rsi.iloc[-3:].mean() - rsi.iloc[-8:-3].mean()
            
            # Define zonas do RSI baseadas na direção
            if direcao == 'CALL':
                if rsi_atual > 65:  # Muito sobrecomprado
                    self.logger.info(f"RSI muito alto para CALL: {rsi_atual:.2f}")
                    return None
                elif rsi_atual < 40:  # Precisa ter força mínima
                    self.logger.info(f"RSI muito baixo para CALL: {rsi_atual:.2f}")
                    return None
                elif 45 <= rsi_atual <= 55:  # Zona neutra
                    # Só aceita se tiver tendência clara de alta
                    if rsi_tendencia < 2:
                        self.logger.info(f"RSI em zona neutra sem tendência clara: {rsi_atual:.2f}")
                        return None
                
            else:  # PUT
                if rsi_atual < 35:  # Muito sobrevendido
                    self.logger.info(f"RSI muito baixo para PUT: {rsi_atual:.2f}")
                    return None
                elif rsi_atual > 60:  # Precisa ter fraqueza mínima
                    self.logger.info(f"RSI muito alto para PUT: {rsi_atual:.2f}")
                    return None
                elif 45 <= rsi_atual <= 55:  # Zona neutra
                    # Só aceita se tiver tendência clara de baixa
                    if rsi_tendencia > -2:
                        self.logger.info(f"RSI em zona neutra sem tendência clara: {rsi_atual:.2f}")
                        return None
                    
            # Verifica divergências
            preco_tendencia = dados['close'].iloc[-3:].mean() - dados['close'].iloc[-8:-3].mean()
            
            # Divergência bearish (preço subindo, RSI caindo)
            if preco_tendencia > 0 and rsi_tendencia < 0 and direcao == 'CALL':
                self.logger.info("Divergência bearish detectada")
                return None
            
            # Divergência bullish (preço caindo, RSI subindo)
            if preco_tendencia < 0 and rsi_tendencia > 0 and direcao == 'PUT':
                self.logger.info("Divergência bullish detectada")
                return None
            
            # 7. MACD mais restritivo
            macd = ta.trend.MACD(dados['close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            macd_hist = macd_line - signal_line
            
            if abs(macd_hist) < 0.0004:  # Aumentado de 0.0003
                self.logger.info(f"MACD muito fraco: {macd_hist:.6f}")
                return None
            
            # Verifica direção do MACD
            if direcao == 'CALL' and macd_hist < 0:
                self.logger.info("MACD contradiz CALL")
                return None
            elif direcao == 'PUT' and macd_hist > 0:
                self.logger.info("MACD contradiz PUT")
                return None
            
            # 8. Bollinger Bands
            bb = ta.volatility.BollingerBands(dados['close'])
            bb_superior = bb.bollinger_hband().iloc[-1]
            bb_inferior = bb.bollinger_lband().iloc[-1]
            bb_medio = bb.bollinger_mavg().iloc[-1]
            
            preco_atual = dados['close'].iloc[-1]
            bb_spread = (bb_superior - bb_inferior) / bb_medio
            
            # Calcula posição relativa nas bandas (0 = banda inferior, 1 = banda superior)
            bb_posicao = (preco_atual - bb_inferior) / (bb_superior - bb_inferior)
            
            # Calcula tendência das bandas
            bb_spread_anterior = (bb.bollinger_hband().iloc[-2] - bb.bollinger_lband().iloc[-2]) / bb.bollinger_mavg().iloc[-2]
            bb_spread_tendencia = bb_spread - bb_spread_anterior
            
            # Define limites mínimos de spread baseado no par
            bb_spread_min = 0.0015  # 0.15% mínimo de spread
            if 'JPY' in ativo:
                bb_spread_min = 0.002  # 0.2% para pares com JPY
            
            # Validações das Bollinger Bands
            if bb_spread < bb_spread_min:
                self.logger.info(f"BB Spread muito pequeno: {bb_spread:.6f}")
                return None
            
            # Verifica squeeze das bandas
            if bb_spread_tendencia < -0.0002:  # Bandas convergindo
                self.logger.info(f"Bandas convergindo: tendência = {bb_spread_tendencia:.6f}")
                return None
            
            # Verifica posição relativa baseada na direção
            if direcao == 'CALL':
                if bb_posicao > 0.7:  # Muito próximo da banda superior
                    self.logger.info(f"Preço muito próximo da banda superior: {bb_posicao:.2f}")
                    return None
                elif bb_posicao < 0.3:  # Precisa ter espaço para subir
                    if bb_spread_tendencia <= 0:  # Se as bandas não estiverem expandindo
                        self.logger.info(f"Posição muito baixa sem expansão das bandas")
                        return None
                    
            else:  # PUT
                if bb_posicao < 0.3:  # Muito próximo da banda inferior
                    self.logger.info(f"Preço muito próximo da banda inferior: {bb_posicao:.2f}")
                    return None
                elif bb_posicao > 0.7:  # Precisa ter espaço para cair
                    if bb_spread_tendencia <= 0:  # Se as bandas não estiverem expandindo
                        self.logger.info(f"Posição muito alta sem expansão das bandas")
                        return None
            
            # Verifica alinhamento com tendência de longo prazo
            if (mom_longo > 0) != (direcao == 'CALL'):
                self.logger.info(f"Momentum não alinhado com tendência de longo prazo")
                return None
            
            # 4. Volume mínimo aumentado
            volume_atual = dados['volume'].iloc[-1]
            volume_medio = dados['volume'].rolling(20).mean()
            volume_ratio = volume_atual / volume_medio.iloc[-1]
            
            # Verifica tendência de volume (últimas 3 barras vs média)
            volume_tendencia = dados['volume'].iloc[-3:].mean() / volume_medio.iloc[-1]
            
            # Define limites de volume baseado no horário
            hora_atual = pd.Timestamp.now().hour
            
            # Ajusta expectativas de volume por período
            if 8 <= hora_atual <= 12:  # Período mais ativo
                vol_min_ratio = 1.2  # Volume 20% acima da média
            elif 13 <= hora_atual <= 17:  # Período intermediário
                vol_min_ratio = 1.1  # Volume 10% acima da média
            else:  # Períodos menos ativos
                vol_min_ratio = 1.3  # Exige mais volume para confirmar movimento
            
            # Validações de volume
            if volume_ratio < vol_min_ratio:
                self.logger.info(f"Volume muito baixo: {volume_ratio:.2f}x média")
                return None
            
            if volume_tendencia < 0.8:  # Volume caindo
                self.logger.info(f"Tendência de queda no volume: {volume_tendencia:.2f}")
                return None
            
            # Verifica consistência do volume
            volume_std = dados['volume'].iloc[-5:].std() / volume_medio.iloc[-1]
            if volume_std > 0.5:  # Volume muito irregular
                self.logger.info(f"Volume muito irregular: std={volume_std:.2f}")
                return None
            
            # Calcula volatilidade
            volatilidade = dados['close'].pct_change().std()
            
            # Define limites de volatilidade baseado no par
            vol_min = 0.015  # Aumentado de ~0.005
            vol_max = 0.25   # Aumentado de 0.12
            
            # Ajusta limites para pares com JPY (20% mais altos)
            if 'JPY' in ativo:
                vol_min *= 1.2
                vol_max *= 1.2
            
            if volatilidade < vol_min:
                self.logger.info(f"Volatilidade muito baixa: {volatilidade:.6f}")
                return None
            elif volatilidade > vol_max:
                self.logger.info(f"Volatilidade muito alta: {volatilidade:.6f}")
                return None
            
            # Adiciona verificação de tendência de volatilidade
            vol_media = dados['close'].pct_change().rolling(20).std().mean()
            vol_ratio = volatilidade / vol_media
            
            if vol_ratio < 0.7:  # Volatilidade caindo muito
                self.logger.info(f"Tendência de queda na volatilidade: {vol_ratio:.2f}")
                return None
            elif vol_ratio > 2.0:  # Volatilidade subindo muito rápido
                self.logger.info(f"Volatilidade aumentando muito rápido: {vol_ratio:.2f}")
                return None
            
            # 6. RSI mais restritivo
            rsi = ta.momentum.RSIIndicator(dados['close']).rsi()
            rsi_atual = rsi.iloc[-1]
            
            # Calcula tendência do RSI
            rsi_tendencia = rsi.iloc[-3:].mean() - rsi.iloc[-8:-3].mean()
            
            # Define zonas do RSI baseadas na direção
            if direcao == 'CALL':
                if rsi_atual > 65:  # Muito sobrecomprado
                    self.logger.info(f"RSI muito alto para CALL: {rsi_atual:.2f}")
                    return None
                elif rsi_atual < 40:  # Precisa ter força mínima
                    self.logger.info(f"RSI muito baixo para CALL: {rsi_atual:.2f}")
                    return None
                elif 45 <= rsi_atual <= 55:  # Zona neutra
                    # Só aceita se tiver tendência clara de alta
                    if rsi_tendencia < 2:
                        self.logger.info(f"RSI em zona neutra sem tendência clara: {rsi_atual:.2f}")
                        return None
                
            else:  # PUT
                if rsi_atual < 35:  # Muito sobrevendido
                    self.logger.info(f"RSI muito baixo para PUT: {rsi_atual:.2f}")
                    return None
                elif rsi_atual > 60:  # Precisa ter fraqueza mínima
                    self.logger.info(f"RSI muito alto para PUT: {rsi_atual:.2f}")
                    return None
                elif 45 <= rsi_atual <= 55:  # Zona neutra
                    # Só aceita se tiver tendência clara de baixa
                    if rsi_tendencia > -2:
                        self.logger.info(f"RSI em zona neutra sem tendência clara: {rsi_atual:.2f}")
                        return None
                    
            # Verifica divergências
            preco_tendencia = dados['close'].iloc[-3:].mean() - dados['close'].iloc[-8:-3].mean()
            
            # Divergência bearish (preço subindo, RSI caindo)
            if preco_tendencia > 0 and rsi_tendencia < 0 and direcao == 'CALL':
                self.logger.info("Divergência bearish detectada")
                return None
            
            # Divergência bullish (preço caindo, RSI subindo)
            if preco_tendencia < 0 and rsi_tendencia > 0 and direcao == 'PUT':
                self.logger.info("Divergência bullish detectada")
                return None
            
            # 7. MACD mais restritivo
            macd = ta.trend.MACD(dados['close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            macd_hist = macd_line - signal_line
            
            if abs(macd_hist) < 0.0004:  # Aumentado de 0.0003
                self.logger.info(f"MACD muito fraco: {macd_hist:.6f}")
                return None
            
            # Verifica direção do MACD
            if direcao == 'CALL' and macd_hist < 0:
                self.logger.info("MACD contradiz CALL")
                return None
            elif direcao == 'PUT' and macd_hist > 0:
                self.logger.info("MACD contradiz PUT")
                return None
            
            # Verifica movimento mínimo potencial baseado no BB spread
            movimento_em_pips = 0
            valor_pip = self._calcular_pips(ativo, preco_atual)
            
            if direcao == 'CALL':
                movimento_em_pips = (bb_superior - preco_atual) / valor_pip
            else:
                movimento_em_pips = (preco_atual - bb_inferior) / valor_pip
            
            # Define mínimos de pips por tipo de par
            min_pips_requerido = 10  # padrão
            if 'JPY' in ativo:
                min_pips_requerido = 15
            elif 'EUR' in ativo and 'GBP' in ativo:  # crosses
                min_pips_requerido = 8
            
            if movimento_em_pips < min_pips_requerido:
                self.logger.info(f"Movimento potencial muito pequeno: {movimento_em_pips:.1f} pips (mínimo: {min_pips_requerido})")
                return None
            
            # Se passou por TODAS as validações, retorna o resultado
            resultado = {
                'probabilidade': float(prob),
                'direcao': direcao,
                'score': float(prob),
                'assertividade': float(prob * 100),
                'indicadores': {
                    'rsi': float(rsi_atual),
                    'rsi_tendencia': float(rsi_tendencia),
                    'prob_call': float(prob_call),
                    'prob_put': float(prob_put),
                    'macd_diff': float(macd_hist),
                    'momentum': float(roc_atual),
                    'volume_ratio': float(volume_ratio),
                    'volume_tendencia': float(volume_tendencia),
                    'volume_std': float(volume_std),
                    'tendencia': 'ALTA' if tendencia_curta and tendencia_longa else 'BAIXA',
                    'bb_spread': float(bb_spread),
                    'bb_posicao': float(bb_posicao),
                    'bb_spread_tendencia': float(bb_spread_tendencia),
                    'movimento_pips': float(movimento_em_pips),
                    'volatilidade': float(volatilidade),
                    'volatilidade_ratio': float(vol_ratio),
                    'momentum_curto': float(mom_curto),
                    'momentum_medio': float(mom_medio),
                    'momentum_longo': float(mom_longo),
                    'momentum_aceleracao': float(mom_aceleracao)
                }
            }
            
            self.logger.info(f"""
            Sinal APROVADO para {ativo}:
            - Direção: {direcao}
            - Probabilidade: {prob:.4f}
            - RSI: {rsi_atual:.2f} (tendência: {rsi_tendencia:.2f})
            - MACD Diff: {macd_hist:.6f}
            - Momentum: {roc_atual:.4f}
            - Volume: {volume_ratio:.2f}x média (tendência: {volume_tendencia:.2f})
            - BB Spread: {bb_spread:.6f}
            - BB Posição: {bb_posicao:.2f}
            - BB Tendência: {bb_spread_tendencia:.6f}
            - Movimento Potencial: {movimento_em_pips:.1f} pips
            - Volatilidade: {volatilidade:.6f} (ratio: {vol_ratio:.2f})
            - Tendência: {'ALTA' if tendencia_curta and tendencia_longa else 'BAIXA'}
            - Momentum Curto: {mom_curto:.4f}
            - Momentum Médio: {mom_medio:.4f}
            - Momentum Longo: {mom_longo:.4f}
            - Aceleração: {mom_aceleracao:.4f}
            """)
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Erro na previsão: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
        
    async def _preparar_features(self, dados: pd.DataFrame, ativo: str) -> Optional[pd.DataFrame]:
        """Prepara features para o modelo"""
        try:
            # Verifica dados mínimos
            if len(dados) < 14:  # Mínimo para RSI
                self.logger.warning(f"Dados insuficientes para {ativo}: {len(dados)} registros. Mínimo 14 necessário.")
                return None
            
            # Padroniza nomes das colunas
            dados.columns = [col.lower() for col in dados.columns]
            
            # Cria DataFrame de features
            features = pd.DataFrame(index=dados.index)
            
            # RSI (precisa de pelo menos 14 períodos)
            rsi = ta.momentum.RSIIndicator(dados['close'], window=14).rsi()
            features['rsi'] = rsi
            
            # MACD com períodos menores (5,13,3 em vez de 12,26,9)
            macd = ta.trend.MACD(
                dados['close'],
                window_fast=5,    # Era 12
                window_slow=13,   # Era 26
                window_sign=3     # Era 9
            )
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
        
            
            # Bollinger Bands (precisa de pelo menos 20 períodos)
            bb = ta.volatility.BollingerBands(dados['close'])
            features['bb_high'] = bb.bollinger_hband()
            features['bb_low'] = bb.bollinger_lband()
            
            # Momentum (precisa de pelo menos 10 períodos)
            features['momentum'] = ta.momentum.ROCIndicator(dados['close']).roc()
            
            # Verifica cada indicador
            indicadores_vazios = features.isna().all()
            if indicadores_vazios.any():
                self.logger.warning(f"""
                Indicadores vazios para {ativo}:
                {indicadores_vazios[indicadores_vazios].index.tolist()}
                Dados disponíveis: {len(dados)} registros
                """)
                return None
            
            # Preenche NaN apenas nos primeiros registros
            features = features.fillna(method='bfill').fillna(method='ffill')
            
            # Log detalhado
            self.logger.info(f"""
            Features geradas para {ativo}:
            - Registros: {len(features)}
            - Indicadores: {features.columns.tolist()}
            - Últimos valores: {features.iloc[-1].to_dict()}
            - Dados originais: {len(dados)} registros
            - Períodos necessários: RSI(14), MACD(26), BB(20), Momentum(10)
            """)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar features: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    # Função auxiliar para calcular pips (adicionada no início da classe)
    def _calcular_pips(self, ativo, preco):
        """Calcula o valor de 1 pip para o ativo"""
        if 'JPY' in ativo:
            return 0.01  # Para pares com JPY, 1 pip = 0.01
        return 0.0001    # Para outros pares, 1 pip = 0.0001