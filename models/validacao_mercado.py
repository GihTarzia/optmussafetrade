import pandas as pd
from typing import Dict 
from datetime import datetime

class ValidacaoMercado:
    def __init__(self, logger):
        self.logger = logger
        # Configurações de validação
        self.horarios_operacao = {
            'inicio': 9,    # 9:00
            'fim': 17,      # 17:00
            'gap_max': 0.005,  # 0.3%
            'vol_min': 0.00005, # Volatilidade mínima
            'vol_max': 0.005,  # Volatilidade máxima
            'volume_min_ratio': 0.3  # 50% do volume médio
        }

    def validar_condicoes(self, dados: pd.DataFrame) -> Dict:
        """Valida todas as condições de mercado"""
        try:
            resultados = {
                'valido': True,
                'mensagem': [],
                'detalhes': {}
            }

            # 1. Validação de Horário
            hora_atual = datetime.now().hour
            if not (self.horarios_operacao['inicio'] <= hora_atual <= self.horarios_operacao['fim']):
                resultados['valido'] = False
                resultados['mensagem'].append(f"Fora do horário de operação: {hora_atual}h")
                return resultados

            # 2. Validação de Volume
            volume_medio = dados['volume'].rolling(20).mean().iloc[-1]
            volume_atual = dados['volume'].iloc[-1]
            volume_ratio = volume_atual / volume_medio if volume_medio > 0 else 0

            resultados['detalhes']['volume_ratio'] = volume_ratio

            if volume_ratio < self.horarios_operacao['volume_min_ratio']:
                resultados['valido'] = False
                resultados['mensagem'].append(f"Volume insuficiente: {volume_ratio:.2f}x média")

            # 3. Verificação de Gaps
            gaps = dados['close'].diff().abs()
            gap_medio = gaps.mean()
            gap_atual = gaps.iloc[-1]
            gap_ratio = gap_atual / gap_medio if gap_medio > 0 else 0

            resultados['detalhes']['gap_ratio'] = gap_ratio

            if gap_ratio > 5:  # Gap 3x maior que a média
                resultados['valido'] = False
                resultados['mensagem'].append(f"Gap anormal detectado: {gap_ratio:.2f}x média")

            # 4. Validação de Volatilidade
            volatilidade = dados['close'].pct_change().std()
            resultados['detalhes']['volatilidade'] = volatilidade

            if not (self.horarios_operacao['vol_min'] <= volatilidade <= self.horarios_operacao['vol_max']):
                resultados['valido'] = False
                resultados['mensagem'].append(f"Volatilidade fora do range: {volatilidade:.6f}")

            # Log detalhado
            if resultados['valido']:
                self.logger.info("Validação de mercado: OK")
            else:
                self.logger.info(f"Validação de mercado falhou: {', '.join(resultados['mensagem'])}")

            return resultados

        except Exception as e:
            self.logger.error(f"Erro na validação de mercado: {str(e)}")
            return {'valido': False, 'mensagem': ['Erro na validação'], 'detalhes': {}}