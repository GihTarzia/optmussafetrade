import asyncio
from telegram import Bot
from typing import Dict, Optional
from datetime import datetime
import json
import logging
from collections import deque
from queue import PriorityQueue
import pytz
import traceback

class NotificationManager:
    """Gerenciador de filas e histórico de notificações"""
    def __init__(self, max_history: int = 1000):
        self.pending = asyncio.Queue()
        self.history = deque(maxlen=max_history)
        self.failed = deque(maxlen=max_history)
        self.statistics = {
            'sent_count': 0,
            'failed_count': 0,
            'last_sent': None,
            'last_error': None
        }

    async def add_notification(self, message: Dict):
        """Adiciona notificação à fila"""
        await self.pending.put({
            'content': message,
            'timestamp': datetime.now(),
            'attempts': 0
        })

    async def get_next_notification(self) -> Optional[Dict]:
        """Recupera próxima notificação da fila"""
        try:
            return await self.pending.get()
        except asyncio.QueueEmpty:
            return None

    def record_success(self, notification: Dict):
        """Registra notificação bem-sucedida"""
        self.history.append({
            **notification,
            'status': 'sent',
            'sent_at': datetime.now()
        })
        self.statistics['sent_count'] += 1
        self.statistics['last_sent'] = datetime.now()

    def record_failure(self, notification: Dict, error: str):
        """Registra falha na notificação"""
        self.failed.append({
            **notification,
            'status': 'failed',
            'error': error,
            'failed_at': datetime.now()
        })
        self.statistics['failed_count'] += 1
        self.statistics['last_error'] = datetime.now()

class Notificador:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.logger = logging.getLogger(__name__)
        self.queue = asyncio.Queue()
        
    async def start(self):
        """Inicia o processamento da fila"""
        asyncio.create_task(self._process_queue())
        
    async def _process_queue(self):
        """Processa mensagens na fila"""
        while True:
            try:
                mensagem = await self.queue.get()
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=mensagem['texto'],
                    parse_mode='Markdown'
                )
                self.queue.task_done()
                await asyncio.sleep(1)  # Evita flood
            except Exception as e:
                self.logger.error(f"Erro ao processar mensagem: {str(e)}")

    async def enviar_sinal(self, sinal: Dict) -> bool:
        """Envia sinal formatado para o Telegram"""
        try:
            direcao_emoji = "🟢" if sinal['direcao'] == 'CALL' else "🔴"

            mensagem = f"""
🎯 *NOVO SINAL*
------------------------
🔸 Ativo: `{sinal['ativo'].replace('=X','')}`
{direcao_emoji} Direção: *{sinal['direcao']}*
⏰ Expiração: {sinal['tempo_expiracao']}min
💰 Entrada: {sinal['preco_entrada']:.5f}
🎲 Prob: {sinal['probabilidade']:.1f}%
📊 Score: {sinal['score']:.2f}
📈 Volatilidade: {sinal['volatilidade']:.1f}%
------------------------
            """
            
            await self.queue.put({
                'texto': mensagem,
                'tipo': 'sinal'
            })
            
            self.logger.info(f"Sinal enfileirado para {sinal['ativo']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao enfileirar sinal: {str(e)}")
            return False

    async def enviar_resultado(self, operacao: Dict) -> bool:
        """Envia resultado formatado para o Telegram"""
        try:
            emoji = "✅" if operacao['resultado'] == 'WIN' else "❌"
            
            mensagem = f"""
{emoji} *RESULTADO OPERAÇÃO*
------------------------
🔸 Ativo: `{operacao['ativo']}`
📈 Direção: *{operacao['direcao']}*
💰 Entrada: `{operacao['preco_entrada']:.5f}`
💰 Saída: `{operacao['preco_saida']:.5f}`
🎯 Resultado: *{operacao['resultado']}*
------------------------
            """
            
            await self.queue.put({
                'texto': mensagem,
                'tipo': 'resultado'
            })
            
            self.logger.info(f"Resultado enfileirado para operação {operacao['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao enfileirar resultado: {str(e)}")
            return False

    def formatar_sinal(self, sinal: Dict) -> str:
        """Formata sinal para envio"""
        try:
            # Validações básicas
            if not isinstance(sinal, dict):
                raise ValueError(f"Sinal inválido: {sinal}")
            
            # Campos obrigatórios
            campos_obrigatorios = ['ativo', 'direcao', 'probabilidade', 'score']
            for campo in campos_obrigatorios:
                if campo not in sinal:
                    raise ValueError(f"Campo obrigatório ausente: {campo}")
            
            # Emojis para melhor visualização
            direcao_emoji = "🟢" if sinal['direcao'] == 'CALL' else "🔴"
            
            mensagem = [
                f"🎯 *SINAL DE ENTRADA*",
                f"",
                f"🔸 *Ativo:* {sinal['ativo'].replace('=X','')}",
                f"{direcao_emoji} *Direção:* {sinal['direcao']}",
                f"📊 *Probabilidade:* {float(sinal['probabilidade']):.2%}",
                f"💯 *Score:* {float(sinal['score']):.2f}",
                f"⏱️ *Expiração:* {sinal.get('tempo_expiracao', 5)} min"
            ]
            
            # Adiciona preço atual se existir
            if 'preco_entrada' in sinal:
                mensagem.append(f"💰 *Preço Atual:* ${float(sinal['preco_entrada']):.6f}")
            
            mensagem.append("")
            mensagem.append(f"📈 *Indicadores:*")
            
            # Adiciona indicadores se existirem
            if 'indicadores' in sinal and sinal['indicadores']:
                try:
                    if isinstance(sinal['indicadores'], str):
                        ind = json.loads(sinal['indicadores'])
                    else:
                        ind = sinal['indicadores']
                        
                    mensagem.extend([
                        f"• RSI: {float(ind.get('rsi', 0)):.2f}",
                        f"• MACD diff: {float(ind.get('macd_diff', 0)):.6f}",
                        f"• BB spread: {float(ind.get('bb_spread', 0)):.6f}",
                        f"• Momentum: {float(ind.get('momentum', 0)):.6f}"
                    ])
                except Exception as e:
                    self.logger.error(f"Erro ao processar indicadores: {str(e)}")
            
            # Adiciona timestamp se existir
            if 'timestamp' in sinal:
                # Converte para horário de São Paulo
                sp_tz = pytz.timezone('America/Sao_Paulo')
                timestamp = sinal['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                if timestamp.tzinfo is None:
                    timestamp = pytz.UTC.localize(timestamp)
                sp_time = timestamp.astimezone(sp_tz)
                
                mensagem.extend([
                    "",
                    f"⏰ *Horário:* {sp_time.strftime('%d/%m/%Y %H:%M:%S')} (SP)"
                ])
            
            return "\n".join(mensagem)
            
        except Exception as e:
            self.logger.error(f"Erro ao formatar sinal: {str(e)}")
            self.logger.error(f"Sinal que causou erro: {str(sinal)}")
            return f"❌ Erro ao formatar mensagem: {str(e)}"
    
    def formatar_resultado(self, operacao: Dict) -> str:
        """Formata resultado de operação para Telegram"""
        try:
            # Emojis e formatação
            resultado_emoji = "✅" if operacao['resultado'] == 'WIN' else "❌"
            direcao_emoji = "🟢" if operacao['direcao'] == 'CALL' else "🔴"
            lucro_emoji = "💰" if operacao['resultado'] == 'WIN' else "💸"
    
            # Indicadores finais
            #indicadores = operacao.get('indicadores', {})
            #padroes_confirmados = operacao.get('padroes_confirmados', [])
            # Formata valores monetários
            preco_entrada = operacao.get('preco_entrada', 0)
            preco_saida = operacao.get('preco_saida', 0)
            # Análise pós-operação
            #variacao = abs(preco_saida - preco_entrada) / preco_entrada * 100
          
            mensagem = [
                f"{resultado_emoji} *RESULTADO OPERAÇÃO*",
                f"",
                f"{direcao_emoji} *Ativo:* {operacao['ativo'].replace('=X','')}",
                f"📈 *Direção:* {operacao['direcao']}",
                f"{lucro_emoji} *Resultado:* {operacao['resultado']}",
                f"",
                #f"📊 *Métricas da Operação:*",
                #f"• Duração: {tempo_operacao:.1f} min",
                #f"• Variação: {variacao:.2f}%",
                f"• Entrada: ${preco_entrada}",
                f"• Saída: ${preco_saida}",
                f"",
                #f"🔍 *Análise Final:*",
                #f"• Score Inicial: {operacao.get('score_entrada', 0):.1f}%",
                #f"• Assertividade Prevista: {operacao.get('assertividade_prevista', 0):.1f}%",
                f"• Id Sinal: {operacao['id']}",
            ]
            
            # Adiciona padrões confirmados se houver
            #if padroes_confirmados:
            #    mensagem.extend([
            #        f"",
            #        f"✨ *Padrões Confirmados:*",
            #        "• " + ", ".join(padroes_confirmados)
            #    ])

            return "\n".join(mensagem)
            
        except Exception as e:
            self.logger.error(f"Erro ao formatar resultado: {str(e)}")
            return ""
