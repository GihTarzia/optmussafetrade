import asyncio
from telegram import Bot
from typing import Dict, Optional
from datetime import datetime
from colorama import Fore, Style
import json
import logging
from collections import deque

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
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token
        self.chat_id = chat_id
        self.manager = NotificationManager()
        self.logger = logging.getLogger('Notificador')
        #self.logger = logger
        
        # Configurações
        self.max_retries = 3
        self.retry_delay = 5  # segundos
        self.rate_limit = 30  # mensagens por minuto
        self.rate_limit_period = 60  # segundos
        
        # Cache de mensagens recentes para evitar duplicatas
        self.recent_messages = deque(maxlen=100)
        
        # Inicializa bot
        try:
            self.bot = Bot(token) if token else None
            if self.bot:
                self.logger.info("Bot Telegram inicializado com sucesso")
            else:
                self.logger.warning("Bot Telegram não configurado")
        except Exception as e:
            self.logger.error(f"Erro ao criar Bot: {str(e)}")
            self.bot = None
            
        # Inicia worker de processamento
        asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Processa fila de notificações"""
        while True:
            try:
                # Verifica rate limit
                recent_sent = sum(1 for msg in self.manager.history 
                                if (datetime.now() - msg['sent_at']).total_seconds() < self.rate_limit_period)
                
                if recent_sent >= self.rate_limit:
                    await asyncio.sleep(1)
                    continue
                
                # Processa próxima notificação
                notification = await self.manager.get_next_notification()
                if notification:
                    success = await self._send_with_retry(notification)
                    if success:
                        self.manager.record_success(notification)
                    else:
                        self.manager.record_failure(notification, "Max retries exceeded")
                
                await asyncio.sleep(0.1)  # Previne CPU alta
                
            except Exception as e:
                self.logger.error(f"Erro no processamento da fila: {str(e)}")
                await asyncio.sleep(1)

    async def _send_with_retry(self, notification: Dict) -> bool:
        """Tenta enviar mensagem com retries"""
        for attempt in range(self.max_retries):
            try:
                if not self.bot or not self.chat_id:
                    return False
                
                message = notification['content']
                if isinstance(message, dict):
                    message = json.dumps(message, indent=2)
                
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                return True
                
            except Exception as e:
                self.logger.warning(f"Tentativa {attempt + 1} falhou: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                continue
        
        return False

    def _is_duplicate(self, mensagem: str) -> bool:
        """Verifica se mensagem é duplicata recente"""
        for recent in self.recent_messages:
            if mensagem == recent['content']:
                time_diff = (datetime.now() - recent['timestamp']).total_seconds()
                if time_diff < 60:  # Ignora duplicatas em 1 minuto
                    return True
        return False

    async def enviar_mensagem(self, mensagem: str) -> bool:
        """Envia mensagem para o Telegram"""
        try:
            # Verifica duplicata
            if self._is_duplicate(mensagem):
                self.logger.warning(f"Mensagem duplicada ignorada")
                return False
            
            # Adiciona à fila
            await self.manager.add_notification(mensagem)
            
            # Registra mensagem recente
            self.recent_messages.append({
                'content': mensagem,
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao enfileirar mensagem: {str(e)}")
            return False

    def formatar_sinal(self, sinal: Dict) -> str:
        """Formata sinal para mensagem do Telegram com visual aprimorado"""
        try:
            # Emojis e indicadores
            indicadores = sinal.get('indicadores', {})
            emoji_direcao = "🟢" if sinal['direcao'] == 'CALL' else "🔴"
            emoji_tendencia = {
                'CALL': '📈',
                'PUT': '📉',
                'NEUTRO': '↔️'
            }.get(indicadores.get('tendencia', 'NEUTRO'), '↔️')       

            # Calcula força do sinal
            score = float(sinal.get('score', 0))
            forca_sinal = "⭐" * max(1, min(5, int(score * 5))) 
            score_formatted = f"{'🟢' if score >= 0.7 else '🟡' if score >= 0.5 else '🔴'} {score*100:.1f}%"
            assertividade = float(sinal.get('assertividade', 50.0))
            assert_formatted = f"{'🟢' if assertividade >= 70 else '🟡' if assertividade >= 50 else '🔴'} {assertividade:.1f}%"

            # Formata indicadores
            prob_ml = indicadores.get('ml_prob', 0) * 100
            forca_padroes = indicadores.get('padroes_forca', 0) * 100   
            # NOVO: Formatação do Score Técnico
            tech_score = float(indicadores.get('tech_score', 0))
            tech_score_formatted = f"{'🟢' if tech_score >= 0.7 else '🟡' if tech_score >= 0.5 else '🔴'} {tech_score*100:.1f}%"

            # NOVO: Contagem de Confirmações por Padrão
            padroes = sinal.get('padroes', [])
            padroes_info = []
            for padrao in padroes:
                nome_padrao = padrao.get('nome', '')
                confirmacoes = padrao.get('confirmacoes', 0)
                padroes_info.append(f"{nome_padrao} ({confirmacoes}✓)")

            padroes_str = ", ".join(padroes_info) if padroes_info else "Sem padrões específicos"


            volume_score = indicadores.get('volume_ratio', 1.0)
            volume_emoji = "📊" if volume_score > 1.2 else "📈" if volume_score > 1.0 else "📉"

            # Momento do mercado
            momento_score = indicadores.get('momento_score', 0.5)
            momento_emoji = "🌟" if momento_score > 0.7 else "⭐" if momento_score > 0.5 else "☆"


            mensagem = [
                f"{'='*35}",
                f"{emoji_direcao} *SINAL DE {sinal['direcao']}* {emoji_tendencia}",
                f"{'='*35}",
                f"",
                f"🎯 *Ativo:* `{sinal['ativo'].replace('=X','')}`",
                f"⏰ *Horário Entrada:* {sinal['momento_entrada']}",
                f"⌛️ *Expiração:* {sinal['tempo_expiracao']} min",
                f"💲  *Valor:* {sinal['preco_entrada']}",
                f"",   
                f"📊 *ANÁLISE DO SINAL:* {forca_sinal}",
                f"➤ Score: {score_formatted}",
                f"➤ Assertividade: {assert_formatted}",
                f"➤ Score Técnico: {tech_score_formatted}",  # NOVO
                #f"",
                #f"📈 *INDICADORES TÉCNICOS:*",
                #f"➤ Prob. ML: {prob_ml:.1f}%",
                #f"➤ Força Padrões: {forca_padroes:.1f}%",
                #f"➤ Tendência: {indicadores.get('tendencia', 'NEUTRO')}",
                #f"",
                #f"🔍 *PADRÕES IDENTIFICADOS:*",
                #f"➤ {padroes_str}",  # Agora inclui contagem de confirmações
                #f"",
                #f"📊 *ANÁLISE DE MERCADO:*",
                #f"➤ Volume: {volume_emoji} {volume_score:.1f}x média",
                #f"➤ Momento: {momento_emoji} {momento_score:.1f}",
                #f"",
                #f"⚠️ *GESTÃO DE RISCO:*",
                #f"➤ Volatilidade: {float(sinal.get('volatilidade', 0))*100:.2f}%",
                f"➤ Id Sinal: {sinal['id']}",
            ]   

            return "\n".join(mensagem)  

        except Exception as e:
            self.logger.error(f"Erro ao formatar sinal: {str(e)}")
            return "Erro ao formatar mensagem"
    
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
            #tempo_operacao = (operacao.get('timestamp_saida', datetime.now()) - 
            #                 operacao.get('timestamp_entrada', datetime.now())).total_seconds() / 60
      
            mensagem = [
                f"{resultado_emoji} *RESULTADO OPERAÇÃO*",
                f"",
                f"{direcao_emoji} *Ativo:* {operacao['ativo'].replace('=X','')}",
                f"📈 *Direção:* {operacao['direcao']}",
                f"{lucro_emoji} *Resultado:* {operacao['resultado']}",
                #f"💵 *Lucro:* ${abs(operacao['lucro']):.2f}",
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
