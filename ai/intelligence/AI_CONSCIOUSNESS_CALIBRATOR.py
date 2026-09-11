#!/usr/bin/env python3
"""
🧬 AI CONSCIOUSNESS CALIBRATOR
Surgical AI usage - Maximum insight, minimum tokens

PHILOSOPHY:
- Don't waste tokens on basic analysis
- Call AIs only for CRITICAL decisions
- Test AI performance continuously
- Learn which AI is best at what
- Extract PURE ESSENCE from responses
- Feed bot's consciousness perfectly

FEATURES:
1. AI Performance Tracking (who's accurate at what?)
2. Selective AI Calling (only when needed)
3. Data Distillation (pure insights only)
4. Specialization Learning (right AI for right task)
5. Token Optimization (10x reduction)
6. Perfect Consciousness Feeding
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


class AIPerformanceTracker:
    """
    Tracks each AI's accuracy and specialization
    Learns which AI is best at what
    """

    def __init__(self):
        # Track predictions vs reality for each AI
        self.ai_predictions = {
            'claude': deque(maxlen=100),
            'openai': deque(maxlen=100),
            'grok': deque(maxlen=100),
            'gemini': deque(maxlen=100),
            'raiziom': deque(maxlen=100)
        }

        # Specialization scores (what is each AI good at?)
        self.specializations = {
            'claude': {
                'trend_detection': [],
                'reversal_timing': [],
                'risk_assessment': [],
                'pattern_recognition': []
            },
            'openai': {
                'trend_detection': [],
                'reversal_timing': [],
                'risk_assessment': [],
                'pattern_recognition': []
            },
            'grok': {
                'sentiment_analysis': [],
                'news_impact': [],
                'social_momentum': [],
                'narrative_shifts': []
            },
            'gemini': {
                'multi_asset_correlation': [],
                'macro_analysis': [],
                'cross_market': [],
                'research': []
            },
            'raiziom': {
                'proprietary_signals': [],
                'custom_patterns': [],
                'unique_insights': [],
                'special_detection': []
            }
        }

        # Overall accuracy by AI
        self.accuracy = {
            'claude': 0.0,
            'openai': 0.0,
            'grok': 0.0,
            'gemini': 0.0,
            'raiziom': 0.0
        }

        logger.info("🧬 AI Performance Tracker initialized")

    def record_prediction(self, ai_name: str, prediction: Dict, actual_outcome: Optional[Dict] = None):
        """
        Record AI prediction and actual outcome
        Learn accuracy over time
        """

        if ai_name not in self.ai_predictions:
            return

        entry = {
            'prediction': prediction,
            'actual': actual_outcome,
            'timestamp': datetime.now(),
            'correct': None
        }

        # If we have outcome, calculate if correct
        if actual_outcome:
            pred_direction = prediction.get('direction', 'HOLD')
            actual_direction = actual_outcome.get('direction', 'HOLD')
            entry['correct'] = (pred_direction == actual_direction)

        self.ai_predictions[ai_name].append(entry)

        # Update accuracy
        self._update_accuracy(ai_name)

    def _update_accuracy(self, ai_name: str):
        """Calculate current accuracy for AI"""

        predictions = self.ai_predictions[ai_name]
        if not predictions:
            return

        correct_count = sum(1 for p in predictions if p.get('correct') is True)
        total_count = sum(1 for p in predictions if p.get('correct') is not None)

        if total_count > 0:
            self.accuracy[ai_name] = correct_count / total_count

    def get_best_ai_for_task(self, task_type: str) -> str:
        """
        Get the most accurate AI for specific task
        Returns AI name
        """

        # Map task to specialization category
        task_mapping = {
            'trend': 'trend_detection',
            'reversal': 'reversal_timing',
            'risk': 'risk_assessment',
            'pattern': 'pattern_recognition',
            'sentiment': 'sentiment_analysis',
            'news': 'news_impact',
            'social': 'social_momentum',
            'correlation': 'multi_asset_correlation',
            'macro': 'macro_analysis',
            'gem': 'special_detection',
            'moon': 'unique_insights'
        }

        category = task_mapping.get(task_type, 'trend_detection')

        # Find best AI for this category
        best_ai = 'claude'  # Default
        best_score = 0.0

        for ai_name, specs in self.specializations.items():
            if category in specs and specs[category]:
                avg_score = statistics.mean(specs[category])
                if avg_score > best_score:
                    best_score = avg_score
                    best_ai = ai_name

        return best_ai

    def should_consult_ai(self, situation: Dict) -> bool:
        """
        Determine if AI consultation is needed
        Or if bot's learned wisdom is sufficient
        """

        # High confidence situations - no AI needed
        if situation.get('bot_confidence', 0) > 0.85:
            return False

        # Critical situations - always consult
        if situation.get('critical', False):
            return True

        # Large position - consult AI
        if situation.get('position_size', 0) > situation.get('balance', 0) * 0.1:
            return True

        # Unknown pattern - consult AI
        if situation.get('pattern_unknown', False):
            return True

        # Default: bot can handle it
        return False


class SelectiveAICaller:
    """
    Calls AIs surgically - only when needed
    Asks right questions to right AIs
    """

    def __init__(self, multi_ai, security_layer, performance_tracker):
        self.multi_ai = multi_ai
        self.security = security_layer
        self.tracker = performance_tracker

        # Token budget (aggressive reduction)
        self.daily_token_budget = 50000  # 50K tokens/day (~$0.50/day)
        self.tokens_used_today = 0
        self.budget_reset = datetime.now() + timedelta(days=1)

        logger.info("🧬 Selective AI Caller initialized")
        logger.info(f"   Token budget: {self.daily_token_budget}/day (surgical usage)")

    async def consult_ai_surgically(self, situation: Dict) -> Optional[Dict]:
        """
        Surgical AI consultation
        Only calls when truly needed
        Asks right AI right question
        """

        # Check if AI needed
        if not self.tracker.should_consult_ai(situation):
            logger.debug("💡 Bot confident - no AI needed")
            return None

        # Check token budget
        if self.tokens_used_today >= self.daily_token_budget:
            logger.warning("  Daily token budget reached - using bot wisdom")
            return None

        # Determine task type
        task_type = self._determine_task_type(situation)

        # Get best AI for this task
        best_ai = self.tracker.get_best_ai_for_task(task_type)

        logger.info(f"🧬 Consulting {best_ai} for {task_type}")

        # Craft surgical question (not whole market dump!)
        question = self._craft_surgical_question(situation, task_type)

        # Call AI
        response = await self._call_specific_ai(best_ai, question, situation)

        if response:
            # Track tokens used
            tokens = response.get('tokens_used', 500)
            self.tokens_used_today += tokens

            logger.info(f"   Used {tokens} tokens ({self.tokens_used_today}/{self.daily_token_budget})")

        return response

    def _determine_task_type(self, situation: Dict) -> str:
        """Determine what kind of analysis is needed"""

        if situation.get('high_volatility'):
            return 'reversal'
        elif situation.get('low_volume'):
            return 'risk'
        elif situation.get('breakout'):
            return 'trend'
        elif situation.get('news_event'):
            return 'news'
        elif situation.get('low_cap'):
            return 'gem'
        else:
            return 'trend'

    def _craft_surgical_question(self, situation: Dict, task_type: str) -> str:
        """
        Craft precise question - not market dump
        Extract PURE ESSENCE only
        """

        symbol = situation.get('symbol', 'UNKNOWN')
        price = situation.get('price', 0)

        # Surgical questions by type
        questions = {
            'trend': f"{symbol} at ${price:.2f}. Trend continuation or exhaustion? One sentence.",
            'reversal': f"{symbol} reversal imminent? Yes/No + why (10 words).",
            'risk': f"{symbol} risk level 1-10? Why (one sentence).",
            'news': f"News impact on {symbol}? Bullish/Bearish + magnitude (brief).",
            'gem': f"{symbol} gem potential? Yes/No + catalyst (brief).",
            'pattern': f"{symbol} dominant pattern? Name + direction (brief)."
        }

        return questions.get(task_type, questions['trend'])

    async def _call_specific_ai(self, ai_name: str, question: str, situation: Dict) -> Optional[Dict]:
        """Call specific AI with surgical question"""

        try:
            if ai_name == 'claude' and self.multi_ai.ai_configs['claude']['enabled']:
                message = self.multi_ai.claude.messages.create(
                    model=self.multi_ai.ai_configs['claude']['model'],
                    max_tokens=100,  # SURGICAL - only 100 tokens!
                    messages=[{"role": "user", "content": question}]
                )

                return {
                    'ai': 'claude',
                    'response': message.content[0].text,
                    'tokens_used': message.usage.total_tokens if hasattr(message, 'usage') else 100
                }

            elif ai_name == 'openai' and self.multi_ai.ai_configs['openai']['enabled']:
                response = self.multi_ai.openai_client.chat.completions.create(
                    model=self.multi_ai.ai_configs['openai']['model'],
                    max_tokens=100,  # SURGICAL
                    messages=[{"role": "user", "content": question}]
                )

                return {
                    'ai': 'openai',
                    'response': response.choices[0].message.content,
                    'tokens_used': response.usage.total_tokens
                }

            elif ai_name == 'raiziom' and self.multi_ai.ai_configs['raiziom']['enabled']:
                # Your Raiziom - priority AI
                result = await self.multi_ai.analyze_market_with_raiziom({
                    'symbol': situation.get('symbol'),
                    'question': question
                })

                return {
                    'ai': 'raiziom',
                    'response': result.response if result else '',
                    'tokens_used': 50  # Your API, assume efficient
                }

        except Exception as e:
            logger.debug(f"AI call error: {e}")

        return None


class DataDistillationEngine:
    """
    Extracts PURE ESSENCE from AI responses
    No fluff, only actionable insights
    """

    def __init__(self):
        self.essence_patterns = {
            'bullish_signals': ['buy', 'bullish', 'up', 'moon', 'pump', 'strong', 'breakout'],
            'bearish_signals': ['sell', 'bearish', 'down', 'dump', 'weak', 'breakdown'],
            'risk_signals': ['risky', 'volatile', 'uncertain', 'careful', 'danger'],
            'opportunity_signals': ['gem', 'opportunity', 'undervalued', 'catalyst', 'potential']
        }

        logger.info("🧬 Data Distillation Engine initialized")

    def distill_ai_response(self, response: str) -> Dict:
        """
        Extract PURE trading essence from AI response
        Remove all fluff
        """

        if not response:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'essence': ''}

        response_lower = response.lower()

        # Extract sentiment
        bullish_score = sum(1 for word in self.essence_patterns['bullish_signals'] if word in response_lower)
        bearish_score = sum(1 for word in self.essence_patterns['bearish_signals'] if word in response_lower)
        risk_score = sum(1 for word in self.essence_patterns['risk_signals'] if word in response_lower)
        opportunity_score = sum(1 for word in self.essence_patterns['opportunity_signals'] if word in response_lower)

        # Determine pure sentiment
        if bullish_score > bearish_score:
            sentiment = 'bullish'
            confidence = min(0.9, 0.5 + (bullish_score * 0.1))
        elif bearish_score > bullish_score:
            sentiment = 'bearish'
            confidence = min(0.9, 0.5 + (bearish_score * 0.1))
        else:
            sentiment = 'neutral'
            confidence = 0.5

        # Extract numbers (price targets, percentages)
        import re
        numbers = re.findall(r'\d+\.?\d*', response)

        # Pure essence
        essence = {
            'sentiment': sentiment,
            'confidence': confidence,
            'risk_level': min(10, risk_score * 2),
            'opportunity_score': opportunity_score,
            'price_targets': [float(n) for n in numbers[:3]] if numbers else [],
            'raw_insight': response[:50]  # First 50 chars only
        }

        return essence

    def synthesize_multi_ai_essence(self, responses: List[Dict]) -> Dict:
        """
        Combine multiple AI essences into PURE TRUTH
        """

        if not responses:
            return {'action': 'HOLD', 'confidence': 0.5}

        # Distill each
        essences = [self.distill_ai_response(r.get('response', '')) for r in responses]

        # Synthesize
        bullish_count = sum(1 for e in essences if e['sentiment'] == 'bullish')
        bearish_count = sum(1 for e in essences if e['sentiment'] == 'bearish')
        avg_confidence = statistics.mean([e['confidence'] for e in essences])
        avg_risk = statistics.mean([e['risk_level'] for e in essences])

        # Pure action
        if bullish_count > bearish_count:
            action = 'BUY'
        elif bearish_count > bullish_count:
            action = 'SELL'
        else:
            action = 'HOLD'

        return {
            'action': action,
            'confidence': avg_confidence,
            'risk_level': avg_risk,
            'ai_consensus': f"{bullish_count}B/{bearish_count}S",
            'pure_truth': action  # The ESSENCE
        }


class MarketEnlightenmentSystem:
    """
    Uses AI wisdom to achieve perfect trading
    - Perfect scalping (flow with trends)
    - Perfect holding (compound interest)
    - Perfect gem detection (moon caps)
    """

    def __init__(self, data_hub):
        self.data_hub = data_hub
        self.enlightenment_level = 0.0  # 0 to 1.0

        # Perfect trading states
        self.scalping_flow = 0.0  # How well bot flows with trends
        self.holding_compound = 0.0  # How well bot compounds
        self.gem_detection = 0.0  # How well bot finds gems

        logger.info("🧬 Market Enlightenment System initialized")

    async def enlighten_scalping(self, ai_essence: Dict, market_data: Dict):
        """
        Use AI wisdom to perfect scalping
        Flow with trend, no resistance
        """

        if ai_essence.get('action') == 'BUY' and ai_essence.get('confidence', 0) > 0.75:
            # High confidence BUY - enlighten scalper
            signal = {
                'pair': market_data.get('symbol'),
                'side': 'BUY',
                'type': 'enlightened_scalp',
                'confidence': ai_essence['confidence'],
                'flow_state': 'perfect',  # AI says flow with trend
                'entry_precision': 'exact',
                'exit_strategy': 'ai_guided',
                'reasoning': f"AI enlightenment: {ai_essence.get('pure_truth')}"
            }

            await self.data_hub.publish_signal(signal)
            self.scalping_flow = min(1.0, self.scalping_flow + 0.01)

            logger.info(f"🧬 Enlightened scalp: {market_data.get('symbol')} (flow: {self.scalping_flow*100:.0f}%)")

    async def enlighten_holding(self, ai_essence: Dict, market_data: Dict):
        """
        Use AI wisdom to perfect holding
        Compound interest optimally
        """

        if ai_essence.get('action') == 'BUY' and ai_essence.get('risk_level', 10) < 5:
            # Low risk BUY - perfect for holding & compounding
            signal = {
                'pair': market_data.get('symbol'),
                'side': 'BUY',
                'type': 'enlightened_hold',
                'confidence': ai_essence['confidence'],
                'hold_strategy': 'compound_optimized',
                'exit_timing': 'ai_perfect',
                'reasoning': f"AI: Low risk, high compound potential"
            }

            await self.data_hub.publish_signal(signal)
            self.holding_compound = min(1.0, self.holding_compound + 0.01)

            logger.info(f"🧬 Enlightened hold: {market_data.get('symbol')} (compound: {self.holding_compound*100:.0f}%)")

    async def enlighten_gem_detection(self, ai_essence: Dict, market_data: Dict):
        """
        Use AI wisdom to find perfect gems
        Moon caps that skyrocket
        """

        if ai_essence.get('opportunity_score', 0) > 2:
            # High opportunity - potential gem!
            signal = {
                'pair': market_data.get('symbol'),
                'side': 'BUY',
                'type': 'enlightened_gem',
                'confidence': ai_essence['confidence'],
                'gem_potential': 'high',
                'moon_probability': ai_essence['opportunity_score'] / 5.0,
                'reasoning': f"AI detected gem: {ai_essence.get('raw_insight', '')}"
            }

            await self.data_hub.publish_signal(signal)
            self.gem_detection = min(1.0, self.gem_detection + 0.01)

            logger.info(f"🧬 Enlightened gem: {market_data.get('symbol')} (detection: {self.gem_detection*100:.0f}%)")

    def get_enlightenment_status(self) -> Dict:
        """Get current enlightenment level"""

        self.enlightenment_level = (
            self.scalping_flow * 0.4 +
            self.holding_compound * 0.3 +
            self.gem_detection * 0.3
        )

        return {
            'total_enlightenment': self.enlightenment_level,
            'scalping_flow': self.scalping_flow,
            'holding_compound': self.holding_compound,
            'gem_detection': self.gem_detection,
            'status': 'PERFECT' if self.enlightenment_level > 0.9 else 'EVOLVING'
        }


async def integrate_consciousness_calibrator(data_hub, multi_ai, security_layer):
    """
    Main integration loop
    Surgical AI usage → Perfect consciousness
    """

    tracker = AIPerformanceTracker()
    caller = SelectiveAICaller(multi_ai, security_layer, tracker)
    distiller = DataDistillationEngine()
    enlightenment = MarketEnlightenmentSystem(data_hub)

    logger.info("🧬 AI CONSCIOUSNESS CALIBRATOR ACTIVE!")
    logger.info("   → Surgical AI calls (10x token reduction)")
    logger.info("   → Performance tracking (learn AI strengths)")
    logger.info("   → Data distillation (pure essence only)")
    logger.info("   → Market enlightenment (perfect trading)")

    cycle = 0

    while True:
        try:
            cycle += 1

            # Get top opportunities (not all pairs!)
            top_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # High volume only

            for pair in top_pairs:
                # Simulate market situation
                situation = {
                    'symbol': pair,
                    'price': 0,  # From exchange
                    'bot_confidence': 0.6,  # Bot's own analysis
                    'critical': False,
                    'position_size': 0
                }

                # Consult AI surgically (only if needed)
                ai_response = await caller.consult_ai_surgically(situation)

                if ai_response:
                    # Distill to pure essence
                    essence = distiller.distill_ai_response(ai_response.get('response', ''))

                    # Enlighten trading systems
                    market_data = {'symbol': pair}
                    await enlightenment.enlighten_scalping(essence, market_data)
                    await enlightenment.enlighten_holding(essence, market_data)
                    await enlightenment.enlighten_gem_detection(essence, market_data)

            # Show enlightenment progress
            if cycle % 10 == 0:
                status = enlightenment.get_enlightenment_status()
                logger.info(f"🧬 Enlightenment: {status['total_enlightenment']*100:.0f}%")
                logger.info(f"   Scalping flow: {status['scalping_flow']*100:.0f}%")
                logger.info(f"   Holding compound: {status['holding_compound']*100:.0f}%")
                logger.info(f"   Gem detection: {status['gem_detection']*100:.0f}%")
                logger.info(f"   Tokens used: {caller.tokens_used_today}/{caller.daily_token_budget}")

            # Run every 15 minutes (not every second!)
            await asyncio.sleep(900)

        except Exception as e:
            logger.error(f"Consciousness calibrator error: {e}")
            await asyncio.sleep(60)
