#!/usr/bin/env python3
"""
ULTIMATE ORCHESTRATOR - EVERYTHING WIRED PROPERLY
All systems working in TRUE fluid unison with complete data flows
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

# Import core orchestrator
from COMPLETE_UNIFIED_ORCHESTRATOR import (
    CompleteUnifiedOrchestrator,
    CentralDataHub,
    LearningOrchestrator,
    ScoutingOrchestrator,
    UnifiedDecisionEngine
)

# Import enhanced data flows
from ENHANCED_DATA_FLOWS import (
    RealTimeLearningPipeline,
    UnifiedScoutingPipeline,
    CollectiveIntelligenceCoordinator,
    UnifiedReportingSystem
)


class UltimateOrchestrator(CompleteUnifiedOrchestrator):
    """
    ULTIMATE ORCHESTRATOR
    Complete system with all enhancements:
    - Real-time learning from trades
    - Unified scouting with aggregation
    - Collective intelligence decision making
    - Comprehensive reporting
    - Fluid unison operation
    """
    
    def __init__(self, mode: str = "testnet"):
        super().__init__(mode)
        
        # Enhanced components
        self.learning_pipeline = None
        self.scouting_pipeline = None
        self.intelligence_coordinator = None
        self.reporting_system = UnifiedReportingSystem()
        
        logger.info("🚀 Ultimate Orchestrator initialized")
    
    async def wire_all_systems(self):
        """Enhanced wiring with complete data flows"""
        
        # First do base wiring
        await super().wire_all_systems()
        
        logger.info("\n🔌 ENHANCING WITH COMPLETE DATA FLOWS...")
        
        # Add real-time learning pipeline
        self.learning_pipeline = RealTimeLearningPipeline(
            self.ai_systems,
            self.ledger
        )
        logger.info("✅ Real-time Learning Pipeline wired")
        
        # Add unified scouting pipeline
        self.scouting_pipeline = UnifiedScoutingPipeline(
            self.trading_engines,
            self.pattern_memory
        )
        logger.info("✅ Unified Scouting Pipeline wired")
        
        # Add collective intelligence coordinator
        self.intelligence_coordinator = CollectiveIntelligenceCoordinator(
            self.ai_systems,
            self.brain,
            self.awareness
        )
        logger.info("✅ Collective Intelligence Coordinator wired")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 COMPLETE DATA FLOWS ESTABLISHED")
        logger.info("=" * 80)
    
    async def enhanced_trading_loop(self):
        """Enhanced main loop with complete orchestration"""
        logger.info("\n🔄 ENHANCED TRADING LOOP ACTIVE...")
        
        cycle = 0
        
        while self.is_running:
            try:
                cycle += 1
                cycle_start = datetime.now()
                
                logger.info(f"\n{'━' * 80}")
                logger.info(f"🔄 UNIFIED CYCLE {cycle} - COMPLETE ORCHESTRATION")
                logger.info(f"{'━' * 80}")
                
                # 1. UNIFIED SCOUTING (parallel across all engines)
                logger.info("🔭 Phase 1: Unified Scouting...")
                findings = await self.scouting_pipeline.aggregate_all_scouting()
                
                if findings:
                    logger.info(f"   ✅ Found {len(findings)} opportunities")
                    await self.reporting_system.report_signal({'count': len(findings)})
                    
                    # Publish all findings to data hub
                    for finding in findings:
                        await self.data_hub.publish_signal(finding)
                
                # 2. COLLECTIVE DECISION MAKING
                logger.info("🧠 Phase 2: Collective Intelligence...")
                
                # Get high priority findings
                priority_findings = self.scouting_pipeline.get_high_priority_findings()
                
                if priority_findings:
                    logger.info(f"   ✅ {len(priority_findings)} high priority signals")
                    
                    # Make collective decision for each
                    for finding in priority_findings[:5]:  # Top 5 only
                        decision = await self.intelligence_coordinator.make_collective_decision(finding)
                        
                        action = decision.get('action', 'hold')
                        confidence = decision.get('confidence', 0)
                        
                        logger.info(f"   🎯 Decision: {action.upper()} (confidence: {confidence:.2f})")
                        
                        if action in ['buy', 'sell'] and confidence > 0.7:
                            logger.info(f"   ⚡ HIGH CONFIDENCE SIGNAL - Ready for execution")
                
                # 3. LEARNING FROM RECENT TRADES
                logger.info("🎓 Phase 3: Learning & Evolution...")
                
                # Get recent trades from ledger
                if len(self.data_hub.recent_trades) > 0:
                    recent_trade = list(self.data_hub.recent_trades)[-1]
                    await self.learning_pipeline.process_trade_for_learning(recent_trade)
                    logger.info(f"   ✅ Learning from {len(self.data_hub.recent_trades)} recent trades")
                
                # 4. SYSTEM STATUS
                logger.info("📊 Phase 4: System Status...")
                
                logger.info(f"   Data Hub:")
                logger.info(f"      • Market Data: {len(self.data_hub.recent_market_data)} recent")
                logger.info(f"      • Signals: {len(self.data_hub.recent_signals)} recent")
                logger.info(f"      • Trades: {len(self.data_hub.recent_trades)} recent")
                logger.info(f"      • Learning Buffer: {len(self.learning_pipeline.training_buffer)} samples")
                logger.info(f"      • Findings Buffer: {len(self.scouting_pipeline.findings_buffer)} findings")
                
                # Swarm status
                if self.ai_systems.get('swarm'):
                    try:
                        consensus = await self.ai_systems['swarm'].collective_decision()
                        if consensus:
                            logger.info(f"   🧠 Swarm: {consensus.get('agent_count', 0)} agents, confidence {consensus.get('confidence', 0):.2f}")
                    except:
                        pass
                
                # Cycle metrics
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                
                # Report cycle completion
                await self.reporting_system.report_cycle(cycle, {
                    'signals_count': len(findings) if findings else 0,
                    'decisions_count': len(priority_findings) if priority_findings else 0,
                    'learning_count': len(self.learning_pipeline.training_buffer),
                    'duration': cycle_duration
                })
                
                logger.info(f"✅ Cycle {cycle} complete in {cycle_duration:.2f}s")
                logger.info(f"{'━' * 80}\n")
                
                await asyncio.sleep(60)  # Main cycle every 60 seconds
                
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Enhanced loop error: {e}")
                await asyncio.sleep(60)
    
    async def start(self):
        """Start the ultimate orchestrator"""
        try:
            # Initialize all systems
            await self.initialize_all_systems()
            
            # Wire everything with enhancements
            await self.wire_all_systems()
            
            # Start all background orchestrators
            background_tasks = []
            
            # Learning loop
            if self.orchestrators.get('learning'):
                background_tasks.append(
                    asyncio.create_task(self.orchestrators['learning'].run_learning_loop())
                )
            
            # Decision loop
            if self.orchestrators.get('decision'):
                background_tasks.append(
                    asyncio.create_task(self.orchestrators['decision'].run_decision_loop())
                )
            
            # Enhanced main loop
            background_tasks.append(
                asyncio.create_task(self.enhanced_trading_loop())
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("🎉 ALL SYSTEMS RUNNING IN COMPLETE UNISON")
            logger.info("=" * 80)
            
            # Run all tasks
            await asyncio.gather(*background_tasks)
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            logger.info("👋 Ultimate Orchestrator shutting down...")


async def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['testnet', 'paper', 'live'], 
                       default='testnet')
    args = parser.parse_args()
    
    orchestrator = UltimateOrchestrator(mode=args.mode)
    await orchestrator.start()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║              ULTIMATE ORCHESTRATOR - FULL INTEGRATION             ║
    ║                                                                   ║
    ║  ✅ Real-time learning from trades                                ║
    ║  ✅ Unified scouting across all engines                           ║
    ║  ✅ Collective intelligence decision making                       ║
    ║  ✅ Complete information flow & reporting                         ║
    ║  ✅ All systems in TRUE fluid unison                              ║
    ║                                                                   ║
    ║              EVERYTHING CONNECTED - LET IT EVOLVE                 ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
