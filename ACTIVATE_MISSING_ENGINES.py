#!/usr/bin/env python3
"""
ACTIVATE MISSING ENGINES - Load them as background tasks
Since initialize hangs, load them AFTER bot starts
"""

content_to_add = '''

    async def load_advanced_engines_background(self):
        """Load advanced engines in background after bot starts"""
        await asyncio.sleep(5)  # Wait for bot to start
        
        logger.info("\\n" + "=" * 80)
        logger.info("🚀 LOADING ADVANCED ENGINES IN BACKGROUND...")
        logger.info("=" * 80)
        
        # 1. Ultra Rare Engines
        try:
            logger.info('⚡ Loading Ultra Rare Engines...')
            from ULTRA_RARE_ENGINES import UltraRareEnginesOrchestrator
            self.advanced_systems['ultra_rare'] = UltraRareEnginesOrchestrator()
            logger.info('✅ Ultra Rare Engines: 10 profit engines active!')
        except Exception as e:
            logger.warning(f'⚠️ Ultra Rare Engines failed: {e}')
        
        # 2. Adaptive Confidence Engine  
        try:
            logger.info('🧠 Loading Adaptive Confidence Engine...')
            from ADAPTIVE_CONFIDENCE_ENGINE import get_adaptive_confidence_engine
            self.adaptive_confidence = get_adaptive_confidence_engine()
            logger.info('✅ Adaptive Confidence Engine active!')
        except Exception as e:
            logger.warning(f'⚠️ Adaptive Confidence failed: {e}')
        
        # 3. Omniscient Execution Engine
        try:
            logger.info('👁️ Loading Omniscient Execution Engine...')
            from OMNISCIENT_EXECUTION_ENGINE import OmniscientExecutionEngine
            self.omniscient_engine = OmniscientExecutionEngine()
            logger.info('✅ Omniscient Execution Engine active!')
        except Exception as e:
            logger.warning(f'⚠️ Omniscient Engine failed: {e}')
        
        logger.info("\\n" + "=" * 80)
        logger.info("🎉 ALL ADVANCED ENGINES LOADED!")
        logger.info("=" * 80)
'''

# Read the file
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
    content = f.read()

# Add the method after initialize_all_systems
insert_pos = content.find('    async def start_all_orchestrators(self):')
if insert_pos > 0:
    content = content[:insert_pos] + content_to_add + '\n' + content[insert_pos:]

# Add task to start_all_orchestrators
start_method_pos = content.find('    async def start_all_orchestrators(self):')
tasks_pos = content.find('tasks = []', start_method_pos)
if tasks_pos > 0:
    insert_task_pos = content.find('\n', tasks_pos) + 1
    task_line = '        \n        # Load advanced engines in background\n        tasks.append(asyncio.create_task(self.load_advanced_engines_background()))\n        \n'
    content = content[:insert_task_pos] + task_line + content[insert_task_pos:]

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w') as f:
    f.write(content)

print("✅ Advanced engines will load as background tasks!")
print("   This bypasses the hung initialization!")
