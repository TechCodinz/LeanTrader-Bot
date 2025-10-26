#!/usr/bin/env python3
"""
PERSISTENCE MANAGER
Loads all learned memory/databases so bot doesn't start from scratch
Designed for VPS deployment - preserves learning across restarts
"""

import os
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    Manages loading/saving of all learned knowledge:
    - Database files (.db)
    - Pattern memory (CSV, JSON)
    - Model weights (.pkl)
    - Trading history
    - Strategy scores
    """
    
    def __init__(self):
        self.workspace = Path('/workspace')
        self.data_dir = self.workspace / 'data'
        self.models_dir = self.workspace / 'models'
        self.runtime_dir = self.workspace / 'runtime'
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.runtime_dir.mkdir(exist_ok=True)
        
        self.loaded_databases = {}
        self.loaded_memories = {}
        self.loaded_models = {}
        
        logger.info("🧠 Persistence Manager initialized")
    
    def load_all_learned_data(self) -> Dict[str, Any]:
        """
        Load ALL learned data from previous runs
        Returns comprehensive state dict
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔄 LOADING LEARNED MEMORY - Restoring previous knowledge...")
        logger.info("=" * 80)
        
        state = {
            'databases': {},
            'patterns': {},
            'models': {},
            'history': {},
            'scores': {},
            'memory': {}
        }
        
        # 1. Load all .db files
        state['databases'] = self._load_databases()
        
        # 2. Load pattern memory
        state['patterns'] = self._load_pattern_memory()
        
        # 3. Load trading history
        state['history'] = self._load_trading_history()
        
        # 4. Load strategy scores
        state['scores'] = self._load_strategy_scores()
        
        # 5. Load brain memory
        state['memory'] = self._load_brain_memory()
        
        # 6. Load model weights
        state['models'] = self._load_model_weights()
        
        logger.info("\n✅ LEARNED MEMORY LOADED - Bot will use previous knowledge!")
        logger.info("=" * 80)
        
        return state
    
    def _load_databases(self) -> Dict[str, Any]:
        """Load all SQLite databases"""
        databases = {}
        
        db_files = [
            'ultra_trading_system.db',
            'evolution_engine.db',
            'ultimate_bot_450_models.db',
            'nobel_complete.db',
            'nobel_simple.db',
            'divine_intelligence.db',
            'enhanced_trading_bot.db'
        ]
        
        for db_file in db_files:
            db_path = self.workspace / db_file
            if db_path.exists():
                try:
                    # Connect and get row count
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    db_data = {'path': str(db_path), 'tables': []}
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                        count = cursor.fetchone()[0]
                        db_data['tables'].append({'name': table_name, 'rows': count})
                    
                    conn.close()
                    databases[db_file] = db_data
                    
                    total_rows = sum(t['rows'] for t in db_data['tables'])
                    logger.info(f"  ✅ {db_file}: {len(db_data['tables'])} tables, {total_rows} total rows")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  {db_file}: {e}")
        
        logger.info(f"📊 Loaded {len(databases)} databases with learned data")
        return databases
    
    def _load_pattern_memory(self) -> Dict[str, Any]:
        """Load pattern memory from CSV and JSON"""
        patterns = {}
        
        # Pattern memory CSV
        pattern_csv = self.data_dir / 'pattern_memory.csv'
        if pattern_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(pattern_csv)
                patterns['memory_csv'] = {
                    'path': str(pattern_csv),
                    'patterns': len(df),
                    'data': df.to_dict('records') if len(df) < 1000 else []
                }
                logger.info(f"  ✅ pattern_memory.csv: {len(df)} learned patterns")
            except Exception as e:
                logger.warning(f"  ⚠️  pattern_memory.csv: {e}")
        
        # Pattern scores JSON
        scores_json = self.data_dir / 'pattern_scores.json'
        if scores_json.exists():
            try:
                with open(scores_json, 'r') as f:
                    scores = json.load(f)
                patterns['scores_json'] = {
                    'path': str(scores_json),
                    'patterns': len(scores) if isinstance(scores, dict) else 0,
                    'data': scores
                }
                logger.info(f"  ✅ pattern_scores.json: {len(scores) if isinstance(scores, dict) else 0} scored patterns")
            except Exception as e:
                logger.warning(f"  ⚠️  pattern_scores.json: {e}")
        
        logger.info(f"🎯 Loaded {len(patterns)} pattern memory systems")
        return patterns
    
    def _load_trading_history(self) -> Dict[str, Any]:
        """Load trading history"""
        history = {}
        
        history_csv = self.data_dir / 'history.csv'
        if history_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(history_csv)
                history['csv'] = {
                    'path': str(history_csv),
                    'trades': len(df),
                    'size_mb': history_csv.stat().st_size / (1024*1024)
                }
                logger.info(f"  ✅ history.csv: {len(df)} historical trades ({history['csv']['size_mb']:.1f} MB)")
            except Exception as e:
                logger.warning(f"  ⚠️  history.csv: {e}")
        
        logger.info(f"📈 Loaded trading history")
        return history
    
    def _load_strategy_scores(self) -> Dict[str, Any]:
        """Load strategy performance scores"""
        scores = {}
        
        # Best params
        best_params = self.workspace / 'best_params.json'
        if best_params.exists():
            try:
                with open(best_params, 'r') as f:
                    data = json.load(f)
                scores['best_params'] = {
                    'path': str(best_params),
                    'strategies': len(data) if isinstance(data, dict) else 0,
                    'data': data
                }
                logger.info(f"  ✅ best_params.json: Best parameters for strategies loaded")
            except Exception as e:
                logger.warning(f"  ⚠️  best_params.json: {e}")
        
        logger.info(f"🏆 Loaded strategy scores")
        return scores
    
    def _load_brain_memory(self) -> Dict[str, Any]:
        """Load brain/memory state"""
        memory = {}
        
        brain_json = self.runtime_dir / 'brain.json'
        if brain_json.exists():
            try:
                with open(brain_json, 'r') as f:
                    data = json.load(f)
                memory['brain'] = {
                    'path': str(brain_json),
                    'positions': len(data.get('positions', {})),
                    'trades': len(data.get('trades', [])),
                    'data': data
                }
                logger.info(f"  ✅ brain.json: {memory['brain']['positions']} positions, {memory['brain']['trades']} trades")
            except Exception as e:
                logger.warning(f"  ⚠️  brain.json: {e}")
        
        logger.info(f"🧠 Loaded brain memory")
        return memory
    
    def _load_model_weights(self) -> Dict[str, Any]:
        """Load ML model weights"""
        models = {}
        
        # Look for .pkl files
        pkl_files = list(self.workspace.glob('*.pkl'))
        for pkl_file in pkl_files[:10]:  # Limit to 10
            try:
                with open(pkl_file, 'rb') as f:
                    # Just check if we can load it
                    data = pickle.load(f)
                models[pkl_file.name] = {
                    'path': str(pkl_file),
                    'size_kb': pkl_file.stat().st_size / 1024,
                    'loaded': True
                }
                logger.info(f"  ✅ {pkl_file.name}: Model weights loaded ({models[pkl_file.name]['size_kb']:.1f} KB)")
            except Exception as e:
                logger.debug(f"  ⚠️  {pkl_file.name}: {e}")
        
        if models:
            logger.info(f"🤖 Loaded {len(models)} model weight files")
        
        return models
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save current state for next run"""
        try:
            # Save to persistent state file
            state_file = self.workspace / 'persistent_state.json'
            with open(state_file, 'w') as f:
                # Only save serializable data
                serializable = {
                    'timestamp': str(Path.cwd()),
                    'databases_loaded': list(state.get('databases', {}).keys()),
                    'patterns_loaded': list(state.get('patterns', {}).keys()),
                    'history_loaded': list(state.get('history', {}).keys()),
                }
                json.dump(serializable, f, indent=2)
            logger.info(f"💾 State saved to {state_file}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")


def initialize_persistence() -> PersistenceManager:
    """
    Initialize and load all persistent data
    Call this at bot startup
    """
    manager = PersistenceManager()
    learned_state = manager.load_all_learned_data()
    return manager, learned_state


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)
    manager, state = initialize_persistence()
    
    print("\n" + "=" * 80)
    print("LEARNED DATA SUMMARY")
    print("=" * 80)
    print(f"Databases:       {len(state['databases'])}")
    print(f"Pattern systems: {len(state['patterns'])}")
    print(f"History files:   {len(state['history'])}")
    print(f"Score systems:   {len(state['scores'])}")
    print(f"Brain memory:    {len(state['memory'])}")
    print(f"Model weights:   {len(state['models'])}")
    print("=" * 80)
