# Deduplication & Integration Plan

## Analysis of Duplicates - Keeping Best Versions

### Critical Files to Deduplicate

#### 1. router.py (4 versions)
- `./router.py` (49,242 bytes) ← **KEEP** - Most complete
- `./traders_core/router.py` (8,343 bytes) - Delete
- `./traders_core/execution/router.py` (6,824 bytes) - Delete  
- `./src/leantrader/execution/router.py` (979 bytes) - Delete

#### 2. pipeline.py (6 versions)
- `./strategies/pipeline.py` (14,128 bytes) ← **KEEP** - Most complete
- `./tools/pipeline.py` (10,765 bytes) - Might have different purpose, analyze
- `./src/leantrader/ta/pipeline.py` (5,434 bytes) - Delete
- `./features/pipeline.py` (2,837 bytes) - Delete
- `./traders_core/features/pipeline.py` (1,353 bytes) - Delete
- `./core/features/pipeline.py` (751 bytes) - Delete

#### 3. app.py (4 versions)
- `./src/leantrader/api/app.py` (20,653 bytes) ← **KEEP** - Most complete
- `./leantrader/src/leantrader/api/app.py` (5,249 bytes) - Delete
- `./_incoming/bundle/leantrader/src/leantrader/api/app.py` (5,381 bytes) - Delete
- `./traders_core/api/app.py` (1,040 bytes) - Delete

#### 4. guards.py (4 versions)
- `./w3guard/guards.py` (15,152 bytes) ← **KEEP** - Most complete
- `./web3_local/guards.py` (2,948 bytes) - Delete
- `./risk/guards.py` (2,881 bytes) - Delete
- `./web3/guards.py` (2,553 bytes) - Delete

#### 5. notifier.py (4 versions)
- `./notifier.py` (6,940 bytes) ← **KEEP** - Most complete
- `./src/leantrader/live/notifier.py` (3,881 bytes) - Delete
- `./leantrader/src/leantrader/live/notifier.py` (2,045 bytes) - Delete
- `./_incoming/bundle/leantrader/src/leantrader/live/notifier.py` (1,717 bytes) - Delete

#### 6. metrics.py (4 versions)
- `./observability/metrics.py` (6,863 bytes) ← **KEEP** - Most complete
- `./traders_core/observability/metrics.py` (1,207 bytes) - Delete
- `./src/leantrader/learn/metrics.py` (927 bytes) - Delete
- `./src/leantrader/backtest/metrics.py` (350 bytes) - Delete

## Integration Dependencies

### Core Files Needed by Engines

1. **ultra_arbitrage_engine.py** needs:
   - `ultra_core.py` - Core functionality
   - `risk_engine.py` - Risk management
   - `scanners/arbitrage.py` - Arbitrage scanning

2. **ultra_scalping_engine.py** needs:
   - `ultra_core.py` - Core functionality
   - `risk_engine.py` - Risk management
   - `pattern_memory.py` - Pattern storage
   - `brain.py` - Decision making

3. **ultra_moon_spotter.py** needs:
   - `aiohttp` - HTTP requests
   - `web3` - Blockchain interaction

4. **EVOLUTION_ENGINE.py** needs:
   - TensorFlow, LangChain, Redis, Celery
   - AI model infrastructure

5. **enhanced_trading_bot.py** needs:
   - `ccxt` - Exchange connections
   - `telegram` - Notifications
   - `sklearn` - ML models

6. **REAL_PROFIT_BOT.py** needs:
   - `ccxt` - Gate.io connection
   - Already self-contained

## Action Items

### Phase 1: Find Missing Dependencies
1. Check if `ultra_core.py` exists
2. Check if `risk_engine.py` exists
3. Check if `pattern_memory.py` exists
4. Check if `brain.py` exists

### Phase 2: Deduplicate
1. Backup duplicates to `_duplicates_backup/`
2. Delete inferior versions
3. Update imports in files that reference deleted files

### Phase 3: Wire Dependencies
1. Create missing dependency files if needed
2. Update import statements
3. Ensure proper initialization order

### Phase 4: Create Proper Orchestrator
1. Initialize each engine with correct dependencies
2. Share common resources (exchanges, risk engine)
3. Coordinate execution
4. Handle signal aggregation
5. Manage risk across all engines

### Phase 5: Test
1. Test each engine individually
2. Test orchestrator with all engines
3. Verify no circular dependencies
4. Check memory usage

## Next Steps

1. Check for existence of core dependencies
2. Analyze if they're complete or stubs
3. Create deduplication script
4. Execute deduplication
5. Build proper integration
