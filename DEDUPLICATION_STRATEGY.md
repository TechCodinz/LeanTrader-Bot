# DEDUPLICATION STRATEGY

## Analysis Summary
- **47 duplicate filenames** found
- **Most critical**: router.py, pipeline.py, app.py, guards.py, notifier.py
- **_incoming/** directory contains 20 files that are duplicates of src/leantrader (pending import - DELETE)
- **backup_before_ultra_20250914_182511/** contains old ultra_scout.py (DELETE - we have newer version)

## DECISION RULES

### 1. Keep the LARGEST/MOST COMPLETE version
### 2. Keep root directory files over subdirectory files (they're usually the main ones)
### 3. Delete _incoming/ duplicates (already imported)
### 4. Delete backup/ duplicates (we have newer versions)

---

## DEDUPLICATION PLAN

### IMMEDIATE DELETIONS (No conflicts):

**_incoming directory (20 files) - ALL DUPLICATES:**
- Delete entire _incoming/bundle/leantrader/src/leantrader/ 
- Reason: Already exists in src/leantrader/ and leantrader/src/leantrader/

**backup directory (1 file):**
- Delete backup_before_ultra_20250914_182511/ultra_scout.py (628 lines)
- Keep: ./ultra_scout.py (927 lines) ✅

---

### FILE-BY-FILE DECISIONS:

#### pipeline.py (6 copies):
**KEEP**: strategies/pipeline.py (364 lines) - Most complete ✅
**DELETE**:
- core/features/pipeline.py (26 lines)
- features/pipeline.py (81 lines)  
- src/leantrader/ta/pipeline.py (156 lines)
- traders_core/features/pipeline.py (33 lines)
- tools/pipeline.py (289 lines) - Tools version, may need

**DECISION**: Keep strategies/pipeline.py AND tools/pipeline.py (different purposes)

#### router.py (4 copies):
**KEEP**: ./router.py (1,154 lines) - Main router ✅
**KEEP**: traders_core/router.py (236 lines) - Different purpose ✅
**KEEP**: traders_core/execution/router.py (204 lines) - Execution routing ✅
**KEEP**: src/leantrader/execution/router.py (25 lines) - Interface ✅
**DECISION**: All serve different purposes, KEEP ALL

#### app.py (4 copies):
**KEEP**: src/leantrader/api/app.py (567 lines) - Most complete API ✅
**DELETE**:
- traders_core/api/app.py (45 lines)
- leantrader/src/leantrader/api/app.py (144 lines)

#### guards.py (4 copies):
**KEEP**: w3guard/guards.py (451 lines) - Most complete ✅
**DELETE**:
- risk/guards.py (82 lines)
- web3/guards.py (87 lines)
- web3_local/guards.py (94 lines)

#### notifier.py (4 copies):
**KEEP**: ./notifier.py (190 lines) - Main notifier ✅
**DELETE**:
- src/leantrader/live/notifier.py (120 lines)
- leantrader/src/leantrader/live/notifier.py (61 lines)

#### metrics.py (4 copies):
**KEEP**: observability/metrics.py (264 lines) - Most complete ✅
**KEEP**: src/leantrader/backtest/metrics.py (10 lines) - Backtest specific ✅
**KEEP**: src/leantrader/learn/metrics.py (32 lines) - Learning specific ✅
**DELETE**:
- traders_core/observability/metrics.py (29 lines)

#### ta.py (4 copies):
**KEEP**: src/leantrader/features/ta.py (29 lines) - Most complete ✅
**DELETE**:
- leantrader/src/leantrader/features/ta.py (29 lines)
- traders_core/utils/ta.py (15 lines)

#### regime.py (3 copies):
**KEEP**: research/regime.py (59 lines) - Most complete ✅
**DELETE**:
- ./regime.py (16 lines)
- traders_core/research/regime.py (6 lines)

#### paper_broker.py (2 copies):
**KEEP**: ./paper_broker.py (398 lines) - Main version ✅
**DELETE**:
- traders_core/sim/paper_broker.py (108 lines)

#### mt5_adapter.py (2 copies):
**KEEP**: ./mt5_adapter.py (335 lines) - Most complete ✅
**DELETE**:
- traders_core/mt5_adapter.py (245 lines)

#### main.py (2 copies):
**KEEP**: ./main.py (244 lines) - Main entry point ✅
**KEEP**: traders_core/main.py (77 lines) - TraderCore entry ✅

#### auto_loop.py (2 copies):
**KEEP**: traders_core/auto_loop.py (137 lines) - More complete ✅
**DELETE**:
- ./auto_loop.py (145 lines) - Actually this is larger, REVERSE
**FINAL**: KEEP ./auto_loop.py (145 lines) ✅

#### All __init__.py files:
**KEEP ALL** - Each serves different module

---

## SUMMARY

**SAFE TO DELETE:**
- Entire _incoming/ directory (20 files)
- backup_before_ultra_20250914_182511/ultra_scout.py
- leantrader/src/leantrader/api/app.py
- traders_core/api/app.py
- risk/guards.py
- web3/guards.py  
- web3_local/guards.py
- src/leantrader/live/notifier.py
- leantrader/src/leantrader/live/notifier.py
- traders_core/observability/metrics.py
- leantrader/src/leantrader/features/ta.py
- traders_core/utils/ta.py
- ./regime.py
- traders_core/research/regime.py
- traders_core/sim/paper_broker.py
- traders_core/mt5_adapter.py
- core/features/pipeline.py
- features/pipeline.py
- traders_core/features/pipeline.py

**TOTAL TO DELETE**: ~35+ files

**KEEP**: All unique functionality, largest/most complete versions
