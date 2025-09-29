import sys

sys.path.append(r"c:\Users\User\Downloads\LeanTrader_ForexPack")

# instantiate with empty lists
tc = TraderCore([], [], [], [], [], [])
print("TraderCore instantiated; mt5:", bool(tc.mt5))
