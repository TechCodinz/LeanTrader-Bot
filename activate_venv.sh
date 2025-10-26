#!/bin/bash
# Convenient script to activate the restored virtual environment

echo "🚀 Activating Trading Bot Virtual Environment"
echo ""
echo "Virtual environment location: /workspace/venv"
echo ""
echo "To activate manually, run:"
echo "  source /workspace/venv/bin/activate"
echo ""
echo "Or use the venv python directly:"
echo "  /workspace/venv/bin/python your_script.py"
echo ""
echo "To run the bot:"
echo "  /workspace/venv/bin/python RUN_BOT.py --testnet"
echo ""
echo "To test the bot:"
echo "  /workspace/venv/bin/python TEST_BOT.py"
echo ""

# Activate if being sourced
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    source /workspace/venv/bin/activate
    echo "✅ Virtual environment activated!"
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
else
    echo "⚠️  To activate in current shell, run:"
    echo "  source $0"
fi
