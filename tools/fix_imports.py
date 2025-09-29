#!/usr/bin/env python3
"""
Ultra Import Fixer
Fixes missing imports across the codebase (opt-in, minimal dependencies).
"""

import logging
import re
from pathlib import Path
from typing import List, Set, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImportFixer:
    """Fixes missing imports across the entire codebase"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.fixed_files = 0
        self.total_files = 0
        self.errors = 0

        # Common imports that are often missing
        self.common_imports = {
            'os': [
                'os.getenv',
                'os.path',
                'os.environ',
                'os.system',
                'os.makedirs',
                'os.path.exists',
            ],
            'datetime': ['datetime.now', 'datetime.timedelta', 'datetime.datetime'],
            'time': ['time.time', 'time.sleep', 'time.strftime'],
            'json': ['json.loads', 'json.dumps', 'json.load', 'json.dump'],
            'sys': ['sys.path', 'sys.exit', 'sys.argv'],
            'logging': ['logging.getLogger', 'logging.info', 'logging.error', 'logging.warning'],
            'asyncio': ['asyncio.run', 'asyncio.create_task', 'asyncio.gather', 'asyncio.sleep'],
            'typing': [
                'typing.Dict',
                'typing.List',
                'typing.Optional',
                'typing.Any',
                'typing.Tuple',
            ],
            'pathlib': ['Path', 'pathlib.Path'],
            'numpy': ['np.', 'numpy.'],
            'pandas': ['pd.', 'pandas.'],
            'requests': ['requests.get', 'requests.post', 'requests.put', 'requests.delete'],
            'sqlite3': ['sqlite3.connect', 'sqlite3.Row'],
            'hashlib': ['hashlib.md5', 'hashlib.sha256', 'hashlib.sha1', 'hashlib.sha512'],
            'math': ['math.sqrt', 'math.log', 'math.exp', 'math.sin', 'math.cos'],
            'random': ['random.random', 'random.choice', 'random.randint'],
            'collections': ['collections.defaultdict', 'collections.deque', 'collections.Counter'],
            'itertools': ['itertools.combinations', 'itertools.permutations'],
            'functools': ['functools.wraps', 'functools.partial'],
            'warnings': ['warnings.warn', 'warnings.filterwarnings'],
            'traceback': ['traceback.print_exc', 'traceback.format_exc'],
            'inspect': ['inspect.getframeinfo', 'inspect.currentframe'],
            'threading': ['threading.Thread', 'threading.Lock'],
            'queue': ['queue.Queue', 'queue.Empty'],
            'subprocess': ['subprocess.run', 'subprocess.Popen', 'subprocess.call'],
            'urllib': ['urllib.parse', 'urllib.request'],
            'base64': ['base64.b64encode', 'base64.b64decode'],
            'uuid': ['uuid.uuid4', 'uuid.uuid1'],
            're': ['re.search', 're.match', 're.findall', 're.sub'],
            'socket': ['socket.socket', 'socket.AF_INET', 'socket.SOCK_STREAM'],
            'ssl': ['ssl.create_default_context', 'ssl.SSLContext'],
            'email': ['email.mime.text', 'email.mime.multipart'],
            'smtplib': ['smtplib.SMTP', 'smtplib.SMTP_SSL'],
            'csv': ['csv.reader', 'csv.writer', 'csv.DictReader'],
            'configparser': ['configparser.ConfigParser'],
            'argparse': ['argparse.ArgumentParser'],
            'pickle': ['pickle.dumps', 'pickle.loads', 'pickle.dump', 'pickle.load'],
            'yaml': ['yaml.load', 'yaml.dump', 'yaml.safe_load'],
            'toml': ['toml.load', 'toml.dump'],
            'xml': ['xml.etree.ElementTree'],
            'html': ['html.escape', 'html.unescape'],
            'urllib3': ['urllib3.PoolManager'],
            'websocket': ['websocket.WebSocketApp'],
            'ccxt': ['ccxt.bybit', 'ccxt.binance', 'ccxt.coinbase'],
            'talib': ['talib.RSI', 'talib.MACD', 'talib.BBANDS'],
            'sklearn': ['sklearn.ensemble', 'sklearn.preprocessing', 'sklearn.model_selection'],
            'tensorflow': ['tensorflow.keras', 'tf.'],
            'torch': ['torch.nn', 'torch.optim', 'torch.tensor'],
            'plotly': ['plotly.graph_objects', 'plotly.express'],
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        for pattern in ['**/*.py', '**/*.pyw']:
            python_files.extend(self.root_dir.glob(pattern))
        return python_files

    def analyze_file(self, file_path: Path) -> Dict[str, Set[str]]:
        """Analyze a file to find missing imports"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return {}

        # Find existing imports
        existing_imports = set()
        import_pattern = r'^(?:from\s+(\w+)\s+import|import\s+(\w+))'
        for line in content.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                existing_imports.add(module)

        # Find used modules
        missing_imports: Dict[str, Set[str]] = {}
        for module, patterns in self.common_imports.items():
            if module not in existing_imports:
                for pattern in patterns:
                    if pattern in content:
                        if module not in missing_imports:
                            missing_imports[module] = set()
                        missing_imports[module].add(pattern)

        return missing_imports

    def fix_file(self, file_path: Path) -> bool:
        """Fix missing imports in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return False

        missing_imports = self.analyze_file(file_path)
        if not missing_imports:
            return False

        # Find the best place to insert imports
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                insert_index = i + 1
            elif line.strip().startswith('#') or line.strip() == '':
                continue
            else:
                break

        # Create import statements
        new_imports = []
        for module in sorted(missing_imports.keys()):
            if module == 'numpy':
                new_imports.append('import numpy as np\n')
            elif module == 'pandas':
                new_imports.append('import pandas as pd\n')
            elif module == 'tensorflow':
                new_imports.append('import tensorflow as tf\n')
            elif module == 'streamlit':
                new_imports.append('import streamlit as st\n')
            elif module == 'typing':
                new_imports.append('from typing import Dict, List, Optional, Any, Tuple, Union\n')
            elif module == 'datetime':
                new_imports.append('from datetime import datetime, timedelta\n')
            elif module == 'collections':
                new_imports.append('from collections import defaultdict, deque, Counter\n')
            elif module == 'pathlib':
                new_imports.append('from pathlib import Path\n')
            elif module == 'json':
                new_imports.append('import json\n')
            elif module == 'requests':
                new_imports.append('import requests\n')
            elif module == 'sqlite3':
                new_imports.append('import sqlite3\n')
            elif module == 'hashlib':
                new_imports.append('import hashlib\n')
            elif module == 'math':
                new_imports.append('import math\n')
            elif module == 'random':
                new_imports.append('import random\n')
            elif module == 'itertools':
                new_imports.append('import itertools\n')
            elif module == 'functools':
                new_imports.append('import functools\n')
            elif module == 'warnings':
                new_imports.append('import warnings\n')
            elif module == 'traceback':
                new_imports.append('import traceback\n')
            elif module == 'inspect':
                new_imports.append('import inspect\n')
            elif module == 'threading':
                new_imports.append('import threading\n')
            elif module == 'queue':
                new_imports.append('import queue\n')
            elif module == 'subprocess':
                new_imports.append('import subprocess\n')
            elif module == 'urllib':
                new_imports.append('import urllib\n')
            elif module == 'base64':
                new_imports.append('import base64\n')
            elif module == 'uuid':
                new_imports.append('import uuid\n')
            elif module == 're':
                new_imports.append('import re\n')
            elif module == 'socket':
                new_imports.append('import socket\n')
            elif module == 'ssl':
                new_imports.append('import ssl\n')
            elif module == 'email':
                new_imports.append('from email.mime.text import MIMEText\n')
            elif module == 'smtplib':
                new_imports.append('import smtplib\n')
            elif module == 'csv':
                new_imports.append('import csv\n')
            elif module == 'configparser':
                new_imports.append('import configparser\n')
            elif module == 'argparse':
                new_imports.append('import argparse\n')
            elif module == 'pickle':
                new_imports.append('import pickle\n')
            elif module == 'yaml':
                new_imports.append('import yaml\n')
            elif module == 'toml':
                new_imports.append('import toml\n')
            elif module == 'xml':
                new_imports.append('import xml.etree.ElementTree as ET\n')
            elif module == 'html':
                new_imports.append('import html\n')
            elif module == 'urllib3':
                new_imports.append('import urllib3\n')
            elif module == 'websocket':
                new_imports.append('import websocket\n')
            elif module == 'ccxt':
                new_imports.append('import ccxt\n')
            elif module == 'talib':
                new_imports.append('import talib\n')
            elif module == 'sklearn':
                new_imports.append('from sklearn.ensemble import RandomForestClassifier\n')
            elif module == 'torch':
                new_imports.append('import torch\n')
            elif module == 'plotly':
                new_imports.append('import plotly.graph_objects as go\n')
            else:
                new_imports.append(f'import {module}\n')

        # Insert new imports
        for i, new_import in enumerate(new_imports):
            lines.insert(insert_index + i, new_import)

        # Write the fixed file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        except Exception as e:
            logger.error(f"Error writing {file_path}: {e}")
            return False

    def run(self):
        """Run the import fixer on all Python files"""
        logger.info("🔍 Starting Ultra Import Fixer...")

        python_files = self.find_python_files()
        self.total_files = len(python_files)

        logger.info(f"📁 Found {self.total_files} Python files")

        for i, file_path in enumerate(python_files):
            if i % 1000 == 0:
                logger.info(f"📊 Progress: {i}/{self.total_files} files processed")

            try:
                if self.fix_file(file_path):
                    self.fixed_files += 1
                    logger.debug(f"✅ Fixed imports in {file_path}")
            except Exception as e:
                self.errors += 1
                logger.error(f"❌ Error processing {file_path}: {e}")

        logger.info("🎉 Import fixing completed!")
        logger.info(f"📊 Files processed: {self.total_files}")
        logger.info(f"✅ Files fixed: {self.fixed_files}")
        logger.info(f"❌ Errors: {self.errors}")


def main():
    """Main function"""
    fixer = ImportFixer()
    fixer.run()


if __name__ == "__main__":
    main()

