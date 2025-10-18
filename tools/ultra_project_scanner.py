#!/usr/bin/env python3
"""
Ultra Project Scanner
Comprehensive analysis of the trading bot project
"""

import ast
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraProjectScanner:
    """Comprehensive scanner for the ultra trading bot project"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.stats = {
            'total_files': 0,
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_imports': 0,
            'syntax_errors': 0,
            'import_errors': 0,
            'undefined_names': 0,
            'unused_imports': 0,
            'duplicate_files': 0,
            'large_files': 0,
            'empty_files': 0,
            'malformed_files': 0,
        }

        self.issues = {'critical': [], 'high': [], 'medium': [], 'low': []}

        self.file_analysis: Dict[str, Dict[str, Any]] = {}
        self.import_usage = defaultdict(set)
        self.duplicate_files = defaultdict(list)
        self.large_files: List[Tuple[str, int]] = []

    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single Python file for issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {'error': str(e), 'malformed': True}

        analysis: Dict[str, Any] = {
            'path': str(file_path),
            'size': len(content),
            'lines': len(content.splitlines()),
            'functions': 0,
            'classes': 0,
            'imports': [],
            'syntax_errors': [],
            'import_errors': [],
            'undefined_names': [],
            'unused_imports': [],
            'issues': [],
        }

        # Count functions and classes
        analysis['functions'] = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
        analysis['classes'] = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))

        # Find imports
        import_pattern = r'^(?:from\s+(\w+)\s+import|import\s+(\w+))'
        for line in content.splitlines():
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                analysis['imports'].append(module)

        # Check for syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            analysis['syntax_errors'].append({'line': e.lineno, 'message': str(e), 'text': e.text})
            analysis['issues'].append(f"Syntax error at line {e.lineno}: {e.msg}")

        # Check for common import issues
        if 'os.getenv' in content and 'import os' not in content:
            analysis['import_errors'].append('Missing import os')
            analysis['issues'].append('Missing import os')

        if (
            'datetime.now' in content
            and 'import datetime' not in content
            and 'from datetime import' not in content
        ):
            analysis['import_errors'].append('Missing import datetime')
            analysis['issues'].append('Missing import datetime')

        if 'json.loads' in content and 'import json' not in content:
            analysis['import_errors'].append('Missing import json')
            analysis['issues'].append('Missing import json')

        if ('np.' in content) and ('import numpy' not in content) and ('import numpy as np' not in content):
            analysis['import_errors'].append('Missing import numpy')
            analysis['issues'].append('Missing import numpy')

        if ('pd.' in content) and ('import pandas' not in content) and ('import pandas as pd' not in content):
            analysis['import_errors'].append('Missing import pandas')
            analysis['issues'].append('Missing import pandas')

        # Check for undefined names (heuristic)
        undefined_patterns = [
            r'\bos\.\w+',
            r'\bdatetime\.\w+',
            r'\bjson\.\w+',
            r'\bnp\.\w+',
            r'\bpd\.\w+',
            r'\btf\.\w+',
            r'\btorch\.\w+',
            r'\bplt\.\w+',
            r'\bst\.\w+',
            r'\brequests\.\w+',
            r'\bsqlite3\.\w+',
            r'\bhashlib\.\w+',
            r'\bmath\.\w+',
            r'\brandom\.\w+',
            r'\bthreading\.\w+',
            r'\basyncio\.\w+',
            r'\blogging\.\w+',
            r'\bpathlib\.\w+',
        ]

        for pattern in undefined_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match not in analysis['undefined_names']:
                    analysis['undefined_names'].append(match)

        # Check for unused imports (very rough)
        for import_name in analysis['imports']:
            body = content.replace(f'import {import_name}', '').replace(f'from {import_name}', '')
            if import_name not in body:
                analysis['unused_imports'].append(import_name)

        # Categorize issues
        if analysis['syntax_errors']:
            self.issues['critical'].append(
                f"{file_path}: {len(analysis['syntax_errors'])} syntax errors"
            )

        if analysis['import_errors']:
            self.issues['high'].append(
                f"{file_path}: {len(analysis['import_errors'])} import errors"
            )

        if analysis['undefined_names']:
            self.issues['medium'].append(
                f"{file_path}: {len(analysis['undefined_names'])} undefined names"
            )

        if analysis['unused_imports']:
            self.issues['low'].append(
                f"{file_path}: {len(analysis['unused_imports'])} unused imports"
            )

        return analysis

    def find_duplicate_files(self) -> Dict[str, List[str]]:
        """Find duplicate files based on content hash"""
        file_hashes: Dict[str, List[str]] = defaultdict(list)

        for file_path in self.root_dir.rglob("*.py"):
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    file_hash = str(hash(content))
                    file_hashes[file_hash].append(str(file_path))
            except Exception:
                continue

        duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
        return duplicates

    def find_large_files(self, threshold: int = 1000) -> List[Tuple[str, int]]:
        """Find files larger than threshold lines"""
        large_files: List[Tuple[str, int]] = []

        for file_path in self.root_dir.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    if lines > threshold:
                        large_files.append((str(file_path), lines))
            except Exception:
                continue

        return sorted(large_files, key=lambda x: x[1], reverse=True)

    def analyze_imports(self) -> Dict[str, Any]:
        """Analyze import usage across the project"""
        import_stats: Dict[str, int] = defaultdict(int)
        missing_imports: Dict[str, int] = defaultdict(int)

        for file_path in self.root_dir.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Count import usage
                for line in content.splitlines():
                    if 'import ' in line or 'from ' in line:
                        import_stats['total_imports'] += 1

                # Check for missing imports
                if 'os.getenv' in content and 'import os' not in content:
                    missing_imports['os'] += 1
                if (
                    'datetime.now' in content
                    and 'import datetime' not in content
                    and 'from datetime import' not in content
                ):
                    missing_imports['datetime'] += 1
                if 'json.loads' in content and 'import json' not in content:
                    missing_imports['json'] += 1
                if 'np.' in content and 'import numpy' not in content:
                    missing_imports['numpy'] += 1
                if 'pd.' in content and 'import pandas' not in content:
                    missing_imports['pandas'] += 1

            except Exception:
                continue

        return {'import_stats': dict(import_stats), 'missing_imports': dict(missing_imports)}

    def scan_project(self) -> Dict[str, Any]:
        """Scan the entire project"""
        logger.info("🔍 Starting Ultra Project Scanner...")

        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        self.stats['total_files'] = len(python_files)

        logger.info(f"📁 Found {self.stats['total_files']} Python files")

        # Scan each file
        for i, file_path in enumerate(python_files):
            if i % 1000 == 0:
                logger.info(f"📊 Progress: {i}/{self.stats['total_files']} files scanned")

            analysis = self.scan_file(file_path)
            self.file_analysis[str(file_path)] = analysis

            # Update stats
            if 'error' in analysis:
                self.stats['malformed_files'] += 1
            else:
                self.stats['total_lines'] += analysis['lines']
                self.stats['total_functions'] += analysis['functions']
                self.stats['total_classes'] += analysis['classes']
                self.stats['total_imports'] += len(analysis['imports'])
                self.stats['syntax_errors'] += len(analysis['syntax_errors'])
                self.stats['import_errors'] += len(analysis['import_errors'])
                self.stats['undefined_names'] += len(analysis['undefined_names'])
                self.stats['unused_imports'] += len(analysis['unused_imports'])

                if analysis['lines'] == 0:
                    self.stats['empty_files'] += 1
                elif analysis['lines'] > 1000:
                    self.stats['large_files'] += 1

        # Find duplicates
        logger.info("🔍 Finding duplicate files...")
        duplicates = self.find_duplicate_files()
        self.stats['duplicate_files'] = len(duplicates)
        self.duplicate_files = duplicates

        # Find large files
        logger.info("🔍 Finding large files...")
        large_files = self.find_large_files(1000)
        self.large_files = large_files

        # Analyze imports
        logger.info("🔍 Analyzing imports...")
        import_analysis = self.analyze_imports()

        return {
            'stats': self.stats,
            'issues': self.issues,
            'duplicates': duplicates,
            'large_files': large_files[:20],  # Top 20 largest files
            'import_analysis': import_analysis,
            'file_analysis': self.file_analysis,
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report"""
        report: List[str] = []
        report.append("=" * 80)
        report.append("🚨 ULTRA TRADING BOT PROJECT - COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Project Statistics
        report.append("📊 PROJECT STATISTICS")
        report.append("-" * 40)
        stats = results['stats']
        report.append(f"Total Python Files: {stats['total_files']:,}")
        report.append(f"Total Lines of Code: {stats['total_lines']:,}")
        report.append(f"Total Functions: {stats['total_functions']:,}")
        report.append(f"Total Classes: {stats['total_classes']:,}")
        report.append(f"Total Imports: {stats['total_imports']:,}")
        report.append("")

        # Critical Issues
        report.append("🚨 CRITICAL ISSUES")
        report.append("-" * 40)
        report.append(f"Syntax Errors: {stats['syntax_errors']:,}")
        report.append(f"Import Errors: {stats['import_errors']:,}")
        report.append(f"Undefined Names: {stats['undefined_names']:,}")
        report.append(f"Malformed Files: {stats['malformed_files']:,}")
        report.append("")

        # Code Quality Issues
        report.append("⚠️ CODE QUALITY ISSUES")
        report.append("-" * 40)
        report.append(f"Unused Imports: {stats['unused_imports']:,}")
        report.append(f"Empty Files: {stats['empty_files']:,}")
        report.append(f"Large Files (>1000 lines): {stats['large_files']:,}")
        report.append(f"Duplicate Files: {stats['duplicate_files']:,}")
        report.append("")

        # Top Issues by Category
        report.append("🔍 TOP ISSUES BY CATEGORY")
        report.append("-" * 40)
        for category, issues in results['issues'].items():
            if issues:
                report.append(f"\n{category.upper()} ({len(issues)} issues):")
                for issue in issues[:10]:  # Show top 10
                    report.append(f"  • {issue}")
                if len(issues) > 10:
                    report.append(f"  ... and {len(issues) - 10} more")
        report.append("")

        # Largest Files
        report.append("📏 LARGEST FILES")
        report.append("-" * 40)
        for file_path, lines in results['large_files'][:10]:
            report.append(f"{lines:>6,} lines: {file_path}")
        report.append("")

        # Import Analysis
        report.append("📦 IMPORT ANALYSIS")
        report.append("-" * 40)
        import_analysis = results['import_analysis']
        report.append(f"Total Imports: {import_analysis['import_stats'].get('total_imports', 0):,}")
        report.append("\nMissing Imports:")
        for module, count in sorted(
            import_analysis['missing_imports'].items(), key=lambda x: x[1], reverse=True
        ):
            report.append(f"  {module}: {count:,} files")
        report.append("")

        # Duplicate Files
        report.append("🔄 DUPLICATE FILES")
        report.append("-" * 40)
        duplicates = results['duplicates']
        if duplicates:
            report.append(f"Found {len(duplicates)} sets of duplicate files:")
            for i, (file_hash, files) in enumerate(list(duplicates.items())[:5]):
                report.append(f"\nSet {i+1} ({len(files)} files):")
                for file_path in files:
                    report.append(f"  • {file_path}")
            if len(duplicates) > 5:
                report.append(f"\n... and {len(duplicates) - 5} more sets")
        else:
            report.append("No duplicate files found")
        report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. IMMEDIATE ACTIONS:")
        report.append("   • Fix all syntax errors (critical)")
        report.append("   • Add missing imports (high priority)")
        report.append("   • Remove duplicate files")
        report.append("   • Split large files into modules")
        report.append("")
        report.append("2. CODE QUALITY:")
        report.append("   • Remove unused imports")
        report.append("   • Fix undefined names")
        report.append("   • Add proper error handling")
        report.append("   • Implement code formatting (black, isort)")
        report.append("")
        report.append("3. ARCHITECTURE:")
        report.append("   • Refactor monolithic files")
        report.append("   • Create proper module structure")
        report.append("   • Add configuration management")
        report.append("   • Implement proper testing")
        report.append("")
        report.append("4. PRODUCTION READINESS:")
        report.append("   • Add security measures")
        report.append("   • Implement monitoring")
        report.append("   • Add logging and error tracking")
        report.append("   • Create deployment scripts")
        report.append("")

        # Summary
        total_issues = sum(len(issues) for issues in results['issues'].values())
        report.append("📋 SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Issues Found: {total_issues:,}")
        report.append(f"Critical Issues: {len(results['issues']['critical']):,}")
        report.append(f"High Priority Issues: {len(results['issues']['high']):,}")
        report.append(f"Medium Priority Issues: {len(results['issues']['medium']):,}")
        report.append(f"Low Priority Issues: {len(results['issues']['low']):,}")
        report.append("")
        report.append("⚠️  This is a MASSIVE project requiring systematic refactoring!")
        report.append("⚠️  Estimated time to fix: 2-4 weeks with a team of developers")
        report.append("⚠️  Consider starting with core functionality only")
        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_detailed_report(
        self, results: Dict[str, Any], output_file: str = "ultra_project_analysis.json"
    ):
        """Save detailed analysis to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 Detailed report saved to {output_file}")


def main():
    """Main function"""
    scanner = UltraProjectScanner()
    results = scanner.scan_project()

    # Generate and print report
    report = scanner.generate_report(results)
    print(report)

    # Save detailed report
    scanner.save_detailed_report(results)

    # Save summary report
    with open("ultra_project_summary.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info("🎉 Project scanning completed!")
    logger.info(f"📊 Total files scanned: {results['stats']['total_files']:,}")
    logger.info(
        f"🚨 Total issues found: {sum(len(issues) for issues in results['issues'].values()):,}"
    )


if __name__ == "__main__":
    main()

