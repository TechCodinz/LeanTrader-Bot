#!/usr/bin/env python3
"""
Git Conflict Fixer
Fixes Git merge conflicts in Python files
"""

import os
import re
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitConflictFixer:
    """Fixes Git merge conflicts in Python files"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.fixed_files = 0
        self.total_files = 0
        self.errors = 0
    
    def fix_git_conflicts(self, file_path: Path) -> bool:
        """Fix Git merge conflicts in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return False
        
        original_content = content
        
        # Remove Git conflict markers and keep the HEAD version conservatively
        # Pattern: <<<<<<< HEAD ... ======= ... >>>>>>> branch
        pattern = re.compile(
            r"<<<<<<< HEAD\n([\s\S]*?)\n=======\n[\s\S]*?\n>>>>>>>[^\n]+",
            re.MULTILINE,
        )
        content = re.sub(pattern, r"\1", content)
        
        # Remove any stray markers if present
        content = re.sub(r"^<<<<<<<.*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"^=======$", "", content, flags=re.MULTILINE)
        content = re.sub(r"^>>>>>>>.*$", "", content, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)
        
        if content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            except Exception as e:
                logger.error(f"Error writing {file_path}: {e}")
                return False
        
        return False
    
    def run(self):
        """Run the Git conflict fixer"""
        logger.info("🔧 Starting Git Conflict Fixer...")
        
        python_files = list(self.root_dir.rglob("*.py"))
        self.total_files = len(python_files)
        
        for i, file_path in enumerate(python_files):
            if i % 1000 == 0:
                logger.info(f"📊 Progress: {i}/{self.total_files} files processed")
            
            try:
                if self.fix_git_conflicts(file_path):
                    self.fixed_files += 1
                    logger.debug(f"✅ Fixed conflicts in {file_path}")
            except Exception as e:
                self.errors += 1
                logger.error(f"❌ Error processing {file_path}: {e}")
        
        logger.info(f"🎉 Git conflict fixing completed!")
        logger.info(f"📊 Files processed: {self.total_files}")
        logger.info(f"✅ Files fixed: {self.fixed_files}")
        logger.info(f"❌ Errors: {self.errors}")

def main():
    """Main function"""
    fixer = GitConflictFixer()
    fixer.run()

if __name__ == "__main__":
    main()
