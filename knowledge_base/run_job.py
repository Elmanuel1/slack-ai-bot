import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from config.settings import Settings
from knowledge_base.job_runner import create_confluence_job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def main():
    try:
        # Create and run the job
        job = create_confluence_job(Settings())
        job.run()
        
    except Exception as e:
        logging.error(f"Failed to run knowledge base job: {str(e)}")
        raise

if __name__ == "__main__":
    main() 