import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRAFFIC_AI_SECRET", "test-secret-key")  # tests never write data/.secret_key
