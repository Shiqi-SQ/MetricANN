import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import PlushieGUI

if __name__ == "__main__":
    app = PlushieGUI()
    app.run()
