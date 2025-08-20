#!/usr/bin/env python3

"""
======================================================================
 Title:                   ASDA - Automated Spot Detection & Anamysis
 Creating Author:         Janan ARSLAN
 Creation Date:           [27-11-2024]
 Latest Modification:     [20-08-32025]
 Modification Author:     Janan ARSLAN
 E-mail:                  janan.arslan@gmail.com
 Version:                 v1.10
======================================================================


"""

# Environment setup
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")

import sys
import warnings
from PyQt5.QtWidgets import QApplication

# Import our GUI
from gui import SpotAnalysisGUI

warnings.filterwarnings("ignore")

def main():
    """Launch the spot analysis application."""
    print("Starting Spot Analysis Tool...")
    
    app = QApplication(sys.argv)
    window = SpotAnalysisGUI()
    window.show()
    
    print("GUI loaded successfully. Ready for analysis!")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()