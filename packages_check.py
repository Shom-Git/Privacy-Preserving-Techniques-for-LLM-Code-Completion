try:
    import torch
    import transformers
    import datasets
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as mpl
    import pandas as pd
except ImportError as e:
    missing_package = str(e).split("No module named ")[1].strip("'")
    print(f"Missing package: {missing_package}. Please install it to proceed.")