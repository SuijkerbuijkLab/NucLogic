from utils import calculate_cutoff
import os
import pandas as pd
import numpy as np


def calculate_phenotypes(input_directory, phenotype_1, phenotype_2, cutoff_method):

    cutoff = calculate_cutoff(properties, f"log_ratio_wt_crc")
