
"""

Description:
Script reads in DNA methylation and clinical files from GEO projects


"""

import pandas as pd
from pdb import set_trace
from read_tcga_data import *


def read_geo_data(path, skip1, rows, skip2, label, counter):
    """Read in DNAm and label data from GEO."""
    # Get the label mappings (diagnoses) for each DNAm sample
    geo_clinical = pd.read_csv(path, sep='\t', skiprows=skip1, nrows=rows)
    geo_clinical.set_index('!Sample_title', inplace=True)
    geo_clinical = geo_clinical.T
    if 'GSE104707' in path:
        diagnosis_map = dict(zip(geo_clinical['!Sample_geo_accession'], geo_clinical[label].iloc[:, 0]))
    else:
    	diagnosis_map = dict(zip(geo_clinical['!Sample_geo_accession'], geo_clinical[label]))
    # Get the CpG DNAm measures
    geo_betas = pd.read_csv(path, sep='\t', skiprows=skip2)
    geo_betas.set_index('ID_REF', inplace=True)
    geo_betas = geo_betas.T
    
    geo_betas['Diagnosis'] = geo_betas.index.map(diagnosis_map)
    geo_betas['Dataset'] = counter
    counter += 1
    return geo_betas, counter


def preprocess_data_sets(df):
    """Drop any samples not used in models."""
    df = df[df['Diagnosis'] != 'tissue: Stomach']
    return df


def geo_data_set_reader():
    """Main function for reading GEO data sets."""
    counter = 1
    
    path = 'C:/Users/User/Desktop/D Drive/ECa_paper/GEO/GSE52826_series_matrix.txt'
    skip1 = 30; rows = 34; skip2 = 65
    geo_GSE52826, counter = read_geo_data(path, skip1, rows, skip2, '!Sample_source_name_ch1', counter)
    # 12 samples in this data set - drop any 'cg' feature with at least 1 missing value
    geo_GSE52826.dropna(axis='columns', inplace=True) # 6478 'cg's dropped 

    path = 'C:/Users/User/Desktop/D Drive/ECa_paper/GEO/GSE74693_series_matrix.txt'
    skip1 = 32; rows = 34; skip2 = 69
    geo_GSE74693, counter = read_geo_data(path, skip1, rows, skip2, '!Sample_source_name_ch1', counter)
    # 10 samples - so drop any 'cg' feature with at least 1 missing value
    geo_GSE74693.dropna(axis='columns', inplace=True)     # 2284 'cg's dropped

    path = 'C:/Users/User/Desktop/D Drive/ECa_paper/GEO/GSE79366_series_matrix.txt'
    skip1 = 29; rows = 31; skip2 = 62
    geo_GSE79366, counter = read_geo_data(path, skip1, rows, skip2, '!Sample_characteristics_ch1', counter)
    # 14 samples - so drop any 'cg' feature with at least 1 missing value
    geo_GSE79366.dropna(axis='columns', inplace=True) 


    # Is the normal esophagus in this study from tumour/BE adjacent or healthy controls?
    # I think these "squamous tissue" are from BE patients ) i.e. adjacent normal
    # The study doi: 10.1136/gutjnl-2017-314544 seems to usee the GSE81334 samples - but
    # not all of them
    path = 'C:/Users/User/Desktop/D Drive/ECa_paper/GEO/GSE81334_series_matrix.txt'
    skip1 = 38; rows = 33; skip2 = 73
    geo_GSE81334, counter = read_geo_data(path, skip1, rows, skip2, '!Sample_description', counter)
    # 148 samples - no missing value - already down to 426K samples
    

    path = 'C:/Users/User/Desktop/D Drive/ECa_paper/GEO/GSE104707_series_matrix.txt'
    skip1 = 37; rows = 33; skip2 = 71
    geo_GSE104707, counter = read_geo_data(path, skip1, rows, skip2, '!Sample_characteristics_ch1', counter)
    # 160 samples - only 385k CpGs - no missing values.


    # Reading the GEO Summary, it indicates 125 EAC, 19 BE, 64 normal adjacent and 21 normal stomach
    # There are 125 tumour, 19 BE, 64 adjacent squamous from EAC or BE patients, 21 normal stomach,
    # 11 squamous esophagus from healthy controls, and 10 squamous esophagus from GERD patients.
    # Drop the Stomach samples, perhaps combine the 64, 11 and 10 into 'normal'
    path = 'C:/Users/User/Desktop/D Drive/ECa_paper/GEO/GSE72872_series_matrix.txt'
    skip1 = 50; rows = 33; skip2 = 85
    geo_GSE72872, counter = read_geo_data(path, skip1, rows, skip2, '!Sample_characteristics_ch1', counter)
    geo_GSE72872 = preprocess_data_sets(geo_GSE72872)
    # 229 samples - only 374k CpGs - approx. 50K have 1 missing value only, none have >1 missing - fill
    geo_GSE72872.fillna(geo_GSE72872.mean(), inplace=True)
    
    path = 'C:/Users/User/Desktop/D Drive/ECa_paper/GEO/GSE164083_series_matrix.txt'
    skip1 = 30; rows = 35; skip2 = 66
    geo_GSE164083, counter = read_geo_data(path, skip1, rows, skip2, '!Sample_source_name_ch1', counter)
    # 159 samples - 706k CpGs - no missing values - EPIC
    

    data_sets = {'geo_GSE52826': geo_GSE52826, 'geo_GSE74693': geo_GSE74693,
                 'geo_GSE79366': geo_GSE79366, 'geo_GSE81334': geo_GSE81334,
                 'geo_GSE104707': geo_GSE104707, 'geo_GSE72872': geo_GSE72872,
                 'geo_GSE164083': geo_GSE164083
                 }
    
    return data_sets
