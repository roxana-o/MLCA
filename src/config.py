# config.py

N_BIDDERS  = 6        
N_ITEMS    = 18      
USE_EXACT_OPT = False  

# CCA phase  
CCA_ROUNDS  = 10  
PRICE_STEP  = 1.0 
SUPPLY      = [1] * N_ITEMS

# ML DQ phase 
ML_DQ_ROUNDS    = 10   
NEXTPRICE_STEPS = 30     
NEXTPRICE_LR    = 0.3    
NEXTPRICE_MU    = 0.5   

# ML VQ phase 
ML_VQ_ROUNDS  = 20     

# MVNN training
TRAIN_EPOCHS = 300
TRAIN_LR     = 0.001

# MVNN architecture
MVNN_HIDDEN_UNITS = 20
MVNN_LAYERS       = 2
MVNN_T_CUTOFF     = 10.0

MODEL_CANDIDATE_SAMPLES = 80          
MODEL_CANDIDATE_KEEP    = 25         
MODEL_CANDIDATE_SIZES   = (4, 5, 6)   


# Multi-run experiment
NUM_RUNS  = 5 
SEED_BASE = 101

CACHED_DQ_FREQ = 5 

MODEL_TYPE = "mvnn"   # options: "mvnn", "linear"
VALUATION_TYPE = "complementarity"   # options: "sats", "additive", "pairwise", "complementarity", 'mixed'

# Training loss weights
DQ_LOSS_WEIGHT = 1.0
VQ_LOSS_WEIGHT = 3.0

# Adam weight decay
WEIGHT_DECAY = 1e-6

QUALITY_SNAPSHOTS = True       
QUALITY_N_TEST    = 200   

SUBSET_MAX_SIZE = 4        
SUBSET_MAX_FEATURES = 4000 