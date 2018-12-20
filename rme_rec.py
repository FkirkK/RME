"""Main class responsible for calling the runner with correct parameters"""
import glob
import os
import helper_methods
import model_runner
import neg_inf
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import time
import pickle_loader
import io

args = helper_methods.get_args_parser().parse_args()
DATA_DIR = os.path.join(args.data_path, args.dataset)
SAVED_MODEL_DIR = args.saved_model_path
PRED_DIR = os.path.join(DATA_DIR, 'prediction-temp')
SHIFTED_K_VALUE = args.s
NEGATIVE_SAMPLE_RATIO = args.neg_sample_ratio
save_dir = os.path.join(DATA_DIR, 'model_tmp_res')
n_components = args.n_factors
lam = args.reg
lam_emb = args.reg_embed
user_cooc = args.user_cooc
item_cooc = args.item_cooc
random_seed = args.random_seed

FXP_weight = args.FXP_weight
FXN_weight = args.FXN_weight
FYP_weight = args.FYP_weight
FYN_weight = args.FYN_weight


unique_movieId, unique_uid = helper_methods.get_unique_users_and_items(DATA_DIR)
n_users = len(unique_uid)
n_items = len(unique_movieId)
print n_users, n_items

if args.neg_item_inference:
    _ = neg_inf.negative_inference(DATA_DIR, save_dir, n_components, n_users, n_items, SHIFTED_K_VALUE,
                                                    NEGATIVE_SAMPLE_RATIO, lam, lam_emb)

LOAD_NEGATIVE_MATRIX = args.model.lower() == 'rme'

recalls = np.zeros(5, dtype=np.float32) #store results of topk recommendation in range [5, 10, 20, 50, 100]
ndcgs = np.zeros(5, dtype=np.float32)
maps = np.zeros(5, dtype=np.float32)
print '*************************************lam =  %.3f ******************************************' % lam
print '*************************************lam embedding =  %.3f ******************************************' % lam_emb

vad_data, vad_raw, vad_df = helper_methods.load_data(os.path.join(DATA_DIR, 'validation.csv'), shape=(n_users, n_items))
test_data, test_raw, test_df = helper_methods.load_data(os.path.join(DATA_DIR, 'test.csv'), shape=(n_users, n_items))
train_data, train_raw, train_df = helper_methods.load_data(os.path.join(DATA_DIR, 'train.csv'), shape=(n_users, n_items))

print 'loading pro_pro_cooc.dat'
t1 = time.time()
X = pickle_loader.load_pickle(os.path.join(DATA_DIR, 'item_item_cooc.dat'))
t2 = time.time()
print '[INFO]: sparse matrix size of item item co-occurrence matrix: %d mb\n' % (
    (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024 * 1024))
print 'Time : %d seconds' % (t2 - t1)

print 'loading user_user_cooc.dat'
t1 = time.time()
Y = pickle_loader.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc.dat'))
t2 = time.time()
print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
    (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / (1024 * 1024))
print 'Time : %d seconds' % (t2 - t1)

################# LOADING NEGATIVE CO-OCCURRENCE MATRIX and converting to Negative SPPMI matrix #######################
X_neg_sppmi = None
Y_neg_sppmi = None
if LOAD_NEGATIVE_MATRIX:
    print 'test loading negative_pro_pro_cooc.dat'
    t1 = time.time()
    X_neg = pickle_loader.load_pickle(os.path.join(DATA_DIR, 'negative_item_item_cooc.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of negative item item co-occurrence matrix: %d mb\n' % (
        (X_neg.data.nbytes + X_neg.indices.nbytes + X_neg.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds' % (t2 - t1)

    print 'converting negative co-occurrence matrix into sppmi matrix'
    t1 = time.time()
    X_neg_sppmi = helper_methods.convert_to_SPPMI_matrix(X_neg, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)

    print 'test loading negative_pro_pro_cooc.dat'
    t1 = time.time()
    Y_neg = pickle_loader.load_pickle(os.path.join(DATA_DIR, 'negative_user_user_cooc.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of negative user user co-occurrence matrix: %d mb\n' % (
            (Y_neg.data.nbytes + Y_neg.indices.nbytes + Y_neg.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds' % (t2 - t1)

    print 'converting negative co-occurrence matrix into sppmi matrix'
    t1 = time.time()
    Y_neg_sppmi = helper_methods.convert_to_SPPMI_matrix(Y_neg, max_row=n_users, shifted_K=SHIFTED_K_VALUE)
    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)


################################################################################################
########## converting CO-OCCURRENCE MATRIX INTO Shifted Positive Pointwise Mutual Information (SPPMI) matrix ###########
####### We already know the user-user co-occurrence matrix Y and item-item co-occurrence matrix X

print 'converting co-occurrence matrix into sppmi matrix'
t1 = time.time()
X_sppmi = helper_methods.convert_to_SPPMI_matrix(X, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
Y_sppmi = helper_methods.convert_to_SPPMI_matrix(Y, max_row=n_users, shifted_K=SHIFTED_K_VALUE)
t2 = time.time()
print 'Time : %d seconds' % (t2 - t1)


######## Finally, we have train_data, vad_data, test_data,
# X_sppmi: item item Shifted Positive Pointwise Mutual Information matrix
# Y_sppmi: user-user       Shifted Positive Pointwise Mutual Information matrix

print 'Training data', train_data.shape
print 'Validation data', vad_data.shape
print 'Testing data', test_data.shape

n_jobs = 1  # default value
model_type = 'model2'  # default value
if os.path.exists(save_dir):
    #cleaning folder
    lst = glob.glob(os.path.join(save_dir, '*.*'))
    for f in lst:
        os.remove(f)
else:
    os.mkdir(save_dir)


if (FXP_weight != -1):
    FXP = FXP_weight
else:
    FXP = None

if (FXN_weight != -1):
    FXN = FXN_weight
else:
    FXN = None

if (FYP_weight != -1):
    FYP = FYP_weight
else:
    FYP = None

if (FYN_weight != -1):
    FYN = FYN_weight
else:
    FYN = None

runner = model_runner.ModelRunner(train_data, vad_data, test_data, X_sppmi, X_neg_sppmi, Y_sppmi, Y_neg_sppmi,
                       save_dir=save_dir, data_dir=DATA_DIR, randomSeed=random_seed,
                       FXP=FXP, FXN = FXN, FYP=FYP, FYN=FYN)

start = time.time()
if args.model == 'wmf':
    (recalls, ndcgs, maps) = runner.run("wmf", n_jobs=n_jobs, lam=lam,
                                                         saved_model = True,
                                                         SAVED_MODEL_DIR=SAVED_MODEL_DIR,
                                                         PRED_DIR=PRED_DIR,
                                                         n_components = n_components)
if args.model == 'cofactor':
    (recalls, ndcgs, maps) = runner.run("cofactor", n_jobs=n_jobs,
                                                        lam=lam,
                                                         saved_model=True,
                                                         SAVED_MODEL_DIR=SAVED_MODEL_DIR,
                                                         PRED_DIR=PRED_DIR,
                                                         n_components=n_components)
if args.model == 'rme':
    (recalls, ndcgs, maps) = runner.run("rme", n_jobs=n_jobs,lam=lam, lam_emb = lam_emb,
                                                         user_cooc = user_cooc, item_cooc = item_cooc,
                                                         saved_model=True,
                                                         SAVED_MODEL_DIR=SAVED_MODEL_DIR,
                                                         PRED_DIR=PRED_DIR,
                                                         n_components=n_components)
end = time.time()
if not os.path.exists('shell_result'):
    os.mkdir('shell_result')
with io.open('shell_result/grid_search.txt', mode='a', encoding='utf-8') as filePointer:
    print ('total running time: %d seconds'%(end-start))
    for idx, topk in enumerate([5, 10, 20, 50, 100]):
        print 'top-%d results: recall@%d = %.4f, ndcg@%d = %.4f, map@%d = %.4f'%(topk,
                                                                                      topk, recalls[idx],
                                                                                      topk, ndcgs[idx],
                                                                                      topk, maps[idx])
        filePointer.write(u'top-%d results: recall@%d = %.4f, ndcg@%d = %.4f, map@%d = %.4f \n'%(topk,
                                                                                      topk, recalls[idx],
                                                                                      topk, ndcgs[idx],
                                                                                      topk, maps[idx]))
    filePointer.write(u'\n')