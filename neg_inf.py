import time
import wmf
import rec_eval
import produce_negative_embedding as pne
import glob
import os
import numpy as np
import joblib
import text_utils
import model_runner
import helper_methods


def softmax(x):
    """Compute softmax values for each ranked list."""
    # We want the item with higher ranking score have lower prob to be withdrawn as negative instances
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=None, vad_data=None):
    n_songs = train_data.shape[1]
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_songs), dtype=bool)
    item_idx[train_data[user_idx].nonzero()] = True
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = True
    X_pred = Et[user_idx].dot(Eb)
    if mu is not None:
        if isinstance(mu, np.ndarray):
            assert mu.size == n_songs  # mu_i
            X_pred *= mu
        elif isinstance(mu, dict):  # func(mu_ui)
            params, func = mu['params'], mu['func']
            args = [params[0][user_idx], params[1]]
            if len(params) > 2:  # for bias term in document or length-scale
                args += [params[2][user_idx]]
            if not callable(func):
                raise TypeError("expecting a callable function")
            X_pred *= func(*args)
        else:
            raise ValueError("unsupported mu type")
    X_pred[item_idx] = np.inf
    return X_pred


def gen_neg_instances(train_data, U, VT, user_idx, DATA_DIR, n_items, neg_ratio=1.0, iter=0, vad_data=None):
    print 'Job start... %d to %d' % (user_idx.start, user_idx.stop)
    # if user_idx.start != 99000: return
    batch_users = user_idx.stop - user_idx.start
    X_pred = _make_prediction(train_data, U, VT, user_idx, batch_users, vad_data=vad_data)

    rows = []
    cols = []
    total_lost = 0
    for idx, uid in enumerate(range(user_idx.start, user_idx.stop)):
        num_pos = train_data[uid].count_nonzero()
        num_neg = int(num_pos * neg_ratio)
        if num_neg <= 0: continue
        ranks = X_pred[idx]
        neg_withdrawn_prob = softmax(np.negative(ranks))
        # print (neg_withdrawn_prob)
        neg_instances = list(set(np.random.choice(range(n_items), num_neg, p=neg_withdrawn_prob)))
        # rows = rows + len(neg_instances)*[uid]
        # uid_dup = np.empty(len(neg_instances))
        # uid_dup.fill(uid)
        if uid < 0: print 'error with %d to %d' % (user_idx.start, user_idx.stop)
        # rows = rows + uid_dup
        rows = np.append(rows, np.full(len(neg_instances), uid))
        cols = np.append(cols, neg_instances)
    # print 'check for neg values: ', np.sum(rows <0)
    # print 'check for neg values: ', np.sum(cols <0)
    if len(rows) > 0:
        path = os.path.join(DATA_DIR, 'sub_dataframe_iter_%d_idxstart_%d.csv' % (iter, user_idx.start))
        assert len(rows) == len(cols)
        with open(path, 'w') as writer:
            for i in range(len(rows)): writer.write(str(rows[i]) + "," + str(cols[i]) + '\n')
            writer.flush()
        # df = pd.DataFrame({'uid':rows, 'sid':cols}, columns=["uid", "sid"], dtype=np.int16)
        # df.to_csv(path, sep=",",header=False, index = False)
    # return df

def negative_inference(DATA_DIR, save_dir, n_components, n_users, n_items, SHIFTED_K_VALUE, NEGATIVE_SAMPLE_RATIO, lam, lam_emb):
    U, V = None, None
    vad_data, vad_raw, vad_df = helper_methods.load_data(os.path.join(DATA_DIR, 'validation.csv'), shape=(n_users, n_items))
    train_data, train_raw, train_df = helper_methods.load_data(os.path.join(DATA_DIR, 'train.csv'), shape=(n_users, n_items))
    test_data, test_raw, test_df = helper_methods.load_data(os.path.join(DATA_DIR, 'test.csv'), shape=(n_users, n_items))
    U, V = wmf.decompose(train_data, vad_data, num_factors=n_components)
    VT = V.T
    iter, max_iter = 0, 10
    # load postivie information
    X = text_utils.load_pickle(os.path.join(DATA_DIR, 'item_item_cooc.dat'))
    Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc.dat'))
    X_sppmi = helper_methods.convert_to_SPPMI_matrix(X, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
    Y_sppmi = helper_methods.convert_to_SPPMI_matrix(Y, max_row=n_users, shifted_K=SHIFTED_K_VALUE)
    best_ndcg100 = 0.0
    best_iter = 1
    early_stopping = False
    while (iter < max_iter and not early_stopping):
        ################ Expectation step: ######################
        user_slices = rec_eval.user_idx_generator(n_users, batch_users=5000)
        print 'GENERATING NEGATIVE INSTANCES ...'
        t1 = time.time()
        df = joblib.Parallel(n_jobs=16)(
            joblib.delayed(gen_neg_instances)(train_data, U, VT, user_idx, neg_ratio=NEGATIVE_SAMPLE_RATIO, iter=iter, vad_data=vad_data)
            for user_idx in user_slices)
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)

        print 'merging to one file ...'
        t1 = time.time()
        neg_file_out = os.path.join(DATA_DIR, 'train_neg_iter_%d.csv' % (iter))
        with open(neg_file_out, 'w') as writer:
            writer.write('userId,movieId\n')
        # os.system("echo uid,sid >> " + neg_file_out)
        for f in glob.glob(os.path.join(DATA_DIR, 'sub_dataframe_iter*')):
            os.system("cat " + f + " >> " + neg_file_out)
            # with open(f, 'rb') as reader:
            #
            #     writer.write(reader.readline())
            # writer.flush()
        # clean
        for f in glob.glob(os.path.join(DATA_DIR, 'sub_dataframe_iter*')):
            os.remove(f)

        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)
        # neg_train_df = pd.concat(df)
        # neg_train_df.to_csv(neg_file_out, index = False)
        #########################################################

        ################ maximization step:######################
        print 'GENERATING NEGATIVE EMBEDDINGS ...'
        t1 = time.time()
        train_neg_data, _, train_neg_df = helper_methods.load_data(neg_file_out, shape=(n_users, n_items))
        # build the negative info:
        X_neg, _ = pne.produce_neg_embeddings(DATA_DIR, train_neg_data, n_users, n_items, iter=iter)
        X_neg_sppmi = helper_methods.convert_to_SPPMI_matrix(X_neg, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
        Y_neg_sppmi = None
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)

        # build the model
        print 'build the model...'
        t1 = time.time()
        runner = model_runner.ModelRunner(train_data, vad_data, None, X_sppmi, X_neg_sppmi, Y_sppmi, None, save_dir=save_dir)
        U, V, ndcg100 = runner.run("rme", n_jobs=1,
                                   lam=lam, lam_emb=lam_emb, n_components=n_components, ret_params_only=1)
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)
        print '*************************************ITER %d ******************************************' % iter
        print 'NDCG@100 at this iter:', ndcg100
        #
        if best_ndcg100 < ndcg100:
            best_iter = iter
            best_ndcg100 = ndcg100
        else:
            early_stopping = True
        iter += 1
        #########################################################
    print 'Max NDCG@100: %f , at iter: %d' % (best_ndcg100, best_iter)
    best_train_neg_file = os.path.join(DATA_DIR, 'train_neg_iter_%d.csv' % (best_iter))
    best_train_neg_file_newname = os.path.join(DATA_DIR, 'train_neg.csv')
    best_train_emb_file = os.path.join(DATA_DIR, 'negative_item_item_cooc_iter%d.dat' % (best_iter))
    best_train_emb_file_newname = os.path.join(DATA_DIR, 'negative_item_item_cooc.dat')
    print 'renaming from %s to %s' % (best_train_neg_file, best_train_neg_file_newname)
    os.rename(best_train_neg_file, best_train_neg_file_newname)
    print 'renaming from %s to %s' % (best_train_emb_file, best_train_emb_file_newname)
    os.rename(best_train_emb_file, best_train_emb_file_newname)
    # cleaning
    for i in range(max_iter):
        if i == best_iter: continue
        if early_stopping and (i > best_iter + 1): break
        del_file = os.path.join(DATA_DIR, 'train_neg_iter_%d.csv' % (i))
        os.remove(del_file)
        del_file = os.path.join(DATA_DIR, 'negative_item_item_cooc_iter%d.dat' % (i))
        os.remove(del_file)
    return glob, os