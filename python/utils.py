import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):
    # Build the interaction index: read data/<dataset>.txt where each line is "u i",
    # and return user-to-items (u2i) and item-to-users (i2u) lookup tables.
    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    # Sample a random integer from [l, r) that is not in the set s.
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    # Sampler worker: pick a random user, build a (seq, pos, neg) triplet and push to the queue.
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        # Multi-process sampler: run sample_function in parallel and expose next_batch() / close().
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    # Train / val / test split: users with < 4 interactions go fully into training;
    # otherwise the last two interactions become valid / test, the rest are training.
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:                          # To be rigorous, the training set needs at least two data points to learn
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    # Test-set evaluation (full-sort): rank the ground-truth item against ALL items
    # the user has not interacted with; report NDCG@10 and Recall@10.
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    REC = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        if len(valid[u]) > 0:
            rated.add(valid[u][0])
        pos_item = test[u][0]
        item_idx = [pos_item]
        for i in range(1, itemnum + 1):
            if i == pos_item: 
                continue
            if i in rated:
                continue
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            REC += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, REC / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    # Validation evaluation with 100 sampled negatives; report NDCG@10 and Recall@10.
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    REC = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            REC += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, REC / valid_user

def evaluate_valid_union200(model, dataset, args):
    # Validation evaluation with 200 sampled negatives; report NDCG@10 and Recall@10.
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    valid_user = 0.0
    REC = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        cnt = 0
        while cnt < 200:
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            cnt += 1
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            REC += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG / valid_user, REC / valid_user


# ====================================================================
# [Cold-start / Long-tail Evaluation]  -- for Sec. IV-D (RQ4)
# Idea:
#   1) Sort items by training-side interaction count and split them into
#      three buckets: head (top 20%) / mid (next 30%) / tail (bottom 50%).
#   2) For every test user, assign that user to a bucket according to
#      which bucket the ground-truth test item falls into, and accumulate
#      Recall@K / NDCG@K within that bucket.
#   3) Use the same full-corpus ranking protocol as evaluate() (1 positive
#      vs. all un-interacted items), so the per-bucket gap directly reflects
#      the model's representation quality at different popularity levels,
#      which is aligned with Proposition 2 of Frequency-aware Rebalancing.
# Usage: call once in main.py after loading the best checkpoint.
# ====================================================================
def evaluate_longtail(model, dataset, args, item2users,
                      head_ratio=0.2, mid_ratio=0.3, ks=(5, 10)):
    """Bucketized evaluation for long-tail / cold-start analysis (§IV-D / RQ4).

    Args:
        model:        trained HarmonRec model exposing the `.predict()` interface.
        dataset:      [user_train, user_valid, user_test, usernum, itemnum].
        args:         argparse namespace, must contain `maxlen`.
        item2users:   list of lists, item2users[i] = list of users who interacted
                      with item i (i2u_index built in main.py).
        head_ratio:   fraction of items in the head bucket (default 0.2).
        mid_ratio:    fraction of items in the mid bucket (default 0.3).
                      tail_ratio is implied = 1 - head_ratio - mid_ratio.
        ks:           tuple of cutoffs to report, default (5, 10).

    Returns:
        dict with the structure:
        {
          'head': {'Recall@5': float, 'Recall@10': float,
                   'NDCG@5':   float, 'NDCG@10':   float,
                   'n_users':  int,   'n_items':   int},
          'mid':  {...},
          'tail': {...},
          'overall': {...},
          'bucket_split': {'head_items': int, 'mid_items': int, 'tail_items': int},
        }
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # ---------- Step 1: bucket items by popularity (training-side count) ----------
    # Use training interactions (item2users covers full interactions; if you want
    # strict train-only popularity, replace below with a per-train counter).
    item_freq = np.zeros(itemnum + 1, dtype=np.int64)
    for i in range(1, itemnum + 1):
        if i < len(item2users):
            item_freq[i] = len(item2users[i])

    # Sort item ids 1..itemnum by frequency desc (stable on ties via id).
    valid_ids = np.arange(1, itemnum + 1)
    order = np.argsort(-item_freq[1:itemnum + 1], kind='stable')
    sorted_ids = valid_ids[order]

    n_head = max(int(itemnum * head_ratio), 1)
    n_mid = max(int(itemnum * mid_ratio), 1)
    head_set = set(sorted_ids[:n_head].tolist())
    mid_set = set(sorted_ids[n_head:n_head + n_mid].tolist())
    tail_set = set(sorted_ids[n_head + n_mid:].tolist())

    print('[LongTail] bucket split: head=%d (%.1f%%)  mid=%d (%.1f%%)  tail=%d (%.1f%%)'
          % (len(head_set), 100.0 * len(head_set) / itemnum,
             len(mid_set), 100.0 * len(mid_set) / itemnum,
             len(tail_set), 100.0 * len(tail_set) / itemnum))

    def _which_bucket(iid):
        if iid in head_set: return 'head'
        if iid in mid_set:  return 'mid'
        return 'tail'

    # ---------- Step 2: per-bucket NDCG/Recall accumulators ----------
    buckets = ['head', 'mid', 'tail']
    stats = {b: {'NDCG': {k: 0.0 for k in ks},
                 'REC':  {k: 0.0 for k in ks},
                 'n':    0} for b in buckets}

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    # ---------- Step 3: standard full-corpus ranking (identical to evaluate()) ----------
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        if len(valid[u]) > 0:
            rated.add(valid[u][0])

        pos_item = test[u][0]
        bucket = _which_bucket(pos_item)

        item_idx = [pos_item]
        for i in range(1, itemnum + 1):
            if i == pos_item:
                continue
            if i in rated:
                continue
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        stats[bucket]['n'] += 1
        for k in ks:
            if rank < k:
                stats[bucket]['NDCG'][k] += 1.0 / np.log2(rank + 2)
                stats[bucket]['REC'][k] += 1.0

        total_n = sum(stats[b]['n'] for b in buckets)
        if total_n % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    print('')

    # ---------- Step 4: normalize & assemble result dict ----------
    results = {}
    overall = {'NDCG': {k: 0.0 for k in ks},
               'REC':  {k: 0.0 for k in ks},
               'n': 0}
    for b in buckets:
        n = max(stats[b]['n'], 1)
        results[b] = {}
        for k in ks:
            results[b]['Recall@%d' % k] = stats[b]['REC'][k] / n
            results[b]['NDCG@%d' % k]   = stats[b]['NDCG'][k] / n
        results[b]['n_users'] = stats[b]['n']
        results[b]['n_items'] = len(head_set if b == 'head' else (mid_set if b == 'mid' else tail_set))
        # aggregate for overall sanity-check
        overall['n'] += stats[b]['n']
        for k in ks:
            overall['NDCG'][k] += stats[b]['NDCG'][k]
            overall['REC'][k]  += stats[b]['REC'][k]

    n_all = max(overall['n'], 1)
    results['overall'] = {}
    for k in ks:
        results['overall']['Recall@%d' % k] = overall['REC'][k] / n_all
        results['overall']['NDCG@%d' % k]   = overall['NDCG'][k] / n_all
    results['overall']['n_users'] = overall['n']

    results['bucket_split'] = {
        'head_items': len(head_set),
        'mid_items':  len(mid_set),
        'tail_items': len(tail_set),
    }

    return results


def print_longtail_table(results, ks=(5, 10)):
    """Pretty-print a long-tail results dict (for log files / stdout)."""
    print('\n================ Cold-start / Long-tail Results ================')
    bs = results['bucket_split']
    print('Items per bucket -> head=%d  mid=%d  tail=%d' %
          (bs['head_items'], bs['mid_items'], bs['tail_items']))
    header = '%-7s | %-9s | ' % ('bucket', '#users')
    for k in ks:
        header += 'R@%d     N@%d     ' % (k, k)
    print(header)
    print('-' * len(header))
    for b in ['head', 'mid', 'tail', 'overall']:
        if b not in results:
            continue
        r = results[b]
        line = '%-7s | %-9d | ' % (b, r['n_users'])
        for k in ks:
            line += '%.4f  %.4f  ' % (r['Recall@%d' % k], r['NDCG@%d' % k])
        print(line)
    print('================================================================\n')

