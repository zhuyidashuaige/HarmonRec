import os
import time
import torch
import argparse

from model import HarmonRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='music4all_onion', required=True)
parser.add_argument('--train_dir', default='v7', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=500, type=int)
parser.add_argument('--hidden_units', default=128, type=int)  # Transformer hidden dimension (kept in sync with the multimodal target dimension)
parser.add_argument('--mm_dim', default=128, type=int)        # Unified multimodal target dimension (id / content / semantic features are all projected here)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--semantic_id_path', default=None, type=str)
parser.add_argument('--text_semantic_id_path', default="data/text_semantic_id.txt", type=str)   # Text-modality semantic IDs (produced by RQ / clustering)
parser.add_argument('--audio_semantic_id_path', default="data/audio_semantic_id.txt", type=str)  # Audio-modality semantic IDs
parser.add_argument('--visual_semantic_id_path', default="data/visual_semantic_id.txt", type=str) # Visual-modality semantic IDs
parser.add_argument('--semantic_dims', default=None, type=str)
parser.add_argument('--codebook_width', default=512, type=int)
parser.add_argument('--num_hierarchies', default=3, type=int)
parser.add_argument('--temp', default=0.7, type=float)                           # Softmax temperature tau for the frequency-aware reweighting
parser.add_argument('--override_fusion_weights', default=None, type=str)           # Manually override the collaborative / content fusion weights (e.g., "0.5,0.5")
parser.add_argument('--override_modal_weights', default=None, type=str)            # Manually override the per-modality weights (e.g., "0.33,0.33,0.33")
parser.add_argument('--override_codebook_weights', default=None, type=str)         # Manually override the per-codebook-layer weights (e.g., "1,0,0")

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # Build u2i / i2u indices so that we can quickly look up which items a user
    # has interacted with and which users have interacted with a given item.
    u2i_index, i2u_index = build_index(args.dataset)

    # Train / validation / test split (last 2 interactions per user -> valid / test).
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg@10, val_recall@10) (test_ndcg@10, test_recall@10)\n')
    
    # Multi-process sampler that yields (uid, seq, pos, neg) tuples; see WarpSampler in utils.py.
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    item2code = None
    item2code_text = None
    item2code_audio = None
    item2code_visual = None
    item_content = None
    item_content_text = None
    item_content_audio = None
    item_content_visual = None
    codebook_sizes = None
    codebook_sizes_text = None
    codebook_sizes_audio = None
    codebook_sizes_visual = None
    code_dims = None

    # Sources of semantic IDs (one file per modality).
    text_path = args.text_semantic_id_path or args.semantic_id_path
    audio_path = args.audio_semantic_id_path
    visual_path = args.visual_semantic_id_path

    # Read 3-layer text-modality semantic IDs (one per line: "item_id \t c1,c2,c3").
    if text_path is not None:
        codes = []
        with open(text_path, 'r') as sf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    iid = int(parts[0])
                    if iid <= itemnum:
                        c_parts = parts[1].split(',')
                        if len(c_parts) >= 3:
                            c1 = int(c_parts[0])
                            c2 = int(c_parts[1])
                            c3 = int(c_parts[2])
                            codes.append((iid, c1, c2, c3))
        max1 = 0
        max2 = 0
        max3 = 0
        item2code_arr = np.zeros((itemnum+1, 3), dtype=np.int64)
        for iid, c1, c2, c3 in codes:
            item2code_arr[iid] = [c1, c2, c3]
            if c1 > max1: max1 = c1
            if c2 > max2: max2 = c2
            if c3 > max3: max3 = c3
        if args.codebook_width is not None:
            w = int(args.codebook_width)
            codebook_sizes_text = [max(max1, w), max(max2, w), max(max3, w)]
        else:
            codebook_sizes_text = [max1, max2, max3]
        item2code_text = torch.from_numpy(item2code_arr)

    # Read 3-layer audio-modality semantic IDs.
    if audio_path is not None:
        codes = []
        with open(audio_path, 'r') as sf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    iid = int(parts[0])
                    if iid <= itemnum:
                        c_parts = parts[1].split(',')
                        if len(c_parts) >= 3:
                            c1 = int(c_parts[0])
                            c2 = int(c_parts[1])
                            c3 = int(c_parts[2])
                            codes.append((iid, c1, c2, c3))
        max1 = 0
        max2 = 0
        max3 = 0
        item2code_arr = np.zeros((itemnum+1, 3), dtype=np.int64)
        for iid, c1, c2, c3 in codes:
            item2code_arr[iid] = [c1, c2, c3]
            if c1 > max1: max1 = c1
            if c2 > max2: max2 = c2
            if c3 > max3: max3 = c3
        if args.codebook_width is not None:
            w = int(args.codebook_width)
            codebook_sizes_audio = [max(max1, w), max(max2, w), max(max3, w)]
        else:
            codebook_sizes_audio = [max1, max2, max3]
        item2code_audio = torch.from_numpy(item2code_arr)

    # Read 3-layer visual-modality semantic IDs.
    if visual_path is not None:
        codes = []
        with open(visual_path, 'r') as sf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    iid = int(parts[0])
                    if iid <= itemnum:
                        c_parts = parts[1].split(',')
                        if len(c_parts) >= 3:
                            c1 = int(c_parts[0])
                            c2 = int(c_parts[1])
                            c3 = int(c_parts[2])
                            codes.append((iid, c1, c2, c3))
        max1 = 0
        max2 = 0
        max3 = 0
        item2code_arr = np.zeros((itemnum+1, 3), dtype=np.int64)
        for iid, c1, c2, c3 in codes:
            item2code_arr[iid] = [c1, c2, c3]
            if c1 > max1: max1 = c1
            if c2 > max2: max2 = c2
            if c3 > max3: max3 = c3
        if args.codebook_width is not None:
            w = int(args.codebook_width)
            codebook_sizes_visual = [max(max1, w), max(max2, w), max(max3, w)]
        else:
            codebook_sizes_visual = [max1, max2, max3]
        item2code_visual = torch.from_numpy(item2code_arr)
    # Load the per-modality raw content embeddings (.pt) regardless of whether semantic IDs exist.
    def load_pt_to_table(path):
        if path is None:
            return None
        loaded = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(loaded, torch.Tensor):
            arr = loaded.detach().cpu().numpy()
        elif isinstance(loaded, dict):
            if 'embs' in loaded and 'item_ids' in loaded:
                ids = np.array(loaded['item_ids'])
                embs = np.array(loaded['embs'])
                arr = np.concatenate([ids.reshape(-1,1), embs], axis=1)
            else:
                raise RuntimeError(f'Unsupported dict structure in {path}')
        else:
            arr = np.array(loaded)
        table = np.zeros((itemnum+1, arr.shape[1]-1), dtype=np.float32)
        for row in arr:
            iid = int(row[0])
            if iid <= itemnum:
                table[iid] = row[1:].astype(np.float32)
        return torch.from_numpy(table)

    if text_path is not None or audio_path is not None or visual_path is not None:
        if args.semantic_dims is not None:
            parts = [int(x) for x in args.semantic_dims.split(',')]
            if len(parts) == 3:
                code_dims = parts
    # Build the multimodal HarmonRec model: feed in per-modality semantic IDs
    # (text / audio / visual) and the i2u index for popularity statistics.
    model = HarmonRec(
        usernum, itemnum, args,
        item2code=item2code, item_content=item_content, codebook_sizes=codebook_sizes, code_dims=code_dims,
        item2code_text=item2code_text, item2code_audio=item2code_audio, item2code_visual=item2code_visual,
        codebook_sizes_text=codebook_sizes_text, codebook_sizes_audio=codebook_sizes_audio, codebook_sizes_visual=codebook_sizes_visual,
        code_dims_text=code_dims, code_dims_audio=code_dims, code_dims_visual=code_dims,
        item2users=i2u_index
    ).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # Fix padding embeddings to zero so they do not inject any signal.
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0


    model.train() # enable model training
    
    # Optionally resume training from an existing checkpoint.
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, Recall@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion: cross-entropy loss for in-batch negative sampling (paper Sec. III-F).
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_recall = 0.0, 0.0
    best_test_ndcg, best_test_recall = 0.0, 0.0
    patience = 10
    no_improve = 0
    T = 0.0
    t0 = time.time()
    best_ckpt_path = None
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break 
        for step in range(num_batch):
            # Inputs: user ids, sequence, positive samples (neg samples unused under in-batch CE).
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            # In-batch negative sampling forward: returns (N_valid, C) logits and (N_valid,) targets.
            logits, targets = model(u, seq, pos)
            adam_optimizer.zero_grad()
            # Cross-entropy loss over the in-batch candidate pool I_batch (paper Eq. 10).
            loss = ce_criterion(logits, targets)
            # Backward + optimizer step (with optional L2 regularization on item embeddings).
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.sum(param ** 2)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        # Run validation every 4 epochs.
        if epoch % 4 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_valid = evaluate_valid_union200(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, Recall@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1]))

            improved = t_valid[0] > best_val_ndcg
            if improved:
                best_val_ndcg = t_valid[0]
                best_val_recall = max(t_valid[1], best_val_recall)
                folder = args.dataset + '_' + args.train_dir
                fname = 'HarmonRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                best_ckpt_path = os.path.join(folder, fname)
                torch.save(model.state_dict(), best_ckpt_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print('Early stopping triggered. Best val NDCG@10: %.4f' % best_val_ndcg)
                    break

            f.write(str(epoch) + ' ' + str(t_valid) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'HarmonRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    if best_ckpt_path is not None:
        model.load_state_dict(torch.load(best_ckpt_path, map_location=torch.device(args.device)))
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('best test (NDCG@10: %.4f, Recall@10: %.4f)' % (t_test[0], t_test[1]))
        f.write('best_test ' + str(t_test) + '\n')
        f.flush()

        # ==================================================================
        # [Cold-start / Long-tail Evaluation]  --  Sec. IV-D / RQ4
        # Bucket items by popularity into head(20%) / mid(30%) / tail(50%)
        # and report Recall@5, Recall@10, NDCG@5, NDCG@10 per bucket.
        # ==================================================================
        print('\n[Cold-start / Long-tail Evaluation] running ...')
        longtail_results = evaluate_longtail(
            model, dataset, args, i2u_index,
            head_ratio=0.2, mid_ratio=0.3, ks=(5, 10)
        )
        print_longtail_table(longtail_results, ks=(5, 10))

        # write per-bucket results into log file for the paper table
        f.write('=== long_tail_eval (head_ratio=0.2, mid_ratio=0.3) ===\n')
        f.write('bucket_split %s\n' % str(longtail_results['bucket_split']))
        for b in ['head', 'mid', 'tail', 'overall']:
            r = longtail_results[b]
            line = 'longtail_%s n_users=%d ' % (b, r['n_users'])
            for k in (5, 10):
                line += 'R@%d=%.4f N@%d=%.4f ' % (k, r['Recall@%d' % k],
                                                  k, r['NDCG@%d' % k])
            f.write(line + '\n')
        f.flush()
    f.close()
    sampler.close()
    print("Done")
