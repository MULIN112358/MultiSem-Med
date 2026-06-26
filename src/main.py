import os
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import args
from models import MultiSemMed
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)


# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        for adm_idx, adm in enumerate(input):
            target_output = model(input[: adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # predioction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=f"../data/{args.dataset}/output/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main():
    data_path = f"../data/{args.dataset}/output/records_final.pkl"
    voc_path = f"../data/{args.dataset}/output/voc_final.pkl"
    ehr_adj_path = f"../data/{args.dataset}/output/ehr_adj_final.pkl"
    ddi_adj_path = f"../data/{args.dataset}/output/ddi_A_final.pkl"

    device = torch.device("cuda:{}".format(args.cuda))

    ehr_adj = dill.load(open(ehr_adj_path, "rb"))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = MultiSemMed(
        voc_size,
        ehr_adj,
        ddi_adj,
        emb_dim=args.dim,
        nhead=args.nhead,
        device=device,
        ddi_in_memory=args.ddi,
        dataset=args.dataset
    )

    if args.Test:
        model.load_state_dict(torch.load(open(f'../saved/{args.dataset}/{args.model_name}/{args.resume_path}', "rb")))
        model.to(device=device)
        tic = time.time()
        result = []
        for _ in range(10):
            test_sample = [data_test[i] for i in np.random.choice(
                len(data_test), round(len(data_test) * 0.1), replace=True
            )]
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, test_sample, voc_size, 0
            )
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f}\t{:.4f}\n".format(m, s)

        print(outstring)
        print("test time: {}".format(time.time() - tic))

        ja = [e[1] for e in result]
        sorted_arr = np.sort(ja)
        middle_values = sorted_arr[2:-2]
        ja_average = np.mean(middle_values)
        avg_f1 = [e[2] for e in result]
        sorted_arr = np.sort(avg_f1)
        middle_values = sorted_arr[2:-2]
        f1_average = np.mean(middle_values)
        prauc = [e[3] for e in result]
        sorted_arr = np.sort(prauc)
        middle_values = sorted_arr[2:-2]
        prauc_average = np.mean(middle_values)
        print()
        print(ja_average, f1_average, prauc_average)
        return

    model.to(device=device)
    print("parameters", get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    if not os.path.exists(f'../saved/{args.dataset}/{args.model_name}'):
        os.makedirs(f'../saved/{args.dataset}/{args.model_name}')

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0
    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))
        model.train()
        for step, input in enumerate(data_train):
            for idx, adm in enumerate(input):
                seq_input = input[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for id, item in enumerate(adm[2]):
                    loss_multi_target[0][id] = item
                target_output1, loss_ddi = model(seq_input)
                loss_bce = F.binary_cross_entropy_with_logits(
                    target_output1, torch.FloatTensor(loss_bce_target).to(device)
                )
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(target_output1),
                    torch.LongTensor(loss_multi_target).to(device),
                )

                if args.ddi:
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0
                    y_label = np.where(target_output1 == 1)[0]
                    current_ddi_rate = ddi_rate_score(
                        [[y_label]], path=f"../data/{args.dataset}/output/ddi_A_final.pkl"
                    )

                    loss_pred = (
                            torch.exp(-model.w_bce) * loss_bce +
                            torch.exp(-model.w_multi) * loss_multi +
                            (model.w_bce + model.w_multi)
                    )
                    ratio = current_ddi_rate / args.target_ddi
                    ratio = torch.tensor(ratio, device=device)
                    beta = torch.sigmoid(- (ratio - 1))
                    loss = beta * loss_pred + (1 - beta) * loss_ddi

                else:
                    loss = (
                            torch.exp(-model.w_bce) * loss_bce +
                            torch.exp(-model.w_multi) * loss_multi +
                            (model.w_bce + model.w_multi)
                    )

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, data_eval, voc_size, epoch
        )
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "../saved",
                    args.dataset,
                    args.model_name,
                    "Epoch_{}_JA_{:.4}_DDI_{:.4}.model".format(epoch, ja, ddi_rate),
                ),
                "wb",
            ),
        )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch))

    dill.dump(
        history,
        open(
            os.path.join(
                "../saved", args.dataset, args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()
