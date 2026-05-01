"""Generate Figure 2-style analysis plots using original ai4ab utils."""
import sys
import os
import numpy as np
import json
import math

sys.path.insert(0, '/tmp/ai4ab/analysis')
from utils import DataLoader, ResultsPlotter

RESULTS_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/trial_daniel/models/train_1-4_val_5_test_6/Plate_6_260501_2136'
OUTPUT_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/trial_daniel/analysis_results'
PARAMS_DIR = os.path.join(OUTPUT_DIR, 'E_coli_params')

os.makedirs(OUTPUT_DIR, exist_ok=True)

class LocalDataLoader:
    def __init__(self, experiment='cross_val', params_dir=PARAMS_DIR):
        self.experiment = experiment
        self.params_dir = params_dir

    def _load_files(self, channels, replicate):
        feat_vecs = np.loadtxt(os.path.join(RESULTS_DIR, 'feat_vecs.txt'))
        labels = np.loadtxt(os.path.join(RESULTS_DIR, 'labels.txt'))
        preds = np.loadtxt(os.path.join(RESULTS_DIR, 'preds.txt'))
        test_outputs = np.loadtxt(os.path.join(RESULTS_DIR, 'test_outputs.txt'))
        return feat_vecs, labels, preds, test_outputs

    def load_files(self, channels_list, replicate_list):
        self.channels_list = channels_list
        feat_vecs = []
        labels = []
        preds = []
        test_outputs = []
        plate_id = []
        channel_id = []

        for ch_id, ch in enumerate(channels_list):
            for p_id, rep in enumerate(replicate_list):
                feat_vecs_, labels_, preds_, test_outputs_ = self._load_files(ch, rep)
                feat_vecs.append(feat_vecs_)
                labels.append(labels_)
                preds.append(preds_)
                test_outputs.append(test_outputs_)
                plate_id.append(np.ones_like(labels_) * p_id)
                channel_id.append(np.ones_like(labels_) * ch_id)

        self.feat_vecs = np.vstack(feat_vecs)
        self.labels = np.hstack(labels)
        self.preds = np.hstack(preds)
        self.test_outputs = np.vstack(test_outputs)
        self.plate_id = np.hstack(plate_id)
        self.channel_id = np.hstack(channel_id)
        self._get_labels(self.params_dir)

    def _load_labels_from_specs(self, params_dir):
        d = []
        for l in ['moa_dict', 'moa_dict_inv', 'dose_dict', 'classes', 'moa_classes', 'labels_srtd_by_moa', 'moa_labels_srtd']:
            with open(os.path.join(params_dir, f'{l}.json'), 'r') as f:
                d.append(json.load(f))
        return tuple(d)

    def _get_labels(self, params_dir):
        self.moa_dict, self.moa_dict_inv, self.dose_dict, self.classes, self.moa_classes, self.labels_srtd_by_moa, self.moa_labels_srtd = self._load_labels_from_specs(params_dir)
        self.moa_dict_w_dose = {k: (v, self.dose_dict[k.split('_')[1]] if k not in ['DMSO'] else 0) for k, v in self.moa_dict.items()}
        self.moa_to_num = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))
        self.label_to_name = dict(zip([i for i in range(len(self.classes))], self.classes))
        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]
        self.moa_labels = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.labels]
        self.moa_preds = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.preds]
        self.labels_as_name = [self.label_to_name[l].split('_')[0] for l in self.labels]
        self.moa_labels_as_name = [[self.moa_dict_w_dose[self.label_to_name[l]][0]][0] for l in self.labels]


class LocalResultsPlotter:
    def __init__(self, loader):
        self.loader = loader
        self.feat_vecs = self.loader.feat_vecs
        self.labels = self.loader.labels
        self.preds = self.loader.preds
        self.plate_id = self.loader.plate_id
        self.channel_id = self.loader.channel_id
        self.test_outputs = self.loader.test_outputs
        self.classes = self.loader.classes
        self.ch_name_list = self.loader.channels_list
        self.moa_classes = self.loader.moa_classes
        self.moa_dict = self.loader.moa_dict
        self.moa_dict_inv = self.loader.moa_dict_inv
        self.dose_dict = self.loader.dose_dict
        self.labels_srtd_by_moa = self.loader.labels_srtd_by_moa
        self.moa_labels_strd = self.loader.moa_labels_srtd
        self.moa_dict_w_dose = self.loader.moa_dict_w_dose
        self.moa_to_num = self.loader.moa_to_num
        self.label_to_name = self.loader.label_to_name
        self.mic_id = self.loader.mic_id
        self.moa_labels = self.loader.moa_labels
        self.moa_preds = self.loader.moa_preds
        self.labels_as_name = self.loader.labels_as_name
        self.moa_labels_as_name = self.loader.moa_labels_as_name

    def index(self, input_maps, input_choices):
        idx_list_ = []
        for maps, choices in zip(input_maps, input_choices):
            if isinstance(choices[0], str) and choices[0] in self.ch_name_list:
                choices = [self.ch_name_list.index(c) for c in choices]
            idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))
        return np.logical_and.reduce(idx_list_)

    def make_confusion_matrix(self, labels, preds, classes_true, classes_pred, save_name, mode='normalize', title='Confusion matrix', label_name='compound', title_fontsize=20, tick_fontsize=12, label_fontsize=14):
        from matplotlib.ticker import MultipleLocator

        if label_name == 'compound':
            label_dict = dict(zip([i for i in range(len(classes_true))], [self.labels_srtd_by_moa.index(c) for c in classes_true]))
            labels = [label_dict[l] for l in labels]
            preds = [label_dict[l] for l in preds]
            classes_true = self.labels_srtd_by_moa
            classes_pred = self.labels_srtd_by_moa

        elif label_name == 'MoA':
            label_dict = dict(zip([i for i in classes_true], [self.moa_labels_strd.index(c) for c in classes_true]))
            labels = [label_dict[l] for l in labels]
            preds = [label_dict[l] for l in preds]
            classes_true = self.moa_labels_strd
            classes_pred = self.moa_labels_strd

        if isinstance(labels[0], str):
            labels_dict = dict(zip(classes_true, [i for i in range(len(classes_true))]))
            labels = [labels_dict[l] for l in labels]
            preds = [labels_dict[l] for l in preds]

        def return_counts_array(labels, preds):
            counts_array = np.zeros((len(classes_pred), len(classes_pred)))
            for l in np.unique(labels):
                x = np.array(preds)[np.array(labels) == l]
                p = np.unique(x, return_counts=True)
                if p[0].size > 0:
                    for p_idx, p_val in zip(p[0], p[1]):
                        counts_array[l, p_idx] = p_val
            return counts_array

        counts_array = return_counts_array(labels, preds)
        counts_array_norm = np.zeros_like(counts_array)
        for i, row in enumerate(counts_array):
            counts_array_norm[i] = row / row.sum()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(counts_array_norm, cmap=plt.cm.Blues)
        plt.gca().xaxis.tick_bottom()

        ax.set_xticks(np.arange(len(classes_true)))
        ax.set_yticks(np.arange(len(classes_pred)))
        ax.set_xticklabels(classes_true, rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels(classes_pred, fontsize=tick_fontsize)

        for i in range(counts_array.shape[1]):
            for j in range(counts_array.shape[0]):
                c = counts_array[j, i]
                c_n = counts_array_norm[j, i]
                ax.text(i, j, f'{int(c)}', va='center', ha='center', c='black' if c_n < 0.5 else 'white')

        plt.title(title, fontsize=title_fontsize)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.xlabel(f'Predicted {label_name}', fontsize=label_fontsize)
        plt.ylabel(f'True {label_name}', fontsize=label_fontsize)
        plt.tight_layout()
        print('Saving to', save_name)
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches='tight')
        plt.close()

    def p_conditional(self, dose, channel, plate):
        from scipy import special
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        p_cmpd_and_dose = special.softmax(self.test_outputs[idx_list])
        idx_list_2 = self.index([[self.moa_dict_w_dose[c][1] for c in self.classes]], [[0, dose]])
        p_dose = (p_cmpd_and_dose[:, idx_list_2]).sum()
        p_cond = p_cmpd_and_dose[:, idx_list_2] / p_dose
        return p_cond, np.array(self.classes)[idx_list_2]

    def plot_cond_confusion_matrix(self, dose, channel, plate, save_name='cond_cmpd_conf_matrix.svg', save=False, title='Confusion matrix', **kwargs):
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]
        cond_classes_ = [c.split('_')[0] for c in cond_classes]
        self.make_confusion_matrix(cond_labels, cond_preds, cond_classes_, cond_classes_, save_name=save_name, title=title, **kwargs)

    def plot_cond_moa_confusion_matrix(self, dose, channel, plate, save_name='cond_moa_conf_matrix.svg', save=False, title='Confusion matrix', **kwargs):
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]
        moa_cond_dict = {k: v for k, v in self.moa_dict.items()}
        moa_cond_labels = [moa_cond_dict[cond_classes[l]] for l in cond_labels]
        moa_cond_preds = [moa_cond_dict[cond_classes[l]] for l in cond_preds]
        self.make_confusion_matrix(moa_cond_labels, moa_cond_preds, sorted(set(moa_cond_dict.values())), sorted(set(moa_cond_dict.values())), save_name=save_name, title=title, **kwargs)

    def _get_umap(self, data, n_components=2, n_neighbors=500, min_dist=1., metric='cosine'):
        import umap
        umap_ = umap.UMAP(n_components=n_components, random_state=1, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        umap_data = umap_.fit_transform(data)
        return umap_data

    def _get_median_vecs(self, dose=4, concatenate_vecs=True):
        feat_vecs_srtd_ = []
        feat_vecs_med_srtd_ = []
        labels_as_name_srtd_ = []
        labels_as_name_srtd_no_med_ = []
        for l in list(set(self.labels_as_name)):
            if l in ['DMSO', 'Water']:
                idx = self.index([self.mic_id, self.labels_as_name], [[0], [l]])
            else:
                idx = self.index([self.mic_id, self.labels_as_name], [[dose], [l]])
            feat_vecs_srtd_.append(self.feat_vecs[idx])
            feat_vecs_med_srtd_.append(np.median(self.feat_vecs[idx], axis=0))
            labels_as_name_srtd_.append(l)
            labels_as_name_srtd_no_med_.append([l] * self.feat_vecs[idx].shape[0])

        feat_vecs_srtd = np.vstack(feat_vecs_srtd_)
        feat_vecs_med_srtd = np.vstack(feat_vecs_med_srtd_)
        labels_as_name_srtd = np.hstack(labels_as_name_srtd_)
        labels_as_name_srtd_no_med = np.hstack(labels_as_name_srtd_no_med_)
        feat_vecs_srtd_concat = np.vstack([feat_vecs_srtd, feat_vecs_med_srtd])

        if concatenate_vecs:
            return feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med
        else:
            feat_vecs_med = []
            labels_med = []
            for l in self.labels_srtd_by_moa:
                l_idx = list(labels_as_name_srtd).index(l)
                feat_vecs_med.append(feat_vecs_med_srtd[l_idx])
                labels_med.append(l)
            feat_vecs_med = np.vstack(feat_vecs_med)
            return feat_vecs_med, labels_med

    def plot_umap(self, dose=4, n_components=2, n_neighbors=500, min_dist=1, metric='cosine', use_moa_labels=False, save_name='umap_plot.svg', title='UMAP'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib as mpl
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(colors=mpl.colormaps['tab20b'].colors + mpl.colormaps['tab20c'].colors)

        feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med = self._get_median_vecs(dose=dose)

        X = self._get_umap(feat_vecs_srtd_concat, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)

        X_no_med, X_med = X[:-labels_as_name_srtd.shape[0]], X[-labels_as_name_srtd.shape[0]:]

        if use_moa_labels:
            colour_list = ['gainsboro', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
            handles = []
            for l, c in zip(self.moa_classes, colour_list):
                handles.append(mpatches.Patch(color=c, label=l))

            plt.figure(figsize=(7, 7))
            for l in labels_as_name_srtd:
                idx = self.index([labels_as_name_srtd_no_med], [[l]])
                c = colour_list[self.moa_to_num[self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0]]]
                plt.scatter(X_no_med[idx][:, 0], X_no_med[idx][:, 1], s=100, alpha=0.2, c=c, edgecolor='grey')

            for l, x in zip(labels_as_name_srtd, X_med):
                c = colour_list[self.moa_to_num[self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0]]]
                plt.scatter(x[0], x[1], s=150, edgecolor='black', alpha=0.75, label=self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0], c=c)
        else:
            handles = []
            for l, c in enumerate(labels_as_name_srtd):
                handles.append(mpatches.Patch(color=cmap(int(l)), label=c))

            plt.figure(figsize=(7, 7))
            for l, c in enumerate(labels_as_name_srtd):
                idx = self.index([labels_as_name_srtd_no_med], [[c]])
                plt.scatter(X_no_med[idx][:, 0], X_no_med[idx][:, 1], s=100, alpha=0.2, color=cmap(int(l)), edgecolor='grey')

            for i, (l, x) in enumerate(zip(labels_as_name_srtd, X_med)):
                plt.scatter(x[0], x[1], s=150, edgecolor='black', alpha=0.75, label=l, color=cmap(int(i)))

        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title)
        plt.tight_layout()
        print('Saving to', save_name)
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches='tight')
        plt.close()

    def _cosine_similarity(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def _make_cosine_similarity_matrix(self, feat_vecs, labels):
        sim_matrix = np.empty((len(labels), len(labels)))
        for x, fvec1 in enumerate(feat_vecs):
            for y, fvec2 in enumerate(feat_vecs):
                sim_matrix[x, y] = self._cosine_similarity(fvec1, fvec2)
        return sim_matrix

    def plot_cosine_similarity_matrix(self, save_name='cosine_similarity.svg', title='Cosine similarity of feature vectors'):
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_std=False)

        feat_vecs_med, labels_med = self._get_median_vecs(concatenate_vecs=False)
        feat_vecs_med = scaler.fit_transform(feat_vecs_med)

        sim_matrix = self._make_cosine_similarity_matrix(feat_vecs_med, labels_med)

        plt.figure(figsize=(10, 10))
        plt.matshow(sim_matrix, cmap='coolwarm', fignum=0)
        plt.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.xticks([i for i in range(len(labels_med))], labels_med)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks([i for i in range(len(labels_med))], labels_med, fontsize=10)
        plt.title(title, fontsize=20)
        plt.tight_layout()
        print('Saving to', save_name)
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    print('Loading data...')
    loader = LocalDataLoader()
    loader.load_files(['BF', 'Hoechst_FM4_BF'], [1, 2, 3, 4])
    plotter = LocalResultsPlotter(loader)

    print('\n--- Confusion Matrices ---')
    ch = 'Hoechst_FM4_BF'
    r = 2
    d = 4
    d_name = '1xIC50'
    ch_name = ch.replace('_', '+')
    plotter.plot_cond_confusion_matrix(d, ch, r,
                                       save_name=f'cond_cmpd_conf_matrix_e_coli_dose_{d_name}_{ch}_rep_{r+1}.svg',
                                       save=True,
                                       title=f'{ch_name} ({d_name})',
                                       title_fontsize=20, tick_fontsize=16, label_fontsize=16)

    plotter.plot_cond_moa_confusion_matrix(d, ch, r,
                                           save_name=f'cond_moa_conf_matrix_e_coli_dose_{d_name}_{ch}_rep_{r+1}.svg',
                                           save=True,
                                           title=f'{ch_name} ({d_name})',
                                           title_fontsize=20, tick_fontsize=16, label_fontsize=16, label_name='MoA')

    print('\n--- UMAP Plots ---')
    loader2 = LocalDataLoader()
    loader2.load_files(['Hoechst_FM4_BF'], [3])
    plotter2 = LocalResultsPlotter(loader2)

    plotter2.plot_umap(title='E. coli by compound',
                       save_name='umap_e_coli_by_cmpd.svg')

    plotter2.plot_umap(title='E. coli by MoA',
                       save_name='umap_e_coli_by_moa.svg',
                       use_moa_labels=True)

    print('\n--- Cosine Similarity ---')
    plotter.plot_cosine_similarity_matrix(title='Cosine similarity of feature vectors',
                                           save_name='cosine_similarity_e_coli.svg')

    print('\n--- All plots saved to:', OUTPUT_DIR)