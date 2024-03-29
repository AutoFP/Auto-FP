import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

end2end_data = pd.DataFrame(data = [['Australian', 'LR', 0.8918, 'Auto-FP'],
                                   ['Australian', 'MLP', 0.8846, 'Auto-FP'],
                                   ['Australian', 'XGB', 0.8846, 'Auto-FP'],
                                   ['Australian', 'LR', 0.8727, 'HPO'],
                                   ['Australian', 'MLP', 0.8527, 'HPO'],
                                   ['Australian', 'XGB', 0.8974, 'HPO'],
                                   ['Australian', 'LR', 0.8709, 'TPOT-FP'],
                                   ['Australian', 'MLP', 0.8818, 'TPOT-FP'],
                                   ['Australian', 'XGB', 0.88, 'TPOT-FP'],
                                   ['Blood', 'LR', 0.795, 'Auto-FP'],
                                   ['Blood', 'MLP', 0.7766, 'Auto-FP'],
                                   ['Blood', 'XGB', 0.7916, 'Auto-FP'],
                                   ['Blood', 'LR', 0.7495, 'HPO'],
                                   ['Blood', 'MLP', 0.768, 'HPO'],
                                   ['Blood', 'XGB', 0.7899, 'HPO'],
                                   ['Blood', 'LR', 0.7478, 'TPOT-FP'],
                                   ['Blood', 'MLP', 0.7747, 'TPOT-FP'],
                                   ['Blood', 'XGB', 0.768, 'TPOT-FP'],
                                   ['Emotion', 'LR', 0.9301, 'Auto-FP'],
                                   ['Emotion', 'MLP', 0.9079, 'Auto-FP'],
                                   ['Emotion', 'XGB', 0.9073, 'Auto-FP'],
                                   ['Emotion', 'LR', 0.8709, 'HPO'],
                                   ['Emotion', 'MLP', 0.8999, 'HPO'],
                                   ['Emotion', 'XGB', 0.8677, 'HPO'],
                                   ['Emotion', 'LR', 0.8548, 'TPOT-FP'],
                                   ['Emotion', 'MLP', 0.8806, 'TPOT-FP'],
                                   ['Emotion', 'XGB', 0.8709, 'TPOT-FP'],
                                   ['Forex', 'LR', 0.6806, 'Auto-FP'],
                                   ['Forex', 'MLP', 0.649, 'Auto-FP'],
                                   ['Forex', 'XGB', 0.649, 'Auto-FP'],
                                   ['Forex', 'LR', 0.5174, 'HPO'],
                                   ['Forex', 'MLP', 0.5353, 'HPO'],
                                   ['Forex', 'XGB', 0.5923, 'HPO'],
                                   ['Forex', 'LR', 0.7041, 'TPOT-FP'],
                                   ['Forex', 'MLP', 0.551, 'TPOT-FP'],
                                   ['Forex', 'XGB', 0.616, 'TPOT-FP'],
                                   ['Heart', 'LR', 0.8897, 'Auto-FP'],
                                   ['Heart', 'MLP', 0.8775, 'Auto-FP'],
                                   ['Heart', 'XGB', 0.8775, 'Auto-FP'],
                                   ['Heart', 'LR', 0.7916, 'HPO'],
                                   ['Heart', 'MLP', 0.8041, 'HPO'],
                                   ['Heart', 'XGB', 0.8666, 'HPO'],
                                   ['Heart', 'LR', 0.8375, 'TPOT-FP'],
                                   ['Heart', 'MLP', 0.8333, 'TPOT-FP'],
                                   ['Heart', 'XGB', 0.8125, 'TPOT-FP'],
                                   ['Jasmine', 'LR', 0.8292, 'Auto-FP'],
                                   ['Jasmine', 'MLP', 0.8087, 'Auto-FP'],
                                   ['Jasmine', 'XGB', 0.8087, 'Auto-FP'],
                                   ['Jasmine', 'LR', 0.7756, 'HPO'],
                                   ['Jasmine', 'MLP', 0.7983, 'HPO'],
                                   ['Jasmine', 'XGB', 0.8301, 'HPO'],
                                   ['Jasmine', 'LR', 0.8058, 'TPOT-FP'],
                                   ['Jasmine', 'MLP', 0.8058, 'TPOT-FP'],
                                   ['Jasmine', 'XGB', 0.8197, 'TPOT-FP'],
                                   ['Madeline', 'LR', 0.8489, 'Auto-FP'],
                                   ['Madeline', 'MLP', 0.6524, 'Auto-FP'],
                                   ['Madeline', 'XGB', 0.6524, 'Auto-FP'],
                                   ['Madeline', 'LR', 0.5677, 'HPO'],
                                   ['Madeline', 'MLP', 0.5949, 'HPO'],
                                   ['Madeline', 'XGB', 0.8525, 'HPO'],
                                   ['Madeline', 'LR', 0.597, 'TPOT-FP'],
                                   ['Madeline', 'MLP', 0.6458, 'TPOT-FP'],
                                   ['Madeline', 'XGB', 0.8469, 'TPOT-FP'],
                                   ['Pd', 'LR', 0.9404, 'Auto-FP'],
                                   ['Pd', 'MLP', 0.9619, 'Auto-FP'],
                                   ['Pd', 'XGB', 0.9619, 'Auto-FP'],
                                   ['Pd', 'LR', 0.8166, 'HPO'],
                                   ['Pd', 'MLP', 0.7766, 'HPO'],
                                   ['Pd', 'XGB', 0.937, 'HPO'],
                                   ['Pd', 'LR', 0.925, 'TPOT-FP'],
                                   ['Pd', 'MLP', 0.9583, 'TPOT-FP'],
                                   ['Pd', 'XGB', 0.9449, 'TPOT-FP'],
                                   ['Thyroid', 'LR', 0.7178, 'Auto-FP'],
                                   ['Thyroid', 'MLP', 0.7464, 'Auto-FP'],
                                   ['Thyroid', 'XGB', 0.7464, 'Auto-FP'],
                                   ['Thyroid', 'LR', 0.689, 'HPO'],
                                   ['Thyroid', 'MLP', 0.7342, 'HPO'],
                                   ['Thyroid', 'XGB', 0.7485, 'HPO'],
                                   ['Thyroid', 'LR', 0.7526, 'TPOT-FP'],
                                   ['Thyroid', 'MLP', 0.7469, 'TPOT-FP'],
                                   ['Thyroid', 'XGB', 0.7181, 'TPOT-FP'],
                                   ['Wine', 'LR', 0.65, 'Auto-FP'],
                                   ['Wine', 'MLP', 0.5896, 'Auto-FP'],
                                   ['Wine', 'XGB', 0.5896, 'Auto-FP'],
                                   ['Wine', 'LR', 0.4812, 'HPO'],
                                   ['Wine', 'MLP', 0.538, 'HPO'],
                                   ['Wine', 'XGB', 0.6509, 'HPO'],
                                   ['Wine', 'LR', 0.5572, 'TPOT-FP'],
                                   ['Wine', 'MLP', 0.5814, 'TPOT-FP'],
                                   ['Wine', 'XGB', 0.6543, 'TPOT-FP']],
                         columns = ['dataset', 'classifier', 'Accuracy', 'pipeline'])

sns.set_theme(style="ticks", font_scale=3)
end2end_fig = sns.catplot(x='classifier', y='Accuracy', hue='pipeline', col='dataset', palette="pastel",
                          legend=False, col_wrap=5, data=end2end_data, kind='bar', height=4, aspect=2)
end2end_fig.set_titles("{col_name}")
end2end_fig.set_axis_labels("")

for ax in end2end_fig.axes:
    for patch in ax.patches:
        patch.set_width(0.2)

plt.legend(loc='lower center', bbox_to_anchor=(-1.65, -0.9), ncol=3)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.savefig('end2end.eps')
#plt.show()
