import os
import matplotlib.pyplot as plt

def draw(step1, step2, dir, dir1, draw_path):
    color_rank = list(step2.iloc[:, 1].values)
    feature_name_rank = list(step2.iloc[:, 0].values)

    color = ['#DA70D6', '#f60c86', '#0000FF', '#2E8B57', '#880e4f', '#113a5d', '#a82ffc', '#1e1548',
             '#ff2e4c', '#D600FF', '#270B5A', '#395AC0', '#AAA500', '#8A008D', '#ffbe00', '#283e56', '#970747',
             '#7B68EE', '#FF6347', '#096001', '#800080', '#D2691E', '#6A5ACD',
             '#FFA500', '#D2691E']
    fig, ax = plt.subplots(figsize=(10, 2))

    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    j = 0
    xx = range(0, len(feature_name_rank))
    ax.set_xticks(xx)
    ax.set_xticklabels(feature_name_rank, rotation=45, ha='right')

    max_value = 0
    for i in range(len(feature_name_rank)):
        curr_color = color[color_rank[i] - 1]
        plt.vlines(i, min(step1.loc[:, feature_name_rank[i]].values),
                   max(step1.loc[:, feature_name_rank[i]].values),
                   linestyles="solid", colors=curr_color)
        plt.scatter(i, round(
            (min(step1.loc[:, feature_name_rank[i]].values) + max(step1.loc[:, feature_name_rank[i]].values)) / 2, 2),
                    color=curr_color)
        if max_value < max(step1.loc[:, feature_name_rank[i]].values):
            max_value = max(step1.loc[:, feature_name_rank[i]].values)

        ax.get_xticklabels()[i].set_color(curr_color)

    if dir1 == 'IFA_effort':
        dir1 = 'IFA'
    if dir1 == 'recall_effort':
        dir1 = 'Recall@20%'

    plt.title(dir1)
    plt.ylabel('Rankings')

    plt.ylim(0.5, max_value)
    plt.yticks(rotation=90)

    plt.subplots_adjust(left=0.07, bottom=0.52, right=0.99)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 保存图片
    draw_path1 = os.path.join(draw_path, "Data_Analysis_Results\RQ3")  + '/' + dir + '/'
    if not os.path.exists(draw_path1):
        os.makedirs(draw_path1)
    plt.savefig(os.path.join(draw_path1, dir + '-' + dir1 + '.pdf'))

