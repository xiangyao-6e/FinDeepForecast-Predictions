import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
from collections import defaultdict
import numpy as np

# --- 用于添加logo的库 ---
import cairosvg
import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- 设置全局字体 ---
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================================================================= #
# =================== 步骤 1: 数据与辅助函数 =================== #
# ================================================================= #

logo_map = {
    'GPT-5': 'logos/openai.svg',
    'Tongyi Deep Research': 'logos/tongyi.svg',
    'Deepseek-v3.2': 'logos/deepseek.svg',
    'Gemini 2.5 Pro': 'logos/gemini.svg',
    'Gemini 3 Pro': 'logos/gemini.svg',
    'GLM 4.6': 'logos/zhipu.svg',
    'GLM 4.7': 'logos/zhipu.svg',
    'Claude-Sonnet-4.5': 'logos/claude.svg',
    'Mistral Deep Research': 'logos/mistral.svg',
    'Grok 4': 'logos/grok.svg',
    'Perplexity Sonar Deep Research': 'logos/perplexity.svg',
    'OpenAI o3-deep-research': 'logos/openai.svg',
}

def get_base_model_name(model_name):
    """从完整的模型名称中提取出基础名称"""
    for key in logo_map.keys():
        if key.lower() in model_name.lower():
            return key
    return model_name

def add_logo_for_group(group_center_x, max_height, base_model_name, ax):
    """在指定的x坐标和y坐标（组内最高点）上方添加Logo"""
    logo_path = logo_map.get(base_model_name)
    if logo_path is None: return
    try:
        png_data = cairosvg.svg2png(url=logo_path, output_height=180)
        with io.BytesIO(png_data) as f:
            logo_img = plt.imread(f, format='png')
        imagebox = OffsetImage(logo_img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (group_center_x, max_height), xybox=(0., 25.), frameon=False, boxcoords="offset points", pad=0)
        ax.add_artist(ab)
    except FileNotFoundError:
        print(f"Logo文件未找到: {logo_path}")
    except Exception as e:
        print(f"为 {base_model_name} 添加Logo时出错: {e}")

# --- 原始数据 ---
# full_score = 350.0
full_score = 100.0
# data = [
#     ("Gemini 2.5 Pro (T)", 74.20, "LLM (Thinking)"),
#     ("Gemini 3 Pro (T)", 81.55, "LLM (Thinking)"),
#     ("GLM 4.6 (T)", 68.36, "LLM (Thinking)"),
#     ("GLM 4.7 (T)", 67.02, "LLM (Thinking)"),
#     ("Deepseek-v3.2 (T)", 67.32, "LLM (Thinking)"),
#     ("Claude-Sonnet-4.5 (T)", 74.56, "LLM (Thinking)"),
#     ("Grok 4 (T)", 76.57, "LLM (Thinking)"),
#     ("OpenAI GPT-5 (T)", 68.60, "LLM (Thinking)"),
#     ("Gemini 2.5 Pro (T+S)", 85.48, "LLM (Thinking + Search)"),
#     ("Gemini 3 Pro (T+S)", 114.52, "LLM (Thinking + Search)"),
#     ("GLM 4.6 (T+S)", 69.69, "LLM (Thinking + Search)"),
#     ("GLM 4.7 (T+S)", 69.26, "LLM (Thinking + Search)"),
#     ("Deepseek-v3.2 (T+S)", 56.12, "LLM (Thinking + Search)"),
#     ("Claude-Sonnet-4.5 (T+S)", 90.36, "LLM (Thinking + Search)"),
#     ("Grok 4 (T+S)", 77.27, "LLM (Thinking + Search)"),
#     ("OpenAI GPT-5 (T+S)", 115.48, "LLM (Thinking + Search)"),
#     ("Perplexity Sonar Deep Research", 86.54, "Deep Research"),
#     ("Mistral Deep Research", 61.82, "Deep Research"),
#     ("Tongyi Deep Research", 112.20, "Deep Research"),
#     ("Gemini 2.5 Pro Deep Research", 126.85, "Deep Research"),
#     ("Gemini 3 Pro Deep Research", 132.97, "Deep Research"),
#     ("Grok 4 DeepSearch", 130.58, "Deep Research"),
#     ("OpenAI o3-deep-research", 132.55, "Deep Research"),
# ]
data = [
    ("Gemini 2.5 Pro (T)", 20.8, "LLM (Thinking)"),
    # ("Gemini 3 Pro (T)", 43.9, "LLM (Thinking)"),
    ("Deepseek-v3.2 (T)", 21.9, "LLM (Thinking)"),
    ("Claude-Sonnet-4.5 (T)", 24.4, "LLM (Thinking)"),
    ("Grok 4 (T)", 23.7, "LLM (Thinking)"),
    ("OpenAI GPT-5 (T)", 25.0, "LLM (Thinking)"),
    ("Gemini 2.5 Pro (T+S)", 35.0, "LLM (Thinking + Search)"),
    # ("Gemini 3 Pro (T+S)", 58.3, "LLM (Thinking + Search)"),
    ("Deepseek-v3.2 (T+S)", 30.3, "LLM (Thinking + Search)"),
    ("Claude-Sonnet-4.5 (T+S)", 35.9, "LLM (Thinking + Search)"),
    ("Grok 4 (T+S)", 31.7, "LLM (Thinking + Search)"),
    ("OpenAI GPT-5 (T+S)", 36.0, "LLM (Thinking + Search)"),
    ("Perplexity Sonar Deep Research", 39.4, "Deep Research"),
    ("Tongyi Deep Research", 33.7, "Deep Research"),
    ("OpenAI o3-deep-research", 39.5, "Deep Research"),
]

# ================================================================= #
# =================== 步骤 2: 数据分组与排序 =================== #
# ================================================================= #

grouped_data = defaultdict(list)
for item in data:
    base_name = get_base_model_name(item[0])
    grouped_data[base_name].append(item)
    print(f"Assigned {item[0]} to group {base_name}")

if "DeepSeek" in grouped_data:
    grouped_data["DeepSeek-v3.2"] = grouped_data.pop("DeepSeek")
if "Grok" in grouped_data:
    grouped_data["Grok 4"] = grouped_data.pop("Grok")
if "Claude" in grouped_data:
    grouped_data["Claude-Sonnet-4.5"] = grouped_data.pop("Claude")
# if "Gemini 2.5 Pro" in grouped_data:
#     grouped_data["Gemini 2.5 Pro"] = grouped_data.pop("Gemini 2.5 Pro")
# if "Gemini 3 Pro" in grouped_data:
#     grouped_data["Gemini 3 Pro"] = grouped_data.pop("Gemini 3 Pro")
# if "GLM" in grouped_data:
#     grouped_data["GLM 4.6"] = grouped_data.pop("GLM")
# if "GLM" in grouped_data:
#     grouped_data["GLM 4.7"] = grouped_data.pop("GLM")
if "Perplexity" in grouped_data:
    grouped_data["Perplexity Sonar Deep Research"] = grouped_data.pop("Perplexity")
if "Mistral" in grouped_data:
    grouped_data["Mistral Deep Research"] = grouped_data.pop("Mistral")
if "Tongyi" in grouped_data:
    grouped_data["Tongyi Deep Research"] = grouped_data.pop("Tongyi")
# for item in grouped_data["GPT"]:
#     if "o3-deep-research" in item[0]:
#         grouped_data["GPT"].remove(item)
#         grouped_data["GPT o3-deep-research"] = [("GPT o3-deep-research", item[1], item[2])]
#         break
sorted_base_names = sorted(
    grouped_data.keys(),
    key=lambda name: max(item[1] for item in grouped_data[name]),
    reverse=False
)

# 颜色映射保持不变
base_color_map = {'LLM (Thinking)': "#00BAAD", 'LLM (Thinking + Search)': "#E8A107", 'Deep Research': "#61ABFF"}



# --- MODIFIED: 调整画布尺寸，使其更“高” ---
fig, ax = plt.subplots(figsize=(12, 9))
plot_linewidth = 1.0

# --- MODIFIED: 减小bar的宽度，使其更“瘦” ---
bar_width = 0.5

n_groups = len(sorted_base_names)
index = np.arange(n_groups)

# --- 修改：绘图主循环 ---
for i, base_name in enumerate(sorted_base_names):
    group = grouped_data[base_name]
    max_height_in_group = 0
    
    # --- 按分数降序排列组内数据 ---
    group.sort(key=lambda item: item[1], reverse=True)

    # ==================== 新增逻辑：为避让重叠做准备 ==================== #
    # last_label_y 用于追踪上一个标签的Y轴位置，初始设为无穷大
    last_label_y = float('inf')
    # min_vertical_distance 定义了标签之间的最小垂直间距（单位是数据坐标）
    # 你可以微调这个值来改变间距大小
    min_vertical_distance = 1.8
    # ================================================================ #

    for j, (model_name, score, category) in enumerate(group):
        normalized_score = score * 100 / full_score
        color = base_color_map[category]
        z = j + 1
        x_pos = index[i]
        
        # 绘制bar
        ax.bar(x_pos, normalized_score, bar_width, color=color, edgecolor='black', linewidth=plot_linewidth, zorder=z)
        
        # ==================== 修改后的代码：添加可自动避让的数值标签 ==================== #

        # 1. 获取bar的实际高度
        actual_y = normalized_score
        # 2. 确定标签的初始Y轴位置
        adjusted_y = actual_y

        # 3. 检查当前标签是否会与上一个标签重叠
        #    因为分数是降序的，所以 last_label_y >= adjusted_y
        if last_label_y - adjusted_y < min_vertical_distance:
            # 如果距离太近，则将当前标签向下移动到安全位置
            adjusted_y = last_label_y - min_vertical_distance

        # 4. 使用 annotate 函数来添加文本和连接线
        ax.annotate(f'{normalized_score:.1f}',           # 要显示的文本
            xy=(x_pos + bar_width / 2, actual_y),         # 箭头指向的位置 (bar的实际顶端)
            xycoords='data',                              # 'xy' 使用数据坐标系
            xytext=(x_pos + bar_width / 2 + 0.1, adjusted_y), # 文本放置的位置 (经过Y轴调整)
            textcoords='data',                            # 'xytext' 也使用数据坐标系
            ha='left',
            va='center',
            fontsize=10,
            # arrowprops 定义了从文本指向bar的连接线样式
            arrowprops=dict(
                arrowstyle='-',                           # 一条直线，没有箭头
                color='gray',                             # 线条颜色
                lw=0.8,                                   # 线条宽度
                shrinkA=4,                                # 线条起点离文本有4个点的距离
                shrinkB=2,                                # 线条终点离bar有2个点的距离
                connectionstyle="arc3,rad=0",             # 连接样式为直线
            ),
            zorder=10)

        # 5. 更新 last_label_y 的值为当前标签放置的位置，为下一个标签做准备
        last_label_y = adjusted_y
        # ======================= 修改后的代码结束 ======================= #

        if normalized_score > max_height_in_group:
            max_height_in_group = normalized_score
            
    # 在bar组的顶部添加Logo
    add_logo_for_group(index[i], max_height_in_group, base_name, ax)

# ================================================================= #
# ================= 步骤 4: 图表美化与收尾 =================== #
# ================================================================= #

ax.set_ylabel('Overall Accuracy', fontsize=24, labelpad=10)
ax.set_ylim(0, 40)

ax.set_xticks(index)
ax.set_xticklabels(sorted_base_names, rotation=45, ha='right', fontsize=22)
ax.tick_params(axis='x', which='major', pad=5)
ax.tick_params(axis='y', labelsize=22)

ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(plot_linewidth)
ax.spines['bottom'].set_linewidth(plot_linewidth)

# --- MODIFIED: 简化图例，因为分层逻辑不再与类别固定挂钩 ---
legend_handles = [
    mpatches.Patch(facecolor=base_color_map['LLM (Thinking)'], label='LLM (Thinking)'),
    mpatches.Patch(facecolor=base_color_map['LLM (Thinking + Search)'], label='LLM (Thinking + Search)'),
    mpatches.Patch(facecolor=base_color_map['Deep Research'], label='Deep Research')
]
ax.legend(
    handles=legend_handles,
    loc='lower center',             # 图例相对于 bbox_to_anchor 的对齐点
    bbox_to_anchor=(0.48, 1.1),     # (x, y): x=0.5 表示水平居中, y>1 表示图表上方
    ncol=3,                         # 图例横排显示
    frameon=False,
    fontsize=20
)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留上方空间
plt.savefig("output/overall-score-finforecast.png", format='png', bbox_inches='tight', dpi=800)
plt.show()